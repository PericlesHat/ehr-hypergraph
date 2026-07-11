from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import HypergraphConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter, softmax


def _glorot(tensor: Optional[Tensor]) -> None:
    if tensor is None:
        return
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    with torch.no_grad():
        tensor.uniform_(-stdv, stdv)


class MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if num_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(in_channels, out_channels)])
        else:
            layers = [nn.Linear(in_channels, hidden_channels)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.Linear(hidden_channels, out_channels))
            self.layers = nn.ModuleList(layers)

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x), inplace=True)
        return self.layers[-1](x)


class AllSetPMAConv(MessagePassing):
    """Pooling by multi-head attention for one hypergraph propagation direction."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 1,
        heads: int = 1,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads")
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.head_channels = hidden_channels // heads
        self.lin_key = nn.Linear(in_channels, hidden_channels)
        self.lin_value = nn.Linear(in_channels, hidden_channels)
        self.seed = Parameter(torch.empty(1, heads, self.head_channels))
        self.feed_forward = MLP(hidden_channels, hidden_channels, out_channels, num_layers)
        self.norm0 = nn.LayerNorm(hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _glorot(self.lin_key.weight)
        _glorot(self.lin_value.weight)
        nn.init.zeros_(self.lin_key.bias)
        nn.init.zeros_(self.lin_value.bias)
        nn.init.xavier_uniform_(self.seed)
        self.feed_forward.reset_parameters()
        self.norm0.reset_parameters()
        self.norm1.reset_parameters()

    def forward(
        self,
        x_src: Tensor,
        edge_index: Tensor,
        size: Tuple[int, int],
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        heads, channels = self.heads, self.head_channels
        key = self.lin_key(x_src).view(-1, heads, channels)
        value = self.lin_value(x_src).view(-1, heads, channels)
        alpha_src = (key * self.seed).sum(dim=-1)
        out = self.propagate(
            edge_index,
            x=(value, None),
            alpha=(alpha_src, None),
            edge_weight=edge_weight,
            size=size,
        )
        out = out + self.seed
        out = self.norm0(out.reshape(-1, self.hidden_channels))
        out = self.norm1(out + F.relu(self.feed_forward(out)))
        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, index: Tensor, ptr, size_i, edge_weight) -> Tensor:
        alpha = F.leaky_relu(alpha_j, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        msg = x_j * alpha.unsqueeze(-1)
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1, 1)
        return msg


class AllSetTransformerEncoder(nn.Module):
    """Minimal AllSetTransformer encoder using V->E->V->E for visit embeddings."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.v2e0 = AllSetPMAConv(hidden_dim, hidden_dim, hidden_dim)
        self.e2v0 = AllSetPMAConv(hidden_dim, hidden_dim, hidden_dim)
        self.v2e1 = AllSetPMAConv(hidden_dim, hidden_dim, hidden_dim)
        self.bn_e = nn.BatchNorm1d(hidden_dim)
        self.bn_v = nn.BatchNorm1d(hidden_dim)
        self.eps = 1e-5

    def _scale(self, x: Tensor) -> Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        return x / (self.eps + x.pow(2).sum(dim=-1).mean()).sqrt()

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_hyperedges: int) -> Tensor:
        num_nodes = x.size(0)
        rev = hyperedge_index.flip(0)
        e = self.v2e0(x, hyperedge_index, size=(num_nodes, num_hyperedges))
        e = self._scale(e)
        e_in = F.relu(self.bn_e(e))
        v = self.e2v0(e_in, rev, size=(num_hyperedges, num_nodes))
        v = self._scale(v)
        v_in = F.relu(self.bn_v(v))
        e = self.v2e1(v_in, hyperedge_index, size=(num_nodes, num_hyperedges))
        return self._scale(e)


class VisitHypergraphBackbone(nn.Module):
    """Visit-level hypergraph encoder with selectable PyG or AllSet backbone."""

    def __init__(self, hidden_dim: int, backbone: str = "pyg") -> None:
        super().__init__()
        self.backbone = backbone.lower()
        if self.backbone == "pyg":
            self.hg_conv = HypergraphConv(hidden_dim, hidden_dim)
        elif self.backbone == "allset":
            self.encoder = AllSetTransformerEncoder(hidden_dim)
        else:
            raise ValueError(f"Unknown hypergraph backbone: {backbone}")

    def forward(self, x_emb: Tensor, edge_index: Tensor, num_hyperedges: int) -> Tensor:
        if self.backbone == "pyg":
            x = self.hg_conv(x_emb, edge_index)
            return scatter(x[edge_index[0]], edge_index[1], dim=0, dim_size=num_hyperedges, reduce="mean")
        return self.encoder(x_emb, edge_index, num_hyperedges)


class DEC(nn.Module):
    """Deep Embedded Clustering head over patient embeddings."""

    def __init__(self, n_clusters: int, hidden_dim: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, hidden_dim))
        nn.init.xavier_normal_(self.cluster_centers)

    def forward(self, z: Tensor) -> Tensor:
        dist = torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_centers, 2), dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        return (q.t() / torch.sum(q, dim=1)).t()


class EHRHyg(nn.Module):
    """Longitudinal EHR model: visit hypergraph encoder + temporal Transformer."""

    def __init__(
        self,
        num_codes: int,
        num_outcomes: int,
        hidden_dim: int = 128,
        n_clusters: int = 5,
        hyg_backbone: str = "pyg",
    ) -> None:
        super().__init__()
        if num_codes < 1:
            raise ValueError("num_codes must include at least the padding index 0")
        self.code_emb = nn.Embedding(num_codes, hidden_dim, padding_idx=0)
        self.visit_hg = VisitHypergraphBackbone(hidden_dim, backbone=hyg_backbone)
        self.time_proj = nn.Linear(1, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.hidden_dim = hidden_dim
        self.hyg_backbone = hyg_backbone
        self.dec = DEC(n_clusters, hidden_dim)
        self.step_head = nn.Linear(hidden_dim, num_outcomes)

    def forward(self, batch: dict, device: torch.device) -> dict:
        batch_size = len(batch["codes"])
        max_len = max(len(patient) for patient in batch["codes"])
        x_seq = torch.zeros(batch_size, max_len, self.hidden_dim, device=device)
        emb_ids, local_nodes, hyper_edges, edge_map = [], [], [], []
        node_offset, edge_id = 0, 0

        for t in range(max_len):
            step_codes, step_patients = [], []
            for patient_index, patient_codes in enumerate(batch["codes"]):
                if t < len(patient_codes):
                    step_codes.append(patient_codes[t])
                    step_patients.append(patient_index)
            if not step_codes:
                continue

            flat_codes = [int(code) for visit in step_codes for code in visit]
            if not flat_codes:
                continue

            unique_codes = sorted(set(flat_codes))
            code_to_local = {code: i + node_offset for i, code in enumerate(unique_codes)}
            emb_ids.extend(unique_codes)

            for patient_index, visit_codes in zip(step_patients, step_codes):
                if not visit_codes:
                    continue
                for code in visit_codes:
                    local_nodes.append(code_to_local[int(code)])
                    hyper_edges.append(edge_id)
                edge_map.append((t, patient_index, edge_id))
                edge_id += 1
            node_offset += len(unique_codes)

        if emb_ids and edge_id > 0:
            x_emb = self.code_emb(torch.tensor(emb_ids, dtype=torch.long, device=device))
            edge_index = torch.tensor([local_nodes, hyper_edges], dtype=torch.long, device=device)
            visit_embeddings = self.visit_hg(x_emb, edge_index, num_hyperedges=edge_id)
            for t, patient_index, visit_edge_id in edge_map:
                x_seq[patient_index, t] = visit_embeddings[visit_edge_id]

        times = batch["times"].to(device)
        x_seq = x_seq + self.time_proj(torch.log(times.unsqueeze(-1) + 1.0))
        cls = self.cls_token.expand(batch_size, -1, -1)
        x_in = torch.cat((cls, x_seq), dim=1)
        pad_mask = torch.cat(
            (
                torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
                batch["mask"].to(device),
            ),
            dim=1,
        )

        seq_len = x_in.size(1)
        causal_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        if seq_len > 1:
            causal_mask[1:, 1:] = torch.triu(
                torch.ones(seq_len - 1, seq_len - 1, dtype=torch.bool, device=device),
                diagonal=1,
            )

        z = self.transformer(x_in, mask=causal_mask, src_key_padding_mask=pad_mask)
        z_patient = z[:, 0, :]
        z_visits = z[:, 1:, :]
        return {
            "z_patient": z_patient,
            "z_visits": z_visits,
            "q_subtype": self.dec(z_patient),
            "step_logits": self.step_head(z_visits),
        }


def cluster_balance_kl(q: Tensor) -> Tensor:
    k = q.shape[1]
    cluster_prop = q.mean(dim=0)
    uniform = torch.full_like(cluster_prop, 1.0 / float(k))
    return torch.sum(cluster_prop * torch.log((cluster_prop + 1e-10) / uniform))
