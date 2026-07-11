from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm, trange

from ehr_hypergraph.data.dataset import StepwiseDataset, collate_fn
from ehr_hypergraph.models.ehr_hyg import EHRHyg, cluster_balance_kl


HIDDEN_DIM = 128
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 10
DEC_WARMUP_EPOCHS = 5
DEC_KMEANS_N_INIT = 10
LOSS_WEIGHTS = {"step": 1.0, "dec": 0.1, "reg": 0.02}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def available_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    valid = array[np.isfinite(array)]
    return float(valid.mean()) if len(valid) else float("nan")


def safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def safe_auprc(labels: np.ndarray, probs: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, probs))


def macro_auc(labels: np.ndarray, probs: np.ndarray, outcome_keys: Iterable[str]) -> tuple[float, dict]:
    per_label = {
        key: safe_auroc(labels[:, j], probs[:, j])
        for j, key in enumerate(outcome_keys)
    }
    return available_mean(per_label.values()), per_label


def per_outcome_metrics(labels: np.ndarray, probs: np.ndarray, outcome_keys: List[str]) -> dict:
    metrics = {}
    for j, key in enumerate(outcome_keys):
        target = labels[:, j]
        prediction = probs[:, j]
        predicted_positive = prediction >= 0.5
        n_positive = int(target.sum())
        true_positive = int(((predicted_positive) & (target == 1)).sum())
        metrics[key] = {
            "n_visits": int(len(target)),
            "n_positive_visits": n_positive,
            "prevalence": float(target.mean()) if len(target) else float("nan"),
            "auroc": safe_auroc(target, prediction),
            "auprc": safe_auprc(target, prediction),
            "f1_at_0_5": float(f1_score(target, predicted_positive, zero_division=0)),
            "sensitivity_at_0_5": float(true_positive / n_positive) if n_positive else float("nan"),
        }
    return metrics


def calculate_loss(
    out: dict,
    batch: dict,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
    loss_w: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if loss_w is None:
        loss_w = {"step": 1.0, "dec": 0.1, "reg": 0.02}

    valid = (~batch["mask"].to(device)).float()
    per_elem = F.binary_cross_entropy_with_logits(
        out["step_logits"],
        batch["step_labels"].to(device),
        pos_weight=pos_weight,
        reduction="none",
    )
    per_visit = per_elem.mean(dim=-1)
    visit_counts = valid.sum(dim=1).clamp_min(1.0)
    per_patient = (per_visit * valid).sum(dim=1) / visit_counts
    loss_step = per_patient.mean()

    q = out["q_subtype"]
    p = (q**2 / q.sum(0)).t()
    p = (p / p.sum(0)).t().detach()
    loss_dec = F.kl_div(q.log(), p, reduction="batchmean")
    loss_reg = cluster_balance_kl(q)
    total = loss_w["step"] * loss_step + loss_w["dec"] * loss_dec + loss_w["reg"] * loss_reg
    return total, loss_step, loss_dec, loss_reg


def compute_full_metrics(
    logits_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    lengths_list: List[int],
    outcome_keys: List[str],
) -> dict:
    flat_logits, flat_labels = [], []
    for i, length in enumerate(lengths_list):
        flat_logits.append(logits_list[i][:length])
        flat_labels.append(labels_list[i][:length])
    logits = np.concatenate(flat_logits)
    labels = np.concatenate(flat_labels)
    probs = sigmoid_np(logits)
    per_label = per_outcome_metrics(labels, probs, outcome_keys)

    return {
        "visit_macro_auc": available_mean(item["auroc"] for item in per_label.values()),
        "visit_macro_auprc": available_mean(item["auprc"] for item in per_label.values()),
        "visit_macro_f1": available_mean(item["f1_at_0_5"] for item in per_label.values()),
        "visit_macro_sensitivity": available_mean(
            item["sensitivity_at_0_5"] for item in per_label.values()
        ),
        "visit_per_label": per_label,
    }


def evaluate(
    model: EHRHyg,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    loss_w: dict,
    outcome_keys: List[str],
    full: bool = False,
) -> dict:
    model.eval()
    total_loss, total_step, total_dec, total_reg = 0.0, 0.0, 0.0, 0.0
    logits_all, labels_all, lengths_all = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch["times"] = batch["times"].to(device)
            batch["mask"] = batch["mask"].to(device)
            out = model(batch, device)
            loss, l_step, l_dec, l_reg = calculate_loss(out, batch, device, pos_weight, loss_w)
            total_loss += loss.item()
            total_step += l_step.item()
            total_dec += l_dec.item()
            total_reg += l_reg.item()
            logits = out["step_logits"].cpu().numpy()
            labels = batch["step_labels"].numpy()
            for batch_index in range(logits.shape[0]):
                logits_all.append(logits[batch_index])
                labels_all.append(labels[batch_index])
                lengths_all.append(batch["lengths"][batch_index])

    n_batches = max(len(loader), 1)
    result = {
        "loss": total_loss / n_batches,
        "step_loss": total_step / n_batches,
        "dec_loss": total_dec / n_batches,
        "reg_loss": total_reg / n_batches,
    }

    if full:
        result.update(compute_full_metrics(logits_all, labels_all, lengths_all, outcome_keys))
    else:
        flat_logits, flat_labels = [], []
        for i, length in enumerate(lengths_all):
            flat_logits.append(logits_all[i][:length])
            flat_labels.append(labels_all[i][:length])
        macro, _ = macro_auc(np.concatenate(flat_labels), sigmoid_np(np.concatenate(flat_logits)), outcome_keys)
        result["visit_macro_auc"] = macro
    return result


def initialize_dec_centers(
    model: EHRHyg,
    loader: DataLoader,
    device: torch.device,
    n_clusters: int,
    n_init: int = 10,
) -> None:
    print("Initializing DEC cluster centers with KMeans ...", flush=True)
    model.eval()
    z_list = []
    with torch.no_grad():
        for batch in loader:
            batch["times"] = batch["times"].to(device)
            batch["mask"] = batch["mask"].to(device)
            z_list.append(model(batch, device)["z_patient"].cpu().numpy())
    z = np.concatenate(z_list)
    if z.shape[0] < n_clusters:
        raise ValueError(f"Need at least {n_clusters} patients to initialize DEC centers; got {z.shape[0]}.")
    km = KMeans(n_clusters=n_clusters, n_init=n_init).fit(z)
    model.dec.cluster_centers.data = torch.tensor(km.cluster_centers_, device=device, dtype=torch.float32)
    print("  done\n", flush=True)


def save_checkpoint(
    path: Path,
    model: EHRHyg,
    best_epoch: int,
    best_val_auc: float,
    num_codes: int,
    outcome_keys: List[str],
    n_clusters: int,
    hyg_backbone: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_auc": float(best_val_auc),
            "num_codes": num_codes,
            "outcome_keys": outcome_keys,
            "num_outcomes": len(outcome_keys),
            "n_clusters": n_clusters,
            "hyg_backbone": hyg_backbone,
        },
        path,
    )


def format_metric(value: float) -> str:
    return f"{value:.4f}" if np.isfinite(value) else "N/A"


def train_model(args: argparse.Namespace) -> tuple[EHRHyg, dict, dict, dict, StepwiseDataset, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_w = LOSS_WEIGHTS
    outcome_keys = args.outcomes.split(",") if args.outcomes else None
    dataset = StepwiseDataset(
        args.visit_file,
        args.label_file,
        outcome_keys=outcome_keys,
    )

    n = len(dataset)
    if n < 3:
        raise ValueError("Need at least 3 patients to create train/validation/test splits.")
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.1 * n))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train = n - 2
        n_val = 1
        n_test = 1

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(args.seed)
    )
    split_indices = {
        "train": list(train_ds.indices),
        "validation": list(val_ds.indices),
        "test": list(test_ds.indices),
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    model = EHRHyg(
        dataset.num_codes,
        num_outcomes=dataset.num_outcomes,
        hidden_dim=HIDDEN_DIM,
        n_clusters=args.n_clusters,
        hyg_backbone=args.backbone,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("Computing visit-level pos_weight ...", flush=True)
    train_labels = np.concatenate([dataset[i]["step_labels"] for i in train_ds.indices], axis=0)
    pos = train_labels.sum(axis=0)
    neg = train_labels.shape[0] - pos
    pos_weight = torch.tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32, device=device)
    print(f"  pos_weight = {[round(x, 2) for x in pos_weight.cpu().tolist()]}\n", flush=True)

    dec_initialized = False
    if DEC_WARMUP_EPOCHS <= 0:
        initialize_dec_centers(model, train_loader, device, args.n_clusters, DEC_KMEANS_N_INIT)
        dec_initialized = True
    else:
        print(
            f"DEC warmup enabled: first {DEC_WARMUP_EPOCHS} epoch(s) train without DEC/reg.\n",
            flush=True,
        )

    history = {key: [] for key in ["train_loss", "val_loss", "train_auc", "val_auc"]}
    best_auc, best_epoch, best_state, stale = float("nan"), -1, None, 0
    checkpoint_path = Path(args.output_dir) / "checkpoint.pt"

    for epoch in trange(args.epochs, desc="Training"):
        if (not dec_initialized) and epoch >= DEC_WARMUP_EPOCHS:
            initialize_dec_centers(model, train_loader, device, args.n_clusters, DEC_KMEANS_N_INIT)
            dec_initialized = True

        epoch_loss_w = loss_w if dec_initialized else {"step": loss_w["step"], "dec": 0.0, "reg": 0.0}
        model.train()
        epoch_loss = 0.0
        flat_preds, flat_labels = [], []

        for batch in tqdm(train_loader, desc=f"Ep {epoch + 1:02d}", leave=False):
            batch["times"] = batch["times"].to(device)
            batch["mask"] = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch, device)
            loss, _, _, _ = calculate_loss(out, batch, device, pos_weight, epoch_loss_w)
            loss.backward()
            if GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            logits = out["step_logits"].detach().cpu().numpy()
            labels = batch["step_labels"].numpy()
            for batch_index in range(logits.shape[0]):
                length = batch["lengths"][batch_index]
                flat_preds.append(logits[batch_index][:length])
                flat_labels.append(labels[batch_index][:length])

        train_probs = sigmoid_np(np.concatenate(flat_preds))
        train_auc, _ = macro_auc(np.concatenate(flat_labels), train_probs, dataset.outcome_keys)
        val_stats = evaluate(model, val_loader, device, pos_weight, epoch_loss_w, dataset.outcome_keys)
        history["train_loss"].append(epoch_loss / max(len(train_loader), 1))
        history["val_loss"].append(val_stats["loss"])
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_stats["visit_macro_auc"])

        val_auc = val_stats["visit_macro_auc"]
        improved = best_state is None or (
            np.isfinite(val_auc) and (not np.isfinite(best_auc) or val_auc > best_auc)
        )
        if improved:
            best_auc = val_auc
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            save_checkpoint(
                checkpoint_path,
                model,
                best_epoch,
                best_auc,
                dataset.num_codes,
                dataset.outcome_keys,
                args.n_clusters,
                args.backbone,
            )
            stale = 0
        else:
            stale += 1
            if stale >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stop at epoch {epoch + 1} "
                    f"(best={best_epoch}, validation AUROC={format_metric(best_auc)})",
                    flush=True,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"Loaded best checkpoint epoch {best_epoch} "
            f"(validation AUROC {format_metric(best_auc)})\n",
            flush=True,
        )

    if not dec_initialized:
        print("Training ended before DEC warmup; initializing DEC cluster centers for export ...", flush=True)
        initialize_dec_centers(model, train_loader, device, args.n_clusters, DEC_KMEANS_N_INIT)

    print("Running test evaluation ...", flush=True)
    test_stats = evaluate(model, test_loader, device, pos_weight, loss_w, dataset.outcome_keys, full=True)
    best_info = {"best_epoch": best_epoch, "best_val_auc": best_auc}
    return model, history, test_stats, best_info, dataset, split_indices


def allocate_run_directory(root: Path, run_name: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_name.strip())
    suffix = f"_{safe_name}" if safe_name else ""
    out_dir = root / "logs" / f"{stamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def shareable_config(args: argparse.Namespace) -> dict:
    config = vars(args).copy()
    config["visit_file"] = Path(str(config["visit_file"])).name
    config["label_file"] = Path(str(config["label_file"])).name
    config["output_dir"] = getattr(args, "log_subfolder", Path(str(config.get("output_dir", ""))).name)
    config.pop("experiment_root", None)
    config.pop("log_subfolder", None)
    config["fixed_training_settings"] = {
        "hidden_dim": HIDDEN_DIM,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "dec_warmup_epochs": DEC_WARMUP_EPOCHS,
        "dec_kmeans_n_init": DEC_KMEANS_N_INIT,
        "loss_weights": LOSS_WEIGHTS,
    }
    return config


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, ensure_ascii=False, indent=2)


def preprocess_metadata(label_file: str) -> dict:
    report_path = Path(label_file).parent / "preprocess_report.json"
    if not report_path.exists():
        return {}
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    return {
        key: report[key]
        for key in ("label_mode", "prediction_window_days", "exclude_outcome_concepts")
        if key in report
    }


def split_overview(dataset: StepwiseDataset, split_indices: Mapping[str, List[int]]) -> dict:
    overview = {}
    for split_name, indices in split_indices.items():
        overview[split_name] = {
            "n_patients": len(indices),
            "n_visits": int(sum(len(dataset[index]["codes"]) for index in indices)),
        }
    return overview


def write_training_history(history: Mapping[str, List[float]], out_dir: Path) -> None:
    history_frame = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "validation_loss": history["val_loss"],
            "train_macro_auroc": history["train_auc"],
            "validation_macro_auroc": history["val_auc"],
        }
    )
    history_frame.to_csv(out_dir / "training_history.csv", index=False)


def save_training_plots(history: Mapping[str, List[float]], out_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plots = [
        ("loss.png", "Loss", history["train_loss"], history["val_loss"], "Training and validation loss"),
        (
            "auc.png",
            "Macro AUROC",
            history["train_auc"],
            history["val_auc"],
            "Training and validation macro AUROC",
        ),
    ]
    for filename, ylabel, train_values, val_values, title in plots:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, train_values, marker="o", linewidth=1.8, label="Training")
        ax.plot(epochs, val_values, marker="o", linewidth=1.8, label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(epochs)
        if ylabel == "Macro AUROC":
            ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)


def write_outcome_metrics(test_stats: Mapping[str, Any], out_dir: Path) -> None:
    rows = []
    for outcome, metrics in test_stats["visit_per_label"].items():
        rows.append({"outcome": outcome, **metrics})
    pd.DataFrame(rows).to_csv(out_dir / "outcome_metrics.csv", index=False)


def write_cluster_outputs(
    model: EHRHyg,
    dataset: StepwiseDataset,
    split_indices: Mapping[str, List[int]],
    batch_size: int,
    device: torch.device,
    out_dir: Path,
) -> dict:
    index_to_split = {
        index: split_name
        for split_name, indices in split_indices.items()
        for index in indices
    }
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    rows = []
    model.eval()
    dataset_index = 0
    with torch.no_grad():
        for batch in loader:
            batch["times"] = batch["times"].to(device)
            batch["mask"] = batch["mask"].to(device)
            q = model(batch, device)["q_subtype"].cpu().numpy()
            for pid, probabilities in zip(batch["pids"], q):
                cluster = int(np.argmax(probabilities)) + 1
                row = {
                    "person_id": pid,
                    "split": index_to_split[dataset_index],
                    "cluster": cluster,
                    "assignment_probability": float(probabilities[cluster - 1]),
                }
                row.update(
                    {
                        f"cluster_{cluster_id}_probability": float(probability)
                        for cluster_id, probability in enumerate(probabilities, start=1)
                    }
                )
                rows.append(row)
                dataset_index += 1

    assignments = pd.DataFrame(rows)
    assignments.to_csv(out_dir / "patient_clusters.csv", index=False)
    summary_rows = []
    n_patients = max(len(assignments), 1)
    for cluster in range(1, model.dec.n_clusters + 1):
        subset = assignments[assignments["cluster"] == cluster]
        summary_rows.append(
            {
                "cluster": cluster,
                "n_patients": int(len(subset)),
                "patient_fraction": float(len(subset) / n_patients),
                "mean_assignment_probability": (
                    float(subset["assignment_probability"].mean()) if len(subset) else float("nan")
                ),
            }
        )
    cluster_summary = pd.DataFrame(summary_rows)
    cluster_summary.to_csv(out_dir / "cluster_summary.csv", index=False)
    return {
        "method": "Deep Embedded Clustering (DEC)",
        "assignment_population": "all modelled patients",
        "n_clusters_requested": int(model.dec.n_clusters),
        "n_nonempty_clusters": int((cluster_summary["n_patients"] > 0).sum()),
        "cluster_distribution": cluster_summary.to_dict(orient="records"),
        "cluster_summary_file": "cluster_summary.csv",
        "patient_assignments_file": "patient_clusters.csv",
    }


def build_summary(
    args: argparse.Namespace,
    dataset: StepwiseDataset,
    split_indices: Mapping[str, List[int]],
    best_info: Mapping[str, Any],
    test_stats: Mapping[str, Any],
    cluster_result: Mapping[str, Any],
) -> dict:
    task = {
        "evaluation_unit": "visit",
        "outcomes": dataset.outcome_keys,
        **preprocess_metadata(args.label_file),
    }
    return {
        "model": {
            "hypergraph_backbone": args.backbone,
            "hidden_dim": HIDDEN_DIM,
        },
        "cohort": {
            "n_patients": int(len(dataset)),
            "n_visits": int(sum(len(sample["codes"]) for sample in dataset.samples)),
            "patient_splits": split_overview(dataset, split_indices),
        },
        "prediction_task": task,
        "prediction_performance": {
            "selection_metric": "validation visit-level macro AUROC",
            "best_epoch": int(best_info["best_epoch"]),
            "best_validation_macro_auroc": best_info["best_val_auc"],
            "test_visit_macro_auroc": test_stats["visit_macro_auc"],
            "test_visit_macro_auprc": test_stats["visit_macro_auprc"],
            "test_visit_macro_f1_at_0_5": test_stats["visit_macro_f1"],
            "per_outcome_metrics_file": "outcome_metrics.csv",
        },
        "clustering": dict(cluster_result),
        "artifacts": {
            "training_history_file": "training_history.csv",
            "loss_plot": "loss.png",
            "auc_plot": "auc.png",
            "configuration_file": "config.json",
        },
    }


def print_run_summary(summary: Mapping[str, Any]) -> None:
    prediction = summary["prediction_performance"]
    clustering = summary["clustering"]
    print(f"\n{'=' * 65}")
    print("Run summary")
    print(f"  Test visit macro AUROC: {format_metric(prediction['test_visit_macro_auroc'])}")
    print(f"  Test visit macro AUPRC: {format_metric(prediction['test_visit_macro_auprc'])}")
    print(f"  Best validation AUROC:  {format_metric(prediction['best_validation_macro_auroc'])}")
    print(
        f"  DEC clusters:            {clustering['n_nonempty_clusters']} non-empty "
        f"of {clustering['n_clusters_requested']} requested"
    )
    print("  Detailed outcomes:       outcome_metrics.csv")
    print("  Cluster assignments:     patient_clusters.csv")
    print(f"{'=' * 65}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the packaged EHR hypergraph model.")
    parser.add_argument("--visit-file", required=True, help="Path to visit_level_data.csv.")
    parser.add_argument("--label-file", required=True, help="Path to visit_stepwise_labels.csv.")
    parser.add_argument("--outcomes", default="", help="Optional comma-separated outcome columns to train.")
    parser.add_argument("--backbone", choices=("allset", "pyg"), default="allset", help="Visit hypergraph encoder; allset is recommended.")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Patients per mini-batch.")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=40, help="Random seed for the patient split and training.")
    parser.add_argument("--output-dir", default="ehr_hypergraph/runs")
    parser.add_argument("--run-name", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    experiment_root = Path(args.output_dir).expanduser().resolve()
    out_dir = allocate_run_directory(experiment_root, args.run_name or args.backbone)
    args.output_dir = str(out_dir)
    args.log_subfolder = str(out_dir.relative_to(experiment_root))

    print(f"Run artifacts directory:\n  {out_dir}\n", flush=True)
    model, history, test_stats, best_info, dataset, split_indices = train_model(args)
    save_checkpoint(
        out_dir / "checkpoint.pt",
        model,
        best_info["best_epoch"],
        best_info["best_val_auc"],
        dataset.num_codes,
        dataset.outcome_keys,
        args.n_clusters,
        args.backbone,
    )

    device = next(model.parameters()).device
    write_training_history(history, out_dir)
    save_training_plots(history, out_dir)
    write_outcome_metrics(test_stats, out_dir)
    cluster_result = write_cluster_outputs(
        model,
        dataset,
        split_indices,
        args.batch_size,
        device,
        out_dir,
    )
    write_json(out_dir / "config.json", shareable_config(args))
    summary = build_summary(args, dataset, split_indices, best_info, test_stats, cluster_result)
    write_json(out_dir / "summary.json", summary)
    print_run_summary(summary)
    print(f"Saved summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
