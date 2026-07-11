from __future__ import annotations

import ast
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


ID_COLUMNS = {"person_id", "visit_occurrence_id", "days_from_index"}


def infer_outcome_keys(label_frame: pd.DataFrame) -> List[str]:
    """Infer outcome columns from a stepwise label CSV."""
    keys = [col for col in label_frame.columns if col not in ID_COLUMNS]
    if not keys:
        raise ValueError("No outcome columns were found in the label file.")
    return keys


def parse_concepts(value) -> List[int]:
    """Parse one visit-level concept list from CSV."""
    if isinstance(value, list):
        return [int(v) for v in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = ast.literal_eval(text)
        return [int(v) for v in parsed]
    except Exception:
        tokens = re.findall(r"-?\d+", text)
        return [int(v) for v in tokens]


class StepwiseDataset(Dataset):
    """Patient-level longitudinal dataset from preprocessed visit CSV files."""

    def __init__(
        self,
        visit_file: str | Path,
        label_file: str | Path,
        outcome_keys: Optional[Iterable[str]] = None,
        max_patients: int = 0,
        sample_seed: int = 40,
        verbose: bool = True,
    ) -> None:
        t0 = time.time()
        self.visit_file = Path(visit_file)
        self.label_file = Path(label_file)
        self.verbose = verbose

        self._log("Loading visit features ...")
        df_vis = pd.read_csv(self.visit_file)
        self._log(f"  {len(df_vis):,} visits ({time.time() - t0:.1f}s)")

        self._log("Loading stepwise labels ...")
        df_lab = pd.read_csv(self.label_file)
        self._log(f"  {len(df_lab):,} label rows ({time.time() - t0:.1f}s)")

        required_visit_cols = {"person_id", "visit_occurrence_id", "days_from_index", "concepts"}
        missing_visit = sorted(required_visit_cols - set(df_vis.columns))
        if missing_visit:
            raise ValueError(f"Visit file is missing required columns: {missing_visit}")

        required_label_cols = {"person_id", "visit_occurrence_id"}
        missing_label = sorted(required_label_cols - set(df_lab.columns))
        if missing_label:
            raise ValueError(f"Label file is missing required columns: {missing_label}")

        self.outcome_keys = list(outcome_keys) if outcome_keys is not None else infer_outcome_keys(df_lab)
        missing_outcomes = [key for key in self.outcome_keys if key not in df_lab.columns]
        if missing_outcomes:
            raise ValueError(f"Label file is missing requested outcomes: {missing_outcomes}")

        df_vis["visit_occurrence_id"] = df_vis["visit_occurrence_id"].astype(str)
        df_lab["visit_occurrence_id"] = df_lab["visit_occurrence_id"].astype(str)
        df = df_vis.merge(
            df_lab[["person_id", "visit_occurrence_id"] + self.outcome_keys],
            on=["person_id", "visit_occurrence_id"],
            how="inner",
        )
        self._log(f"  {len(df):,} merged rows ({time.time() - t0:.1f}s)")

        df["concepts"] = df["concepts"].apply(parse_concepts)
        df = df.sort_values(["person_id", "days_from_index", "visit_occurrence_id"])

        valid_pids = df["person_id"].unique().tolist()
        if max_patients and 0 < max_patients < len(valid_pids):
            rng = np.random.default_rng(sample_seed)
            valid_pids = rng.choice(np.array(valid_pids), size=max_patients, replace=False).tolist()
            df = df[df["person_id"].isin(set(valid_pids))]

        self.samples = []
        max_code_index = 0
        global_min_days = float(df["days_from_index"].min()) if len(df) else 0.0

        self._log("Building patient sequences ...")
        for pid, group in df.groupby("person_id"):
            codes = group["concepts"].tolist()
            labels = group[self.outcome_keys].values.astype(np.float32)
            times = (group["days_from_index"].values - global_min_days).astype(np.float32)
            visit_ids = group["visit_occurrence_id"].astype(str).tolist()
            days_from_index = group["days_from_index"].values.astype(np.float32)

            for visit_codes in codes:
                if visit_codes:
                    max_code_index = max(max_code_index, max(int(c) for c in visit_codes))

            self.samples.append(
                {
                    "pid": pid,
                    "codes": codes,
                    "times": times,
                    "step_labels": labels,
                    "visit_ids": visit_ids,
                    "days_from_index": days_from_index,
                }
            )

        self.max_code_index = int(max_code_index)
        self.num_codes = self.max_code_index + 1
        self.num_outcomes = len(self.outcome_keys)
        self._log(
            f"  {len(self.samples):,} patients | num_codes={self.num_codes} "
            f"| outcomes={self.num_outcomes} ({time.time() - t0:.1f}s)\n"
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: List[dict]) -> dict:
    """Pad patient visit sequences while keeping per-visit concept sets as lists."""
    pids, codes, times, labels, lengths = [], [], [], [], []
    visit_ids, days_from_index = [], []

    for sample in batch:
        pids.append(sample["pid"])
        codes.append(sample["codes"])
        times.append(torch.tensor(sample["times"], dtype=torch.float))
        labels.append(torch.tensor(sample["step_labels"], dtype=torch.float))
        lengths.append(len(sample["codes"]))
        visit_ids.append(list(sample.get("visit_ids", [])))
        dfi = sample.get("days_from_index")
        days_from_index.append(
            np.asarray(dfi, dtype=np.float32) if dfi is not None else np.zeros(0, dtype=np.float32)
        )

    padded_t = pad_sequence(times, batch_first=True, padding_value=0)
    padded_l = pad_sequence(labels, batch_first=True, padding_value=0)
    mask = torch.zeros(len(batch), padded_t.size(1), dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, length:] = True

    return {
        "pids": pids,
        "codes": codes,
        "times": padded_t,
        "step_labels": padded_l,
        "mask": mask,
        "lengths": lengths,
        "visit_ids": visit_ids,
        "days_from_index": days_from_index,
    }
