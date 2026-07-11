from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd


DEFAULT_FEATURE_TABLES = {
    "condition_occurrence": {
        "domain_id": "Condition",
        "concept_col": "condition_concept_id",
        "date_col": "condition_start_date",
    },
    "drug_exposure": {
        "domain_id": "Drug",
        "concept_col": "drug_concept_id",
        "date_col": "drug_exposure_start_date",
    },
    "procedure_occurrence": {
        "domain_id": "Procedure",
        "concept_col": "procedure_concept_id",
        "date_col": "procedure_date",
    },
    "measurement": {
        "domain_id": "Measurement",
        "concept_col": "measurement_concept_id",
        "date_col": "measurement_date",
    },
    "observation": {
        "domain_id": "Observation",
        "concept_col": "observation_concept_id",
        "date_col": "observation_date",
    },
}

VISIT_DATA_FILENAME = "visit_level_data.csv"
STEPWISE_LABEL_FILENAME = "visit_stepwise_labels.csv"


@dataclass(frozen=True)
class TableSpec:
    table_name: str
    domain_id: str
    concept_col: str
    date_col: str


def _read_csv_table(input_dir: Path, table_name: str, required: bool) -> Optional[pd.DataFrame]:
    candidates = [
        input_dir / f"{table_name}.csv",
        input_dir / f"{table_name.upper()}.csv",
        input_dir / f"{table_name}.csv.gz",
        input_dir / f"{table_name.upper()}.csv.gz",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    if required:
        expected = ", ".join(str(p.name) for p in candidates)
        raise FileNotFoundError(f"Missing required OMOP table {table_name}; expected one of: {expected}")
    return None


def _normalize_visit_table(visit: pd.DataFrame) -> pd.DataFrame:
    required = ["visit_occurrence_id", "person_id", "visit_start_date", "visit_end_date"]
    missing = [col for col in required if col not in visit.columns]
    if missing:
        raise ValueError(f"visit_occurrence is missing required columns: {missing}")
    out = visit.copy()
    out["visit_occurrence_id"] = out["visit_occurrence_id"].astype(str)
    out["person_id"] = out["person_id"].astype(int)
    out["visit_start_date"] = pd.to_datetime(out["visit_start_date"]).dt.date
    out["visit_end_date"] = pd.to_datetime(out["visit_end_date"]).dt.date
    out = out.sort_values(["person_id", "visit_start_date", "visit_occurrence_id"])
    first_dates = out.groupby("person_id")["visit_start_date"].transform("min")
    out["days_from_index"] = [
        int((visit_date - first_date).days)
        for visit_date, first_date in zip(out["visit_start_date"], first_dates)
    ]
    return out


def _assign_events_to_visits_by_date(events: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    assigned = []
    visit_groups = {
        int(pid): group.sort_values(["visit_start_date", "visit_occurrence_id"]).copy()
        for pid, group in visits.groupby("person_id")
    }
    for row in events.itertuples(index=False):
        person_visits = visit_groups.get(int(row.person_id))
        if person_visits is None or person_visits.empty:
            continue
        event_date: date = row.event_date
        candidates = person_visits[
            (person_visits["visit_start_date"] <= event_date)
            & (person_visits["visit_end_date"] >= event_date)
        ]
        if candidates.empty:
            continue
        selected = candidates.iloc[-1]
        item = row._asdict()
        item["visit_occurrence_id"] = str(selected["visit_occurrence_id"])
        assigned.append(item)
    return pd.DataFrame(assigned)


def _extract_events(
    table: pd.DataFrame,
    spec: TableSpec,
    known_visits: pd.DataFrame,
    assign_missing_visit_ids: bool,
) -> pd.DataFrame:
    required = ["person_id", spec.concept_col, spec.date_col]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise ValueError(f"{spec.table_name} is missing required columns: {missing}")

    columns = ["person_id", spec.concept_col, spec.date_col]
    if "visit_occurrence_id" in table.columns:
        columns.append("visit_occurrence_id")

    events = table[columns].copy()
    events = events.rename(
        columns={
            spec.concept_col: "concept_id",
            spec.date_col: "event_date",
        }
    )
    events["person_id"] = events["person_id"].astype(int)
    events["concept_id"] = pd.to_numeric(events["concept_id"], errors="coerce").fillna(0).astype(int)
    events = events[events["concept_id"] > 0]
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.date
    events = events.dropna(subset=["event_date"])
    events["domain_id"] = spec.domain_id
    events["source_table"] = spec.table_name

    if "visit_occurrence_id" in events.columns:
        events["visit_occurrence_id"] = events["visit_occurrence_id"].astype(str)
    else:
        events["visit_occurrence_id"] = ""

    events.loc[events["visit_occurrence_id"].isin(["", "nan", "None", "<NA>"]), "visit_occurrence_id"] = ""
    if assign_missing_visit_ids:
        with_visit = events[events["visit_occurrence_id"] != ""].copy()
        missing_visit = events[events["visit_occurrence_id"] == ""].copy()
        if not missing_visit.empty:
            assigned = _assign_events_to_visits_by_date(missing_visit, known_visits)
            if not assigned.empty:
                with_visit = pd.concat([with_visit, assigned], ignore_index=True)
        events = with_visit
    else:
        events = events[events["visit_occurrence_id"] != ""].copy()

    if events.empty:
        return events

    visit_keys = known_visits[["person_id", "visit_occurrence_id"]].drop_duplicates()
    events = events.merge(visit_keys, on=["person_id", "visit_occurrence_id"], how="inner")
    return events


def _load_feature_events(
    input_dir: Path,
    visit: pd.DataFrame,
    table_names: Iterable[str],
    assign_missing_visit_ids: bool,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    frames = []
    counts: Dict[str, int] = {}
    for table_name in table_names:
        if table_name not in DEFAULT_FEATURE_TABLES:
            raise ValueError(f"Unsupported feature table: {table_name}")
        raw = _read_csv_table(input_dir, table_name, required=False)
        if raw is None:
            counts[table_name] = 0
            continue
        info = DEFAULT_FEATURE_TABLES[table_name]
        spec = TableSpec(
            table_name=table_name,
            domain_id=str(info["domain_id"]),
            concept_col=str(info["concept_col"]),
            date_col=str(info["date_col"]),
        )
        events = _extract_events(raw, spec, visit, assign_missing_visit_ids=assign_missing_visit_ids)
        counts[table_name] = int(len(events))
        if not events.empty:
            frames.append(events)
    if not frames:
        return pd.DataFrame(columns=["person_id", "visit_occurrence_id", "concept_id", "event_date", "domain_id", "source_table"]), counts
    return pd.concat(frames, ignore_index=True), counts


def _load_outcome_concepts(path: Path) -> Dict[str, List[int]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, Mapping):
        raise ValueError("Outcome concept file must be a JSON object mapping label names to concept_id lists.")
    out: Dict[str, List[int]] = {}
    for label, values in raw.items():
        if not isinstance(values, list):
            raise ValueError(f"Outcome label {label} must map to a list of OMOP concept IDs.")
        out[str(label)] = [int(v) for v in values]
    return out


def _build_labels(
    input_dir: Path,
    visit: pd.DataFrame,
    outcome_concepts: Mapping[str, List[int]],
    label_mode: str,
    prediction_window_days: int,
) -> pd.DataFrame:
    condition = _read_csv_table(input_dir, "condition_occurrence", required=True)
    if condition is None:
        raise FileNotFoundError("condition_occurrence is required for outcome label construction.")
    required = ["person_id", "condition_concept_id", "condition_start_date"]
    missing = [col for col in required if col not in condition.columns]
    if missing:
        raise ValueError(f"condition_occurrence is missing required columns for labels: {missing}")

    condition = condition[required].copy()
    condition["person_id"] = condition["person_id"].astype(int)
    condition["condition_concept_id"] = pd.to_numeric(condition["condition_concept_id"], errors="coerce").fillna(0).astype(int)
    condition["condition_start_date"] = pd.to_datetime(condition["condition_start_date"], errors="coerce").dt.date
    condition = condition.dropna(subset=["condition_start_date"])

    label_dates: Dict[str, Dict[int, List[object]]] = {}
    for label, concept_ids in outcome_concepts.items():
        concept_set = set(int(c) for c in concept_ids)
        subset = condition[condition["condition_concept_id"].isin(concept_set)]
        label_dates[label] = {
            int(pid): sorted(group["condition_start_date"].tolist())
            for pid, group in subset.groupby("person_id")
        }

    rows = []
    for row in visit.itertuples(index=False):
        base = {
            "person_id": int(row.person_id),
            "visit_occurrence_id": str(row.visit_occurrence_id),
            "days_from_index": int(row.days_from_index),
        }
        current_date = row.visit_start_date
        for label in outcome_concepts.keys():
            dates = label_dates[label].get(int(row.person_id), [])
            if label_mode == "cumulative":
                value = any(d <= current_date for d in dates)
            elif label_mode == "future_window":
                value = any(
                    current_date < d <= current_date + pd.Timedelta(days=prediction_window_days).to_pytimedelta()
                    for d in dates
                )
            else:
                raise ValueError(f"Unsupported label mode: {label_mode}")
            base[label] = int(value)
        rows.append(base)
    return pd.DataFrame(rows)


def _build_concept_map(events: pd.DataFrame, concept: Optional[pd.DataFrame]) -> pd.DataFrame:
    unique = sorted(int(c) for c in events["concept_id"].dropna().unique().tolist())
    concept_map = pd.DataFrame({"concept_id": unique})
    concept_map["code_index"] = range(1, len(concept_map) + 1)

    if concept is not None and "concept_id" in concept.columns:
        keep_cols = [
            col
            for col in ["concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_code", "concept_class_id"]
            if col in concept.columns
        ]
        meta = concept[keep_cols].drop_duplicates("concept_id").copy()
        meta["concept_id"] = pd.to_numeric(meta["concept_id"], errors="coerce").fillna(0).astype(int)
        concept_map = concept_map.merge(meta, on="concept_id", how="left")

    event_domains = events[["concept_id", "domain_id"]].drop_duplicates("concept_id")
    concept_map = concept_map.merge(event_domains, on="concept_id", how="left", suffixes=("", "_from_event"))
    if "domain_id_from_event" in concept_map.columns:
        concept_map["domain_id"] = concept_map.get("domain_id").fillna(concept_map["domain_id_from_event"])
        concept_map = concept_map.drop(columns=["domain_id_from_event"])
    return concept_map[["code_index"] + [col for col in concept_map.columns if col != "code_index"]]


def preprocess_omop(
    input_dir: Path,
    out_dir: Path,
    outcome_concepts_path: Path,
    label_mode: str,
    prediction_window_days: int,
    feature_tables: List[str],
    exclude_outcome_concepts: bool,
    min_concepts_per_visit: int,
    assign_missing_visit_ids: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    visit = _normalize_visit_table(_read_csv_table(input_dir, "visit_occurrence", required=True))
    concept = _read_csv_table(input_dir, "concept", required=False)
    outcome_concepts = _load_outcome_concepts(outcome_concepts_path)

    events, table_counts = _load_feature_events(
        input_dir=input_dir,
        visit=visit,
        table_names=feature_tables,
        assign_missing_visit_ids=assign_missing_visit_ids,
    )

    outcome_concept_set = {int(c) for values in outcome_concepts.values() for c in values}
    if exclude_outcome_concepts and not events.empty:
        events = events[~events["concept_id"].isin(outcome_concept_set)].copy()

    if events.empty:
        raise ValueError("No feature events remained after loading and filtering OMOP event tables.")

    concept_map = _build_concept_map(events, concept)
    dense_lookup = dict(zip(concept_map["concept_id"].astype(int), concept_map["code_index"].astype(int)))
    events["code_index"] = events["concept_id"].map(dense_lookup).astype(int)

    feature_groups = (
        events.groupby(["person_id", "visit_occurrence_id"])["code_index"]
        .apply(lambda values: sorted(set(int(v) for v in values)))
        .reset_index()
    )
    feature_groups["visit_occurrence_id"] = feature_groups["visit_occurrence_id"].astype(str)

    visit_features = visit[["person_id", "visit_occurrence_id", "days_from_index"]].copy()
    visit_features = visit_features.merge(feature_groups, on=["person_id", "visit_occurrence_id"], how="left")
    visit_features["code_index"] = visit_features["code_index"].apply(lambda x: x if isinstance(x, list) else [])
    visit_features = visit_features.rename(columns={"code_index": "concepts"})
    if min_concepts_per_visit > 0:
        visit_features = visit_features[visit_features["concepts"].apply(len) >= min_concepts_per_visit].copy()
    visit_features["concepts"] = visit_features["concepts"].apply(lambda values: "[" + ", ".join(str(v) for v in values) + "]")

    labels = _build_labels(
        input_dir=input_dir,
        visit=visit,
        outcome_concepts=outcome_concepts,
        label_mode=label_mode,
        prediction_window_days=prediction_window_days,
    )
    labels = labels.merge(
        visit_features[["person_id", "visit_occurrence_id"]],
        on=["person_id", "visit_occurrence_id"],
        how="inner",
    )

    visit_features.to_csv(out_dir / VISIT_DATA_FILENAME, index=False)
    labels.to_csv(out_dir / STEPWISE_LABEL_FILENAME, index=False)
    concept_map.to_csv(out_dir / "concept_map.csv", index=False)

    table_summary = pd.DataFrame(
        [
            {
                "table_name": table_name,
                "loaded_events": int(table_counts.get(table_name, 0)),
                "events_per_visit": float(table_counts.get(table_name, 0) / max(len(visit), 1)),
            }
            for table_name in feature_tables
        ]
    )
    table_summary.to_csv(out_dir / "event_table_summary.csv", index=False)

    report = {
        "input_dir": input_dir.name,
        "n_patients": int(visit_features["person_id"].nunique()),
        "n_visits": int(len(visit_features)),
        "n_feature_events_after_filtering": int(len(events)),
        "n_dense_codes": int(len(concept_map)),
        "feature_tables": feature_tables,
        "event_table_counts_before_outcome_filter": table_counts,
        "exclude_outcome_concepts": bool(exclude_outcome_concepts),
        "label_mode": label_mode,
        "prediction_window_days": int(prediction_window_days),
        "label_prevalence": {
            label: float(labels[label].mean()) if label in labels.columns and len(labels) else 0.0
            for label in outcome_concepts.keys()
        },
        "outputs": {
            "visit_features": VISIT_DATA_FILENAME,
            "stepwise_labels": STEPWISE_LABEL_FILENAME,
            "concept_map": "concept_map.csv",
            "event_table_summary": "event_table_summary.csv",
        },
    }
    with (out_dir / "preprocess_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess OMOP CDM CSV tables into visit-code hypergraph inputs.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing OMOP CSV tables.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for model-ready CSV outputs.")
    parser.add_argument(
        "--outcome-concepts",
        type=Path,
        required=True,
        help="JSON mapping each outcome label to OMOP condition_concept_id values.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["future_window", "cumulative"],
        default="future_window",
        help="How to create stepwise labels at each visit.",
    )
    parser.add_argument(
        "--prediction-window-days",
        type=int,
        default=365,
        help="Future prediction window used when --label-mode future_window.",
    )
    parser.add_argument(
        "--feature-tables",
        nargs="+",
        default=list(DEFAULT_FEATURE_TABLES.keys()),
        help="OMOP event tables to convert into visit concepts.",
    )
    parser.add_argument(
        "--include-outcome-concepts",
        action="store_true",
        help="Keep target outcome concepts as input features. The default excludes them to reduce label leakage.",
    )
    parser.add_argument(
        "--min-concepts-per-visit",
        type=int,
        default=0,
        help="Drop visits with fewer than this many feature concepts after filtering.",
    )
    parser.add_argument(
        "--assign-missing-visit-ids",
        action="store_true",
        help="Assign events without visit_occurrence_id to a same-person visit interval using event dates.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    preprocess_omop(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        outcome_concepts_path=args.outcome_concepts,
        label_mode=args.label_mode,
        prediction_window_days=args.prediction_window_days,
        feature_tables=args.feature_tables,
        exclude_outcome_concepts=not args.include_outcome_concepts,
        min_concepts_per_visit=args.min_concepts_per_visit,
        assign_missing_visit_ids=args.assign_missing_visit_ids,
    )


if __name__ == "__main__":
    main()
