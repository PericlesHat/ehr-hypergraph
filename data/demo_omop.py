from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd


EVENT_TYPE_CONCEPT_ID = 32817
OBSERVATION_PERIOD_TYPE_CONCEPT_ID = 44814724


OUTCOME_CONDITION_NAMES: Mapping[str, List[str]] = {
    "CVD": [
        "Myocardial infarction outcome",
        "Heart failure outcome",
        "Coronary artery disease outcome",
    ],
    "Cerebrovascular": [
        "Ischemic stroke outcome",
        "Transient ischemic attack outcome",
    ],
    "CKD": [
        "Chronic kidney disease outcome",
        "End stage renal disease outcome",
    ],
    "Microvascular": [
        "Diabetic retinopathy outcome",
        "Diabetic neuropathy outcome",
    ],
    "FootAndSkin": [
        "Diabetic foot ulcer outcome",
        "Lower extremity cellulitis outcome",
    ],
    "DKA": [
        "Diabetic ketoacidosis outcome",
    ],
}


NON_TARGET_CONDITIONS = [
    "Essential hypertension",
    "Type 2 diabetes mellitus",
    "Hyperlipidemia",
    "Obesity",
    "Nicotine dependence",
    "Atrial fibrillation",
    "Peripheral vascular disease",
    "Chronic obstructive pulmonary disease",
    "Depressive disorder",
    "Anxiety disorder",
    "Osteoarthritis",
    "Sleep apnea",
    "Nonalcoholic fatty liver disease",
    "Hypothyroidism",
    "Gastroesophageal reflux disease",
    "Urinary tract infection",
    "Pneumonia",
    "Acute kidney injury",
    "Anemia",
    "Back pain",
    "Chest pain",
    "Shortness of breath",
    "Edema",
    "Proteinuria",
    "Abnormal glucose level",
    "Hyperkalemia",
    "Hypoglycemia",
    "Dehydration",
    "Visual disturbance",
    "Skin infection",
]


DRUGS = [
    "Metformin",
    "Insulin glargine",
    "Insulin lispro",
    "Atorvastatin",
    "Lisinopril",
    "Losartan",
    "Amlodipine",
    "Hydrochlorothiazide",
    "Aspirin",
    "Clopidogrel",
    "Furosemide",
    "Empagliflozin",
    "Semaglutide",
    "Gabapentin",
    "Cephalexin",
    "Amoxicillin clavulanate",
    "Albuterol",
    "Levothyroxine",
    "Omeprazole",
    "Sertraline",
]


PROCEDURES = [
    "Electrocardiogram",
    "Echocardiography",
    "Cardiac catheterization",
    "Retinal photography",
    "Foot examination",
    "Debridement of wound",
    "Dialysis procedure",
    "Kidney ultrasound",
    "Chest radiography",
    "Computed tomography head",
    "Influenza vaccination procedure",
    "Diabetes education",
]


MEASUREMENTS = [
    "Hemoglobin A1c",
    "Serum creatinine",
    "Estimated glomerular filtration rate",
    "Urine albumin creatinine ratio",
    "Low density lipoprotein cholesterol",
    "High density lipoprotein cholesterol",
    "Triglycerides",
    "Systolic blood pressure",
    "Diastolic blood pressure",
    "Body mass index",
    "Serum potassium",
    "Blood glucose",
    "Hemoglobin",
    "White blood cell count",
    "Troponin I",
]


OBSERVATIONS = [
    "Current smoker",
    "Former smoker",
    "Family history of cardiovascular disease",
    "Food insecurity screening",
    "Medication adherence concern",
    "Lives alone",
    "Limited English proficiency",
    "Exercise counseling",
]


VISIT_PROFILES = [
    {
        "visit_concept_id": 9202,
        "visit_name": "Outpatient Visit",
        "probability": 0.70,
        "length_of_stay_lambda": 0.0,
        "condition_mean": 3.4,
        "drug_mean": 1.6,
        "procedure_mean": 0.35,
        "measurement_mean": 2.0,
        "observation_mean": 0.25,
    },
    {
        "visit_concept_id": 9203,
        "visit_name": "Emergency Room Visit",
        "probability": 0.12,
        "length_of_stay_lambda": 0.0,
        "condition_mean": 5.2,
        "drug_mean": 1.8,
        "procedure_mean": 0.70,
        "measurement_mean": 3.5,
        "observation_mean": 0.20,
    },
    {
        "visit_concept_id": 9201,
        "visit_name": "Inpatient Visit",
        "probability": 0.10,
        "length_of_stay_lambda": 3.0,
        "condition_mean": 8.5,
        "drug_mean": 4.5,
        "procedure_mean": 1.8,
        "measurement_mean": 6.0,
        "observation_mean": 0.35,
    },
    {
        "visit_concept_id": 581458,
        "visit_name": "Telehealth Visit",
        "probability": 0.08,
        "length_of_stay_lambda": 0.0,
        "condition_mean": 2.2,
        "drug_mean": 1.1,
        "procedure_mean": 0.05,
        "measurement_mean": 0.4,
        "observation_mean": 0.45,
    },
]


def _as_date_string(value: date) -> str:
    return value.isoformat()


def _make_concepts() -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], Dict[str, List[int]]]:
    rows = []
    lookup: Dict[str, Dict[str, int]] = {
        "Condition": {},
        "Drug": {},
        "Procedure": {},
        "Measurement": {},
        "Observation": {},
        "Visit": {},
        "Type Concept": {},
    }

    def add(domain: str, name: str, concept_id: int, vocabulary: str, concept_class: str) -> None:
        lookup[domain][name] = concept_id
        rows.append(
            {
                "concept_id": concept_id,
                "concept_name": name,
                "domain_id": domain,
                "vocabulary_id": vocabulary,
                "concept_class_id": concept_class,
                "standard_concept": "S",
                "concept_code": f"DEMO-{concept_id}",
                "valid_start_date": "1970-01-01",
                "valid_end_date": "2099-12-31",
                "invalid_reason": "",
            }
        )

    for i, name in enumerate(NON_TARGET_CONDITIONS + [n for values in OUTCOME_CONDITION_NAMES.values() for n in values]):
        add("Condition", name, 1_000_000 + i, "DemoSNOMED", "Clinical Finding")
    for i, name in enumerate(DRUGS):
        add("Drug", name, 2_000_000 + i, "DemoRxNorm", "Clinical Drug")
    for i, name in enumerate(PROCEDURES):
        add("Procedure", name, 3_000_000 + i, "DemoCPT", "Procedure")
    for i, name in enumerate(MEASUREMENTS):
        add("Measurement", name, 4_000_000 + i, "DemoLOINC", "Lab Test")
    for i, name in enumerate(OBSERVATIONS):
        add("Observation", name, 5_000_000 + i, "DemoOMOP", "Observable Entity")
    for profile in VISIT_PROFILES:
        add("Visit", profile["visit_name"], int(profile["visit_concept_id"]), "Visit", "Visit")
    add("Type Concept", "EHR record", EVENT_TYPE_CONCEPT_ID, "Type Concept", "Type Concept")
    add("Type Concept", "EHR observation period", OBSERVATION_PERIOD_TYPE_CONCEPT_ID, "Type Concept", "Type Concept")

    outcome_concepts = {
        label: [lookup["Condition"][name] for name in names]
        for label, names in OUTCOME_CONDITION_NAMES.items()
    }
    return pd.DataFrame(rows), lookup, outcome_concepts


def _choice(rng: np.random.Generator, values: Iterable[str], size: int, replace: bool = False) -> List[str]:
    values = list(values)
    if size <= 0 or not values:
        return []
    size = min(size, len(values)) if not replace else size
    return rng.choice(np.array(values, dtype=object), size=size, replace=replace).tolist()


def _sample_visit_dates(
    rng: np.random.Generator,
    start_date: date,
    end_date: date,
    n_visits: int,
) -> List[date]:
    total_days = max((end_date - start_date).days, n_visits + 1)
    if n_visits <= 1:
        return [start_date + timedelta(days=int(total_days * 0.1))]
    raw = rng.gamma(shape=1.5, scale=1.0, size=n_visits)
    offsets = np.cumsum(raw)
    offsets = (offsets / offsets[-1] * (total_days - 1)).astype(int)
    jitter = rng.integers(-10, 11, size=n_visits)
    offsets = np.clip(offsets + jitter, 0, total_days - 1)
    offsets = np.unique(offsets)
    while len(offsets) < n_visits:
        offsets = np.unique(np.append(offsets, rng.integers(0, total_days)))
    offsets = np.sort(offsets[:n_visits])
    return [start_date + timedelta(days=int(offset)) for offset in offsets]


def _sample_profile(rng: np.random.Generator, high_acuity: bool) -> Mapping[str, object]:
    probabilities = np.array([float(p["probability"]) for p in VISIT_PROFILES], dtype=float)
    if high_acuity:
        probabilities = probabilities * np.array([0.85, 1.25, 1.55, 0.80])
    probabilities = probabilities / probabilities.sum()
    index = int(rng.choice(np.arange(len(VISIT_PROFILES)), p=probabilities))
    return VISIT_PROFILES[index]


def _add_condition(
    rows: List[dict],
    event_id: int,
    person_id: int,
    visit_occurrence_id: int,
    concept_id: int,
    event_date: date,
    source_prefix: str,
) -> int:
    rows.append(
        {
            "condition_occurrence_id": event_id,
            "person_id": person_id,
            "condition_concept_id": concept_id,
            "condition_start_date": _as_date_string(event_date),
            "condition_start_datetime": f"{_as_date_string(event_date)} 00:00:00",
            "condition_end_date": "",
            "condition_end_datetime": "",
            "condition_type_concept_id": EVENT_TYPE_CONCEPT_ID,
            "stop_reason": "",
            "provider_id": "",
            "visit_occurrence_id": visit_occurrence_id,
            "visit_detail_id": "",
            "condition_source_value": f"{source_prefix}-{concept_id}",
            "condition_source_concept_id": 0,
            "condition_status_source_value": "",
            "condition_status_concept_id": 0,
        }
    )
    return event_id + 1


def _add_drug(
    rows: List[dict],
    event_id: int,
    person_id: int,
    visit_occurrence_id: int,
    concept_id: int,
    start_date: date,
    days_supply: int,
) -> int:
    end_date = start_date + timedelta(days=max(days_supply - 1, 0))
    rows.append(
        {
            "drug_exposure_id": event_id,
            "person_id": person_id,
            "drug_concept_id": concept_id,
            "drug_exposure_start_date": _as_date_string(start_date),
            "drug_exposure_start_datetime": f"{_as_date_string(start_date)} 00:00:00",
            "drug_exposure_end_date": _as_date_string(end_date),
            "drug_exposure_end_datetime": f"{_as_date_string(end_date)} 00:00:00",
            "verbatim_end_date": "",
            "drug_type_concept_id": EVENT_TYPE_CONCEPT_ID,
            "stop_reason": "",
            "refills": 0,
            "quantity": "",
            "days_supply": days_supply,
            "sig": "",
            "route_concept_id": 0,
            "lot_number": "",
            "provider_id": "",
            "visit_occurrence_id": visit_occurrence_id,
            "visit_detail_id": "",
            "drug_source_value": f"DRUG-{concept_id}",
            "drug_source_concept_id": 0,
            "route_source_value": "",
            "dose_unit_source_value": "",
        }
    )
    return event_id + 1


def _add_procedure(
    rows: List[dict],
    event_id: int,
    person_id: int,
    visit_occurrence_id: int,
    concept_id: int,
    event_date: date,
) -> int:
    rows.append(
        {
            "procedure_occurrence_id": event_id,
            "person_id": person_id,
            "procedure_concept_id": concept_id,
            "procedure_date": _as_date_string(event_date),
            "procedure_datetime": f"{_as_date_string(event_date)} 00:00:00",
            "procedure_end_date": "",
            "procedure_end_datetime": "",
            "procedure_type_concept_id": EVENT_TYPE_CONCEPT_ID,
            "modifier_concept_id": 0,
            "quantity": 1,
            "provider_id": "",
            "visit_occurrence_id": visit_occurrence_id,
            "visit_detail_id": "",
            "procedure_source_value": f"PROC-{concept_id}",
            "procedure_source_concept_id": 0,
            "modifier_source_value": "",
        }
    )
    return event_id + 1


def _add_measurement(
    rows: List[dict],
    event_id: int,
    person_id: int,
    visit_occurrence_id: int,
    concept_id: int,
    event_date: date,
    value: float,
) -> int:
    rows.append(
        {
            "measurement_id": event_id,
            "person_id": person_id,
            "measurement_concept_id": concept_id,
            "measurement_date": _as_date_string(event_date),
            "measurement_datetime": f"{_as_date_string(event_date)} 00:00:00",
            "measurement_time": "",
            "measurement_type_concept_id": EVENT_TYPE_CONCEPT_ID,
            "operator_concept_id": 0,
            "value_as_number": round(float(value), 3),
            "value_as_concept_id": 0,
            "unit_concept_id": 0,
            "range_low": "",
            "range_high": "",
            "provider_id": "",
            "visit_occurrence_id": visit_occurrence_id,
            "visit_detail_id": "",
            "measurement_source_value": f"MEAS-{concept_id}",
            "measurement_source_concept_id": 0,
            "unit_source_value": "",
            "value_source_value": "",
        }
    )
    return event_id + 1


def _add_observation(
    rows: List[dict],
    event_id: int,
    person_id: int,
    visit_occurrence_id: int,
    concept_id: int,
    event_date: date,
) -> int:
    rows.append(
        {
            "observation_id": event_id,
            "person_id": person_id,
            "observation_concept_id": concept_id,
            "observation_date": _as_date_string(event_date),
            "observation_datetime": f"{_as_date_string(event_date)} 00:00:00",
            "observation_type_concept_id": EVENT_TYPE_CONCEPT_ID,
            "value_as_number": "",
            "value_as_string": "",
            "value_as_concept_id": 0,
            "qualifier_concept_id": 0,
            "unit_concept_id": 0,
            "provider_id": "",
            "visit_occurrence_id": visit_occurrence_id,
            "visit_detail_id": "",
            "observation_source_value": f"OBS-{concept_id}",
            "observation_source_concept_id": 0,
            "unit_source_value": "",
            "qualifier_source_value": "",
            "value_source_value": "",
        }
    )
    return event_id + 1


def generate_demo_omop(
    out_dir: Path,
    n_patients: int,
    seed: int,
    max_visits: int,
    condition_rate_scale: float,
    drug_rate_scale: float,
    procedure_rate_scale: float,
    measurement_rate_scale: float,
    observation_rate_scale: float,
) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    concept_df, concept_lookup, outcome_concepts = _make_concepts()
    condition_pool = NON_TARGET_CONDITIONS
    condition_weights = np.array(
        [1.6, 1.5, 1.3, 1.2, 0.8, 0.5, 0.5, 0.5, 0.8, 0.7, 0.7, 0.5, 0.6, 0.5, 0.7,
         0.8, 0.5, 0.5, 0.6, 0.8, 0.8, 0.8, 0.7, 0.5, 0.8, 0.4, 0.5, 0.4, 0.3, 0.4],
        dtype=float,
    )
    condition_weights = condition_weights / condition_weights.sum()

    person_rows = []
    observation_period_rows = []
    visit_rows = []
    condition_rows: List[dict] = []
    drug_rows: List[dict] = []
    procedure_rows: List[dict] = []
    measurement_rows: List[dict] = []
    observation_rows: List[dict] = []

    condition_id = 1
    drug_id = 1
    procedure_id = 1
    measurement_id = 1
    observation_id = 1
    visit_id = 1

    latent_groups = ["lower_risk", "cardiometabolic", "renal", "vascular", "fragile_high_use"]
    latent_probs = np.array([0.32, 0.30, 0.17, 0.13, 0.08])

    for person_id in range(1, n_patients + 1):
        group = str(rng.choice(latent_groups, p=latent_probs))
        birth_year = int(rng.integers(1935, 1996))
        gender_concept_id = int(rng.choice([8507, 8532], p=[0.49, 0.51]))
        start_year = int(rng.integers(2015, 2020))
        start = date(start_year, int(rng.integers(1, 13)), int(rng.integers(1, 28)))
        followup_days = int(rng.integers(900, 3000))
        end = min(start + timedelta(days=followup_days), date(2025, 12, 31))

        base_visits = int(rng.negative_binomial(2, 0.23) + 3)
        if group in {"cardiometabolic", "renal", "vascular"}:
            base_visits += int(rng.poisson(5))
        if group == "fragile_high_use":
            base_visits += int(rng.poisson(14))
        n_visits = int(np.clip(base_visits, 2, max_visits))
        visit_dates = _sample_visit_dates(rng, start, end, n_visits)

        person_rows.append(
            {
                "person_id": person_id,
                "gender_concept_id": gender_concept_id,
                "year_of_birth": birth_year,
                "month_of_birth": int(rng.integers(1, 13)),
                "day_of_birth": int(rng.integers(1, 28)),
                "birth_datetime": "",
                "race_concept_id": 0,
                "ethnicity_concept_id": 0,
                "location_id": "",
                "provider_id": "",
                "care_site_id": "",
                "person_source_value": f"DEMO-P{person_id:06d}",
                "gender_source_value": "F" if gender_concept_id == 8532 else "M",
                "gender_source_concept_id": 0,
                "race_source_value": "",
                "race_source_concept_id": 0,
                "ethnicity_source_value": "",
                "ethnicity_source_concept_id": 0,
            }
        )
        observation_period_rows.append(
            {
                "observation_period_id": person_id,
                "person_id": person_id,
                "observation_period_start_date": _as_date_string(start),
                "observation_period_end_date": _as_date_string(end),
                "period_type_concept_id": OBSERVATION_PERIOD_TYPE_CONCEPT_ID,
            }
        )

        outcome_plan: Dict[str, Tuple[int, int]] = {}
        for label, ids in outcome_concepts.items():
            base_risk = {
                "CVD": 0.16,
                "Cerebrovascular": 0.08,
                "CKD": 0.20,
                "Microvascular": 0.18,
                "FootAndSkin": 0.16,
                "DKA": 0.08,
            }[label]
            group_multiplier = {
                "lower_risk": 0.55,
                "cardiometabolic": 1.20,
                "renal": 1.55 if label in {"CKD", "Microvascular"} else 1.05,
                "vascular": 1.65 if label in {"CVD", "Cerebrovascular", "FootAndSkin"} else 1.10,
                "fragile_high_use": 1.85,
            }[group]
            risk = min(base_risk * group_multiplier, 0.72)
            if rng.random() < risk and n_visits > 2:
                onset_visit_index = int(rng.integers(max(1, n_visits // 5), n_visits))
                outcome_plan[label] = (int(rng.choice(ids)), onset_visit_index)

        chronic_conditions = set()
        chronic_seed_count = {
            "lower_risk": 2,
            "cardiometabolic": 5,
            "renal": 5,
            "vascular": 5,
            "fragile_high_use": 7,
        }[group]
        chronic_conditions.update(_choice(rng, condition_pool[:15], chronic_seed_count, replace=False))

        for visit_index, visit_date in enumerate(visit_dates):
            high_acuity = group in {"renal", "vascular", "fragile_high_use"}
            profile = _sample_profile(rng, high_acuity)
            los_lambda = float(profile["length_of_stay_lambda"])
            los_days = int(rng.poisson(los_lambda)) if los_lambda > 0 else 0
            visit_end = min(visit_date + timedelta(days=los_days), end)
            current_visit_id = visit_id
            visit_id += 1
            visit_rows.append(
                {
                    "visit_occurrence_id": current_visit_id,
                    "person_id": person_id,
                    "visit_concept_id": int(profile["visit_concept_id"]),
                    "visit_start_date": _as_date_string(visit_date),
                    "visit_start_datetime": f"{_as_date_string(visit_date)} 00:00:00",
                    "visit_end_date": _as_date_string(visit_end),
                    "visit_end_datetime": f"{_as_date_string(visit_end)} 00:00:00",
                    "visit_type_concept_id": EVENT_TYPE_CONCEPT_ID,
                    "provider_id": "",
                    "care_site_id": "",
                    "visit_source_value": str(profile["visit_name"]),
                    "visit_source_concept_id": 0,
                    "admitted_from_concept_id": 0,
                    "admitted_from_source_value": "",
                    "discharged_to_concept_id": 0,
                    "discharged_to_source_value": "",
                    "preceding_visit_occurrence_id": current_visit_id - 1 if visit_index > 0 else "",
                }
            )

            n_condition = int(rng.poisson(float(profile["condition_mean"]) * condition_rate_scale))
            sampled_conditions = _choice(
                rng,
                rng.choice(np.array(condition_pool, dtype=object), size=len(condition_pool), replace=False, p=condition_weights),
                n_condition,
            )
            sampled_conditions = list(dict.fromkeys(list(chronic_conditions) + sampled_conditions))
            for name in sampled_conditions:
                condition_id = _add_condition(
                    condition_rows,
                    condition_id,
                    person_id,
                    current_visit_id,
                    concept_lookup["Condition"][name],
                    visit_date,
                    "COND",
                )

            for label, (concept_id, onset_index) in outcome_plan.items():
                if visit_index == onset_index or (visit_index > onset_index and rng.random() < 0.18):
                    condition_id = _add_condition(
                        condition_rows,
                        condition_id,
                        person_id,
                        current_visit_id,
                        concept_id,
                        visit_date,
                        label.upper(),
                    )

            n_drug = int(rng.poisson(float(profile["drug_mean"]) * drug_rate_scale))
            if "Type 2 diabetes mellitus" in chronic_conditions:
                n_drug += int(rng.random() < 0.75)
            for name in _choice(rng, DRUGS, n_drug):
                drug_id = _add_drug(
                    drug_rows,
                    drug_id,
                    person_id,
                    current_visit_id,
                    concept_lookup["Drug"][name],
                    visit_date,
                    int(rng.choice([7, 14, 30, 60, 90], p=[0.10, 0.10, 0.50, 0.10, 0.20])),
                )

            n_procedure = int(rng.poisson(float(profile["procedure_mean"]) * procedure_rate_scale))
            for name in _choice(rng, PROCEDURES, n_procedure):
                procedure_id = _add_procedure(
                    procedure_rows,
                    procedure_id,
                    person_id,
                    current_visit_id,
                    concept_lookup["Procedure"][name],
                    visit_date,
                )

            n_measurement = int(rng.poisson(float(profile["measurement_mean"]) * measurement_rate_scale))
            for name in _choice(rng, MEASUREMENTS, n_measurement):
                value = rng.normal(0.0, 1.0)
                measurement_id = _add_measurement(
                    measurement_rows,
                    measurement_id,
                    person_id,
                    current_visit_id,
                    concept_lookup["Measurement"][name],
                    visit_date,
                    value,
                )

            n_observation = int(rng.poisson(float(profile["observation_mean"]) * observation_rate_scale))
            for name in _choice(rng, OBSERVATIONS, n_observation):
                observation_id = _add_observation(
                    observation_rows,
                    observation_id,
                    person_id,
                    current_visit_id,
                    concept_lookup["Observation"][name],
                    visit_date,
                )

    tables = {
        "person": pd.DataFrame(person_rows),
        "observation_period": pd.DataFrame(observation_period_rows),
        "visit_occurrence": pd.DataFrame(visit_rows),
        "condition_occurrence": pd.DataFrame(condition_rows),
        "drug_exposure": pd.DataFrame(drug_rows),
        "procedure_occurrence": pd.DataFrame(procedure_rows),
        "measurement": pd.DataFrame(measurement_rows),
        "observation": pd.DataFrame(observation_rows),
        "concept": concept_df,
        "cdm_source": pd.DataFrame(
            [
                {
                    "cdm_source_name": "Synthetic OMOP demo for EHR hypergraph experiments",
                    "cdm_source_abbreviation": "SYNHYG",
                    "cdm_holder": "Local demo",
                    "source_description": "Synthetic OMOP-like data with no real patients.",
                    "source_documentation_reference": "https://ohdsi.github.io/CommonDataModel/cdm54.html",
                    "cdm_etl_reference": "ehr_hypergraph.data.demo_omop",
                    "source_release_date": date.today().isoformat(),
                    "cdm_release_date": date.today().isoformat(),
                    "cdm_version": "5.4",
                    "cdm_version_concept_id": 0,
                    "vocabulary_version": "Synthetic demo vocabulary",
                }
            ]
        ),
    }

    for name, frame in tables.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)

    with (out_dir / "outcome_concepts.json").open("w", encoding="utf-8") as f:
        json.dump(outcome_concepts, f, indent=2)

    summary = {
        "n_patients": int(n_patients),
        "seed": int(seed),
        "tables": {name: int(len(frame)) for name, frame in tables.items()},
        "events_per_visit": {
            name: float(len(tables[name]) / max(len(tables["visit_occurrence"]), 1))
            for name in [
                "condition_occurrence",
                "drug_exposure",
                "procedure_occurrence",
                "measurement",
                "observation",
            ]
        },
        "note": (
            "This demo intentionally makes condition_occurrence the densest event table. "
            "Measurement can be denser than conditions in ICU or lab-heavy databases."
        ),
    }
    with (out_dir / "demo_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic OMOP-like multi-table EHR demo.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory where OMOP CSV tables will be written.")
    parser.add_argument("--n-patients", type=int, default=600, help="Number of synthetic patients.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max-visits", type=int, default=90, help="Maximum visits per patient.")
    parser.add_argument("--condition-rate-scale", type=float, default=1.0, help="Multiplier for condition table density.")
    parser.add_argument("--drug-rate-scale", type=float, default=1.0, help="Multiplier for drug table density.")
    parser.add_argument("--procedure-rate-scale", type=float, default=1.0, help="Multiplier for procedure table density.")
    parser.add_argument("--measurement-rate-scale", type=float, default=1.0, help="Multiplier for measurement table density.")
    parser.add_argument("--observation-rate-scale", type=float, default=1.0, help="Multiplier for observation table density.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    generate_demo_omop(
        out_dir=args.out_dir,
        n_patients=args.n_patients,
        seed=args.seed,
        max_visits=args.max_visits,
        condition_rate_scale=args.condition_rate_scale,
        drug_rate_scale=args.drug_rate_scale,
        procedure_rate_scale=args.procedure_rate_scale,
        measurement_rate_scale=args.measurement_rate_scale,
        observation_rate_scale=args.observation_rate_scale,
    )


if __name__ == "__main__":
    main()
