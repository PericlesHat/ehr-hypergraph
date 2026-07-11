# ehr-hypergraph

## 1. Overview

`ehr-hypergraph` is a deep learning pipeline for longitudinal electronic health record data. It represents the clinical concepts observed in each visit as a hypergraph, then models each patient's ordered sequence of visits. The same learned patient representation supports two complementary goals:

- Predict one or more user-defined clinical outcomes from prior visits.
- Discover patient subtypes through unsupervised clustering.

The pipeline is designed around OMOP Common Data Model (CDM) exports, but the underlying idea is general: each patient needs a timeline of visits, and each visit needs a set of coded clinical events. Diagnoses, medications, procedures, measurements, and observations can all contribute to a visit representation.

The implementation uses Python, PyTorch, and PyTorch Geometric. A CUDA-capable GPU server is recommended for real cohorts. A modest research GPU is usually sufficient, although memory use still grows with the number of patients, visits, and unique clinical concepts. CPU execution is supported for small checks and data preparation, but training on a large cohort will be much slower.

## 2. Package Structure

```text
requirements.txt              Python dependencies

ehr_hypergraph/
  data/
    demo_omop.py              Generate synthetic OMOP-style tables
    preprocess_omop.py        Convert OMOP tables into model-ready visit files
    dataset.py                Load visit files as longitudinal patient timelines

  models/
    ehr_hyg.py                Hypergraph encoders, Transformer, and DEC clustering

  train.py                    Training entry point
  README.md                   This guide
```

## 3. Requirements

Use Python 3.10 or 3.11. The package was developed with PyTorch 2.1 and PyTorch Geometric 2.7; the allowed version ranges are listed in `requirements.txt`.

On a Linux GPU server, create an isolated environment first:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Alternatively, use Conda or Miniconda:

```bash
conda create -n ehr-hypergraph python=3.11 -y
conda activate ehr-hypergraph
python -m pip install --upgrade pip
```

Install a PyTorch build matched to the server's CUDA driver using the [official PyTorch selector](https://pytorch.org/get-started/locally/). Then install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Confirm that PyTorch and PyTorch Geometric can see the intended environment:

```bash
python -c "import torch, torch_geometric; print(torch.__version__); print(torch.cuda.is_available()); print(torch_geometric.__version__)"
```

For a CPU-only smoke test, `pip install -r requirements.txt` is sufficient. A GPU build should be installed before the requirements file so that `pip` does not replace it with an unintended CPU build.

## 4. OMOP EHR Data

The preprocessor follows the [OMOP CDM v5.4 specification](https://ohdsi.github.io/CommonDataModel/cdm54.html). It uses `visit_occurrence` to define the longitudinal timeline and aggregates clinical concepts from event tables into one set per visit.

### Expected Tables

The minimum input is:

```text
visit_occurrence.csv
condition_occurrence.csv
outcome_concepts.json
```

`visit_occurrence.csv` must provide `person_id`, `visit_occurrence_id`, `visit_start_date`, and `visit_end_date`. `condition_occurrence.csv` provides both input concepts and the currently supported outcome labels.

The following tables are optional but recommended as additional input features:

```text
drug_exposure.csv
procedure_occurrence.csv
measurement.csv
observation.csv
concept.csv
```

Raw OMOP `concept_id` values are large and sparse. Preprocessing maps them to compact `code_index` values for the embedding layer and preserves the mapping in `concept_map.csv` for clinical interpretation.

### Define Outcomes

Create `outcome_concepts.json` as a clinically reviewed mapping from an outcome name to OMOP condition concept IDs:

```json
{
  "CVD": [1000030, 1000031, 1000032],
  "CKD": [1000035, 1000036]
}
```

The current label builder defines outcomes from `condition_occurrence.condition_concept_id`. For the default `future_window` task, a visit is positive when the outcome occurs after that visit and within the following 365 days. By default, target outcome concepts are removed from the input visit concepts to reduce direct label leakage.

### Generate a Synthetic Demo

The demo generator creates synthetic OMOP-like CSV tables with 600 patients, unequal follow-up lengths, and uneven event-table density. It contains no real patients or PHI.

```bash
python -m ehr_hypergraph.data.demo_omop \
  --out-dir demo_omop \
  --n-patients 600 \
  --seed 7
```

### Preprocess a Demo or Real OMOP Export

Place a real OMOP export in one directory, alongside its `outcome_concepts.json`, then run:

```bash
python -m ehr_hypergraph.data.preprocess_omop \
  --input-dir your_omop_export \
  --out-dir processed_ehr \
  --outcome-concepts your_omop_export/outcome_concepts.json
```

The same command works for the synthetic demo by replacing `your_omop_export` with `demo_omop`.

The preprocessor creates:

```text
processed_ehr/
  visit_level_data.csv          One row per visit with its code_index list
  visit_stepwise_labels.csv     One row per visit with outcome labels
  concept_map.csv               code_index to OMOP concept mapping
  event_table_summary.csv       Event-table counts after loading
  preprocess_report.json        Cohort and label-generation metadata
```

`visit_level_data.csv` is deliberately named for its data content. The default leakage prevention is a preprocessing behavior, not a property that needs to be encoded in the filename.

Advanced preprocessing options are available when needed. For example, `--prediction-window-days 180` changes the future prediction window, and `--assign-missing-visit-ids` date-matches events without a visit ID to a same-person visit interval. These are not needed for the normal workflow.

### Non-OMOP Data

The ideas are not restricted to OMOP. A non-OMOP EHR source can be adapted when it can produce the same logical fields: de-identified patient ID, visit ID, visit time, clinical code ID, and outcome definition. OMOP is recommended because its table and vocabulary conventions make this conversion explicit and portable across institutions.

## 5. Train a Model

Run training from the repository root after preprocessing:

```bash
python -m ehr_hypergraph.train \
  --visit-file processed_ehr/visit_level_data.csv \
  --label-file processed_ehr/visit_stepwise_labels.csv \
  --output-dir runs
```

The default backbone is `allset`, the recommended model for this package. The trainer makes a patient-level 70%/10%/20% train/validation/test split. A patient appears in only one split; all visits for that patient stay together.

The user-facing options are intentionally small:

| Option | Default | Purpose |
| --- | --- | --- |
| `--visit-file` | required | Preprocessed `visit_level_data.csv`. |
| `--label-file` | required | Matching `visit_stepwise_labels.csv`. |
| `--outcomes` | all label columns | Comma-separated subset of outcome columns. |
| `--backbone` | `allset` | `allset` is recommended; use `pyg` as the PyG baseline. |
| `--epochs` | `60` | Maximum training epochs; early stopping can end sooner. |
| `--batch-size` | `128` | Number of patients per mini-batch; lower it when GPU memory is limited. |
| `--n-clusters` | `5` | Number of DEC patient subtypes to discover. |
| `--seed` | `40` | Reproducible split and training seed. |
| `--output-dir` | `ehr_hypergraph/runs` | Parent directory for timestamped runs. |
| `--run-name` | empty | Optional readable suffix for the run directory. |

The remaining optimizer, regularization, and DEC settings use conservative package defaults and are recorded in every run's `config.json`.

## 6. Training Outputs

Each run creates a timestamped folder under `--output-dir/logs/`:

```text
summary.json             Concise cohort, prediction, and clustering summary
outcome_metrics.csv      Test prevalence, AUROC, AUPRC, F1, and sensitivity per outcome
cluster_summary.csv      Patient count and fraction for each DEC cluster
patient_clusters.csv     Patient-level cluster assignment and probabilities
training_history.csv     Training and validation loss and macro AUROC by epoch
loss.png                 Training and validation loss by epoch
auc.png                  Training and validation macro AUROC by epoch
config.json              User options and fixed training settings
checkpoint.pt            Best validation checkpoint
```

Start with `summary.json`. It reports the cohort split, prediction task, validation-selected epoch, test visit-level macro AUROC/AUPRC, and the number and distribution of non-empty patient clusters. Use `outcome_metrics.csv` to inspect individual outcomes and `cluster_summary.csv` plus `patient_clusters.csv` to begin subtype characterization.

Metrics that cannot be calculated because a split contains only one label class are left blank rather than reported as an artificial baseline score.

## 7. Model Pipeline

At each visit, every distinct EHR concept is a hypergraph node and the complete set of concepts observed in that visit is one hyperedge. This differs from a standard graph: a hyperedge can directly connect many diagnoses, medications, tests, and procedures from the same encounter.

```text
OMOP event tables
  -> visit-level concept sets
  -> dynamic hypergraph
  -> visit embeddings
  -> time-aware Transformer over each patient timeline
  -> outcome prediction and patient clustering
```

During a mini-batch forward pass, the model constructs the relevant incidence structure from the visits in that batch rather than storing one fixed cohort-wide adjacency matrix. The hypergraph encoder produces one embedding per visit; a causal Transformer then combines visit embeddings and visit timing into a patient representation. Prediction heads emit visit-level outcome logits, while the DEC head assigns a soft patient subtype probability.

Two hypergraph backbones are available:

- `allset` (recommended): an AllSet/HypCAST-style attention encoder with `V -> E -> V -> E` propagation. It uses learned multiset pooling to aggregate clinical concepts into visits, then visits back into updated concept states. See the [AllSet paper](https://openreview.net/forum?id=hpBTIv2uy_E).
- `pyg`: a baseline built on PyTorch Geometric's [`HypergraphConv`](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/hypergraph_conv.html).

## 8. Citation

Please credit Ziyang Zhang for the EHR Hypergraph software packaging work. The earliest version of this code was developed from work by Ziyang Zhang and Ran Xu. The AllSet-style hypergraph encoder is based on the AllSet formulation and draws on the [official AllSet repository](https://github.com/jianhao2016/AllSet). The related EHR hypergraph direction is represented by the [TACCO repository](https://github.com/PericlesHat/TACCO).

### Recommended Citation

If this package contributes to a research project, please cite the following work. 

```bibtex
@inproceedings{zhang2024tacco,
  title={Tacco: Task-guided co-clustering of clinical concepts and patient visits for disease subtyping based on ehr data},
  author={Zhang, Ziyang and Cui, Hejie and Xu, Ran and Xie, Yuzhang and Ho, Joyce C and Yang, Carl},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={6324--6334},
  year={2024}
}
```

### Additional related citations:

```bibtex
@article{xu2023hypergraph,
  title={Hypergraph transformers for ehr-based clinical predictions},
  author={Xu, Ran and Ali, Mohammed K and Ho, Joyce C and Yang, Carl},
  journal={AMIA Summits on Translational Science Proceedings},
  volume={2023},
  pages={582},
  year={2023}
}
```

```bibtex
@inproceedings{zhang2025type,
  title={Type 2 Diabetes Subtyping via Phenotype and Genotype Co-Learning},
  author={Zhang, Ziyang and Wang, Lily and Meng, Weimin and Liu, Chang and Shao, Hui and Sun, Yan V and Guo, Jingchuan and Bian, Jiang and Yin, Rui and Yang, Carl},
  booktitle={Medinfo 2025-Healthcare Smart x Medicine Deep: Proceedings of the 20th World Congress on Medical and Health Informatics},
  pages={1064--1068},
  year={2025},
  organization={SAGE Publications 1 Oliver's Yard, 55 City Road, London, EC1Y 1SP}
}
```

```bibtex
@inproceedings{chien2022allset,
  title={You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks},
  author={Chien, Eli and Pan, Chao and Peng, Jianhao and Milenkovic, Olgica},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
