# Self-Explainable GNN Diagnosis on MIMIC-IV ED Patient Graphs

This repository compares self-explainable graph neural networks for diagnosis
prediction on multi-visit clinical graphs built from MIMIC-IV emergency-department
data. The active task is the max6 cohort (patients with at most 6 total visits)
with the 10 most frequent diagnoses by TRAIN-fold frequency as classes. All methods
share one artifact, class set, patient-disjoint split and seed, and select
checkpoints on validation only.

## Methods

Each method lives in `comparison/standardized/clinical_graph_v2/methods/<m>/`.

| Method | Folder `<m>` | Role |
|---|---|---|
| ProtGNN | `protgnn` | prototype-based self-explaining GNN |
| CEI-GNN | `cei` | evidence-interaction GNN (v1/v2/v3) and its studies |
| GSAT | `gsat` | stochastic attention / information bottleneck |
| GraphCare | `graphcare` | native clinical adapter, visit-conditioned attention |
| GCHM-PNA | `gchm_pna` | hub-gated, relation-aware PNA (v2/v3) and protocol |
| XGBoost | `xgboost` | tabular control |

## Code layout

```text
comparison/
  canonical_split.json            fixed class order and subject folds (never regenerate)
  standardized/
    clinical_graph_v2/
      core/                       shared build, train, audit, tensorize, model
      methods/<m>/                adapter.py or plugin_<m>.py, studies/
      paths.py                    PACKAGE_ROOT / REPO_ROOT
      <old module>.py             compatibility shims
    icd_mapping.py, event_graph_v1/, enriched_input_v1/spec.py,
    event_graph_gchm_xgb_v1/      label chain (kept unchanged)
shared/                           lib/ (contracts) and data_prep/
tests/
environment.yml                   sole dependency file
external/                         vendored third-party code (GraphXAI)
```

## Setup

```bash
conda env create -f environment.yml
conda activate protgnn-mimic
```

## Safety rules

- The held-out test fold is closed: never loaded, never evaluated.
- No preprocessing, cache build or training without explicit approval.
- Every run writes to a new, unoccupied output directory; filled ones are never reused.
- Data is local and never committed.

Project notes, reports and decisions are kept outside this repository.
