# Self-Explainable GNN Diagnosis on MIMIC-IV ED Patient Graphs

This repository compares self-explainable graph neural networks for diagnosis
prediction on multi-visit clinical graphs built from MIMIC-IV emergency-department
data. The active task is the max6 cohort (patients with at most 6 total visits)
with the 10 most frequent diagnoses by TRAIN-fold frequency as classes. All methods
share one artifact, class set, patient-disjoint split and seed, and select
checkpoints on validation only.

## Methods

Each method lives in its own top-level folder.

| Method | Folder `<m>` | Role |
|---|---|---|
| ProtGNN | `protgnn` | prototype-based self-explaining GNN |
| CEI-GNN | `cei` | evidence-interaction GNN (v1/v2/v3) and its studies |
| GSAT | `gsat` | stochastic attention / information bottleneck |
| GraphCare | `graphcare` | native clinical adapter, visit-conditioned attention |
| GCHM-PNA | `gchm_pna` | hub-gated, relation-aware PNA (v2/v3) |

## Code layout

```text
data_pipeline/
  s1_clean/                      frozen raw ED CSV cleaning scripts (never run in cleanup)
  s2_events/                     events, cohort and metadata repair
  s3_labels/                     diagnosis labels and ICD mapping
  s4_graph/                      clinical graph build, storage and audit
  s5_filter_split/               TRAIN Top-10 labels, 10k sampling and dev selection
core/                            training engine, contracts, tensorization, model
  explain/                       shared explanation contracts and GraphXAI integration
protgnn/, gsat/, graphcare/, gchm_pna/, cei/
                                 method implementations and their studies
comparison/
  canonical_split.json           fixed class order and subject folds (never regenerate)
  top3/                          top-three performance comparison package
  standardized/                  local ignored inputs and results (not a code package)
data/                            local raw inputs (not moved)
tests/
environment.yml                  sole dependency file
external/                        vendored third-party code (GraphXAI)
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
