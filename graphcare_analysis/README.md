# GraphCare analysis (Method B)

Comparison method for the thesis: **GraphCare** (Jiang et al., ICLR 2024) — a
knowledge-graph-augmented, bi-attention GNN (BAT-GNN) for EHR prediction. It
sits in the same competitive set as ProtGNN/ProtoEHR (graph + KG + attention
interpretability) and runs on our **single-visit** ED cohort.

Unlike `protgnn_analysis/` (written here), GraphCare's model is **vendored
upstream** under `external/GraphCare/`; this folder only adapts our data, builds
the KG, and wraps training/evaluation.

## Layout
- `config.py`   — hyperparameters; reuses shared seed/split.
- `adapter.py`  — `merged_ed.csv` -> GraphCare/PyHealth per-patient records (Phase 1).
- `build_kg.py` — personalized KG (ontology/PMI MVP; LLM+UMLS full) (Phase 2).
- `run.py`      — train/eval wrapper -> `shared/lib` metrics -> `outputs/` (Phase 3).
- `outputs/`    — runs/results.

## Setup — use an ISOLATED venv (important)
GraphCare pins **old** deps (`torch==1.12.0`, `torch-geometric==2.3.0`,
`pyhealth==1.1.2`) that **conflict** with this project's `torch 2.8`. Installing
them into the main environment would break `protgnn_analysis`. Use a separate
venv — the comparison only needs the output report, not a shared env:

```bash
python3 -m venv .venv-graphcare
source .venv-graphcare/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch==1.12.0 torch-geometric==2.3.0 pyhealth==1.1.2 scikit-learn==1.2.1
```
(If `torch==1.12.0` has no Apple-Silicon wheel, fall back to a modern stack —
`torch 2.x` + latest `pyhealth`/`torch-geometric` — and patch GraphCare's code
for the PyG/pyhealth API changes.)

Upstream model: BAT-GNN is in `external/GraphCare/graphcare_/model.py`.

## Run (inside the GraphCare venv)
```bash
PYTHONPATH=.:external/GraphCare python3 graphcare_analysis/run.py
```

## Fair comparison
Same `data/merged_ed.csv`, same subject-aware split (`shared/lib/splits.py`,
seed 1234), same metrics (`shared/lib/metrics.py`) as `protgnn_analysis/`.

## Roadmap
| Phase | Goal | File |
|-------|------|------|
| 0 | Clone upstream + install deps | `external/GraphCare/` |
| 1 | Data adapter -> GraphCare input | `adapter.py` |
| 2 | Personalized KG (ontology/PMI) | `build_kg.py` |
| 3 | Wire training + shared metrics | `run.py` |
| 4 | Comparison ladder + interpretability (BAT attention vs ProtGNN/GraphXAI) | — |
