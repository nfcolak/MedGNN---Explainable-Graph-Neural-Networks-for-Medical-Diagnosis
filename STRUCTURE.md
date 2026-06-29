# Repository structure

The project is organised as **two self-contained analyses** that share the same
data and evaluation protocol, so they can be compared fairly:

- **`protgnn_analysis/`** — Method A: the intra-patient ProtGNN (current work).
- **`graphcare_analysis/`** — Method B: the GraphCare (KG + BAT-GNN) comparison.
  The model is vendored upstream under `external/GraphCare/`; this folder holds
  the data adapter, KG builder and run wrapper (work in progress; see roadmap).

```text
.
├── data/                      SHARED raw + processed data (merged_ed.csv, processed_*/)
├── external/                  SHARED 3rd-party code (GraphXAI-main, GraphCare)
├── docs/                      SHARED literature + RUN_WITH_OWN_DATA.md
│
├── shared/                    SHARED code imported by BOTH analyses
│   ├── data_prep/             MIMIC-IV-ED tables -> merged_ed.csv
│   │                          (merge_ed.py + med/chiefcomplaint standardizers, extract_ed_labs)
│   └── lib/                   The pieces that MUST match for a fair comparison:
│       ├── config_base.py     DATA_DIR, SEED (1234), SPLIT_RATIO, device
│       ├── splits.py          subject-aware 80/10/10 split
│       └── metrics.py         multi-class metrics (macro/micro-F1, top-3/5)
│
├── protgnn_analysis/          METHOD A — intra-patient ProtGNN
│   ├── config.py              hyperparameters (was configs/config.py)
│   ├── load_dataset.py        IntraPatientHeteroDataset + graph builders
│   ├── models/                GCN / GAT / GIN + GnnNets wrapper
│   ├── my_mcts.py             MCTS prototype projection
│   ├── explainability/        GraphXAI wrapper + integration
│   ├── train.py               main train + explain entry point
│   ├── scripts/               eval / summarize / confusion / hpo / clinical explanations / ...
│   └── outputs/               checkpoints, runs, results
│
├── graphcare_analysis/        METHOD B — GraphCare (KG + BAT-GNN); model vendored upstream
│   ├── config.py              hyperparameters (reuses shared seed/split)
│   ├── adapter.py             merged_ed.csv -> GraphCare/PyHealth records     [Phase 1]
│   ├── build_kg.py            personalized KG (ontology/PMI; LLM+UMLS later)  [Phase 2]
│   ├── run.py                 train/eval wrapper reusing shared lib           [Phase 3]
│   └── outputs/               GraphCare runs/results
│
├── baselines/                 SHARED tabular baseline (XGBoost / HistGB)
├── README.md  STRUCTURE.md  environment.yml  requirements*.txt  TODO.txt
```

## Running

Both analyses run from the **repo root** with the repo root on `PYTHONPATH`
(this replaces the old `PYTHONPATH=src:...`):

```bash
# Method A — ProtGNN (current)
PYTHONPATH=.:external/GraphXAI-main python3 protgnn_analysis/train.py --explain_n 100

# Method B — GraphCare (comparison; clone external/GraphCare first)
PYTHONPATH=.:external/GraphCare python3 graphcare_analysis/run.py

# Shared data prep (MIMIC-IV-ED -> merged_ed.csv)
python3 shared/data_prep/merge_ed.py
```

## Fair-comparison guarantee

Both methods consume the **same** `data/merged_ed.csv`, the **same**
subject-aware split (`shared/lib/splits.py`, seed 1234) and the **same** metrics
(`shared/lib/metrics.py`), so differences in results come from the *method*, not
the data or the split.

> **Note (cleanup TODO):** `protgnn_analysis/` currently keeps its own copies of
> the split/metric helpers (unchanged, working). `shared/lib/` holds identical
> canonical copies that `graphcare_analysis/` uses. A later pass should migrate
> `protgnn_analysis/` to import from `shared/lib/` and delete the duplicates.

## GraphCare roadmap (where the work is)

| Phase | Goal | File |
|-------|------|------|
| 0 | Clone upstream + install deps (pyhealth, torch-geometric) | `external/GraphCare/` |
| 1 | Data adapter: merged_ed.csv -> GraphCare records | `graphcare_analysis/adapter.py` |
| 2 | Personalized KG (ontology/PMI; LLM+UMLS later) | `graphcare_analysis/build_kg.py` |
| 3 | Wire training/eval + shared metrics/report | `graphcare_analysis/run.py` |
| 4 | Comparison ladder + interpretability (BAT attention vs GraphXAI) | `baselines/`, `shared/lib` |
