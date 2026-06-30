# Self-Explainable Graph Neural Networks for Medical Diagnosis

This project adapts ProtGNN-style prototype learning for medical diagnosis experiments, with patient-similarity graph construction and optional GraphXAI explanation workflows.

The repository is split into **two self-contained analyses** that share the same
data and evaluation protocol (see `STRUCTURE.md`):

* `protgnn_analysis/`   — Method A, the intra-patient ProtGNN (current work).
* `graphcare_analysis/` — Method B, the GraphCare (KG + BAT-GNN) comparison.

## Project Structure

```text
.
├── data/                     Shared raw + processed data (merged_ed.csv)
├── shared/                   Code shared by BOTH analyses
│   ├── data_prep/            MIMIC-IV-ED -> merged_ed.csv (merge_ed.py + standardizers)
│   └── lib/                  Split / metrics / config base (identical across methods)
├── protgnn_analysis/         METHOD A — intra-patient ProtGNN (current)
│   ├── config.py             Hyperparameters
│   ├── load_dataset.py       IntraPatientHeteroDataset + graph builders
│   ├── models/               GCN / GAT / GIN + GnnNets
│   ├── my_mcts.py            Prototype subgraph projection (MCTS)
│   ├── explainability/       GraphXAI wrappers
│   ├── train.py              Train + explain entry point
│   ├── scripts/              eval / summarize / confusion / hpo / ...
│   └── outputs/              Checkpoints, runs, results
├── graphcare_analysis/       METHOD B — GraphCare (KG + BAT-GNN); model vendored upstream
│   ├── config.py  adapter.py  build_kg.py  run.py
│   └── outputs/
├── baselines/                Tabular baseline (XGBoost / HistGB), shared by both
├── external/                 External research code, including GraphXAI
└── docs/                     Literature, figures, and project references
```

## Requirements

For a reproducible setup with local MIMIC ED files, use:

```bash
conda env create -f environment.yml
conda activate protgnn-mimic
```

or:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

`requirements.txt` contains flexible version ranges. `requirements-lock.txt`
and `environment.yml` contain pinned versions for collaboration.

GraphXAI is kept as external research code under `external/GraphXAI-main/`. The training scripts add this folder to Python's import path automatically.

See `docs/RUN_WITH_OWN_DATA.md` for the full data placement and rerun workflow.

## Usage

All commands run from the repo root with the repo root on `PYTHONPATH`
(so the `protgnn_analysis` / `graphcare_analysis` / `shared` packages resolve).

### Method A — ProtGNN (current)

Configuration lives in `protgnn_analysis/config.py`.

Run the main train-and-explain workflow:

```bash
PYTHONPATH=.:external/GraphXAI-main python3 protgnn_analysis/train.py --explain_n 10 --no_prot
```

Run with prototype learning enabled:

```bash
PYTHONPATH=.:external/GraphXAI-main python3 protgnn_analysis/train.py --explain_n 100
```

### Method B — GraphCare (comparison)

Clone the upstream model first (see `external/GraphCare/README.md`), then:

```bash
PYTHONPATH=.:external/GraphCare python3 graphcare_analysis/run.py
```

When prototypes are enabled, each enriched explanation JSON includes
`prototype_evidence` with the nearest learned prototypes, prototype class, distance,
activation, and contribution to the predicted class.

Run hyperparameter optimization:

```bash
PYTHONPATH=.:external/GraphXAI-main python3 protgnn_analysis/scripts/hyperparameter_opt.py --n_trials 50 --max_epochs 80
```

Run the older training loop:

```bash
PYTHONPATH=.:external/GraphXAI-main python3 protgnn_analysis/scripts/train_gnns.py
```

Generate English clinical-language explanations from the latest GraphXAI outputs:

```bash
PYTHONPATH=.:external/GraphXAI-main python3 protgnn_analysis/scripts/generate_clinical_explanations.py --limit 5
```

Clinical explanation files include patient-level GraphXAI signals, similar-patient
context, and prototype evidence when the latest run used prototype learning.

Generated artifacts are written under `outputs/`.

## Reference

This work builds on the AAAI 2022 paper "ProtGNN: Towards Self-Explaining Graph Neural Networks".

```bibtex
@article{zhang2021protgnn,
  title={ProtGNN: Towards Self-Explaining Graph Neural Networks},
  author={Zhang, Zaixi and Liu, Qi and Wang, Hao and Lu, Chengqiang and Lee, Cheekong},
  journal={arXiv preprint arXiv:2112.00911},
  year={2021}
}
```
