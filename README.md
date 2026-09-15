# Self-Explainable Graph Neural Networks for Medical Diagnosis

The current four-method comparison uses **one exact native ProtGNN/GSAT input**:
ProtGNN, GSAT, GraphCare, and PNA (plain or medication–complaint interaction).
All consume the same 74,511 ordered patient stars, 30 ordered labels, canonical
split, and **331 native feature slots**, including the original **132 hub fields**.
GraphCare has an explicit continuous hub encoder before BAT message passing.

This is a **source-snapshot reproduction**, not a temporally clean early-diagnosis
benchmark. Native history, visit counts, stay-wide vitals/labs and their upstream
limitations are retained—not silently replaced by the earlier 127-field enrichment.
See the [exact-input contract and verification](docs/native-identical-input-v1.md).

## Start here — current identical-input comparison

Run from the repository root. The launcher selects the main Python interpreter
for ProtGNN/GSAT/PNA and `.venv-graphcare/bin/python` for GraphCare.

```bash
# Instantiates every model with the full cohort; creates no run output.
python3 -m comparison.standardized.train_identical \
  --output comparison/standardized/native_runs/full_v1 --dry-run

# Full-data training, ONLY when intentionally requested (not run during wiring).
python3 -m comparison.standardized.train_identical \
  --output comparison/standardized/native_runs/full_v1 \
  --epochs 30 --seed 1234 --batch-size 128 --loss ce --execute

# Explicitly bounded integration check through the SAME production path.
python3 -m comparison.standardized.train_identical \
  --output comparison/standardized/native_runs/my_wiring_check \
  --epochs 2 --limit 16 --batch-size 8 --execute
```

`--resume --execute` restores the epoch-committed optimizer/RNG state; `--replay
--execute` reloads the selected checkpoint and verifies exact saved validation
logits. Use the same scientific arguments when resuming/replaying. No test loader
is used for selection. `--loss native` declares method-native class weighting;
`--loss ce` and `--loss sqrt_inverse` provide shared classification-loss policies
without removing prototype or information-bottleneck auxiliary losses.

Historical reproduction commands (`run_all`, `run_benchmark`, `run_common_input`,
`pna_benchmark`, and `enriched_input_v1`) are **not current equal-input defaults**.
Their inputs/results/checkpoints remain unchanged and cannot be mixed with this
version. The [historical runbook](comparison/standardized/README.md) documents
those older runs; their three-method topology parity did not establish equal hub
payloads. The current entrypoint rejects any other artifact fingerprint.

- [Repository map and output ownership](STRUCTURE.md)
- [Exact verification evidence and remaining limitations](docs/usability-verification.md)
- [Browser graph viewer](visualizer/README.md)
- [Working-tree cleanup and reversible restoration](docs/cleanup-working-tree.md)
- [Raw data workflow (legacy caveats)](docs/RUN_WITH_OWN_DATA.md)

## Project Structure

```text
.
├── data/                     Shared raw + processed data (merged_ed.csv)
├── shared/                   Shared code and standardized contracts
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
├── gsat_analysis/            METHOD C — GSAT on shared patient graphs
├── comparison/standardized/  Current benchmark CLI, cache audit, explanations, summary
├── tests/                    Maintained contract, fixture and model smoke tests
├── visualizer/               React/TypeScript graph viewer
├── baselines/                Exploratory tabular baseline (XGBoost / HistGB)
├── external/                 External research code, including GraphXAI
└── docs/                     Operating runbooks, current evidence, cleanup manifest
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

## Legacy / exploratory usage

The commands below are historical method-specific workflows. They may train
models or generate caches, and do not use the current native identical-input
artifact. For the four-method comparison, use `train_identical` above.

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
PYTHONPATH=.:external/GraphXAI-main python3 protgnn_analysis/scripts/hpo_disease.py --n_trials 50 --max_epochs 80
```

Generate English clinical-language explanations from the latest GraphXAI outputs:

```bash
PYTHONPATH=.:external/GraphXAI-main python3 protgnn_analysis/scripts/generate_clinical_explanations.py --limit 5
```

Clinical explanation files include patient-level GraphXAI signals, similar-patient
context, and prototype evidence when the latest run used prototype learning.

Legacy generated artifacts are written under each method's `outputs/` folder.
Standardized results belong under `comparison/standardized/results/`.

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
