# Self-Explainable Graph Neural Networks for Medical Diagnosis

This project adapts ProtGNN-style prototype learning for medical diagnosis experiments, with patient-similarity graph construction and optional GraphXAI explanation workflows.

The repository has been consolidated into a single clean project structure. The previous duplicate `ProtGNN-main` and `prot_gnn_core` folders have been merged into the layout below.

## Project Structure

```text
.
├── configs/                  Shared experiment configuration
├── data/                     Local datasets and dataset metadata
├── docs/                     Literature, figures, and project references
├── external/                 External research code, including GraphXAI
├── notebooks/                Exploratory notebooks or scratch analysis
├── outputs/                  Generated checkpoints, metrics, and reports
├── scripts/                  Runnable training, HPO, testing, and baseline scripts
└── src/prot_gnn/             Main Python package
    ├── explainability/       GraphXAI integration wrappers
    ├── models/               GCN, GAT, GIN, and ProtGNN model code
    ├── load_dataset.py       Dataset loaders and graph builders
    ├── my_mcts.py            Prototype subgraph search
    └── utils.py              Plotting and helper utilities
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

Configuration lives in `configs/config.py`.

Run the main train-and-explain workflow:

```bash
PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/train_and_explain.py --explain_n 10 --no_prot
```

Run with prototype learning enabled:

```bash
PYTHONPATH=src:external/GraphXAI-main:. python scripts/train_and_explain.py --explain_n 100
```

When prototypes are enabled, each enriched explanation JSON includes
`prototype_evidence` with the nearest learned prototypes, prototype class, distance,
activation, and contribution to the predicted class.

Run hyperparameter optimization:

```bash
python scripts/hyperparameter_opt.py --n_trials 50 --max_epochs 80
```

Run the older training loop:

```bash
python scripts/train_gnns.py
```

Generate English clinical-language explanations from the latest GraphXAI outputs:

```bash
PYTHONPATH=src:external/GraphXAI-main:. python scripts/generate_clinical_explanations.py --limit 5
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
