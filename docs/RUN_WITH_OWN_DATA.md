# Run With Your Own Data

The raw MIMIC ED files are not included in GitHub. To regenerate the dataset,
model outputs, GraphXAI JSON files, and summary files, place the raw CSV files
under:

```text
data/Original CSVs/
```

Required files:

```text
edstays.csv
triage.csv
vitalsign.csv
medrecon.csv
pyxis.csv
diagnosis.csv
icd9_to_icd10_mapping.csv
```

Install dependencies. Preferred reproducible setup:

```bash
conda env create -f environment.yml
conda activate protgnn-mimic
```

Alternative pip setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

`requirements.txt` contains flexible version ranges. `requirements-lock.txt` and
`environment.yml` contain the pinned environment used for the current working
run. If PyTorch Geometric fails to install, install PyTorch first from the
official PyTorch selector for your operating system, then rerun the dependency
installation.

Generate merged datasets:

```bash
python3 data/merge_ed.py
```

Train the model and generate GraphXAI JSON explanations for the full test set:

```bash
PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/train_and_explain.py \
  --clst 0.02 \
  --sep 0.0 \
  --explain_n -1 \
  --archive_tag test_only_all_explanations
```

Generate the readable clinical summary from the archived run:

```bash
PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/generate_clinical_explanations.py \
  --results_dir outputs/runs/<RUN_FOLDER>
```

Important outputs:

```text
outputs/runs/<RUN_FOLDER>/explanations/graph_*.json
outputs/runs/<RUN_FOLDER>/clinical_explanations/summary.xlsx
outputs/runs/<RUN_FOLDER>/clinical_explanations/summary.csv
outputs/runs/<RUN_FOLDER>/clinical_explanations/graph_*.txt
```

Note: `--explain_n -1` explains all graphs in the test split, not all 20,000
patients. With the default 80/10/10 split, this is about 2,000 patients.
