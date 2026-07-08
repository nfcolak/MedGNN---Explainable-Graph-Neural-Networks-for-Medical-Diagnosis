# 🧬 MedProtoGNN

> **Self-explainable Graph Neural Networks for patient-level medical diagnosis**

MedProtoGNN is a research-oriented machine learning project that explores how
**prototype-based Graph Neural Networks** can support interpretable medical
diagnosis on emergency department data. The project builds intra-patient graphs
from MIMIC-IV-ED style records, trains ProtGNN-inspired models, compares against
GraphCare, and provides a polished interactive web visualizer for inspecting
patient-level graph explanations.

The central idea is simple: instead of treating an ED visit as a flat table, each
encounter becomes a graph where the patient is connected to vital signs,
medications, chief complaints, and diagnosis-related clinical attributes.

---

## 🔴 Project Overview

This repository combines three connected goals:

| Goal | Description |
|---|---|
| 🧠 **Explainable Medical GNNs** | Train ProtGNN-style models that classify patient encounters while exposing prototype-based evidence. |
| 🏥 **Clinical Graph Construction** | Transform ED visits into intra-patient graphs containing vitals, medications, chief complaints, and diagnosis labels. |
| 🔍 **Interactive Graph Visualization** | Provide a React/Cytoscape.js interface for exploring patient graphs, medication PMI bundles, normalized features, and clinical node descriptions. |

The project is designed for experimentation, model comparison, and visual
inspection of why a medical graph model may associate a patient encounter with a
diagnosis class.

---

## ✨ Key Features

- 🧬 Converts emergency department encounters into **patient-centered medical graphs**.
- 🧠 Implements a **ProtGNN-inspired explainable GNN pipeline** with prototype evidence.
- 🔬 Integrates **GraphXAI explanation workflows** for graph-level interpretability.
- 🩺 Supports clinical feature groups: **vital signs, medications, chief complaints, diagnosis attributes**.
- 🟢 Adds **medication PMI co-occurrence edges** to capture drug bundle relationships.
- 📊 Includes shared evaluation utilities for fair comparison across methods.
- ⚖️ Compares the current ProtGNN pipeline with a **GraphCare / BAT-GNN** analysis track.
- 🖥️ Provides a dark, publication-ready **interactive patient graph visualizer**.
- 📁 Loads **77,697 local graph JSON records** through lazy sharded frontend loading.
- 🧾 Adds human-readable descriptions for diagnoses, merged diagnosis groups, medications, vitals, and chief complaints.

---

## 🧱 System Architecture

```text
MIMIC-IV-ED style CSVs
        │
        ▼
Shared preprocessing pipeline
        │
        ├── merged_ed.csv
        │
        ├── medication + complaint standardization
        │
        └── merged diagnosis classes
        │
        ▼
Patient-centered graph construction
        │
        ├── PATIENT center node
        ├── vital-sign nodes
        ├── medication nodes
        ├── chief-complaint nodes
        └── PMI medication bundle edges
        │
        ▼
Explainable graph learning
        │
        ├── ProtGNN-style prototype learning
        ├── GCN / GAT / GIN backbones
        ├── GraphXAI explanations
        └── GraphCare comparison track
        │
        ▼
Interactive clinical graph visualizer
```

---

## 📂 Repository Structure

```text
.
├── data/                         Shared raw and processed data
├── shared/                       Data preparation + shared metrics/splits
│   ├── data_prep/                MIMIC-IV-ED preprocessing pipeline
│   └── lib/                      Shared split, metrics, and config helpers
│
├── protgnn_analysis/             Method A: intra-patient ProtGNN pipeline
│   ├── load_dataset.py           Patient graph construction
│   ├── train.py                  Main train + explain entry point
│   ├── models/                   GCN, GAT, GIN, GnnNets wrappers
│   ├── explainability/           GraphXAI integration
│   ├── scripts/                  Evaluation, HPO, summaries, clinical text
│   └── outputs/                  Checkpoints, runs, explanations
│
├── graphcare_analysis/           Method B: GraphCare comparison
│   ├── adapter.py                Converts shared data to GraphCare records
│   ├── build_kg.py               Personalized KG / PMI graph construction
│   ├── run.py                    Train/evaluate GraphCare wrapper
│   └── outputs/                  Comparison outputs
│
├── baselines/                    Tabular disease-prediction baselines
├── external/                     Vendored third-party research code
├── docs/                         Literature, figures, and run instructions
│
└── visualizer/                   React + TypeScript graph visualizer
    ├── public/graphs/            Browser-readable graph JSON shards
    ├── scripts/                  Dataset export script
    └── src/                      UI, graph canvas, clinical descriptions
```

---

## 👨🏾‍🔧 Technologies Used

### 🧠 Machine Learning & Data

| Technology | Purpose |
|---|---|
| **Python** | Main data processing, training, and evaluation language |
| **PyTorch** | Neural network training backend |
| **PyTorch Geometric** | Graph data representation and GNN operations |
| **GraphXAI** | Graph explanation workflows and explanation artifacts |
| **scikit-learn** | Metrics, baselines, and evaluation utilities |
| **pandas / NumPy** | MIMIC-IV-ED preprocessing and feature engineering |

### 🖥️ Frontend Visualizer

| Technology | Purpose |
|---|---|
| **React 18** | Component-based interactive UI |
| **TypeScript** | Safer frontend state and graph data typing |
| **Vite** | Fast local development and production builds |
| **Cytoscape.js** | Patient graph rendering, zooming, panning, and styling |
| **Lucide React** | Clean icon system for controls and UI actions |
| **Plain CSS** | Custom dark publication-style interface |

---

## 🧩 Main Components

| Component | Purpose |
|---|---|
| 🧬 **IntraPatientHeteroDataset** | Builds one graph per ED patient encounter. |
| 🧠 **ProtGNN Training Pipeline** | Trains GNN models and produces prototype-based evidence. |
| 🔬 **GraphXAI Integration** | Generates explainability outputs for patient-level graphs. |
| ⚖️ **GraphCare Analysis** | Provides a comparison track using KG + BAT-GNN style modeling. |
| 📊 **Shared Metrics** | Keeps split logic and metrics consistent across methods. |
| 🖥️ **Graph Visualizer** | Lets users inspect patient graphs, node types, values, PMI edges, and clinical metadata. |
| 🧾 **Clinical Description Layer** | Adds readable explanations for diagnosis groups, vitals, medications, and chief complaints. |

---

## 🧬 Medical Graph Design

Each graph represents one emergency department encounter.

| Graph Element | Meaning |
|---|---|
| 🔴 **PATIENT** | The central ED visit / stay node. |
| 🔵 **Vital sign** | Triage and visit-level vital measurements. |
| 🟢 **Medication** | Drugs administered or prescribed during the encounter context. |
| 🟠 **Chief complaint** | The main triage complaint category. |
| 🟢╌╌ **Medication PMI** | A co-occurrence edge between two drugs whose PMI exceeds the bundle threshold. |

The visualizer uses fixed radial positioning: vital signs stay closer to the
patient, medications are placed farther away, and the patient node remains fixed
in the center.

---

## 🖥️ Interactive Visualizer

The visualizer is a standalone frontend app under `visualizer/`.

### 🌟 Visualizer Features

- 🔍 Patient graph selector with searchable graph metadata.
- 🧭 Zoom, pan, reset view, and fit-to-screen controls.
- 🧬 Fixed graph layout with the patient centered.
- 🏷️ Visible labels and normalized feature values on nodes.
- 🧾 Hover tooltips for node type, value, feature name, and importance.
- 📋 Click side panel with full node details.
- 🩺 Clinical descriptions for medications, vitals, and chief complaints.
- 🧠 Diagnosis card with short explanation and merged diagnosis categories.
- 🟢 Dashed PMI edges for medication co-occurrence bundles.
- 🌙 Dark, publication-ready interface designed for graph inspection.

### 🚀 Run the Visualizer

```bash
cd visualizer
npm install
npm run dev
```

Then open the local Vite URL, usually:

```text
http://localhost:5173/
```

### 🏗️ Build the Visualizer

```bash
cd visualizer
npm run build
```

---

## 🧪 Run the ML Pipeline

### 1️⃣ Create the Environment

Recommended reproducible setup:

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

### 2️⃣ Prepare the Data

Raw MIMIC-IV-ED files are not included in the repository. Place them under:

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

Generate the merged ED dataset:

```bash
python3 shared/data_prep/extract_ed_labs.py
python3 shared/data_prep/merge_ed.py
```

### 3️⃣ Train ProtGNN

Run the main train-and-explain workflow:

```bash
PYTHONPATH=.:external/GraphXAI-main python3 protgnn_analysis/train.py --explain_n 100
```

Run without prototype learning:

```bash
PYTHONPATH=.:external/GraphXAI-main python3 protgnn_analysis/train.py --explain_n 10 --no_prot
```

Generate clinical-language explanation summaries:

```bash
PYTHONPATH=.:external/GraphXAI-main python3 protgnn_analysis/scripts/generate_clinical_explanations.py
```

### 4️⃣ Run GraphCare Comparison

GraphCare uses a separate dependency stack. See `graphcare_analysis/README.md`
before installing its environment.

```bash
PYTHONPATH=.:external/GraphCare python3 graphcare_analysis/run.py
```

---

## 📊 Dataset Export for the Frontend

The frontend reads static graph JSON files from `visualizer/public/graphs/`.
To regenerate these files from the processed PyG dataset:

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

This work builds on the ProtGNN paper:

```bibtex
@article{zhang2021protgnn,
  title={ProtGNN: Towards Self-Explaining Graph Neural Networks},
  author={Zhang, Zaixi and Liu, Qi and Wang, Hao and Lu, Chengqiang and Lee, Cheekong},
  journal={arXiv preprint arXiv:2112.00911},
  year={2021}
}
```

---

## ⚠️ Medical Disclaimer

This project is a research and visualization tool. It is **not** a clinical
decision-support system and should not be used to diagnose, treat, or triage
patients.

---

## 👷 Author

| Contributor | Role |
|---|---|
| **Necati Furkan Colak** | 
| **Leyla Khasiyeva** | 

