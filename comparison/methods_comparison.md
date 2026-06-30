# What each method does — ProtGNN vs GraphCare

Both predict the 30-class primary diagnosis from a single MIMIC-IV-ED visit, on
the same `merged_ed.csv`, same canonical split, same metrics. They differ in
*how* they model and explain.

| Dimension | **ProtGNN** (Method A) | **GraphCare** (Method B) |
|---|---|---|
| **Core idea** | Prototype / case-based reasoning — self-explaining *by design* | KG-augmented **Bi-Attention** GNN — attention-based interpretability |
| **One-liner** | "This patient looks like learned heart-failure prototypes" | "Weight the patient's KG-expanded codes by learned attention, then classify" |
| **Graph unit** | 1 patient = 1 intra-patient **heterogeneous** graph (the visit) | 1 patient = 1 **personalized KG subgraph** (their codes + KG neighbours) |
| **Nodes** | typed: patient + vital + medication + symptom + chief-complaint | medical-code nodes + their KG entity neighbours |
| **Edges** | patient↔entity (star) + med↔med **PMI** co-occurrence | KG relations (here: `co_occurs` PMI + `same_class` drug class) |
| **Graph source** | hand-built (star + PMI threshold) | knowledge graph (ours: ontology+PMI; original: LLM/GPT-4 + UMLS) |
| **Encoder** | 3-layer **GCN** [128,128,128], mean pooling | **BAT-GNN**: GINE-style conv + α-attention (over visit nodes) + β-attention (visit time-decay) + edge attention |
| **Patient representation** | global mean pool → one graph embedding `h_G` | `joint`: graph mean-pool ⊕ direct EHR-node embedding |
| **Classification / reasoning** | log-distance similarity to **90 prototypes** (30×3) → linear layer | attention-weighted message passing → MLP head |
| **Interpretability** | **built-in**: which prototype + **MCTS projection** onto a real patient subgraph; plus post-hoc **GraphXAI** (Grad/IG/GNNExplainer) | **built-in**: α (which node/visit), β (which time), edge-attention scores |
| **Training loss** | CE + **cluster** + **separation** + ℓ1 (prototype-shaping) | **CE only** (task loss; attention is the mechanism) |
| **Prototype projection** | yes — 90 prototypes projected to real subgraphs via MCTS (slow step) | none |
| **Output** | 30-class probs + nearest-prototype evidence + node importance | 30-class probs + attention maps |
| **In our setup — data/KG** | merged_ed.csv → intra-patient graph, PMI edges | **same** merged_ed.csv → 259-node KG (PMI+ontology), model-direct (we bypass GraphCare's hardwired data pipeline) |
| **Environment** | torch 2.8 (main env) | torch 1.12 (isolated `.venv-graphcare`) |
| **Empirical strength (aligned)** | **balanced_acc 0.572 / macro-F1 0.496** → rare/minority classes | **accuracy 0.578 / top5 0.840** → overall correctness + ranking |

## Takeaway
- **ProtGNN** reasons by *similarity to learned clinical archetypes* and grounds
  every decision in a real patient subgraph (case-based, minority-sensitive).
- **GraphCare** reasons by *attention over a knowledge graph* of the patient's
  codes (relation-aware, accuracy-oriented).
- Given the same KG signal, the difference is purely the **modelling paradigm**:
  prototype case-based reasoning vs KG bi-attention — and they show
  **complementary strengths** (balance vs overall accuracy).
