"""GraphCare analysis configuration.

Reuses the shared data location / seed / split ratio so the comparison with
protgnn_analysis is fair. The GraphCare model itself is vendored upstream under
`external/GraphCare/` (see external/GraphCare/README.md) — we do NOT reimplement
it, we adapt our data into its input format and wrap its training.
"""
from pathlib import Path

from shared.lib.config_base import DATA_DIR, SEED, SPLIT_RATIO, get_device

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
UPSTREAM_DIR = Path(__file__).resolve().parents[1] / "external" / "GraphCare"


class GraphCareConfig:
    # --- shared (must match protgnn_analysis for a fair comparison) ---
    device = get_device()
    seed = SEED
    split_ratio = SPLIT_RATIO
    data_dir = DATA_DIR

    # --- paths ---
    OUTPUTS_DIR = OUTPUTS_DIR
    UPSTREAM_DIR = UPSTREAM_DIR

    # --- data ---
    csv_filename = "merged_ed.csv"
    target = "disease"                                   # 30-class disease_1
    num_classes = 30
    # Input eligibility by mode. Split-aware standardized preprocessing excludes
    # diagnosis-derived symptom_* while preserving pre-diagnosis chiefcomplaint_*.
    standardized_code_prefixes = ("med_", "pyx_", "chiefcomplaint_")
    legacy_code_prefixes = ("med_", "pyx_", "symptom_", "chiefcomplaint_")
    code_prefixes = legacy_code_prefixes
    # KG construction (mirror protgnn vocab/PMI so both methods share the KG signal)
    med_min_prev = 0.01          # keep meds with prevalence >= 1% (== protgnn)
    pmi_threshold = 2.0          # PMI > this -> "co_occurs" edge (== protgnn)

    # --- graph structure (topology) ---
    # One of shared.lib.graph_structures supported by graphcare
    # (star / cooccur / ontology / full / full_kg_expanded). Controls which
    # relations each patient subgraph includes. Common structures always add an
    # explicit patient hub and remain record-local; only full_kg_expanded adds
    # record-external one-hop KG neighbours. Selected via --graph_structure on
    # the CLI (or interactively). Per-structure KG caches live under
    # data/graphs/<structure>/graphcare/.
    graph_structure = "full"

    # --- personalized knowledge graph ---
    kg_path = OUTPUTS_DIR / "kg.pt"
    # MVP: ontology/PMI KG (cheap, reproducible).
    # True -> reproduce GraphCare's LLM(GPT-4)+UMLS personalized KG (heavier).
    use_llm_kg = False

    # --- BAT-GNN (upstream hyperparameters; tune) ---
    emb_dim = 128
    num_layers = 2
    dropout = 0.3
    lr = 1e-3
    weight_decay = 1e-5
    batch_size = 32
    max_epochs = 100


cfg = GraphCareConfig()
