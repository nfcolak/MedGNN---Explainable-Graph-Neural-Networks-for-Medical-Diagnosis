"""GSAT analysis configuration.

Reuses the shared data location / seed / split ratio so the comparison with
protgnn_analysis and graphcare_analysis is fair. GSAT is reimplemented in
`gsat_analysis/models/` (the upstream Graph-COM/GSAT repo is vendored under
`external/GSAT/` as reference only — it is pinned to old torch/PyG and its own
dataset classes, so it is not importable here). See docs/PROJECT_CONTEXT.md.
"""
from pathlib import Path

from shared.lib.config_base import DATA_DIR, SEED, SPLIT_RATIO, get_device

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
# GSAT rides ProtGNN's per-patient graph cache verbatim — no separate builder.
UPSTREAM_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "external" / "GSAT"


class GSATConfig:
    # --- shared (must match the other two analyses for a fair comparison) ---
    device = get_device()
    seed = SEED
    split_ratio = SPLIT_RATIO
    data_dir = DATA_DIR

    # --- paths ---
    OUTPUTS_DIR = OUTPUTS_DIR

    # --- data (identical to graphcare_analysis / protgnn disease target) ---
    dataset_name = "mimic_intra_patient_disease"          # 30-class disease_1
    num_classes = 30

    # --- graph structure (topology) ---
    # One of shared.lib.graph_structures (star / cooccur / ontology / full).
    # Selected via --graph_structure on the CLI (or interactively). Per-structure
    # graph caches are ProtGNN's: data/graphs/<structure>/protgnn/.
    graph_structure = "star"

    # --- GIN backbone (GSAT's paper-default encoder, sized to match ProtGNN
    #     capacity: latent_dim [128,128,128]). mean readout to match both other
    #     methods' graph-level pooling (paper uses add-pool; deviation noted in
    #     docs/PROJECT_CONTEXT.md — readout is held constant across methods so
    #     the explanation mechanism is the only variable). ---
    hidden_dim = 128
    num_layers = 3
    dropout = 0.3
    readout = "mean"

    # --- stochastic attention / GIB (Miao et al. 2022, Sec. 4.2 + App. C.2.2) ---
    # Node-level attention (App. C.3: used for molecular / large-graph datasets;
    # our intra-patient graphs are exactly that regime). p_v per node is the
    # inherent explanation and plugs straight into shared/lib/fidelity.py.
    attention_level = "node"        # "node" | "edge"
    temperature = 1.0               # Gumbel-sigmoid temperature (not tuned, =1)
    info_loss_coef = 1.0            # beta in Eq. (8); paper leaves it at 1
    init_r = 0.9                    # curriculum: start loose ...
    final_r = 0.7                   # ... decay to r=0.7 (paper default, non-motif)
    decay_interval = 10             # every N epochs ...
    decay_r = 0.1                   # ... drop r by 0.1

    # --- optimisation ---
    lr = 1e-3
    weight_decay = 0.0
    batch_size = 128
    max_epochs = 100
    patience = 10                   # early-stop on val macro-F1 (post-r-decay)
    min_delta = 0.005


cfg = GSATConfig()
