import os
from pathlib import Path
import torch
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# data/ stays shared at the repo root; outputs/ lives inside this analysis folder
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


class DataParser():
    def __init__(self):
        super().__init__()
        # Intra-patient heterogeneous graph: each patient's own clinical record
        # becomes a graph (PATIENT/VITAL/MED/... nodes + concept co-occurrence
        # edges), giving the GNN informative structure and prototypes that map
        # to interpretable clinical sub-patterns.
        # Ablation alternatives:
        #   'mimic_intra_patient_with_los'    — leakage ablation
        #   'mimic_intra_patient_full'        — full 309k patients
        #   'mimic_intra_patient_disease'     — predict primary diagnosis
        self.dataset_name = 'mimic_intra_patient'
        # Graph structure (topology) — one of shared.lib.graph_structures
        # (star / cooccur / ontology / full). Selected via --graph_structure on
        # the CLI (or interactively if omitted); controls which concept↔concept
        # edges the intra-patient graph gets. Graph caches are collected under
        # data/graphs/<structure>/protgnn/.
        self.graph_structure = 'star'
        self.dataset_dir = str(DATA_DIR)
        self.task = None
        self.random_split: bool = True
        self.data_split_ratio: List = [0.8, 0.1, 0.1]   # the ratio of training, validation and testing set for random split
        self.seed = 1


class GATParser():# hyper-parameter for gat model
    def __init__(self):
        super().__init__()
        self.gat_dropout = 0.6    # dropout in gat layer
        self.gat_heads = 10         # multi-head
        self.gat_hidden = 10        # the hidden units for each head
        self.gat_concate = True    # the concatenation of the multi-head feature
        self.num_gat_layer = 3


class ModelParser():
    def __init__(self):
        super().__init__()
        # MPS (Apple Silicon) > CPU
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model_name: str = 'gcn'
        self.checkpoint: str = str(OUTPUTS_DIR / 'checkpoints')
        self.concate: bool = False
        # 3 GCN layers so messages can travel
        #   PATIENT → MED → MED (PMI-linked) → PATIENT
        # i.e. drug-cluster information propagates back to the patient node.
        # Width kept modest to prevent the early-epoch overfit seen on the
        # 20k sample.
        # HPO (disease, 20-trial best = Trial 15, val macro-F1=0.4801):
        #   3 layers × hidden 128, readout=max, dropout=0.51, emb_norm=True,
        #   lr=0.00179, batch=128, weight_decay=4.3e-5.
        # Top-5 trials all chose readout=max + emb_norm=True with high dropout.
        self.latent_dim: List[int] = [128, 128, 128]
        # HPO chose 'max' (on a 12k plain-GCN), but with prototypes + full data
        # max+emb_norm gave CATASTROPHIC eval instability (acc swung 0.05–0.28).
        # Reverted to 'mean' (repo's stable default). A/B-able vs max.
        self.readout: 'str' = 'mean'
        self.mlp_hidden: List[int] = [64]              # one MLP layer before output
        self.gnn_dropout: float = 0.5                  # (unused by current model)
        self.dropout: float = 0.51                     # HPO-selected (disease)
        self.adj_normlize: bool = True
        # HPO picked True (paired with max), but that combo destabilised eval on
        # the prototype/full-data run. L2-normalised embeddings + prototype L2
        # distances squish dynamic range → poor separability. Back to False.
        self.emb_normlize: bool = False
        self.enable_prot = True
        self.num_prototypes_per_class = 3              # 5→3: faster projection (90 vs 150 prototypes)
        self.gat_dropout = 0.6
        self.gat_heads = 10
        self.gat_hidden = 10
        self.gat_concate = True
        self.num_gat_layer = 3

    def process_args(self) -> None:
        pass


class MCTSParser(DataParser, ModelParser):
    # Lightweight MCTS profile — first run took ~20 min for a single
    # projection pass at rollout=10/max_atoms=8/expand=12 and the
    # process was killed before training completed. These settings
    # cut projection cost ~3-4x with negligible prototype-quality loss
    # on graphs of ~20 nodes.
    rollout: int = 3                          # 10 → 3
    high2low: bool = False
    c_puct: float = 5
    min_atoms: int = 3
    max_atoms: int = 6                        # 8 → 6
    expand_atoms: int = 8                     # 12 → 8

    def process_args(self) -> None:
        self.explain_model_path = os.path.join(self.checkpoint,
                                               self.dataset_name,
                                               f"{self.model_name}_best.pth")


class RewardParser():
    def __init__(self):
        super().__init__()
        self.reward_method: str = 'mc_l_shapley'                         # Liberal, gnn_score, mc_shapley, l_shapley， mc_l_shapley
        self.local_raduis: int = 4                                       # (n-1) hops neighbors for l_shapley
        self.subgraph_building_method: str = 'zero_filling'
        self.sample_num: int = 100                                       # sample time for monte carlo approximation


class TrainParser():
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.00179       # HPO-selected (disease, Trial 15)
        self.batch_size = 128              # HPO-selected (disease, Trial 15)
        self.weight_decay = 4.3e-5         # HPO value (5e-4 made eval WORSE+unstable, reverted)
        self.max_epochs = 300
        self.warm_epochs = 20
        self.proj_epochs = 50
        self.early_stopping = 10           # patience (epochs without meaningful gain)
        self.early_stop_min_delta = 0.005  # macro-F1 must improve by >0.5% to reset patience
        self.last_layer_optimizer_lr = 1e-4            # the learning rate of the last layer
        self.joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}      # the learning rates of the joint training optimizer
        self.warm_epochs = 10                          # the number of warm epochs
        self.proj_epochs = 20                          # near the eval peak (~ep13-17), reachable w/ early_stop=20
        self.proj_interval = 25                        # re-project every N epochs (projection is slow)
        self.sampling_epochs = 100                     # the epoch to start sampling edges
        self.nearest_graphs = 10                       # number of graphs in projection


data_args = DataParser()
model_args = ModelParser()
mcts_args = MCTSParser()
reward_args = RewardParser()
train_args = TrainParser()

import torch
import random
import numpy as np
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
