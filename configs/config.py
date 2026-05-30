import os
from pathlib import Path
import torch
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class DataParser():
    def __init__(self):
        super().__init__()
        # Intra-patient heterogeneous graph (Option A).
        # Patient-similarity star graphs carried near-zero label-relevant
        # topology (neighbour-label agreement ≈ class prior). The intra-patient
        # graph has PATIENT/VITAL/MED nodes + medication co-occurrence edges,
        # giving the GNN structure that's actually informative and prototypes
        # that map to interpretable clinical sub-patterns.
        # Ablation alternatives:
        #   'mimic_patient_sim_no_los'        — original star graph, no LOS
        #   'mimic_intra_patient_with_los'    — leakage ablation
        #   'mimic_intra_patient_full'        — full 309k patients
        self.dataset_name = 'mimic_intra_patient'
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
        self.latent_dim: List[int] = [128, 96, 64]
        # 'mean' (or 'sum') is sane for tabular star graphs; 'max' on L2-normalised
        # embeddings collapses information.
        self.readout: 'str' = 'mean'
        self.mlp_hidden: List[int] = [64]              # one MLP layer before output
        self.gnn_dropout: float = 0.5                  # regularise message passing
        self.dropout: float = 0.5
        self.adj_normlize: bool = True
        # L2-normalised node embeddings + L2 prototype distances squish dynamic range
        # onto the unit sphere → poor prototype separability. Disabled.
        self.emb_normlize: bool = False
        self.enable_prot = True
        self.num_prototypes_per_class = 5
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
        self.learning_rate = 0.001
        self.batch_size = 256              # larger batch → better GPU utilisation on MPS
        self.weight_decay = 5e-4           # stronger L2 regularisation to combat overfit
        self.max_epochs = 300
        self.warm_epochs = 20
        self.proj_epochs = 50
        self.early_stopping = 20           # stop sooner if no improvement
        self.last_layer_optimizer_lr = 1e-4            # the learning rate of the last layer
        self.joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}      # the learning rates of the joint training optimizer
        self.warm_epochs = 10                          # the number of warm epochs
        self.proj_epochs = 25                          # the epoch to start mcts
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
