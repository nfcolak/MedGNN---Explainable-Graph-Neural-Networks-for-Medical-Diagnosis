"""Shared configuration used by BOTH analyses (protgnn_analysis, graphcare_analysis).

Keeps the things that must match across methods for a fair comparison:
the data location, the random seed, and the train/val/test split ratio.
"""
import random
from pathlib import Path

import numpy as np
import torch

# shared/lib/config_base.py  ->  parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"          # shared raw + processed data (merged_ed.csv)

SEED = 1234                            # governs split + model init across methods
SPLIT_RATIO = [0.8, 0.1, 0.1]          # train / val / test


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
