"""Shared configuration used by all standardized analysis entry points.

Keeps the things that must match across methods for a fair comparison:
the data location, the random seed, and the train/val/test split ratio.
"""
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import os
from pathlib import Path
import random
import sys

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


@contextmanager
def preserve_process_state():
    """Restore process-global paths, environment, and supported RNG streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    environment = dict(os.environ)
    python_path = list(sys.path)

    cuda_state = None
    try:
        if torch.cuda.is_available():
            cuda_state = [state.clone() for state in torch.cuda.get_rng_state_all()]
    except (AssertionError, RuntimeError):
        cuda_state = None

    mps_state = None
    mps = getattr(torch, "mps", None)
    try:
        if mps is not None and hasattr(mps, "get_rng_state"):
            mps_state = mps.get_rng_state().clone()
    except (AssertionError, RuntimeError):
        mps_state = None

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None and mps is not None:
            mps.set_rng_state(mps_state)
        os.environ.clear()
        os.environ.update(environment)
        sys.path[:] = python_path


@contextmanager
def preserve_config_state(*config_objects):
    """Restore complete instance configuration dictionaries on every exit."""
    snapshots = [deepcopy(vars(config)) for config in config_objects]
    try:
        yield
    finally:
        for config, snapshot in zip(config_objects, snapshots):
            vars(config).clear()
            vars(config).update(snapshot)


def isolated_callable(*config_objects):
    """Decorate a public callable with process and configuration isolation."""
    def decorate(function):
        @wraps(function)
        def isolated(*args, **kwargs):
            with preserve_process_state(), preserve_config_state(*config_objects):
                return function(*args, **kwargs)
        return isolated
    return decorate
