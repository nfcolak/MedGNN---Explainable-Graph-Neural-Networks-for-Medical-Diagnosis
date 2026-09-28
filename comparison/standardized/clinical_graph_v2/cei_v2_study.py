"""Bounded CEI-GNN v2 pair-interaction development study."""
from __future__ import annotations

from dataclasses import dataclass

STUDY_METHOD = "cei_gnn_v2"
MODES = ("product", "additive", "off")
SEEDS = (1234, 2025, 7)
SAMPLE_SEED = 1234
PAIR_RANK = 16
SMOKE_BUDGET = (256, 128, 2)
FULL_BUDGET = (10000, 5000, 40)
FULL_STAGE_NAMES = tuple(f"{mode}_seed{seed}" for seed in SEEDS for mode in MODES)

@dataclass(frozen=True)
class Stage:
    name: str
    argv: list
    output: str
    seed: int
    budget: tuple
    pair_mode: str

def stage_specs() -> tuple:
    raise NotImplementedError("stub")
def build_plan(*, artifact, targets, canonical, output_root) -> list[Stage]:
    raise NotImplementedError("stub")
def validate_exact_plan(stages):
    raise NotImplementedError("stub")
def validate_v2_method_config(binding):
    raise NotImplementedError("stub")
def validate_v2_binding(binding, *, budget, seed, mode):
    raise NotImplementedError("stub")
def assert_arm_parity(bindings: dict[str, dict]):
    raise NotImplementedError("stub")
