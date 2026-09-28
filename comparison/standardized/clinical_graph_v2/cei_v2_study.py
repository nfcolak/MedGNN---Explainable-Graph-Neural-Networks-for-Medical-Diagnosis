"""Bounded CEI-GNN v2 pair-interaction development study.

Default CLI output is a print-only plan. Training needs `--execute smoke` or
`--execute full`; validation and test folds are never evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from comparison.standardized.clinical_graph_v2 import cei_pilot as pilot

STUDY_METHOD = "cei_gnn_v2"
MODES = ("product", "additive", "off")
SEEDS = (1234, 2025, 7)
SAMPLE_SEED = 1234
PAIR_RANK = 16
SMOKE_BUDGET = (256, 128, 2)
FULL_BUDGET = (10000, 5000, 40)
FULL_STAGE_NAMES = tuple(f"{mode}_seed{seed}" for seed in SEEDS for mode in MODES)
_PARITY_FIELDS = (
    "artifact_graphs_sha256", "artifact_visit_membership_sha256", "targets_sha256",
    "target_binding_sha256", "label_order", "source_code", "preprocessing_sha256",
    "split_sample_ids_sha256", "sample_seed", "train_limit", "dev_limit", "epochs",
    "patience", "selection_fold", "final_eval", "weights", "edges", "edge_direction",
    "top_k_labels", "num_classes", "input_contract_version", "parameter_count", "lr",
    "weight_decay", "batch_size", "min_delta", "hidden", "layers", "dropout",
    "vocabulary_size", "node_dim", "edge_dim", "num_relations", "num_meta_relations",
)


@dataclass(frozen=True)
class Stage:
    name: str
    argv: list
    output: str
    seed: int
    budget: tuple
    pair_mode: str


def stage_specs():
    specs = [("v2_smoke", SMOKE_BUDGET, 1234, "product")]
    specs.extend((f"{mode}_seed{seed}", FULL_BUDGET, seed, mode)
                 for seed in SEEDS for mode in MODES)
    return tuple(specs)


def _stage(name, artifact, targets, canonical, output, budget, seed, mode):
    train_limit, dev_limit, epochs = budget
    argv = [
        sys.executable, "-m", "comparison.standardized.clinical_graph_v2.train",
        "--artifact", str(artifact), "--targets", str(targets),
        "--canonical", str(canonical), "--output", str(output),
        "--method", STUDY_METHOD,
        "--train-limit", str(train_limit), "--dev-limit", str(dev_limit),
        "--sample-seed", str(SAMPLE_SEED), "--seed", str(seed),
        "--top-k-labels", "10", "--edges", "all",
        "--edge-direction", "forward", "--weights", "sqrt_inverse",
        "--selection-fold", "dev", "--final-eval", "none",
        "--epochs", str(epochs), "--patience", "40",
        "--method-option", f"pair_mode={mode}",
    ]
    return Stage(name, argv, str(output), seed, budget, mode)


def build_plan(*, artifact, targets, canonical, output_root):
    """Build exactly the approved ten stages without opening input contents."""
    artifact = pilot._absolute_existing(artifact, "artifact", directory=True)
    targets = pilot._absolute_existing(targets, "targets")
    canonical = pilot._absolute_existing(canonical, "canonical")
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        raise ValueError(f"output_root path must be absolute: {output_root}")
    root = root.resolve()
    stages = []
    for name, budget, seed, mode in stage_specs():
        output = root / name
        if output.exists():
            raise FileExistsError(f"Refusing occupied stage output {output}")
        stages.append(_stage(name, artifact, targets, canonical, output, budget, seed, mode))
    return stages


def validate_exact_plan(stages):
    """Reject any edit to the approved ten-stage plan."""
    specs = stage_specs()
    if len(stages) != len(specs):
        raise ValueError("execution requires the exact approved ten-stage plan")
    shared = None
    for stage, (name, budget, seed, mode) in zip(stages, specs):
        if (stage.name, tuple(stage.budget), stage.seed, stage.pair_mode) != (name, budget, seed, mode):
            raise ValueError(f"unauthorized stage tuple/order: {stage.name}")
        try:
            paths = tuple(stage.argv[stage.argv.index(flag) + 1]
                          for flag in ("--artifact", "--targets", "--canonical"))
        except (ValueError, IndexError) as error:
            raise ValueError(f"{name} argv is missing a required input path") from error
        if shared is None:
            shared = paths
        elif paths != shared:
            raise ValueError("stage plan input paths differ")
        expected = _stage(name, *paths, Path(stage.output), budget, seed, mode)
        if stage.argv != expected.argv:
            raise ValueError(f"unauthorized argv for {name}")


def validate_v2_method_config(binding):
    """Rebuild the adapter from source defaults and require an identical run_config."""
    config = binding.get("method_config")
    if not isinstance(config, dict) or config.get("method") != STUDY_METHOD \
            or binding.get("method") != STUDY_METHOD:
        raise ValueError("binding is not a cei_gnn_v2 run")
    settings = config.get("effective_settings")
    if not isinstance(settings, dict) or set(settings) != {"pair_rank", "pair_mode"}:
        raise ValueError("cei_gnn_v2 effective_settings must contain pair_rank and pair_mode")
    architecture = config.get("architecture")
    required = {"num_tokens", "node_dim", "edge_dim", "num_classes", "hidden", "layers",
                "dropout", "token_dim", "num_triples", "num_relations", "parameter_count",
                "active_parameter_count"}
    if not isinstance(architecture, dict) or not required.issubset(architecture):
        raise ValueError("cei_gnn_v2 architecture is missing dimensions/counts")
    top_level = {"num_tokens": "vocabulary_size", "node_dim": "node_dim", "edge_dim": "edge_dim",
                 "num_triples": "num_meta_relations", "num_classes": "num_classes",
                 "hidden": "hidden", "layers": "layers", "dropout": "dropout",
                 "num_relations": "num_relations"}
    for key, field in top_level.items():
        if binding.get(field) is None or architecture[key] != binding[field]:
            raise ValueError(f"runner binding {field} differs from method architecture")
    from comparison.standardized.clinical_graph_v2 import train
    from comparison.standardized.clinical_graph_v2.methods import build_method

    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args([
        "--artifact", "<bound>", "--targets", "<bound>", "--output", "<bound>",
        "--method", STUDY_METHOD,
        "--train-limit", str(binding.get("train_limit")),
        "--dev-limit", str(binding.get("dev_limit")),
        "--sample-seed", str(binding.get("sample_seed")), "--seed", str(binding.get("seed")),
        "--top-k-labels", "10", "--edges", "all", "--edge-direction", "forward",
        "--weights", "sqrt_inverse", "--selection-fold", "dev", "--final-eval", "none",
        "--epochs", str(binding.get("epochs")), "--patience", str(binding.get("patience")),
        "--method-option", f"pair_mode={settings['pair_mode']}"]), parser)
    model = build_method(
        STUDY_METHOD, num_tokens=architecture["num_tokens"], node_dim=architecture["node_dim"],
        edge_dim=architecture["edge_dim"], num_classes=architecture["num_classes"],
        hidden=args.hidden, layers=args.layers, dropout=args.dropout,
        token_dim=args.token_dim, num_triples=architecture["num_triples"], args=args)
    if model.run_config() != config:
        raise ValueError("cei_gnn_v2 source-derived run_config drift")
    if binding.get("early_stopping_start_epoch_index") != train.early_stopping_start_epoch(
            STUDY_METHOD, model):
        raise ValueError("cei_gnn_v2 early-stopping schedule start differs from source")
    for field in ("lr", "weight_decay", "batch_size", "min_delta"):
        if binding.get(field) != getattr(args, field):
            raise ValueError(f"runner binding {field} differs from source defaults")


def validate_v2_binding(binding, *, budget, seed, mode):
    """Validate one stage binding against the approved study policy."""
    for field in pilot._REQUIRED_BINDINGS:
        if field not in binding or binding[field] is None:
            raise ValueError(f"binding missing required field: {field}")
    train_limit, dev_limit, epochs = budget
    expected = {
        "method": STUDY_METHOD, "train_limit": train_limit, "dev_limit": dev_limit,
        "epochs": epochs, "patience": 40, "seed": seed, "sample_seed": SAMPLE_SEED,
        "selection_fold": "dev", "final_eval": "none", "test_evaluated": False,
        "weights": "sqrt_inverse", "edges": "all", "edge_direction": "forward",
        "top_k_labels": 10,
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"study policy mismatch for {key}: expected {value!r}")
    architecture = binding.get("method_config", {}).get("architecture", {})
    for field in ("parameter_count", "active_parameter_count"):
        if binding.get(field) != architecture.get(field):
            raise ValueError(f"{field} differs from this arm's method architecture")
    settings = binding.get("method_config", {}).get("effective_settings", {})
    if settings.get("pair_mode") != mode:
        raise ValueError(f"pair_mode mismatch: expected {mode!r}")
    if settings.get("pair_rank") != PAIR_RANK:
        raise ValueError(f"pair_rank mismatch: expected {PAIR_RANK}")
    validate_v2_method_config(binding)


def assert_arm_parity(bindings):
    """All full arms share inputs, splits, source and capacity; only seed/mode differ."""
    items = sorted(bindings.items())
    if not items:
        raise ValueError("no arms to compare")
    reference_name, reference = items[0]

    def shared_architecture(binding):
        architecture = dict(binding["method_config"]["architecture"])
        for key in ("active_parameter_count", "inactive_parameter_count"):
            architecture.pop(key, None)
        return architecture

    for name, binding in items[1:]:
        for field in _PARITY_FIELDS:
            if binding.get(field) != reference.get(field):
                raise ValueError(f"arm parity differs: {field} ({name} vs {reference_name})")
        if shared_architecture(binding) != shared_architecture(reference):
            raise ValueError(f"arm parity differs: architecture ({name})")
    return True
