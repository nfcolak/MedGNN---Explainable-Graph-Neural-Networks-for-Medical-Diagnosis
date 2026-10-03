"""Synthetic regression tests for CEI-GNN v2 inactive-parameter parity."""
from __future__ import annotations

import importlib.util
import copy
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "cei/studies/cei_v2_study.py"


def _module():
    assert MODULE_PATH.is_file(), "CEI-GNN v2 study module must exist"
    return importlib.import_module("cei.studies.cei_v2_study")


@lru_cache(maxsize=None)
def _source(mode, seed):
    from core import train
    from core.registry import build_method
    from core.tensorize import PAYLOAD_WIDTH

    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args([
        "--artifact", "a", "--targets", "t", "--output", "o", "--method", "cei_gnn_v2",
        "--train-limit", "10000", "--dev-limit", "5000", "--sample-seed", "1234",
        "--seed", str(seed), "--top-k-labels", "10", "--edges", "all",
        "--edge-direction", "forward", "--weights", "sqrt_inverse", "--selection-fold", "dev",
        "--final-eval", "none", "--epochs", "40", "--patience", "40",
        "--method-option", f"pair_mode={mode}"]), parser)
    model = build_method("cei_gnn_v2", num_tokens=8, node_dim=3, edge_dim=PAYLOAD_WIDTH,
                         num_classes=2, hidden=args.hidden, layers=args.layers,
                         dropout=args.dropout, token_dim=args.token_dim, num_triples=2, args=args)
    return json.loads(json.dumps(model.run_config())), {
        "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "min_delta": args.min_delta,
        "early_stopping_start_epoch_index": train.early_stopping_start_epoch("cei_gnn_v2", model)}


def _binding(mode="product", seed=2025):
    config, runner = copy.deepcopy(_source(mode, seed))
    arch = config["architecture"]
    return {
        "method": "cei_gnn_v2", "method_config": config,
        "method_native_defaults": config["native_defaults"],
        "selected_dev": {"epoch": 3, "epoch_index": 2, "metric": "macro_f1",
                         "metric_value": .5, "prediction_sha256": "pred", "sample_ids_sha256": "ids"},
        "artifact_graphs_sha256": "g", "artifact_visit_membership_sha256": "m",
        "targets_sha256": "t", "target_binding_sha256": "tb", "label_order": ["A", "B"],
        "source_code": {"x.py": "h"}, "preprocessing_sha256": "p",
        "split_sample_ids_sha256": {"train": "tr", "dev": "dv", "validation": "va"},
        "seed": seed, "sample_seed": 1234, "selection_fold": "dev", "final_eval": "none",
        "train_limit": 10000, "dev_limit": 5000, "epochs": 40, "patience": 40,
        "test_evaluated": False, "parameter_count": arch["parameter_count"],
        "active_parameter_count": arch["active_parameter_count"], "weights": "sqrt_inverse",
        "edge_direction": "forward", "edges": "all", "top_k_labels": 10,
        "message_passing": True, "edge_payload": True, "num_classes": 2,
        "input_contract_version": "v", "vocabulary_size": 8, "node_dim": 3,
        "edge_dim": arch["edge_dim"], "hidden": arch["hidden"], "layers": arch["layers"],
        "dropout": arch["dropout"], "num_relations": arch["num_relations"],
        "num_meta_relations": 2, **runner,
    }


def _arms():
    return {f"{mode}_seed{seed}": _binding(mode, seed)
            for seed in (1234, 2025, 7) for mode in ("product", "additive", "off")}


def _set_inactive(binding, count, top_level=False):
    binding["method_config"]["architecture"]["inactive_parameter_count"] = count
    if top_level:
        binding["inactive_parameter_count"] = count


def test_valid_three_mode_three_seed_set_is_accepted():
    assert _module().assert_arm_parity(_arms())


def test_rejects_nested_inactive_drift_product_vs_additive():
    arms = _arms()
    b = arms["additive_seed1234"]
    _set_inactive(b, b["method_config"]["architecture"]["inactive_parameter_count"] + 11)
    # The earliest rejection is the arithmetic guard; this is implied by active/total parity + P4 check; mutant P2 is equivalent (validator report §4).
    with pytest.raises(ValueError, match=r"arm parity"):
        _module().assert_arm_parity(arms)


def test_rejects_nested_inactive_drift_between_product_seeds():
    arms = _arms()
    b = arms["product_seed2025"]
    _set_inactive(b, b["method_config"]["architecture"]["inactive_parameter_count"] + 11)
    # Guard at cei_v2_study.py:258 raises "arm parity differs: inactive_parameter_count within product arms".
    with pytest.raises(ValueError, match=r"arm parity"):
        _module().assert_arm_parity(arms)


def test_rejects_inactive_not_total_minus_active():
    arms = _arms()
    for b in arms.values():
        architecture = b["method_config"]["architecture"]
        _set_inactive(b, architecture["inactive_parameter_count"] + 1)
    with pytest.raises(ValueError, match=re.escape("inactive_parameter_count = total - active")):
        _module().assert_arm_parity(arms)


def test_rejects_top_level_nested_inactive_mismatch():
    b = _binding("product", 1234)
    nested = b["method_config"]["architecture"]["inactive_parameter_count"]
    b["inactive_parameter_count"] = nested + 1
    with pytest.raises(ValueError, match=re.escape("top-level inactive_parameter_count differs from architecture")):
        _module().assert_arm_parity({"product_seed1234": b})


def test_rejects_off_inactive_not_larger():
    arms = _arms()
    product = arms["product_seed1234"]["method_config"]["architecture"]["inactive_parameter_count"]
    for b in arms.values():
        if b["method_config"]["effective_settings"]["pair_mode"] == "off":
            architecture = b["method_config"]["architecture"]
            delta = product - architecture["inactive_parameter_count"]
            architecture["active_parameter_count"] -= delta
            b["active_parameter_count"] -= delta
            _set_inactive(b, product)
    # Earliest rejection is off active-count ordering; scenario implied by active/total parity + P4 check; mutant P3 is equivalent (validator report §4).
    with pytest.raises(ValueError, match=r"arm parity"):
        _module().assert_arm_parity(arms)


def test_rejects_architecture_only_active_count_drift():
    arms = _arms()
    b = arms["product_seed1234"]
    b["method_config"]["architecture"]["active_parameter_count"] += 1
    with pytest.raises(ValueError, match=re.escape("parameter counts disagree with architecture")):
        _module().assert_arm_parity(arms)
