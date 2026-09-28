"""Synthetic contract tests for the CEI-GNN v2 pair study."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "comparison/standardized/clinical_graph_v2/cei_v2_study.py"


def _module():
    assert MODULE_PATH.is_file(), "CEI-GNN v2 study module must implement the locked planner"
    spec = importlib.util.spec_from_file_location("cei_v2_study_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _inputs(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    targets, canonical = tmp_path / "targets.csv", tmp_path / "canonical.json"
    targets.write_text("x")
    canonical.write_text("{}")
    return artifact, targets, canonical


@lru_cache(maxsize=None)
def _source(mode, seed, budget):
    from comparison.standardized.clinical_graph_v2 import train
    from comparison.standardized.clinical_graph_v2.methods import build_method
    from comparison.standardized.clinical_graph_v2.tensorize import PAYLOAD_WIDTH

    train_limit, dev_limit, epochs = budget
    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args([
        "--artifact", "a", "--targets", "t", "--output", "o", "--method", "cei_gnn_v2",
        "--train-limit", str(train_limit), "--dev-limit", str(dev_limit),
        "--sample-seed", "1234", "--seed", str(seed), "--top-k-labels", "10",
        "--edges", "all", "--edge-direction", "forward", "--weights", "sqrt_inverse",
        "--selection-fold", "dev", "--final-eval", "none", "--epochs", str(epochs),
        "--patience", "40", "--method-option", f"pair_mode={mode}"]), parser)
    model = build_method("cei_gnn_v2", num_tokens=8, node_dim=3, edge_dim=PAYLOAD_WIDTH,
                         num_classes=2, hidden=args.hidden, layers=args.layers,
                         dropout=args.dropout, token_dim=args.token_dim, num_triples=2,
                         args=args)
    runner = {"lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
              "min_delta": args.min_delta,
              "early_stopping_start_epoch_index": train.early_stopping_start_epoch("cei_gnn_v2", model)}
    return json.dumps(model.run_config()), json.dumps(runner)


def _binding(mode="product", seed=2025, budget=(10000, 5000, 40), **updates):
    config_json, runner_json = _source(mode, seed, budget)
    config, runner = json.loads(config_json), json.loads(runner_json)
    arch = config["architecture"]
    train_limit, dev_limit, epochs = budget
    binding = {
        "method": "cei_gnn_v2", "method_config": config,
        "method_native_defaults": config["native_defaults"],
        "artifact_graphs_sha256": "g", "artifact_visit_membership_sha256": "m",
        "targets_sha256": "t", "target_binding_sha256": "tb", "label_order": ["A", "B"],
        "source_code": {"x.py": "h"}, "preprocessing_sha256": "p",
        "split_sample_ids_sha256": {"train": "tr", "dev": "dv", "validation": "va"},
        "seed": seed, "sample_seed": 1234, "selection_fold": "dev", "final_eval": "none",
        "train_limit": train_limit, "dev_limit": dev_limit, "epochs": epochs, "patience": 40,
        "test_evaluated": False, "parameter_count": arch["parameter_count"],
        "active_parameter_count": arch["active_parameter_count"], "weights": "sqrt_inverse",
        "edge_direction": "forward", "edges": "all", "top_k_labels": 10,
        "message_passing": True, "edge_payload": True, "num_classes": 2,
        "input_contract_version": "v", "vocabulary_size": 8, "node_dim": 3,
        "edge_dim": arch["edge_dim"], "hidden": arch["hidden"], "layers": arch["layers"],
        "dropout": arch["dropout"], "num_relations": arch["num_relations"],
        "num_meta_relations": 2, **runner,
    }
    binding.update(updates)
    return binding


def test_plan_has_exact_ten_dev_only_stages(tmp_path):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    expected = [("v2_smoke", (256, 128, 2), 1234, "product")] + [
        (f"{mode}_seed{seed}", (10000, 5000, 40), seed, mode)
        for seed in (1234, 2025, 7) for mode in ("product", "additive", "off")]
    assert [(s.name, s.budget, s.seed, s.pair_mode) for s in stages] == expected
    for stage in stages:
        argv = stage.argv
        assert argv[argv.index("--method") + 1] == "cei_gnn_v2"
        assert argv[argv.index("--seed") + 1] == str(stage.seed)
        assert argv[argv.index("--sample-seed") + 1] == "1234"
        assert argv[argv.index("--final-eval") + 1] == "none"
        assert argv[argv.index("--selection-fold") + 1] == "dev"
        assert argv[argv.index("--method-option") + 1] == f"pair_mode={stage.pair_mode}"
        assert Path(argv[argv.index("--output") + 1]) == Path(stage.output)
        flags = [token for token in argv if token.startswith("--")]
        values = [argv[i + 1] for i, token in enumerate(argv[:-1]) if token.startswith("--")]
        assert "none" == argv[argv.index("--final-eval") + 1]
        assert "dev" == argv[argv.index("--selection-fold") + 1]
        assert not any("test" in flag for flag in flags)
        assert not any(value == "test" for value in values)


def test_plan_refuses_occupied_output_and_relative_paths(tmp_path):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    (tmp_path / "runs" / "off_seed7").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                          output_root=tmp_path / "runs")
    with pytest.raises(ValueError, match="absolute"):
        module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                          output_root="relative/runs")


def test_exact_plan_validator_rejects_edited_stage(tmp_path):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    module.validate_exact_plan(stages)
    edited = list(stages)
    edited[5] = dataclasses.replace(edited[5], argv=edited[5].argv[:-1] + ["pair_mode=product"])
    with pytest.raises(ValueError, match="argv"):
        module.validate_exact_plan(edited)
    with pytest.raises(ValueError, match="ten-stage"):
        module.validate_exact_plan(stages[:-1])


def test_binding_policy_accepts_source_config_and_rejects_drift():
    module = _module()
    module.validate_v2_binding(_binding("additive", 2025), budget=(10000, 5000, 40),
                               seed=2025, mode="additive")
    with pytest.raises(ValueError, match="seed"):
        module.validate_v2_binding(_binding("additive", 2025), budget=(10000, 5000, 40),
                                   seed=7, mode="additive")
    with pytest.raises(ValueError, match="pair_mode"):
        module.validate_v2_binding(_binding("additive", 2025), budget=(10000, 5000, 40),
                                   seed=2025, mode="product")
    with pytest.raises(ValueError, match="test_evaluated"):
        module.validate_v2_binding(_binding("off", 7, test_evaluated=True),
                                   budget=(10000, 5000, 40), seed=7, mode="off")
    drifted = _binding("product", 1234)
    drifted["method_config"]["architecture"]["pair_rank"] = 8
    with pytest.raises(ValueError, match="run_config"):
        module.validate_v2_binding(drifted, budget=(10000, 5000, 40), seed=1234, mode="product")


def test_binding_parameter_counts_match_each_arm_architecture():
    module = _module()
    binding = _binding("off", 7)
    binding["parameter_count"] += 1
    with pytest.raises(ValueError, match="parameter_count"):
        module.validate_v2_binding(binding, budget=(10000, 5000, 40), seed=7, mode="off")
    binding = _binding("off", 7)
    binding["active_parameter_count"] += 1
    with pytest.raises(ValueError, match="active_parameter_count"):
        module.validate_v2_binding(binding, budget=(10000, 5000, 40), seed=7, mode="off")


def test_arm_parity_allows_seed_and_mode_but_rejects_other_differences():
    module = _module()
    arms = {f"{mode}_seed{seed}": _binding(mode, seed)
            for seed in (1234, 2025, 7) for mode in ("product", "additive", "off")}
    assert module.assert_arm_parity(arms)
    arms["off_seed7"] = _binding("off", 7, split_sample_ids_sha256={
        "train": "other", "dev": "dv", "validation": "va"})
    with pytest.raises(ValueError, match="split_sample_ids_sha256"):
        module.assert_arm_parity(arms)
