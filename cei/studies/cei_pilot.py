"""Fail-closed planner and executor for the approved four-stage CEI dev pilot.

Default CLI behavior is print-only. Training is possible only with --execute and
is limited to the fixed smoke/control/candidate/product-off stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict, dataclass

import numpy as np

from core.paths import CODE_ROOTS


@dataclass(frozen=True)
class Stage:
    name: str
    argv: list[str]
    output: str
    seed: int
    budget: tuple[int, int, int]
    treatment: str


# These are the only paths whose values are expected to vary by arm. No generic
# key filtering is used; all other manifest fields (including unknown ones) compare.
_EXCLUDED_TOP_LEVEL = {
    "method", "adaptation_version", "method_native_defaults", "method_overrides",
    "conv", "parameter_count", "active_parameter_count",
    "selected_dev", "selected_validation", "selection", "device",
    "total_seconds", "output", "output_path",
    "use_interactions",
    "treatment",
    "early_stopping_start_epoch_index",
    "hidden", "layers", "dropout", "lr", "weight_decay", "batch_size", "min_delta",
}
_METHOD_CONFIG_KEYS = {"method", "adaptation_version", "native_defaults",
                       "effective_settings", "architecture", "native_schedule",
                       "objective_coefficients", "mechanism_settings"}
_CEI_EFFECTIVE_KEYS = {"num_tokens", "node_dim", "edge_dim", "num_classes", "hidden",
                       "token_dim", "num_triples", "num_relations", "dropout",
                       "interaction_rank", "use_interactions", "num_node_types"}
_CEI_ARCH_KEYS = {"parameter_count", "active_parameter_count"}
_CEI_ARCH_KEYS.update({"node_dim", "edge_dim", "num_tokens", "num_triples",
                       "num_relations", "num_node_types", "num_classes", "hidden",
                       "layers", "dropout", "token_dim", "interaction_rank",
                       "inactive_parameter_count"})
_PROTGNN_EFFECTIVE_KEYS = {"warm_epochs", "proj_epochs", "proj_interval", "nearest_graphs",
                           "prototypes_per_class", "cluster_weight", "separation_weight",
                           "margin", "rollout", "min_atoms", "max_atoms", "expand_atoms", "c_puct"}
_REQUIRED_BINDINGS = (
    "method", "artifact_graphs_sha256", "artifact_visit_membership_sha256",
    "targets_sha256", "target_binding_sha256", "label_order", "source_code",
    "preprocessing_sha256", "split_sample_ids_sha256", "seed", "sample_seed",
    "selection_fold", "final_eval", "train_limit", "dev_limit", "epochs",
    "test_evaluated", "parameter_count", "active_parameter_count", "weights",
    "edge_direction", "edges", "top_k_labels", "message_passing",
    "edge_payload", "num_classes", "input_contract_version",
)


def _sha_json(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(data).hexdigest()


def _absolute_existing(path, label, *, directory=False) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{label} path must be absolute: {path}")
    if not candidate.exists():
        raise FileNotFoundError(f"{label} path does not exist: {candidate}")
    candidate = candidate.resolve(strict=True)
    if directory and not candidate.is_dir():
        raise ValueError(f"{label} must be an existing directory: {candidate}")
    if not directory and not candidate.is_file():
        raise ValueError(f"{label} must be an existing file: {candidate}")
    return candidate


def _stage(name, method, artifact, targets, canonical, out, budget, treatment,
           *, product_off=False):
    train_limit, dev_limit, epochs = budget
    argv = [
        sys.executable, "-m", "core.train",
        "--artifact", str(artifact), "--targets", str(targets),
        "--canonical", str(canonical), "--output", str(out),
        "--method", method,
        "--train-limit", str(train_limit), "--dev-limit", str(dev_limit),
        "--sample-seed", "1234", "--seed", "1234",
        "--top-k-labels", "10", "--edges", "all",
        "--edge-direction", "forward", "--weights", "sqrt_inverse",
        "--selection-fold", "dev", "--final-eval", "none",
        "--epochs", str(epochs), "--patience", "40",
    ]
    if method == "cei_gnn":
        argv.extend(["--method-option", f"use_interactions={'false' if product_off else 'true'}"])
    return Stage(name, argv, str(out), 1234, budget, treatment)


def build_plan(*, artifact, targets, canonical, output_root):
    """Build exactly the authorized four stages, without touching input contents."""
    artifact = _absolute_existing(artifact, "artifact", directory=True)
    targets = _absolute_existing(targets, "targets")
    canonical = _absolute_existing(canonical, "canonical")
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        raise ValueError(f"output_root path must be absolute: {output_root}")
    root = root.resolve()
    plans = (
        ("cei_smoke", "cei_gnn", (256, 128, 2), "smoke", False),
        ("protgnn_control", "protgnn", (10000, 5000, 40), "control", False),
        ("cei_candidate", "cei_gnn", (10000, 5000, 40), "candidate", False),
        ("cei_product_off", "cei_gnn", (10000, 5000, 40), "product_off", True),
    )
    stages = []
    for name, method, budget, treatment, product_off in plans:
        output = root / name
        if output.exists():
            raise FileExistsError(f"Refusing occupied stage output {output}")
        stages.append(_stage(name, method, artifact, targets, canonical, output,
                             budget, treatment, product_off=product_off))
    return stages


def _first_difference(left, right, path=""):
    if type(left) is not type(right):
        return path or "value"
    if isinstance(left, dict):
        keys = sorted(set(left) | set(right))
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            ignored = {"use_interactions"} if path.endswith("effective_settings") else set()
            if key in ignored:
                continue
            if key not in left or key not in right:
                return child
            difference = _first_difference(left[key], right[key], child)
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return path
        for index, (a, b) in enumerate(zip(left, right)):
            difference = _first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
        return None
    return None if left == right else (path or "value")


def assert_common_bindings(left, right):
    """Require complete manifests and compare all non-treatment common fields."""
    for side_name, manifest in (("left", left), ("right", right)):
        if not isinstance(manifest, dict):
            raise ValueError(f"{side_name} binding must be an object")
        for field in _REQUIRED_BINDINGS:
            if field not in manifest or manifest[field] is None:
                raise ValueError(f"{side_name} binding missing required field: {field}")
            if field == "source_code" and (not isinstance(manifest[field], dict) or not manifest[field]):
                raise ValueError(f"{side_name} binding missing required field: source_code")
            if field == "label_order" and (not isinstance(manifest[field], list) or not manifest[field]):
                raise ValueError(f"{side_name} binding missing required field: label_order")
            if field == "split_sample_ids_sha256":
                value = manifest[field]
                if not isinstance(value, dict) or not {"train", "dev"}.issubset(value):
                    raise ValueError(f"{side_name} binding missing required field: split_sample_ids_sha256.train/dev")
    left_keys, right_keys = set(left), set(right)
    differing_keys = sorted((left_keys ^ right_keys) - _EXCLUDED_TOP_LEVEL)
    if differing_keys:
        raise ValueError(f"binding field set differs: {differing_keys[0]}")
    # The runner intentionally emits heterogeneous method_config schemas. Validate
    # each against its source-derived allowlist, then compare CEI treatment arms
    # without erasing unknown nested settings.
    for binding in (left, right):
        config = binding.get("method_config")
        if not isinstance(config, dict):
            raise ValueError("binding method_config must be an object")
        unknown = set(config) - _METHOD_CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown method_config key: {sorted(unknown)[0]}")
        if config.get("method") == "cei_gnn":
            settings = config.get("effective_settings", {})
            architecture = config.get("architecture", {})
            if not isinstance(settings, dict) or set(settings) - _CEI_EFFECTIVE_KEYS:
                raise ValueError("unknown CEI effective_settings key")
            if not isinstance(architecture, dict) or set(architecture) - _CEI_ARCH_KEYS:
                raise ValueError("unknown CEI architecture key")
        elif config.get("method") == "protgnn":
            settings = config.get("effective_settings", {})
            if not isinstance(settings, dict) or set(settings) - _PROTGNN_EFFECTIVE_KEYS:
                raise ValueError("unknown ProtGNN effective_settings key")
        _validate_method_config(binding)
    for key in sorted((left_keys & right_keys) - _EXCLUDED_TOP_LEVEL - {"method_config"}):
        difference = _first_difference(left[key], right[key], key)
        if difference:
            raise ValueError(f"common binding differs: {difference}")
    if left.get("method") == right.get("method") == "cei_gnn":
        a, b = left["method_config"], right["method_config"]
        normalized_a, normalized_b = json.loads(json.dumps(a)), json.loads(json.dumps(b))
        normalized_a.get("effective_settings", {}).pop("use_interactions", None)
        normalized_b.get("effective_settings", {}).pop("use_interactions", None)
        for normalized in (normalized_a, normalized_b):
            architecture = normalized.get("architecture", {})
            architecture.pop("active_parameter_count", None)
            architecture.pop("inactive_parameter_count", None)
        if normalized_a != normalized_b:
            raise ValueError("common binding differs: method_config")
    return True


def _validate_method_config(binding):
    """Recreate method configuration from train.py defaults and frozen pilot argv."""
    config = binding.get("method_config")
    if not isinstance(config, dict):
        raise ValueError("binding method_config must be an object")
    method = config.get("method", binding.get("method"))
    if method not in {"cei_gnn", "protgnn"} or binding.get("method") != method:
        raise ValueError(f"unsupported pilot method_config method: {method}")
    architecture = config.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("method_config architecture must be an object")
    required_dimensions = {"num_tokens", "node_dim", "edge_dim", "num_classes", "hidden",
                           "layers", "dropout", "token_dim", "num_triples", "num_relations",
                           "parameter_count"}
    active_count_key = "joint_active_parameter_count" if method == "protgnn" else "active_parameter_count"
    required_dimensions.add(active_count_key)
    if not required_dimensions.issubset(architecture):
        missing = sorted(required_dimensions - set(architecture))
        raise ValueError(f"method_config architecture is missing dimensions/counts: {missing[0]}")

    top_level_dimensions = {
        "num_tokens": binding.get("vocabulary_size"),
        "node_dim": binding.get("node_dim"),
        "edge_dim": binding.get("edge_dim"),
        "num_triples": binding.get("num_meta_relations"),
        "num_classes": binding.get("num_classes"),
        "hidden": binding.get("hidden"),
        "layers": binding.get("layers"),
        "dropout": binding.get("dropout"),
        "num_relations": binding.get("num_relations"),
    }
    for key, value in top_level_dimensions.items():
        if value is None or architecture[key] != value:
            raise ValueError(f"runner binding {key} differs from method architecture")
    if binding.get("token_dim") is not None and binding["token_dim"] != architecture["token_dim"]:
        raise ValueError("runner binding token_dim differs from method architecture")

    # These are the only policy overrides permitted for the pilot. In particular,
    # method-native defaults and effective settings below are never used as inputs.
    epochs, patience = binding.get("epochs"), binding.get("patience")
    if epochs not in (2, 40) or patience != 40:
        raise ValueError("method configuration violates the frozen epoch/patience policy")
    expected_train, expected_dev = (256, 128) if epochs == 2 else (10000, 5000)
    frozen_policy = {
        "train_limit": expected_train, "dev_limit": expected_dev,
        "seed": 1234, "sample_seed": 1234, "top_k_labels": 10,
        "edges": "all", "edge_direction": "forward", "weights": "sqrt_inverse",
        "selection_fold": "dev", "final_eval": "none", "test_evaluated": False,
    }
    for key, value in frozen_policy.items():
        if binding.get(key) != value:
            raise ValueError(f"runner binding violates frozen pilot policy for {key}")
    argv = ["--artifact", "<bound-artifact>", "--targets", "<bound-targets>",
            "--output", "<bound-output>", "--method", method,
            "--train-limit", str(binding.get("train_limit")),
            "--dev-limit", str(binding.get("dev_limit")),
            "--sample-seed", str(binding.get("sample_seed")),
            "--seed", str(binding.get("seed")),
            "--top-k-labels", str(binding.get("top_k_labels")),
            "--edges", str(binding.get("edges")),
            "--edge-direction", str(binding.get("edge_direction")),
            "--weights", str(binding.get("weights")),
            "--selection-fold", str(binding.get("selection_fold")),
            "--final-eval", str(binding.get("final_eval")),
            "--epochs", str(epochs), "--patience", str(patience)]
    if method == "cei_gnn":
        effective = config.get("effective_settings")
        if not isinstance(effective, dict) or set(effective) != {"interaction_rank", "use_interactions"}:
            raise ValueError("CEI effective settings must contain only the product treatment")
        if type(effective.get("use_interactions")) is not bool:
            raise ValueError("CEI use_interactions must be the sole boolean treatment")
        argv.extend(["--method-option", f"use_interactions={'true' if effective['use_interactions'] else 'false'}"])

    from core import train
    parser = train.parser()
    args = parser.parse_args(argv)
    args = train.normalize_method_args(args, parser)
    if architecture["hidden"] != args.hidden or architecture["layers"] != args.layers:
        raise ValueError("method architecture differs from source runner defaults")
    if architecture["dropout"] != args.dropout or architecture["token_dim"] != args.token_dim:
        raise ValueError("method architecture differs from source runner dimensions")
    args.num_relations = architecture["num_relations"]
    args.num_triples = architecture["num_triples"]
    from core.registry import build_method
    model = build_method(
        method, num_tokens=architecture["num_tokens"], node_dim=architecture["node_dim"],
        edge_dim=architecture["edge_dim"], num_classes=architecture["num_classes"],
        hidden=args.hidden, layers=args.layers, dropout=args.dropout,
        token_dim=args.token_dim, num_triples=architecture["num_triples"], args=args)
    if model.run_config() != config:
        raise ValueError(f"{method} source-derived defaults/schedule/run_config drift")
    expected_start = train.early_stopping_start_epoch(method, model)
    if binding.get("early_stopping_start_epoch_index") != expected_start:
        raise ValueError(f"{method} early-stopping schedule start differs from source")
    for field in ("lr", "weight_decay", "batch_size", "min_delta"):
        if binding.get(field) != getattr(args, field):
            raise ValueError(f"runner binding {field} differs from source defaults/pilot policy")


def validate_pilot_binding(binding, *, expected_budget, expected_seed=1234):
    """Validate complete budget/fold policy before allowing any comparison."""
    for field in _REQUIRED_BINDINGS:
        if field not in binding or binding[field] is None:
            raise ValueError(f"binding missing required field: {field}")
    train_limit, dev_limit, epochs = expected_budget
    expected = {
        "train_limit": train_limit, "dev_limit": dev_limit, "epochs": epochs,
        "patience": 40,
        "seed": expected_seed, "sample_seed": 1234, "selection_fold": "dev",
        "final_eval": "none", "test_evaluated": False, "weights": "sqrt_inverse",
        "edges": "all", "edge_direction": "forward", "top_k_labels": 10,
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            label = "test-fold guard" if key == "test_evaluated" else "pilot budget/policy"
            raise ValueError(f"{label} mismatch for {key}: expected {value!r}")


def _load_json(path, label):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object: {path}")
    return value


def _validate_treatment(binding, treatment):
    expected_method = "protgnn" if treatment == "control" else "cei_gnn"
    if binding.get("method") != expected_method:
        raise ValueError(f"{treatment} method mismatch: expected {expected_method}")
    if treatment in {"candidate", "product_off", "smoke"}:
        method_config = binding.get("method_config")
        settings = method_config.get("effective_settings") if isinstance(method_config, dict) else None
        value = settings.get("use_interactions") if isinstance(settings, dict) else binding.get("use_interactions")
        expected = treatment != "product_off"
        if type(value) is not bool or value is not expected:
            raise ValueError(f"{treatment} use_interactions treatment mismatch; expected {expected}")


def _validate_exact_stage_plan(stages):
    """Reject any argv/metadata edit to the four frozen build_plan tuples."""
    specs = (
        ("cei_smoke", "cei_gnn", (256, 128, 2), "smoke", False),
        ("protgnn_control", "protgnn", (10000, 5000, 40), "control", False),
        ("cei_candidate", "cei_gnn", (10000, 5000, 40), "candidate", False),
        ("cei_product_off", "cei_gnn", (10000, 5000, 40), "product_off", True),
    )
    if len(stages) != len(specs):
        raise ValueError("execution requires the exact authorized four-stage plan")
    shared_paths = None
    for stage, (name, method, budget, treatment, product_off) in zip(stages, specs):
        if (stage.name, stage.budget, stage.seed, stage.treatment) != (
                name, budget, 1234, treatment):
            raise ValueError(f"unauthorized stage tuple/order: {stage.name}")
        try:
            paths = tuple(stage.argv[stage.argv.index(flag) + 1]
                          for flag in ("--artifact", "--targets", "--canonical"))
        except (ValueError, IndexError) as error:
            raise ValueError(f"{name} argv is missing required input path") from error
        if shared_paths is None:
            shared_paths = paths
        elif paths != shared_paths:
            raise ValueError("stage plan input paths differ")
        if not all(Path(path).is_absolute() for path in paths):
            raise ValueError("stage plan inputs must use build_plan's absolute paths")
        expected = _stage(name, method, *paths, Path(stage.output), budget, treatment,
                          product_off=product_off)
        if stage.argv != expected.argv:
            raise ValueError(f"unauthorized argv for {name}")
        if Path(stage.output).resolve() != Path(
                stage.argv[stage.argv.index("--output") + 1]).resolve():
            raise ValueError(f"stage output differs from argv for {name}")


def validate_completed_stages(stage_dirs):
    """Read complete dev-stage manifests, verify bindings, then expose summaries.

    Smoke is never an aggregate arm. Only the three 40-epoch arms are accepted.
    """
    directories = [Path(path).expanduser().resolve(strict=True) for path in stage_dirs]
    if len(directories) != 3:
        raise ValueError("report/validate requires exactly the three non-smoke stage directories")
    expected_names = {"protgnn_control", "cei_candidate", "cei_product_off"}
    if {path.name for path in directories} != expected_names:
        raise ValueError("stage directories must be control, candidate and product-off; exclude smoke")
    loaded = {}
    for directory in directories:
        binding_path, result_path = directory / "binding.json", directory / "result.json"
        if not binding_path.is_file():
            raise ValueError(f"binding.json missing in {directory}")
        if not result_path.is_file():
            raise ValueError(f"result.json missing in {directory}")
        binding = _load_json(binding_path, "binding.json")
        validate_pilot_binding(binding, expected_budget=(10000, 5000, 40))
        treatment = {"protgnn_control": "control", "cei_candidate": "candidate",
                     "cei_product_off": "product_off"}[directory.name]
        _validate_treatment(binding, treatment)
        loaded[directory.name] = {"binding": binding, "result_path": result_path}
    arms = [(name, loaded[name]["binding"]) for name in sorted(expected_names)]
    for index, (name, binding) in enumerate(arms):
        for other_name, other in arms[index + 1:]:
            assert_common_bindings(binding, other)
    candidate = loaded["cei_candidate"]["binding"]
    product_off = loaded["cei_product_off"]["binding"]
    if "parameter_shapes" in candidate or "parameter_shapes" in product_off:
        if candidate.get("parameter_shapes") != product_off.get("parameter_shapes"):
            raise ValueError("candidate/product-off total parameter_shapes differs")
    if "architecture" in candidate or "architecture" in product_off:
        left_arch = candidate.get("architecture", {})
        right_arch = product_off.get("architecture", {})
        if not isinstance(left_arch, dict) or not isinstance(right_arch, dict):
            raise ValueError("candidate/product-off architecture binding is malformed")
        for shape_field in ("parameter_count", "parameter_shapes"):
            if shape_field in left_arch or shape_field in right_arch:
                if left_arch.get(shape_field) != right_arch.get(shape_field):
                    raise ValueError(f"candidate/product-off total {shape_field} differs")
    for field in ("parameter_count", "method_config"):
        if candidate.get(field) != product_off.get(field):
            if field == "method_config":
                left_arch = candidate[field].get("architecture", {})
                right_arch = product_off[field].get("architecture", {})
                for shape_field in ("parameter_shapes", "parameter_count"):
                    if shape_field in left_arch or shape_field in right_arch:
                        if left_arch.get(shape_field) != right_arch.get(shape_field):
                            raise ValueError(f"candidate/product-off total {shape_field} differs")
                if left_arch.get("parameter_count") != right_arch.get("parameter_count"):
                    raise ValueError("candidate/product-off total parameter_count differs")
            else:
                raise ValueError("candidate/product-off total parameter_count differs")
    for name in sorted(expected_names):
        directory = next(path for path in directories if path.name == name)
        binding = loaded[name]["binding"]
        result = _load_json(loaded[name]["result_path"], "result.json")
        if result.get("status") != "completed":
            raise ValueError(f"incomplete result status in {loaded[name]['result_path']}")
        if result.get("binding") != binding:
            raise ValueError(f"result/binding mismatch in {name}")
        if result.get("test_evaluated", binding.get("test_evaluated")) is not False:
            raise ValueError(f"test-fold guard failed in {name}")
        if result.get("metrics") is not None or not isinstance(result.get("dev_metrics"), dict):
            raise ValueError(f"dev-only result contract mismatch in {name}")
        _validate_artifacts(directory, binding, result)
        loaded[name]["replay"] = replay_stage(directory, binding, result)
        loaded[name]["result"] = result
    summaries = []
    for name in sorted(expected_names):
        binding, result = loaded[name]["binding"], loaded[name]["result"]
        summaries.append({
            "stage": name, "status": "completed", "dev_metrics": result["dev_metrics"],
            "parameter_count": binding["parameter_count"],
            "active_parameter_count": binding["active_parameter_count"],
            "dev_sample_ids_sha256": binding["split_sample_ids_sha256"].get("dev"),
        })
    return {"status": "compatible", "arms": summaries, "test_evaluated": False}


def _finite_tree(value, path="result"):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"nonfinite value in {path}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")


def _validate_artifacts(directory, binding, result):
    """Check train.py's actual history and npz result contract before reporting."""
    history_path, dev_path, checkpoint = (directory / "history.json", directory / "dev.npz",
                                          directory / "best.pt")
    if not history_path.is_file() or not dev_path.is_file() or not checkpoint.is_file():
        raise ValueError(f"required history.json, dev.npz, or best.pt missing in {directory}")
    history = json.loads(history_path.read_text())
    if not isinstance(history, list) or len(history) != result["binding"]["epochs"]:
        raise ValueError(f"history length differs from bound epochs in {directory}")
    _finite_tree(history, "history")
    _finite_tree(result.get("dev_metrics"), "dev_metrics")
    if not isinstance(result.get("total_seconds"), (int, float)) or not math.isfinite(result["total_seconds"]):
        raise ValueError(f"finite total_seconds missing in {directory}")
    for row in history:
        if not isinstance(row, dict) or not {"epoch", "train_loss", "seconds"}.issubset(row):
            raise ValueError(f"incomplete optimization history in {directory}")
    with np.load(dev_path, allow_pickle=False) as saved:
        required = {"proba", "y", "sample_ids"}
        if not required.issubset(saved.files):
            raise ValueError(f"dev.npz missing {sorted(required - set(saved.files))}")
        proba, y = saved["proba"], saved["y"]
        ids = saved["sample_ids"]
    if proba.ndim != 2 or y.ndim != 1 or ids.ndim != 1 or not (len(proba) == len(y) == len(ids)):
        raise ValueError(f"dev prediction alignment invalid in {directory}")
    if not np.isfinite(proba).all() or not np.isfinite(y).all() or (proba < 0).any():
        raise ValueError(f"nonfinite/invalid dev predictions in {directory}")
    digest = hashlib.sha256(np.ascontiguousarray(proba).tobytes()).hexdigest()
    selected = binding.get("selected_dev")
    if not isinstance(selected, dict) or selected.get("prediction_sha256") != digest:
        raise ValueError(f"dev prediction digest mismatch in {directory}")
    if result.get("dev_proba_sha256") != digest or selected.get("sample_ids_sha256") != binding["split_sample_ids_sha256"]["dev"]:
        raise ValueError(f"dev prediction binding mismatch in {directory}")
    if ids.size == 0:
        raise ValueError(f"empty dev sample IDs in {directory}")
    id_digest = hashlib.sha256(json.dumps(ids.tolist(), separators=(",", ":"),
                                ensure_ascii=True).encode("utf-8")).hexdigest()
    if id_digest != binding["split_sample_ids_sha256"]["dev"]:
        raise ValueError(f"dev sample IDs differ from binding in {directory}")
    _finite_tree(proba.tolist(), "dev_proba")


def replay_stage(output_dir, binding, result):
    """Rebuild the bound adapter and replay only saved dev predictions exactly.

    Dataset construction may materialize validation tensors as part of the runner
    helper contract, but this function never creates test tensors or evaluates
    validation. A proof is written with exclusive-create semantics.
    """
    output = Path(output_dir)
    if result.get("status") != "completed" or result.get("binding") != binding:
        raise ValueError("replay requires a completed result bound to binding.json")
    if (binding.get("selection_fold") != "dev" or binding.get("final_eval") != "none"
            or binding.get("test_evaluated") is not False or result.get("metrics") is not None):
        raise ValueError("replay is restricted to dev-selected, no-final-evaluation runs")
    if result.get("validation_evaluations") != 0:
        raise ValueError("validation evaluation is forbidden during replay")
    if result.get("test_evaluated", False) is not False:
        raise ValueError("test evaluation is forbidden during replay")
    if (output / "validation.npz").exists():
        raise ValueError("validation prediction artifact is forbidden")

    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from types import SimpleNamespace
    from core import train
    from core.contracts import code_source_hashes, sample_ids_sha256
    from core.registry import build_method
    from core.schema import sha256
    from core.tensorize import preprocessing_state

    source = code_source_hashes()
    if source != binding.get("source_code"):
        raise ValueError("replay source code differs from run binding")
    artifact = Path(binding["artifact"])
    if sha256(artifact / "graphs.jsonl") != binding["artifact_graphs_sha256"]:
        raise ValueError("replay graph artifact hash differs")
    if sha256(artifact / binding["artifact_visit_membership_file"]) != binding["artifact_visit_membership_sha256"]:
        raise ValueError("replay membership artifact hash differs")
    if sha256(binding["targets_path"]) != binding["targets_sha256"]:
        raise ValueError("replay targets hash differs")
    prep_path = output / "preprocessing.json"
    if sha256(prep_path) != binding["preprocessing_sha256"]:
        raise ValueError("replay preprocessing file hash differs")

    targets = train.load_targets(binding["targets_path"])
    targets, kept, _ = train.select_top_labels(targets, binding["top_k_labels"])
    if kept != binding["kept_label_indices"]:
        raise ValueError("replay top-label order differs")
    splits, prep = train.build_dataset(
        artifact, targets, binding["edges"], binding["train_limit"],
        binding["token_min_count"], binding["seed"],
        drop_relations=tuple(binding["dropped_relations"]),
        rewire_relations=tuple(binding["rewired_relations"]),
        min_prior_visits=binding["min_prior_visits"],
        edge_direction=binding["edge_direction"], dev_limit=binding["dev_limit"],
        sample_seed=binding["sample_seed"])
    if "test" in splits or set(("train", "dev", "validation")) - set(splits):
        raise ValueError("replay dataset splits violate train/dev-only selection contract")
    hashes = {fold: sample_ids_sha256(row.sample_id for row in rows)
              for fold, rows in splits.items()}
    if hashes != binding["split_sample_ids_sha256"]:
        raise ValueError("replayed split sample identities differ")
    if preprocessing_state(prep) != json.loads(prep_path.read_text()):
        raise ValueError("replayed preprocessing state differs")
    if {row.subject for row in splits["train"]} & {row.subject for row in splits["dev"]}:
        raise ValueError("replayed train/dev patient overlap")

    method_config = binding["method_config"]
    _validate_method_config(binding)
    architecture = method_config["architecture"]
    args = dict(binding)
    args.update(method_config.get("effective_settings", {}))
    args["num_relations"] = architecture["num_relations"]
    if binding["method"] == "cei_gnn":
        args["method_options"] = dict(method_config.get("effective_settings", {}))
    model = build_method(
        binding["method"], num_tokens=binding["vocabulary_size"],
        node_dim=binding["node_dim"], edge_dim=binding["edge_dim"],
        num_classes=binding["num_classes"], hidden=binding["hidden"],
        layers=binding["layers"], dropout=binding["dropout"],
        token_dim=architecture["token_dim"], num_triples=binding["num_meta_relations"],
        args=SimpleNamespace(**args))
    checkpoint = output / "best.pt"
    checkpoint_hash = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    if sum(param.numel() for param in model.parameters()) != binding["parameter_count"]:
        raise ValueError("replayed parameter count differs")
    loader = DataLoader(splits["dev"], batch_size=binding["batch_size"], shuffle=False)
    probabilities, labels = train.evaluate(
        model, loader, torch.device("cpu"), epoch=binding["selected_dev"]["epoch_index"])
    metrics = train.metrics(labels, probabilities, binding["num_classes"])
    dev_path = output / "dev.npz"
    with np.load(dev_path, allow_pickle=False) as saved:
        ids = np.asarray([row.sample_id for row in splits["dev"]])
        if not np.array_equal(ids, saved["sample_ids"]):
            raise ValueError("replayed dev sample IDs differ")
        if not np.array_equal(labels, saved["y"]):
            raise ValueError("replayed dev labels differ")
        if not np.array_equal(probabilities, saved["proba"]):
            raise ValueError("checkpoint replay probabilities differ")
        prediction_hash = hashlib.sha256(np.ascontiguousarray(saved["proba"]).tobytes()).hexdigest()
        labels_hash = hashlib.sha256(np.ascontiguousarray(saved["y"]).tobytes()).hexdigest()
        ids_hash = hashlib.sha256(np.ascontiguousarray(saved["sample_ids"]).tobytes()).hexdigest()
    if prediction_hash != binding["selected_dev"]["prediction_sha256"]:
        raise ValueError("replayed probabilities do not match selected checkpoint proof")
    if metrics != result.get("dev_metrics"):
        raise ValueError("replayed dev metrics differ")
    proof_path = output / "replay.json"
    proof = {"status": "verified", "method": binding["method"],
             "checkpoint_sha256": checkpoint_hash, "proba_sha256": prediction_hash,
             "labels_sha256": labels_hash, "sample_ids_sha256": ids_hash,
             "source_code": source, "preprocessing_sha256": binding["preprocessing_sha256"],
             "input_hashes": {"graphs": binding["artifact_graphs_sha256"],
                              "visit_membership": binding["artifact_visit_membership_sha256"],
                              "targets": binding["targets_sha256"]},
             "inputs_verified": True, "source_verified": True,
             "preprocessing_verified": True,
             "split_sample_ids_sha256": hashes, "dev_count": len(labels),
             "exact_probabilities": True, "exact_labels": True,
             "exact_sample_identity": True, "patient_disjoint": True,
             "validation_evaluated": False, "test_evaluated": False,
             "dev_metrics": metrics}
    if proof_path.exists():
        existing = _load_json(proof_path, "replay.json")
        for key in ("checkpoint_sha256", "proba_sha256", "labels_sha256", "sample_ids_sha256",
                    "source_code", "preprocessing_sha256", "split_sample_ids_sha256",
                    "input_hashes", "inputs_verified", "source_verified",
                    "preprocessing_verified", "dev_metrics", "validation_evaluated",
                    "test_evaluated"):
            if existing.get(key) != proof[key]:
                raise ValueError(f"existing replay proof {key} differs")
        if existing.get("status") != "verified":
            raise ValueError("existing replay proof is not verified")
        return existing
    with proof_path.open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return proof


def capture_bindings(stage):
    """Capture execution identity before launch without opening clinical inputs."""
    revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain=v1"], check=True,
                            capture_output=True, text=True).stdout
    root_value = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True,
                                capture_output=True, text=True).stdout.strip()
    repo_root = Path(root_value).resolve(strict=True)
    try:
        Path(__file__).resolve(strict=True).relative_to(repo_root)
    except ValueError as error:
        raise ValueError("active runner file is outside git's reported worktree") from error
    executable_sources = _capture_executable_sources(repo_root)
    source_state = _sha_json({"revision": revision, "working_tree": status,
                              "clinical_graph_v2_source": executable_sources["source_hashes"]})
    inputs = {"artifact": stage.argv[stage.argv.index("--artifact") + 1],
              "targets": stage.argv[stage.argv.index("--targets") + 1],
              "canonical": stage.argv[stage.argv.index("--canonical") + 1]}
    input_hashes = {}
    for label, raw_path in inputs.items():
        path = Path(raw_path).resolve(strict=True)
        if path.is_dir():
            files = {child.relative_to(path).as_posix(): hashlib.sha256(child.read_bytes()).hexdigest()
                     for child in sorted(path.rglob("*")) if child.is_file()}
            input_hashes[label] = _sha_json(files)
        else:
            input_hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    removed = {"--output", "--train-limit", "--dev-limit", "--epochs", "--method",
               "--method-option", "--patience"}
    shared_argv = []
    skip_value = False
    for token in stage.argv:
        if skip_value:
            skip_value = False
            continue
        if token in removed:
            skip_value = True
            continue
        shared_argv.append(token)
    return {
        "git_revision": revision,
        "source_state_sha256": source_state,
        "executable_sources": executable_sources,
        "argv_sha256": _sha_json(stage.argv),
        "common_config_sha256": _sha_json(shared_argv),
        "input_state_sha256": _sha_json(input_hashes),
        "inputs": inputs,
        "input_hashes": input_hashes,
    }


def _execution_identity(captured):
    return (captured["source_state_sha256"],
            captured.get("input_state_sha256", _sha_json(captured.get("inputs", {}))),
            captured.get("common_config_sha256", ""))


def _write_journal(path, journal):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(journal, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _capture_executable_sources(repo_root):
    """Hash executable source bytes rooted at the active repository worktree."""
    repo_root = Path(repo_root).resolve(strict=True)
    package = repo_root / "comparison/standardized/clinical_graph_v2"
    required = (package / "methods/cei/studies/cei_pilot.py", package / "methods/cei/plugin_cei_gnn.py")
    if any(not path.is_file() for path in required):
        raise ValueError("active worktree is missing CEI runner or plugin source")
    source_roots = (package, repo_root / "shared/lib", repo_root / "external")
    source_hashes = {}
    for root in source_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" not in path.parts and ".venv" not in path.parts:
                source_hashes[path.relative_to(repo_root).as_posix()] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
    prefix = "comparison/standardized/clinical_graph_v2/"
    clinical = {path[len(prefix):]: digest for path, digest in source_hashes.items()
                if path.startswith(prefix)}
    if not {"methods/cei/studies/cei_pilot.py", "methods/cei/plugin_cei_gnn.py"} <= set(clinical):
        raise ValueError("executable source map omits the CEI runner or plugin")
    return {"source_hashes": source_hashes,
            "clinical_source_hashes": clinical,
            "source_state_sha256": _sha_json(source_hashes)}


def _assert_runner_source_binding(snapshot, binding):
    """Require package bytes used by the executor to equal the runner's binding."""
    runner_source = binding.get("source_code")
    if not isinstance(runner_source, dict) or not runner_source:
        raise ValueError("runner source binding is missing")
    if runner_source != snapshot.get("clinical_source_hashes"):
        raise ValueError("runner source binding differs from active worktree bytes")
    return True


def execute_plan(stages, *, journal_path):
    """Run fixed stages sequentially, journaling every outcome and refusing drift."""
    _validate_exact_stage_plan(stages)
    journal_path = Path(journal_path)
    if journal_path.exists():
        raise FileExistsError(f"Refusing occupied journal {journal_path}")
    journal = {"status": "running", "stages": {}}
    expected_identity = None
    completed_bindings = {}
    _write_journal(journal_path, journal)
    for stage in stages:
        output = Path(stage.output)
        if output.exists():
            journal["stages"][stage.name] = {"status": "refused", "reason": "occupied output"}
            journal["status"] = "failed"
            _write_journal(journal_path, journal)
            raise SystemExit(2)
        before = capture_bindings(stage)
        if expected_identity is not None and _execution_identity(before) != expected_identity:
            journal["stages"][stage.name] = {"status": "refused", "reason": "source/input/config drift"}
            journal["status"] = "failed"
            _write_journal(journal_path, journal)
            raise SystemExit(3)
        if expected_identity is None:
            expected_identity = _execution_identity(before)
        journal["stages"][stage.name] = {"status": "running", "preflight": before}
        _write_journal(journal_path, journal)
        try:
            invocation = list(stage.argv)
            if "--execute" not in invocation:
                invocation.append("--execute")
            completed = subprocess.run(invocation, check=True, capture_output=True, text=True)
            after = capture_bindings(stage)
            if _execution_identity(after) != expected_identity:
                raise RuntimeError("source/input/config drift detected after stage execution")
            binding_file = output / "binding.json"
            result_file = output / "result.json"
            if not binding_file.is_file() or not result_file.is_file():
                raise RuntimeError("training returned without complete binding/result files")
            binding = _load_json(binding_file, "binding.json")
            _assert_runner_source_binding(after["executable_sources"], binding)
            validate_pilot_binding(binding, expected_budget=stage.budget, expected_seed=stage.seed)
            _validate_treatment(binding, stage.treatment)
            result = _load_json(result_file, "result.json")
            if result.get("status") != "completed" or result.get("binding") != binding:
                raise RuntimeError(f"{stage.name} did not produce a completed bound result")
            if result.get("metrics") is not None or not isinstance(result.get("dev_metrics"), dict):
                raise RuntimeError(f"{stage.name} violated dev-only result contract")
            if result.get("test_evaluated", False) is not False:
                raise RuntimeError(f"{stage.name} evaluated test fold")
            _validate_artifacts(output, binding, result)
            replay = replay_stage(output, binding, result)
            completed_bindings[stage.name] = binding
            journal["stages"][stage.name].update({"status": "bound", "postflight": after,
                                                  "replay": replay,
                                                  "stdout": completed.stdout,
                                                  "stderr": completed.stderr})
        except Exception as error:
            journal["stages"][stage.name].update({"status": "failed", "error": str(error)})
            journal["status"] = "failed"
            _write_journal(journal_path, journal)
            raise SystemExit(1) from error
        _write_journal(journal_path, journal)
    try:
        expected_names = {"cei_smoke", "protgnn_control", "cei_candidate", "cei_product_off"}
        if set(completed_bindings) != expected_names:
            raise ValueError("executor did not complete the exact four pilot stages")
        output_root = Path(stages[0].output).parent
        validate_completed_stages([
            output_root / "protgnn_control", output_root / "cei_candidate",
            output_root / "cei_product_off",
        ])
        smoke = _load_json(output_root / "cei_smoke" / "result.json", "smoke result.json")
        if smoke.get("status") != "completed" or smoke.get("binding") != completed_bindings["cei_smoke"]:
            raise ValueError("smoke result incomplete or binding mismatch")
        if (smoke.get("metrics") is not None or not isinstance(smoke.get("dev_metrics"), dict)
                or smoke.get("test_evaluated", False) is not False):
            raise ValueError("smoke result violates dev-only/test-fold policy")
    except Exception as error:
        journal["status"] = "failed"
        journal["validation_error"] = str(error)
        _write_journal(journal_path, journal)
        raise SystemExit(1) from error
    for entry in journal["stages"].values():
        entry["status"] = "completed"
    journal["status"] = "completed"
    _write_journal(journal_path, journal)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact")
    result.add_argument("--targets")
    result.add_argument("--canonical")
    result.add_argument("--output-root")
    result.add_argument("--execute", action="store_true", help="execute only the fixed four stages")
    result.add_argument("--journal")
    result.add_argument("--validate", nargs=3, metavar=("CONTROL", "CANDIDATE", "PRODUCT_OFF"),
                        help="validate completed non-smoke stage directories")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.validate:
        print(json.dumps(validate_completed_stages(args.validate), indent=2, sort_keys=True))
        return 0
    if not all((args.artifact, args.targets, args.canonical, args.output_root)):
        parser().error("--artifact, --targets, --canonical, and --output-root are required")
    stages = build_plan(artifact=args.artifact, targets=args.targets,
                        canonical=args.canonical, output_root=args.output_root)
    if not args.execute:
        print(json.dumps({"status": "not_executed", "stages": [asdict(stage) for stage in stages]},
                         indent=2, sort_keys=True))
        return 0
    journal_path = args.journal or str(Path(args.output_root).resolve() / "cei_pilot_journal.json")
    return execute_plan(stages, journal_path=journal_path)


if __name__ == "__main__":
    raise SystemExit(main())
