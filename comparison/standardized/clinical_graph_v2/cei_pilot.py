"""Fail-closed planner and executor for the approved four-stage CEI dev pilot.

Default CLI behavior is print-only. Training is possible only with --execute and
is limited to the fixed smoke/control/candidate/product-off stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict, dataclass


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
    "hidden", "layers", "dropout", "lr", "weight_decay", "batch_size", "optimizer",
    "epochs", "patience", "min_delta", "early_stopping_start_epoch_index",
    "selected_dev", "selected_validation", "selection", "device",
    "total_seconds", "output", "output_path",
    "parameter_shapes", "use_interactions", "interaction_rank",
    "treatment",
}
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
        sys.executable, "-m", "comparison.standardized.clinical_graph_v2.train",
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
            ignored = set()
            if path == "method_config":
                ignored = {"method", "adaptation_version", "native_defaults", "effective_settings"}
            elif path == "architecture" or path.endswith(".architecture"):
                ignored = {
                    "parameter_count", "active_parameter_count", "parameter_shapes",
                    "hidden", "layers", "dropout", "lr", "weight_decay", "batch_size",
                    "optimizer", "heads", "interaction_rank", "use_interactions",
                }
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
    for key in sorted((left_keys & right_keys) - _EXCLUDED_TOP_LEVEL):
        difference = _first_difference(left[key], right[key], key)
        if difference:
            raise ValueError(f"common binding differs: {difference}")
    return True


def validate_pilot_binding(binding, *, expected_budget, expected_seed=1234):
    """Validate complete budget/fold policy before allowing any comparison."""
    for field in _REQUIRED_BINDINGS:
        if field not in binding or binding[field] is None:
            raise ValueError(f"binding missing required field: {field}")
    train_limit, dev_limit, epochs = expected_budget
    expected = {
        "train_limit": train_limit, "dev_limit": dev_limit, "epochs": epochs,
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


def capture_bindings(stage):
    """Capture execution identity before launch without opening clinical inputs."""
    revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain=v1"], check=True,
                            capture_output=True, text=True).stdout
    package_root = Path(__file__).parent
    source_hashes = {
        path.relative_to(package_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    source_state = _sha_json({"revision": revision, "working_tree": status,
                              "clinical_graph_v2_source": source_hashes})
    inputs = {"artifact": stage.argv[stage.argv.index("--artifact") + 1],
              "targets": stage.argv[stage.argv.index("--targets") + 1],
              "canonical": stage.argv[stage.argv.index("--canonical") + 1]}
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
        "argv_sha256": _sha_json(stage.argv),
        "common_config_sha256": _sha_json(shared_argv),
        "input_state_sha256": _sha_json(inputs),
        "inputs": inputs,
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


def execute_plan(stages, *, journal_path):
    """Run fixed stages sequentially, journaling every outcome and refusing drift."""
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
            validate_pilot_binding(binding, expected_budget=stage.budget, expected_seed=stage.seed)
            _validate_treatment(binding, stage.treatment)
            completed_bindings[stage.name] = binding
            journal["stages"][stage.name].update({"status": "bound", "postflight": after,
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
