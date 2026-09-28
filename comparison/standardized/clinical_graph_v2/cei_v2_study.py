"""Bounded CEI-GNN v2 pair-interaction development study.

Default CLI output is a print-only plan. Training needs `--execute smoke` or
`--execute full`; validation and test folds are never evaluated. The read-only
guarantee applies to default planning and analysis; explicit `--preflight` output is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
_ARM_VARYING_FIELDS = {
    "seed": "The approved study compares three independently specified random seeds.",
    "selected_dev": "Checkpoint and prediction outcomes are trained separately for each arm.",
    "active_parameter_count": "The off arm intentionally disables active interaction parameters.",
    "method_config": "Only pair_mode and active/inactive counts may vary; all other config is shared.",
    "method_overrides": "Only the pair_mode override may vary between study arms.",
}


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


def build_plan(*, artifact, targets, canonical, output_root, allow_existing_smoke=False):
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
        if output.exists() and not (allow_existing_smoke and name == "v2_smoke"):
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
    """Compare every binding key; allow only explicitly documented arm variation."""
    items = sorted(bindings.items())
    if not items:
        raise ValueError("no arms to compare")
    reference_name, reference = items[0]

    def normalized(field, value):
        if field not in ("method_config", "method_overrides"):
            return value
        value = json.loads(json.dumps(value))
        if field == "method_config":
            value.get("effective_settings", {}).pop("pair_mode", None)
            architecture = value.get("architecture", {})
            architecture.pop("active_parameter_count", None)
            architecture.pop("inactive_parameter_count", None)
        elif isinstance(value, dict):
            value.pop("pair_mode", None)
        return value

    for name, binding in items[1:]:
        for field in sorted(set(reference) | set(binding)):
            if field in _ARM_VARYING_FIELDS:
                if field in ("method_config", "method_overrides"):
                    left = normalized(field, reference.get(field))
                    right = normalized(field, binding.get(field))
                    if left != right:
                        raise ValueError(f"arm parity differs: {field} ({name} vs {reference_name})")
                continue
            if (field not in reference or field not in binding or
                    binding[field] != reference[field]):
                raise ValueError(f"arm parity differs: {field} ({name} vs {reference_name})")
    selected_keys = {"epoch", "epoch_index", "metric", "metric_value",
                     "prediction_sha256", "sample_ids_sha256"}
    for name, binding in items:
        selected = binding.get("selected_dev")
        if not isinstance(selected, dict) or set(selected) != selected_keys:
            raise ValueError(f"selected_dev keys differ from the six-field contract: {name}")
    for name, binding in items[1:]:
        first = reference["selected_dev"]
        other = binding["selected_dev"]
        for field in ("metric", "sample_ids_sha256"):
            if first[field] != other[field]:
                raise ValueError(f"arm parity differs: selected_dev.{field} ({name} vs {reference_name})")
    groups = {}
    for name, binding in items:
        mode = binding["method_config"]["effective_settings"]["pair_mode"]
        architecture = binding["method_config"]["architecture"]
        counts = (binding["parameter_count"], binding["active_parameter_count"],
                  architecture["parameter_count"], architecture["active_parameter_count"])
        groups.setdefault(mode, []).append((name, counts))
    all_total = {counts[0] for values in groups.values() for _, counts in values}
    if len(all_total) != 1:
        raise ValueError("arm parity differs: total parameter_count")
    for mode in ("product", "additive", "off"):
        values = groups.get(mode, [])
        if values and len({counts[1:] for _, counts in values}) != 1:
            raise ValueError(f"arm parity differs: active_parameter_count within {mode} arms")
        for name, counts in values:
            if counts[0] != counts[2] or counts[1] != counts[3]:
                raise ValueError(f"arm parity differs: {name} parameter counts disagree with architecture")
    interactions = [counts[1] for mode in ("product", "additive")
                    for _, counts in groups.get(mode, [])]
    if interactions and len(set(interactions)) != 1:
        raise ValueError("arm parity requires equal active_parameter_count across interaction arms")
    off_counts = [counts[1] for _, counts in groups.get("off", [])]
    if interactions and off_counts and max(off_counts) >= min(interactions):
        raise ValueError("arm parity requires off active_parameter_count < interaction")
    return True



def _capture(stage):
    return pilot.capture_bindings(stage)


def _identity(captured):
    # Seeds and pair modes differ between stages by design; the exact plan is
    # validated separately, so identity covers source bytes and input bytes only.
    return captured["source_state_sha256"], captured["input_state_sha256"]


def replay_v2_stage(output_dir, binding, result, *, persist=True):
    """Rebuild the bound adapter and replay the saved dev predictions exactly."""
    output = Path(output_dir)
    if result.get("status") != "completed" or result.get("binding") != binding:
        raise ValueError("replay requires a completed result bound to binding.json")
    if (binding.get("selection_fold") != "dev" or binding.get("final_eval") != "none"
            or binding.get("test_evaluated") is not False or result.get("metrics") is not None):
        raise ValueError("replay is restricted to dev-selected, no-final-evaluation runs")
    if result.get("test_evaluated", False) is not False:
        raise ValueError("test_evaluated must be false in result")
    if result.get("validation_evaluations") != 0 or (output / "validation.npz").exists():
        raise ValueError("validation evaluation is forbidden")
    import torch
    from types import SimpleNamespace
    from torch_geometric.loader import DataLoader
    from comparison.standardized.clinical_graph_v2 import train
    from comparison.standardized.clinical_graph_v2.contracts import recursive_source_hashes, sample_ids_sha256
    from comparison.standardized.clinical_graph_v2.methods import build_method
    from comparison.standardized.clinical_graph_v2.schema import sha256
    from comparison.standardized.clinical_graph_v2.tensorize import preprocessing_state
    from comparison.standardized.clinical_graph_v2.methods.cei_gnn_v2 import within_visit_pairs

    source = recursive_source_hashes(Path(train.__file__).parent)
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
    if "test" in splits or {"train", "dev", "validation"} - set(splits):
        raise ValueError("replay dataset splits violate the train/dev-only contract")
    hashes = {fold: sample_ids_sha256(row.sample_id for row in rows) for fold, rows in splits.items()}
    if hashes != binding["split_sample_ids_sha256"]:
        raise ValueError("replayed split sample identities differ")
    if preprocessing_state(prep) != json.loads(prep_path.read_text()):
        raise ValueError("replayed preprocessing state differs")
    if {row.subject for row in splits["train"]} & {row.subject for row in splits["dev"]}:
        raise ValueError("replayed train/dev patient overlap")
    validate_v2_method_config(binding)
    config = binding["method_config"]
    architecture, settings = config["architecture"], config["effective_settings"]
    args = dict(binding)
    args["method_options"] = dict(settings)
    model = build_method(
        STUDY_METHOD, num_tokens=binding["vocabulary_size"], node_dim=binding["node_dim"],
        edge_dim=binding["edge_dim"], num_classes=binding["num_classes"],
        hidden=binding["hidden"], layers=binding["layers"], dropout=binding["dropout"],
        token_dim=architecture["token_dim"], num_triples=binding["num_meta_relations"],
        args=SimpleNamespace(**args))
    checkpoint = output / "best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    if sum(p.numel() for p in model.parameters()) != binding["parameter_count"]:
        raise ValueError("replayed parameter count differs")
    loader = DataLoader(splits["dev"], batch_size=binding["batch_size"], shuffle=False)
    probabilities, labels = train.evaluate(model, loader, torch.device("cpu"),
                                           epoch=binding["selected_dev"]["epoch_index"])
    metrics = train.metrics(labels, probabilities, binding["num_classes"])
    with np.load(output / "dev.npz", allow_pickle=False) as saved:
        ids = np.asarray([row.sample_id for row in splits["dev"]])
        if not np.array_equal(ids, saved["sample_ids"]):
            raise ValueError("replayed dev sample IDs differ")
        expected_subjects = np.asarray([row.subject for row in splits["dev"]]).astype(str)
        if "subjects" not in saved.files or not np.array_equal(
                expected_subjects, saved["subjects"].astype(str)):
            raise ValueError("replayed dev subjects differ or are missing")
        if not np.array_equal(labels, saved["y"]):
            raise ValueError("replayed dev labels differ")
        if not np.array_equal(probabilities, saved["proba"]):
            raise ValueError("checkpoint replay probabilities differ")
        prediction_hash = hashlib.sha256(np.ascontiguousarray(saved["proba"]).tobytes()).hexdigest()
    if prediction_hash != binding["selected_dev"]["prediction_sha256"]:
        raise ValueError("replayed probabilities do not match the selected checkpoint proof")
    if metrics != result.get("dev_metrics"):
        raise ValueError("replayed dev metrics differ")
    proof = {"status": "verified", "method": STUDY_METHOD,
             "pair_mode": settings["pair_mode"], "seed": binding["seed"],
             "checkpoint_sha256": sha256(checkpoint), "proba_sha256": prediction_hash,
             "split_sample_ids_sha256": hashes, "dev_count": int(len(labels)),
             "exact_probabilities": True, "exact_labels": True, "exact_sample_identity": True,
             "patient_disjoint": True, "validation_evaluated": False, "test_evaluated": False,
             "dev_metrics": metrics}
    proof_path = output / "replay.json"
    dev_pair_counts = [int(within_visit_pairs(row.visit_membership_index, row.node_type,
                                               int(row.num_nodes)).size(1))
                       for row in splits["dev"]]
    if proof_path.exists():
        existing = pilot._load_json(proof_path, "replay.json")
        if {k: existing.get(k) for k in proof} != proof:
            raise ValueError("existing replay proof differs")
        return {**existing, "_dev_pair_counts": dev_pair_counts}
    if not persist:
        raise ValueError("existing replay proof is required for non-persisting replay")
    with proof_path.open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {**proof, "_dev_pair_counts": dev_pair_counts}


def _check_stage_result(stage, binding, result):
    if result.get("status") != "completed" or result.get("binding") != binding:
        raise RuntimeError(f"{stage.name} did not produce a completed bound result")
    if result.get("metrics") is not None or not isinstance(result.get("dev_metrics"), dict):
        raise RuntimeError(f"{stage.name} violated the dev-only result contract")
    if result.get("test_evaluated", False) is not False:
        raise RuntimeError(f"{stage.name} evaluated the test fold")


def _verify_smoke(smoke_stage, journal_path, persist_replay=True):
    """Revalidate completed smoke binding, artifacts, replay, and source/input identity."""
    smoke_dir = Path(smoke_stage.output)
    journal = pilot._load_json(journal_path, "smoke journal")
    entry = journal.get("stages", {}).get("v2_smoke", {})
    if (journal.get("status") != "completed" or entry.get("status") != "completed"
            or entry.get("replay", {}).get("status") != "verified"):
        raise ValueError("a completed verified smoke replay is required")
    binding = pilot._load_json(smoke_dir / "binding.json", "smoke binding.json")
    result = pilot._load_json(smoke_dir / "result.json", "smoke result.json")
    current = _capture(smoke_stage)
    recorded = entry.get("postflight")
    if not recorded or _identity(current) != _identity(recorded):
        raise ValueError("smoke source/input identity is stale")
    pilot._assert_runner_source_binding(current["executable_sources"], binding)
    validate_v2_binding(binding, budget=smoke_stage.budget,
                        seed=smoke_stage.seed, mode=smoke_stage.pair_mode)
    _check_stage_result(smoke_stage, binding, result)
    pilot._validate_artifacts(smoke_dir, binding, result)
    if persist_replay:
        replay_v2_stage(smoke_dir, binding, result)
    else:
        replay_v2_stage(smoke_dir, binding, result, persist=False)
    return journal


def execute_plan(stages, *, journal_path, phase):
    """Run the smoke stage, or the nine full stages, sequentially and fail closed."""
    validate_exact_plan(stages)
    if phase not in ("smoke", "full"):
        raise ValueError("phase must be 'smoke' or 'full'")
    selected = list(stages[:1]) if phase == "smoke" else list(stages[1:])
    if phase == "full":
        _verify_smoke(stages[0], Path(stages[0].output).parent / "journal_smoke.json")
    journal_path = Path(journal_path)
    if journal_path.exists():
        raise FileExistsError(f"Refusing occupied journal {journal_path}")
    journal = {"status": "running", "phase": phase, "stages": {}}
    pilot._write_journal(journal_path, journal)
    expected, bindings = None, {}
    for stage in selected:
        output = Path(stage.output)
        if output.exists():
            journal["stages"][stage.name] = {"status": "refused", "reason": "occupied output"}
            journal["status"] = "failed"
            pilot._write_journal(journal_path, journal)
            raise SystemExit(2)
        before = _capture(stage)
        if expected is not None and _identity(before) != expected:
            journal["stages"][stage.name] = {"status": "refused", "reason": "source/input drift"}
            journal["status"] = "failed"
            pilot._write_journal(journal_path, journal)
            raise SystemExit(3)
        expected = expected or _identity(before)
        journal["stages"][stage.name] = {"status": "running", "preflight": before}
        pilot._write_journal(journal_path, journal)
        try:
            completed = subprocess.run(list(stage.argv) + ["--execute"], check=True,
                                       capture_output=True, text=True)
            after = _capture(stage)
            if _identity(after) != expected:
                raise RuntimeError("source/input drift detected after stage execution")
            binding = pilot._load_json(output / "binding.json", "binding.json")
            result = pilot._load_json(output / "result.json", "result.json")
            pilot._assert_runner_source_binding(after["executable_sources"], binding)
            validate_v2_binding(binding, budget=stage.budget, seed=stage.seed, mode=stage.pair_mode)
            _check_stage_result(stage, binding, result)
            pilot._validate_artifacts(output, binding, result)
            replay = replay_v2_stage(output, binding, result)
            bindings[stage.name] = binding
            journal["stages"][stage.name].update({
                "status": "bound", "postflight": after, "replay": replay,
                "stdout": completed.stdout, "stderr": completed.stderr})
            if phase == "smoke":
                seconds = result.get("total_seconds", binding.get("total_seconds"))
                if seconds is not None:
                    batch_size = int(binding.get("batch_size", 128))
                    smoke_steps = math.ceil(stage.budget[0] / batch_size) * stage.budget[2]
                    full_steps = math.ceil(FULL_BUDGET[0] / batch_size) * FULL_BUDGET[2]
                    journal["eta_minutes_nine_runs_smoke"] = round(
                        9 * float(seconds) * full_steps / max(smoke_steps, 1) / 60, 1)
                    journal["eta_basis"] = "rough; smoke fixed costs dominate"
        except Exception as error:
            journal["stages"][stage.name].update({"status": "failed", "error": str(error)})
            journal["status"] = "failed"
            pilot._write_journal(journal_path, journal)
            raise SystemExit(1) from error
        pilot._write_journal(journal_path, journal)
    if phase == "full":
        try:
            assert_arm_parity(bindings)
        except ValueError as error:
            journal["status"] = "failed"
            journal["validation_error"] = str(error)
            pilot._write_journal(journal_path, journal)
            raise SystemExit(1) from error
    for entry in journal["stages"].values():
        entry["status"] = "completed"
    journal["status"] = "completed"
    pilot._write_journal(journal_path, journal)
    return 0


def validate_completed_study(output_root, *, include_pair_counts=False):
    """Re-validate and replay all nine full arms; return their bindings."""
    root = Path(output_root).expanduser().resolve(strict=True)
    specs = {name: (budget, seed, mode) for name, budget, seed, mode in stage_specs()[1:]}
    bindings, common_pair_counts = {}, None
    for name, (budget, seed, mode) in specs.items():
        directory = root / name
        binding = pilot._load_json(directory / "binding.json", "binding.json")
        result = pilot._load_json(directory / "result.json", "result.json")
        validate_v2_binding(binding, budget=budget, seed=seed, mode=mode)
        _check_stage_result(Stage(name, [], str(directory), seed, budget, mode), binding, result)
        pilot._validate_artifacts(directory, binding, result)
        replay = replay_v2_stage(directory, binding, result, persist=False)
        counts = replay.get("_dev_pair_counts")
        if counts is not None:
            if common_pair_counts is None:
                common_pair_counts = counts
            elif counts != common_pair_counts:
                raise ValueError("dev pair counts differ across arms")
        bindings[name] = binding
    assert_arm_parity(bindings)
    return (bindings, common_pair_counts) if include_pair_counts else bindings



V1_MEAN_RUN_SECONDS = (127.7 + 123.9) / 2  # recorded CEI v1 pilot full-run total_seconds
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
COMPARISONS = {"product_minus_additive": ("product", "additive"),
               "product_minus_off": ("product", "off"),
               "additive_minus_off": ("additive", "off")}


def pair_count_summary(rows):
    from comparison.standardized.clinical_graph_v2.methods.cei_gnn_v2 import within_visit_pairs

    counts = np.asarray([within_visit_pairs(row.visit_membership_index, row.node_type,
                                            int(row.num_nodes)).size(1) for row in rows],
                        dtype=float)
    return {"graphs": int(counts.size), "mean": float(counts.mean()),
            "quantiles": {str(q): float(np.quantile(counts, q)) for q in QUANTILES}}


def dev_pair_counts(rows):
    from comparison.standardized.clinical_graph_v2.methods.cei_gnn_v2 import within_visit_pairs

    return [int(within_visit_pairs(row.visit_membership_index, row.node_type,
                                   int(row.num_nodes)).size(1)) for row in rows]


def _assert_dev_pair_count_parity(counts_by_arm):
    if counts_by_arm and any(counts != counts_by_arm[0] for counts in counts_by_arm[1:]):
        raise ValueError("dev pair counts differ across arms")


def step_time_ratio(rows, prep, *, batches=5):
    """Forward+backward seconds of v2 product over v1 on identical batches; no updates."""
    import time
    from types import SimpleNamespace
    import torch
    from torch_geometric.loader import DataLoader
    from comparison.standardized.clinical_graph_v2.methods import build_method

    def build(method, options):
        torch.manual_seed(1234)
        return build_method(method, num_tokens=len(prep["vocabulary"]),
                            node_dim=rows[0].x.shape[1], edge_dim=rows[0].edge_attr.shape[1],
                            num_classes=10, hidden=128, layers=1, dropout=0.3, token_dim=32,
                            num_triples=len(prep["triples"]) + 1,
                            args=SimpleNamespace(method_options=options))

    loader = list(DataLoader(rows, batch_size=128, shuffle=False))[:batches]
    seconds = {}
    for label, model in (("v1", build("cei_gnn", {"use_interactions": True})),
                         ("v2", build(STUDY_METHOD, {"pair_mode": "product"}))):
        model.train()
        start = time.perf_counter()
        for batch in loader:
            model.zero_grad(set_to_none=True)
            logits = model(batch, epoch=0).logits
            torch.nn.functional.cross_entropy(logits, batch.y.view(-1) % 10).backward()
        seconds[label] = time.perf_counter() - start
    return seconds["v2"] / max(seconds["v1"], 1e-9)


def preflight(*, artifact, targets, output_json):
    """Train-only pair counts and a timing ratio; no training, no dev/validation scores."""
    output = Path(output_json)
    if output.exists():
        raise FileExistsError(f"Refusing occupied preflight output {output}")
    from comparison.standardized.clinical_graph_v2 import train

    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args([
        "--artifact", str(artifact), "--targets", str(targets), "--output", "<unused>",
        "--method", STUDY_METHOD]), parser)
    labels, _, _ = train.select_top_labels(train.load_targets(targets), 10)
    splits, prep = train.build_dataset(Path(artifact), labels, "all", FULL_BUDGET[0],
                                       args.token_min_count, 1234, edge_direction="forward",
                                       dev_limit=FULL_BUDGET[1], sample_seed=SAMPLE_SEED)
    if "test" in splits:
        raise ValueError("preflight must never build test tensors")
    rows = splits["train"]
    ratio = step_time_ratio(rows, prep)
    report = {"status": "verified",
              "scope": "train-only pair counts and step timing; no training or dev/validation scores",
              "pair_counts": pair_count_summary(rows), "step_time_ratio": ratio,
              "eta_minutes_nine_runs_preflight": round(9 * V1_MEAN_RUN_SECONDS * ratio / 60, 1),
              "eta_basis": "pre-smoke extrapolation",
              "test_tensors_loaded": False, "validation_evaluated": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def weighted_macro_f1(y, pred, num_classes, weights=None):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    confusion = np.bincount(y * num_classes + pred, weights=w,
                            minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = np.diag(confusion)
    fp, fn = confusion.sum(0) - tp, confusion.sum(1) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros(num_classes), where=denominator > 0)
    return float(f1.mean())


def load_arm_predictions(output_root):
    root = Path(output_root)
    arms, reference = {}, None
    for name in FULL_STAGE_NAMES:
        with np.load(root / name / "dev.npz", allow_pickle=False) as saved:
            required = {"proba", "y", "subjects", "sample_ids"}
            if not required.issubset(saved.files):
                raise ValueError(f"dev rows differ across arms: missing arrays in {name}")
            rows = (saved["y"].copy(), saved["subjects"].astype(str), saved["sample_ids"].astype(str))
            arms[name] = saved["proba"].copy()
        if reference is None:
            reference = rows
        elif not all(np.array_equal(a, b) for a, b in zip(rows, reference)):
            raise ValueError(f"dev rows differ across arms: {name}")
    return arms, reference[0], reference[1]


def paired_bootstrap(arms, y, subjects, *, num_classes, resamples=1000, seed=2026):
    patients, inverse = np.unique(np.asarray(subjects).astype(str), return_inverse=True)
    predictions = {name: np.asarray(proba).argmax(1) for name, proba in arms.items()}
    point = {name: weighted_macro_f1(y, pred, num_classes) for name, pred in predictions.items()}
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in COMPARISONS}
    for _ in range(resamples):
        drawn = rng.integers(0, len(patients), len(patients))
        weights = np.bincount(drawn, minlength=len(patients))[inverse].astype(float)
        scores = {name: weighted_macro_f1(y, pred, num_classes, weights)
                  for name, pred in predictions.items()}
        for key, (first, second) in COMPARISONS.items():
            samples[key].append(float(np.mean([scores[f"{first}_seed{s}"] - scores[f"{second}_seed{s}"]
                                               for s in SEEDS])))
    comparisons = {}
    for key, (first, second) in COMPARISONS.items():
        values = np.asarray(samples[key])
        comparisons[key] = {
            "point": float(np.mean([point[f"{first}_seed{s}"] - point[f"{second}_seed{s}"]
                                    for s in SEEDS])),
            "interval_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "resamples": int(resamples), "seed": int(seed)}
    return point, comparisons


def decide(point, comparisons):
    per_seed = {str(s): {m: point[f"{m}_seed{s}"] for m in MODES} for s in SEEDS}
    means = {m: float(np.mean([per_seed[str(s)][m] for s in SEEDS])) for m in MODES}
    checks = {
        "product_beats_additive_each_seed": all(
            per_seed[str(s)]["product"] > per_seed[str(s)]["additive"] for s in SEEDS),
        "product_mean_above_both_controls": (means["product"] > means["additive"]
                                             and means["product"] > means["off"]),
        "product_minus_additive_lower_bound_above_zero":
            comparisons["product_minus_additive"]["interval_95"][0] > 0,
    }
    return {"per_seed": per_seed, "seed_means": means, "checks": checks,
            "interaction_useful": all(checks.values())}


def seed_delta_summary(point):
    """Descriptive per-seed deltas in percentage points; never a decision input."""
    result = {}
    for key, (first, second) in COMPARISONS.items():
        per_seed = {str(seed): 100.0 * (point[f"{first}_seed{seed}"]
                                       - point[f"{second}_seed{seed}"])
                    for seed in SEEDS}
        values = np.asarray(list(per_seed.values()), dtype=float)
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        half_width = 4.302653 * sd / math.sqrt(3)
        result[key] = {"per_seed": per_seed, "mean": mean, "sd": sd,
                       "t_interval_df2": [mean - half_width, mean + half_width],
                       "label": "seed-level, descriptive, not decisive", "decisive": False}
    return result


PAIR_COUNT_BANDS = (("0", 0, 0), ("1-179", 1, 179),
                    ("180-1044", 180, 1044), (">1044", 1045, None))
ANALYSIS_NOTES = [
    "Exploratory dev-only screen: intervals are conditional on the selected dev checkpoints and seeds 1234/2025/7 (spec §6.1).",
    "A failed rule means benefit not demonstrated, not evidence of no benefit.",
    "product vs off is directional; off is a structural ablation with fewer active parameters.",
    "GraphXAI edge masks do not attribute within-visit pairs; pair contributions are model accounting, not causal importance.",
]


def pair_count_band_summary(arms, y, subjects, num_classes, pair_counts):
    """Summarize dev rows by injected per-graph counts, without scoring graphs."""
    if pair_counts is None:
        return {name: {"graphs": None, "patients": None,
                       "per_arm_seed_mean_macro_f1": None,
                       "product_minus_additive": None, "decisive": False,
                       "status": "pair counts unavailable"}
                for name, _, _ in PAIR_COUNT_BANDS}
    counts = np.asarray(pair_counts)
    if counts.ndim != 1 or len(counts) != len(y) or not np.isfinite(counts).all() \
            or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("pair_counts must be one non-negative integer per dev graph")
    counts = counts.astype(int)
    output = {}
    for name, low, high in PAIR_COUNT_BANDS:
        mask = counts == 0 if high == 0 else counts >= low
        if high is not None and high != 0:
            mask &= counts <= high
        graph_count = int(mask.sum())
        band = {"graphs": graph_count,
                "patients": int(len(np.unique(np.asarray(subjects)[mask]))),
                "per_arm_seed_mean_macro_f1": None,
                "product_minus_additive": None, "decisive": False}
        if graph_count >= 2:
            arm_scores = {}
            for mode in MODES:
                arm_scores[mode] = float(np.mean([
                    weighted_macro_f1(np.asarray(y)[mask],
                                      np.asarray(arms[f"{mode}_seed{seed}"])[mask].argmax(axis=1),
                                      num_classes)
                    for seed in SEEDS]))
            band["per_arm_seed_mean_macro_f1"] = arm_scores
            band["product_minus_additive"] = (
                arm_scores["product"] - arm_scores["additive"])
        output[name] = band
    return output


def training_curve_summary(output_root, bindings, results):
    curves = {}
    for name, binding in bindings.items():
        history_path = Path(output_root) / name / "history.json"
        if not history_path.is_file():
            raise ValueError(f"history.json missing for {name}: {history_path}")
        try:
            history = json.loads(history_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read history for {name}: {history_path}") from error
        epochs = binding["epochs"]
        if (not isinstance(history, list) or len(history) != epochs
                or any(not isinstance(row, dict) or row.get("selection_fold") != "dev"
                       for row in history)):
            raise ValueError(f"history for {name} must have exactly {epochs} dev-selected epochs")
        try:
            train_loss = [float(row["train_loss"]) for row in history]
            dev_macro_f1 = [float(row["macro_f1"]) for row in history]
            total_seconds = float(results[name]["total_seconds"])
            selected_epoch = int(binding["selected_dev"]["epoch"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"history or binding metrics are incomplete for {name}") from error
        curves[name] = {"train_loss": train_loss, "dev_macro_f1": dev_macro_f1,
                        "selected_epoch": selected_epoch,
                        "selected_at_last_epoch": selected_epoch == epochs,
                        "total_seconds": total_seconds, "decisive": False}
    return curves


def analyze(output_root, output_json, *, pair_counts=None):
    output = Path(output_json)
    root = Path(output_root).expanduser().resolve(strict=True)
    resolved_output = output.expanduser().resolve()
    for name in FULL_STAGE_NAMES:
        arm_dir = (root / name).resolve()
        try:
            resolved_output.relative_to(arm_dir)
        except ValueError:
            continue
        raise ValueError(f"analysis output must not be inside study arm: {arm_dir}")
    if output.exists():
        raise FileExistsError(f"Refusing occupied analysis output {output}")
    import inspect
    validator = validate_completed_study
    if "include_pair_counts" in inspect.signature(validator).parameters:
        bindings, replay_pair_counts = validator(output_root, include_pair_counts=True)
    else:
        bindings, replay_pair_counts = validator(output_root), None
    arms, y, subjects = load_arm_predictions(output_root)
    num_classes = next(iter(bindings.values()))["num_classes"]
    point, comparisons = paired_bootstrap(arms, y, subjects, num_classes=num_classes)
    for name, binding in bindings.items():
        recorded = binding["selected_dev"]["metric_value"]
        if abs(point[name] - recorded) > 1e-6:
            raise ValueError(f"recomputed macro-F1 differs from the bound result: {name}")
    for key, entry in comparisons.items():
        entry["decisive"] = key == "product_minus_additive"
    if pair_counts is None:
        # This transient replay value is reporting-only; it is never a decision input.
        pair_counts = replay_pair_counts
    # Keep the decision tied only to the declared score and paired-interval inputs;
    # reporting-only counts, seed deltas, curves, and secondary results follow later.
    report = {"status": "analyzed", "decision": decide(point, comparisons),
              "comparisons": comparisons, "dev_macro_f1": point,
              "dev_rows": int(len(y)), "patients": int(len(np.unique(subjects))),
              "test_evaluated": False, "validation_evaluated": False,
              "seed_deltas": seed_delta_summary(point),
              "notes": list(ANALYSIS_NOTES),
              "parameter_counts": {},
              "pair_count_bands": pair_count_band_summary(
                  arms, y, subjects, num_classes, pair_counts),
              "pair_count_source": ("injected per-graph dev counts" if pair_counts is not None
                                    and replay_pair_counts is None else
                                    "dev rows reloaded by replay (structural count, no scoring)"
                                    if pair_counts is not None else
                                    "pair counts unavailable in real runs; no dev graph reload")}
    from comparison.standardized.clinical_graph_v2 import train
    secondary = {}
    results = {name: pilot._load_json(Path(output_root) / name / "result.json", "result.json")
               for name in bindings}
    for name, binding in bindings.items():
        proba = arms[name]
        architecture = binding.get("method_config", {}).get("architecture", {})
        inactive_count = binding.get(
            "inactive_parameter_count",
            architecture.get("inactive_parameter_count",
            binding["parameter_count"] - binding["active_parameter_count"]))
        report["parameter_counts"][name] = {
            "parameter_count": binding["parameter_count"],
            "active_parameter_count": binding["active_parameter_count"],
            "inactive_parameter_count": inactive_count}
        patient_metrics = train.patient_equal_metrics(y, proba, subjects, num_classes)
        per_class = train.per_class_table(y, proba, binding["label_order"], num_classes)
        predictions = proba.argmax(axis=1)
        for class_index, row in enumerate(per_class):
            row["false_positives"] = int(np.count_nonzero(
                (predictions == class_index) & (np.asarray(y) != class_index)))
        secondary[name] = {
            "decisive": False,
            "patient_equal_macro_f1": patient_metrics["macro_f1"],
            "per_class": per_class,
            "pair_count_summary": binding.get("pair_count_summary"),
            "parameter_count": binding["parameter_count"],
            "active_parameter_count": binding["active_parameter_count"],
            "inactive_parameter_count": inactive_count,
            "total_seconds": results[name].get("total_seconds"),
        }
    report["secondary_results"] = secondary
    report["training_curves"] = training_curve_summary(output_root, bindings, results)
    report["product_minus_off_bootstrap"] = comparisons["product_minus_off"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact")
    result.add_argument("--targets")
    result.add_argument("--canonical")
    result.add_argument("--output-root")
    result.add_argument("--execute", choices=("smoke", "full"))
    result.add_argument("--preflight", metavar="OUTPUT_JSON")
    result.add_argument("--analyze", metavar="OUTPUT_ROOT")
    result.add_argument("--analysis-output", metavar="OUTPUT_JSON")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.analyze:
        if not args.analysis_output:
            parser().error("--analyze requires --analysis-output")
        print(json.dumps(analyze(args.analyze, args.analysis_output), indent=2, sort_keys=True))
        return 0
    if args.preflight:
        if not (args.artifact and args.targets):
            parser().error("--preflight requires --artifact and --targets")
        print(json.dumps(preflight(artifact=args.artifact, targets=args.targets,
                                   output_json=args.preflight), indent=2, sort_keys=True))
        return 0
    if not all((args.artifact, args.targets, args.canonical, args.output_root)):
        parser().error("--artifact, --targets, --canonical and --output-root are required")
    root = Path(args.output_root).expanduser().resolve()
    smoke_path = root / "journal_smoke.json"
    smoke_completed = (args.execute == "full" or
                       (not args.execute and smoke_path.is_file()
                        and (root / "v2_smoke").is_dir()))
    if args.execute == "full":
        artifact = pilot._absolute_existing(args.artifact, "artifact", directory=True)
        targets = pilot._absolute_existing(args.targets, "targets")
        canonical = pilot._absolute_existing(args.canonical, "canonical")
        stages = [_stage(name, artifact, targets, canonical, root / name, budget, seed, mode)
                  for name, budget, seed, mode in stage_specs()]
    else:
        stages = build_plan(artifact=args.artifact, targets=args.targets,
                            canonical=args.canonical, output_root=root,
                            allow_existing_smoke=smoke_completed)
    smoke_journal = None
    if smoke_completed:
        smoke_journal = _verify_smoke(stages[0], smoke_path, False)
    if not args.execute:
        plan = {"status": "not_executed", "stages": [asdict(s) for s in stages]}
        if (smoke_journal and "eta_minutes_nine_runs_smoke" in smoke_journal):
            plan["eta_minutes_nine_runs_smoke"] = smoke_journal["eta_minutes_nine_runs_smoke"]
            plan["eta_basis"] = smoke_journal.get("eta_basis", "rough; smoke fixed costs dominate")
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    journal = str(root / f"journal_{args.execute}.json")
    return execute_plan(stages, journal_path=journal, phase=args.execute)

if __name__ == "__main__":
    raise SystemExit(main())
