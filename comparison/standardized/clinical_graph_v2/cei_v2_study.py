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



def _capture(stage):
    return pilot.capture_bindings(stage)


def _identity(captured):
    # Seeds and pair modes differ between stages by design; the exact plan is
    # validated separately, so identity covers source bytes and input bytes only.
    return captured["source_state_sha256"], captured["input_state_sha256"]


def replay_v2_stage(output_dir, binding, result):
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
    if proof_path.exists():
        existing = pilot._load_json(proof_path, "replay.json")
        if {k: existing.get(k) for k in proof} != proof:
            raise ValueError("existing replay proof differs")
        return existing
    with proof_path.open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return proof


def _check_stage_result(stage, binding, result):
    if result.get("status") != "completed" or result.get("binding") != binding:
        raise RuntimeError(f"{stage.name} did not produce a completed bound result")
    if result.get("metrics") is not None or not isinstance(result.get("dev_metrics"), dict):
        raise RuntimeError(f"{stage.name} violated the dev-only result contract")
    if result.get("test_evaluated", False) is not False:
        raise RuntimeError(f"{stage.name} evaluated the test fold")


def execute_plan(stages, *, journal_path, phase):
    """Run the smoke stage, or the nine full stages, sequentially and fail closed."""
    validate_exact_plan(stages)
    if phase not in ("smoke", "full"):
        raise ValueError("phase must be 'smoke' or 'full'")
    selected = list(stages[:1]) if phase == "smoke" else list(stages[1:])
    if phase == "full":
        smoke_stage = stages[0]
        smoke_dir = Path(smoke_stage.output)
        smoke_journal_path = smoke_dir.parent / "journal_smoke.json"
        if not smoke_journal_path.is_file():
            raise ValueError("a completed verified smoke journal is required before full stages")
        smoke_journal = pilot._load_json(smoke_journal_path, "smoke journal")
        smoke_entry = smoke_journal.get("stages", {}).get("v2_smoke", {})
        if (smoke_journal.get("status") != "completed" or
                smoke_entry.get("status") != "completed" or
                smoke_entry.get("replay", {}).get("status") != "verified"):
            raise ValueError("a completed verified smoke replay is required before full stages")
        binding = pilot._load_json(smoke_dir / "binding.json", "smoke binding.json")
        result = pilot._load_json(smoke_dir / "result.json", "smoke result.json")
        current = _capture(smoke_stage)
        recorded = smoke_entry.get("postflight")
        if not recorded or _identity(current) != _identity(recorded):
            raise ValueError("smoke source/input identity is stale")
        pilot._assert_runner_source_binding(current["executable_sources"], binding)
        validate_v2_binding(binding, budget=smoke_stage.budget,
                            seed=smoke_stage.seed, mode=smoke_stage.pair_mode)
        _check_stage_result(smoke_stage, binding, result)
        pilot._validate_artifacts(smoke_dir, binding, result)
        replay_v2_stage(smoke_dir, binding, result)
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


def validate_completed_study(output_root):
    """Re-validate and replay all nine full arms; return their bindings."""
    root = Path(output_root).expanduser().resolve(strict=True)
    specs = {name: (budget, seed, mode) for name, budget, seed, mode in stage_specs()[1:]}
    bindings = {}
    for name, (budget, seed, mode) in specs.items():
        directory = root / name
        binding = pilot._load_json(directory / "binding.json", "binding.json")
        result = pilot._load_json(directory / "result.json", "result.json")
        validate_v2_binding(binding, budget=budget, seed=seed, mode=mode)
        _check_stage_result(Stage(name, [], str(directory), seed, budget, mode), binding, result)
        pilot._validate_artifacts(directory, binding, result)
        replay_v2_stage(directory, binding, result)
        bindings[name] = binding
    assert_arm_parity(bindings)
    return bindings
