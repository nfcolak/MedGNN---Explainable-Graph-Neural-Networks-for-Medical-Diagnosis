"""Checkpointed orchestration for the standardized benchmark pipeline.

This module only coordinates the maintained runners; it does not implement
training, cache generation, explanation, or summary science.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.lib.explanation_contract import COHORT_SIZE, validate_cohort_size
from comparison.standardized.run_explanations import DEFAULT_COHORT

CHECKPOINT_DIR = Path(__file__).with_name("checkpoint")
STATE_PATH = CHECKPOINT_DIR / "state.json"
EVENTS_PATH = CHECKPOINT_DIR / "events.jsonl"
PHASES = ("preflight", "cache", "train", "explain", "summarize")


def item_id(phase: str, item: str = "default") -> str:
    return f"{phase}:{item}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Checkpoint:
    """Small durable state machine with append-only audit events."""
    def __init__(self, state_path: Path = STATE_PATH, events_path: Path = EVENTS_PATH):
        self.state_path, self.events_path = Path(state_path), Path(events_path)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {"schema": 1, "updated_at": now(), "items": {}}

    def _save(self) -> None:
        self.state["updated_at"] = now()
        atomic_write(self.state_path, self.state)

    def event(self, event: str, **fields: Any) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": now(), "event": event, **fields}, sort_keys=True) + "\n")

    def status(self, key: str) -> str | None:
        return self.state["items"].get(key, {}).get("status")

    def start(self, key: str, *, resume: bool = False, retry_failed: bool = False) -> bool:
        old = self.state["items"].get(key, {})
        status = old.get("status")
        if resume and status == "completed":
            return False
        if status == "failed" and not retry_failed:
            return False
        if status == "running":
            self.event("stale_running_restarted", item=key)
        self.state["items"][key] = {**old, "status": "running", "started_at": now()}
        self.event("started", item=key)
        self._save()
        return True

    def finish(self, key: str, status: str, error: str | None = None) -> None:
        record = self.state["items"].setdefault(key, {})
        record.update({"status": status, "finished_at": now()})
        if error: record["error"] = error
        self.event(status, item=key, error=error)
        self._save()


def command_plan(root: Path = ROOT, *, cohort_size: int = COHORT_SIZE, cohort_path: Path = DEFAULT_COHORT) -> dict[str, list[list[str]]]:
    """Return argv-only commands, delegating all scientific behavior."""
    validate_cohort_size(cohort_size)
    py = sys.executable
    return {
        "preflight": [[py, "-m", "comparison.standardized.run_benchmark", "--dry-run"]],
        "cache": [
            [py, "-m", "comparison.standardized.build_caches", "--execute"],
            [py, "-m", "comparison.standardized.audit_caches"],
        ],
        "train": [[py, "-m", "comparison.standardized.run_benchmark",
                   "--methods", method, "--structures", topology, "--seeds", str(seed)]
                  for method in ("protgnn", "gsat", "graphcare")
                  for topology in ("star", "cooccur") for seed in (1234, 1235, 1236)],
        "explain": [[py, "-m", "comparison.standardized.run_explanations",
                     "--topology", topology, "--seed", str(seed),
                     "--cohort-size", str(cohort_size), "--cohort", str(cohort_path)]
                    for topology in ("star", "cooccur") for seed in (1234, 1235, 1236)],
        "summarize": [[py, "-m", "comparison.standardized.summarize", "--results-dir", "comparison/standardized/results", "--summary-csv", "comparison/standardized/summary.csv", "--summary-aggregate-csv", "comparison/standardized/summary_aggregate.csv", "--summary-md", "comparison/standardized/summary.md"]],
    }


def recover_output(phase: str, command: list[str], *, root: Path = ROOT,
                   resume: bool = False, archive_incomplete: bool = False) -> bool:
    """Recover completed children or explicitly preserve an incomplete attempt.

    Return True only for validated, completed output. This is run-level restart:
    an archived attempt is rerun from epoch zero, never an optimizer resume.
    """
    method = "gsat"
    if phase == "train" and "--methods" in command:
        method = command[command.index("--methods") + 1]
        topology = command[command.index("--structures") + 1]
        seed = int(command[command.index("--seeds") + 1])
        relative = Path("results") / method / topology / f"seed_{seed}"
    elif phase == "explain" and "--topology" in command:
        topology = command[command.index("--topology") + 1]
        seed = int(command[command.index("--seed") + 1])
        relative = Path("explanations") / topology / f"seed_{seed}"
    else:
        return False
    from shared.lib.benchmark_contract import BenchmarkSpec, file_sha256
    from shared.lib.run_manifest import _artifact_relative_to_run
    BenchmarkSpec(method, topology, seed)
    base = root / "comparison/standardized"
    output = base / relative
    if not output.exists() and not output.is_symlink():
        return False
    if not resume:
        raise FileExistsError(f"Existing output {output}; use --resume to validate completed work.")
    # Never move a symlink or an output that escapes its documented location.
    if output.resolve() != root.resolve() / "comparison/standardized" / relative:
        raise ValueError(f"Output path must not traverse symlinks: {output}")
    if phase == "train":
        manifest_path = output / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        completed = manifest.get("status") == "completed"
        if completed:
            from comparison.standardized.summarize import _completed_run
            run = _completed_run(manifest_path, manifest)
            if (run.method, run.topology, run.seed) != (method, topology, seed):
                raise ValueError(f"Manifest identity differs from selected cell: {output}")
            if (run.dataset_hash != file_sha256(root / "data/merged_ed.csv") or
                    run.split_hash != file_sha256(root / "comparison/canonical_split.json")):
                raise ValueError(f"Completed run provenance differs from current inputs: {output}")
            _artifact_relative_to_run(manifest["metrics_path"], output, "metrics")
            _artifact_relative_to_run(manifest["checkpoint_path"], output, "checkpoint")
            return True
        if manifest and manifest.get("status") not in {"running", "failed"}:
            raise ValueError(f"Unknown run status; inspect {manifest_path}")
    else:
        completed = (output / "validation.json").is_file()
        if completed:
            from comparison.standardized.run_explanations import (
                validate_cross_method_outputs, checkpoint_for,
            )
            validation = validate_cross_method_outputs(
                output, topology=topology, seed=seed,
                cohort_path=root / (Path(command[command.index("--cohort") + 1]) if "--cohort" in command else DEFAULT_COHORT),
                cohort_size=int(command[command.index("--cohort-size") + 1]) if "--cohort-size" in command else COHORT_SIZE,
                split_path=root / "comparison/canonical_split.json",
                dataset_path=root / "data/merged_ed.csv", require_checkpoints=True,
            )
            for name, digest in validation["checkpoint_sha256"].items():
                if digest != file_sha256(root / checkpoint_for(name, topology, seed)):
                    raise ValueError(f"Explanation checkpoint provenance differs for {name}.")
            if json.loads((output / "validation.json").read_text()) != validation:
                raise ValueError(f"Explanation validation artifact differs: {output}")
            return True
    if not archive_incomplete:
        raise FileExistsError(
            f"Incomplete output {output}. Preserve it elsewhere manually, or after ensuring "
            "no worker is running use --resume --retry-failed --archive-incomplete. "
            "This restarts the run, NOT its last epoch."
        )
    import uuid
    destination = base / "attempts" / relative / uuid.uuid4().hex
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.rename(destination)
    print(f"Preserved incomplete attempt: {destination}")
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run preprocessing, training, explanations and summary (expensive).")
    mode.add_argument("--dry-run", action="store_true", help="Print commands only; create nothing (default).")
    p.add_argument("--resume", action="store_true", help="Validate/reuse completed runs; not epoch-level resume.")
    p.add_argument("--retry-failed", action="store_true", help="Retry failed commands, stopping again on any failure.")
    p.add_argument("--archive-incomplete", action="store_true", help="Preserve incomplete run/set under attempts/ before restarting from scratch; requires --resume --retry-failed and no active workers.")
    p.add_argument("--cohort-size", type=int, default=COHORT_SIZE, help="Expected explanation subjects (default: 500).")
    p.add_argument("--cohort", dest="cohort_path", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--state", type=Path, default=STATE_PATH)
    p.add_argument("--events", type=Path, default=EVENTS_PATH)
    args = p.parse_args(argv)
    if args.archive_incomplete and not (args.resume and args.retry_failed):
        p.error("--archive-incomplete requires --resume --retry-failed")
    plan = command_plan(cohort_size=args.cohort_size, cohort_path=args.cohort_path)
    print(json.dumps({"phases": PHASES, "commands": plan}, indent=2, sort_keys=True))
    if args.dry_run or not args.execute:
        return 0
    cp = Checkpoint(args.state, args.events)
    for phase in PHASES:
        for index, command in enumerate(plan[phase]):
            key = item_id(phase, "default" if phase == "preflight" else str(index))
            old = cp.state["items"].get(key, {})
            if old.get("command", command) != command:
                print(f"Command changed for {key}; use a new --state/--events pair.", file=sys.stderr)
                return 1
            if cp.status(key) == "failed" and not args.retry_failed:
                print(f"Blocked by {key}; inspect the failure and use --retry-failed.", file=sys.stderr)
                return 1
            try:
                if recover_output(phase, command, root=ROOT, resume=args.resume,
                                  archive_incomplete=args.archive_incomplete):
                    if cp.status(key) != "completed":
                        if cp.status(key) == "running":
                            cp.event("stale_running_recovered", item=key)
                        cp.finish(key, "completed")
                    continue
                if args.resume and cp.status(key) == "completed" and (
                        "--methods" in command or "--topology" in command):
                    raise ValueError(f"Completed {key} is missing its output; inspect before retrying.")
            except Exception as exc:
                cp.finish(key, "failed", f"{type(exc).__name__}: {exc}")
                print(f"{key}: {exc}", file=sys.stderr)
                return 1
            if not cp.start(key, resume=args.resume, retry_failed=args.retry_failed):
                if cp.status(key) != "completed":
                    print(f"Blocked by {key}; inspect the failure and use --retry-failed.", file=sys.stderr)
                    return 1
                continue
            cp.state["items"][key]["command"] = command
            cp._save()
            try:
                result = subprocess.run(command, cwd=ROOT, shell=False, check=False)
                if result.returncode:
                    raise RuntimeError(f"exit code {result.returncode}: {command!r}")
                cp.finish(key, "completed")
            except BaseException as exc:
                cp.finish(key, "failed", f"{type(exc).__name__}: {exc}")
                print(f"{key}: {exc}", file=sys.stderr)
                if not isinstance(exc, Exception):
                    raise
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
