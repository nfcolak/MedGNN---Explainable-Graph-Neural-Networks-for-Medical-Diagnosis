"""Auditable, non-interactive standardized benchmark runner."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import errno
from itertools import product
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.lib.benchmark_contract import (
    ALLOWED_SEEDS,
    BenchmarkSpec,
    EXPECTED_FOLD_COUNTS,
    load_canonical_split,
    validate_primary_matrix,
)
from shared.lib.run_manifest import RunManifest
from comparison.standardized.audit_caches import audit_standardized_caches


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("benchmark_config.json")


@dataclass(frozen=True)
class RunnerOptions:
    """Command-generation controls shared by the CLI and tests."""

    config: Mapping[str, Any]
    limit: Optional[int] = None
    max_epochs: Optional[int] = None
    loss_weighting: Optional[str] = None

    def __post_init__(self) -> None:
        for name, value in (("limit", self.limit), ("max_epochs", self.max_epochs)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive exact integer.")
        if self.loss_weighting is not None and self.loss_weighting not in ("sqrt_inverse", "none"):
            raise ValueError("loss_weighting must be 'sqrt_inverse' or 'none' when provided.")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the checked-in runner configuration as a JSON object."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Benchmark config must be a JSON object.")
    return config


def build_run_specs(config: Mapping[str, Any]) -> list[BenchmarkSpec]:
    """Build and validate the configured primary method/topology/seed matrix."""
    try:
        methods = config["methods_order"]
        structures = config["structures"]
        seeds = config["seeds"]
    except KeyError as exc:
        raise ValueError(f"Benchmark config is missing {exc.args[0]!r}.") from exc
    if not all(isinstance(values, list) and values for values in (methods, structures, seeds)):
        raise ValueError("methods_order, structures, and seeds must be nonempty JSON arrays.")
    specs = [
        BenchmarkSpec(method, structure, seed)
        for method, structure, seed in product(methods, structures, seeds)
    ]
    validate_primary_matrix(specs)
    return specs


def run_output_dir(spec: BenchmarkSpec, config: Mapping[str, Any]) -> Path:
    """Return the repository-relative isolated directory for one run."""
    return (
        Path(config["results_root"])
        / spec.method
        / spec.structure
        / f"seed_{spec.seed}"
    )


def command_for(spec: BenchmarkSpec, options: RunnerOptions) -> list[str]:
    """Build one method's explicit argv list without shell interpolation."""
    config = options.config
    try:
        method = config["methods"][spec.method]
        base_command = method["command"]
        flags = method["flags"]
        split_path = config["split_path"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Incomplete command config for {spec.method!r}.") from exc
    if not isinstance(base_command, list) or not base_command or not all(
        isinstance(part, str) and part for part in base_command
    ):
        raise ValueError(f"Command for {spec.method!r} must be a nonempty argv list.")
    required_flags = {"structure", "split", "seed", "output_dir", "limit", "max_epochs", "loss_weighting"}
    if not isinstance(flags, dict) or set(flags) != required_flags:
        raise ValueError(f"Command flags for {spec.method!r} are incomplete or unexpected.")

    command = list(base_command)
    command.extend(
        [
            flags["structure"],
            spec.structure,
            flags["split"],
            str(split_path),
            flags["seed"],
            str(spec.seed),
            flags["output_dir"],
            str(run_output_dir(spec, config)),
        ]
    )
    if options.limit is not None:
        command.extend([flags["limit"], str(options.limit)])
    if options.max_epochs is not None:
        command.extend([flags["max_epochs"], str(options.max_epochs)])
    if options.loss_weighting is not None:
        command.extend([flags["loss_weighting"], options.loss_weighting])
    return command


def _contract_paths(config: Mapping[str, Any], project_root: Path) -> tuple[Path, Path, Path]:
    expected = {
        "dataset_path": Path("data/merged_ed.csv"),
        "split_path": Path("comparison/canonical_split.json"),
        "results_root": Path("comparison/standardized/results"),
    }
    resolved = []
    for key, required in expected.items():
        configured = config.get(key)
        if type(configured) is not str or Path(configured) != required:
            raise ValueError(f"{key} must be exactly {required.as_posix()!r}.")
        resolved.append(project_root / required)
    return resolved[0], resolved[1], resolved[2]


def _validate_dataset(dataset_path: Path, split: Mapping[str, Any]) -> None:
    if not dataset_path.exists() or not dataset_path.is_file() or dataset_path.stat().st_size == 0:
        raise ValueError(f"Dataset must be a nonempty regular file: {dataset_path}")
    required_subjects = split["fold"]
    class_names = set(split["classes"])
    seen: set[str] = set()
    try:
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if "subject_id" not in fields or "disease_1" not in fields:
                raise ValueError("Dataset must contain subject_id and disease_1 columns.")
            for row_number, row in enumerate(reader, start=2):
                subject = row.get("subject_id", "")
                if subject not in required_subjects:
                    continue
                if subject in seen:
                    raise ValueError(
                        f"Dataset contains duplicate canonical subject_id {subject!r}."
                    )
                label = row.get("disease_1")
                if label not in class_names:
                    raise ValueError(
                        f"Dataset row {row_number} has invalid canonical disease_1 {label!r}."
                    )
                seen.add(subject)
    except UnicodeDecodeError as exc:
        raise ValueError("Dataset must be valid UTF-8 CSV.") from exc
    missing = set(required_subjects) - seen
    if missing:
        sample = sorted(missing)[:3]
        raise ValueError(
            f"Dataset is missing {len(missing)} canonical subjects; sample={sample}."
        )


def _validate_topology_parameters(config: Mapping[str, Any]) -> None:
    parameters = config.get("topology_parameters")
    if not isinstance(parameters, dict):
        raise ValueError("topology_parameters must be a JSON object.")
    star = parameters.get("star")
    cooccur = parameters.get("cooccur")
    if star != {
        "patient_hub": True,
        "patient_concept_edges": "bidirectional",
        "concept_concept_edges": "none",
    }:
        raise ValueError("Invalid star topology parameters.")
    if cooccur != {
        "patient_hub": True,
        "patient_concept_edges": "bidirectional",
        "concept_concept_edges": "training_fold_pmi",
        "pmi_threshold": 2.0,
    }:
        raise ValueError("Invalid cooccur topology parameters.")


def preflight(
    config: Mapping[str, Any], specs: Sequence[BenchmarkSpec], project_root: Path
) -> tuple[Path, Path, Path, Mapping[str, Any]]:
    """Validate all immutable inputs and matrix cells before any launch."""
    dataset_path, split_path, results_root = _contract_paths(config, project_root)
    split = load_canonical_split(split_path)
    _validate_dataset(dataset_path, split)
    _validate_topology_parameters(config)
    if not specs:
        raise ValueError("Filters selected no benchmark runs.")
    if len(specs) != len(set(specs)):
        raise ValueError("Selected benchmark matrix contains duplicate runs.")
    for spec in specs:
        BenchmarkSpec(spec.method, spec.structure, spec.seed)
    return dataset_path, split_path, results_root, split


_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _require_descriptor_safety() -> None:
    """Fail closed on hosts without the macOS/POSIX openat primitives we require."""
    if not getattr(os, "O_NOFOLLOW", 0) or os.open not in os.supports_dir_fd:
        raise RuntimeError(
            "Descriptor-safe benchmark reservation requires O_NOFOLLOW and dir_fd support."
        )


def _run_components(spec: BenchmarkSpec, config: Mapping[str, Any]) -> tuple[str, ...]:
    relative = run_output_dir(spec, config)
    expected = Path(
        "comparison/standardized/results"
    ) / spec.method / spec.structure / f"seed_{spec.seed}"
    if relative != expected or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("Run output must stay inside standardized results.")
    return relative.parts


def _path_error(component: str, exc: OSError) -> ValueError:
    return ValueError(
        f"Run output path may not traverse a symlink or file: {component}"
    )


def _open_existing_child(parent_fd: int, component: str) -> int:
    try:
        return os.open(component, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise _path_error(component, exc) from exc
        raise


def _open_or_create_child(parent_fd: int, component: str) -> int:
    try:
        return _open_existing_child(parent_fd, component)
    except FileNotFoundError:
        try:
            os.mkdir(component, 0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
        return _open_existing_child(parent_fd, component)


def _open_project_root(project_root: Path) -> int:
    _require_descriptor_safety()
    try:
        descriptor = os.open(project_root, _DIRECTORY_FLAGS)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("Trusted project root must be a non-symlink directory.") from exc
        raise
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("Trusted project root must be a directory.")
    return descriptor


def _cell_exists_no_follow(
    spec: BenchmarkSpec, config: Mapping[str, Any], project_root: Path
) -> bool:
    descriptor = _open_project_root(project_root)
    try:
        for component in _run_components(spec, config):
            try:
                next_descriptor = _open_existing_child(descriptor, component)
            except FileNotFoundError:
                return False
            os.close(descriptor)
            descriptor = next_descriptor
        return True
    finally:
        os.close(descriptor)


def _reject_occupied_outputs(
    specs: Sequence[BenchmarkSpec], config: Mapping[str, Any], project_root: Path
) -> None:
    """Preflight existing cells; exclusive reservation remains the final authority."""
    for spec in specs:
        if _cell_exists_no_follow(spec, config, project_root):
            run_dir = project_root / run_output_dir(spec, config)
            raise FileExistsError(f"Refusing to use existing run output: {run_dir}")


@dataclass
class _RunReservation:
    run_dir: Path
    parent_fd: int
    run_fd: int
    cell_name: str
    identity: tuple[int, int]
    started: bool = False

    def mark_started(self) -> None:
        self.started = True

    def release_unstarted(self) -> None:
        """Remove only this invocation's still-empty, exact reserved inode."""
        if self.started or self.run_fd < 0:
            return
        try:
            current = os.stat(
                self.cell_name, dir_fd=self.parent_fd, follow_symlinks=False
            )
            if (current.st_dev, current.st_ino) != self.identity:
                return
            if os.listdir(self.run_fd):
                return
            os.close(self.run_fd)
            self.run_fd = -1
            os.rmdir(self.cell_name, dir_fd=self.parent_fd)
            os.fsync(self.parent_fd)
        except FileNotFoundError:
            pass

    def close(self) -> None:
        if self.run_fd >= 0:
            os.close(self.run_fd)
            self.run_fd = -1
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1


def _validate_reservation_path(
    reservation: _RunReservation,
    spec: BenchmarkSpec,
    config: Mapping[str, Any],
    project_root: Path,
) -> None:
    descriptor = _open_project_root(project_root)
    try:
        for component in _run_components(spec, config):
            next_descriptor = _open_existing_child(descriptor, component)
            os.close(descriptor)
            descriptor = next_descriptor
        current = os.fstat(descriptor)
        if (current.st_dev, current.st_ino) != reservation.identity:
            raise ValueError("Run output path changed during reservation.")
    finally:
        os.close(descriptor)


def _reserve_run_output(
    spec: BenchmarkSpec, config: Mapping[str, Any], project_root: Path
) -> _RunReservation:
    """Atomically reserve one exact cell through descriptor-relative operations."""
    components = _run_components(spec, config)
    descriptor = _open_project_root(project_root)
    try:
        for component in components[:-1]:
            next_descriptor = _open_or_create_child(descriptor, component)
            os.close(descriptor)
            descriptor = next_descriptor
        cell_name = components[-1]
        os.mkdir(cell_name, 0o755, dir_fd=descriptor)
        run_fd = _open_existing_child(descriptor, cell_name)
        metadata = os.fstat(run_fd)
        reservation = _RunReservation(
            project_root / Path(*components),
            descriptor,
            run_fd,
            cell_name,
            (metadata.st_dev, metadata.st_ino),
        )
        descriptor = -1
        try:
            _validate_reservation_path(reservation, spec, config, project_root)
        except BaseException:
            reservation.release_unstarted()
            reservation.close()
            raise
        return reservation
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to use existing run output: {project_root / Path(*components)}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _reserve_run_outputs(
    specs: Sequence[BenchmarkSpec], config: Mapping[str, Any], project_root: Path
) -> _RunReservation:
    """Compatibility wrapper restricted to one immediately-started run cell."""
    if len(specs) != 1:
        raise ValueError("Run cells must be reserved one at a time immediately before start.")
    return _reserve_run_output(specs[0], config, project_root)


def collect_git_state(project_root: Path) -> tuple[str, bool]:
    """Read the exact repository commit and dirty state without a shell."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, shell=False,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=project_root, shell=False,
        check=True, capture_output=True, text=True,
    ).stdout)
    if len(commit) != 40:
        raise ValueError("git rev-parse did not return a 40-character commit.")
    return commit, dirty


_RUNTIME_METADATA_CODE = r'''
import importlib.metadata
import json
import platform
import sys

method = sys.argv[1]
versions = {"python": platform.python_version()}
for distribution in ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn"):
    try:
        versions[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        versions[distribution] = "unavailable"
try:
    import torch
    if torch.cuda.is_available():
        device_type = "cuda"
    else:
        mps = getattr(torch.backends, "mps", None)
        mps_available = bool(mps is not None and mps.is_available())
        device_type = "mps" if mps_available and method != "graphcare" else "cpu"
except Exception:
    device_type = "unavailable"
print(json.dumps({
    "device": {
        "type": device_type,
        "python_executable": sys.executable,
        "host_platform": f"{platform.system()}-{platform.release()}",
        "machine": platform.machine() or "unknown",
    },
    "package_versions": versions,
}, sort_keys=True))
'''


def collect_runtime_metadata(
    spec: BenchmarkSpec, method_config: Mapping[str, Any], project_root: Path
) -> tuple[dict[str, str], dict[str, str]]:
    """Query versions and selected device from the method's own interpreter."""
    command = method_config.get("command")
    if not isinstance(command, list) or not command or type(command[0]) is not str:
        raise ValueError(f"Missing interpreter command for {spec.method!r}.")
    result = subprocess.run(
        [command[0], "-c", _RUNTIME_METADATA_CODE, spec.method],
        cwd=project_root,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(result.stdout)
        device = payload["device"]
        versions = payload["package_versions"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid runtime metadata from {spec.method!r} interpreter.") from exc
    if (
        not isinstance(device, dict)
        or not device
        or any(type(key) is not str or type(value) is not str for key, value in device.items())
        or not isinstance(versions, dict)
        or not versions
        or any(type(key) is not str or type(value) is not str for key, value in versions.items())
    ):
        raise ValueError(f"Invalid runtime metadata types from {spec.method!r} interpreter.")
    return device, versions


def _fold_counts(split: Mapping[str, Any]) -> dict[str, int]:
    counts = {fold_id: 0 for fold_id in EXPECTED_FOLD_COUNTS}
    for fold_id in split["fold"].values():
        counts[fold_id] += 1
    return {"train": counts[0], "validation": counts[1], "test": counts[2]}


def _effective_fold_counts(
    canonical_fold_counts: Mapping[str, int], limit: Optional[int]
) -> dict[str, int]:
    if limit is None:
        return dict(canonical_fold_counts)
    return {
        fold: min(limit, count) for fold, count in canonical_fold_counts.items()
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--methods", nargs="+", choices=("protgnn", "gsat", "graphcare"))
    parser.add_argument("--structures", nargs="+", choices=("star", "cooccur"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--loss-weighting", choices=("sqrt_inverse", "none"), default=None,
                        help="Class-imbalance weighting applied uniformly to all three "
                             "methods' training loss (train-fold-only). Omit to use each "
                             "method's own unchanged default.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    options = RunnerOptions(config=config, limit=args.limit, max_epochs=args.max_epochs,
                            loss_weighting=args.loss_weighting)
    specs = build_run_specs(config)
    for name, requested in (
        ("methods", args.methods), ("structures", args.structures), ("seeds", args.seeds)
    ):
        if requested is not None and len(requested) != len(set(requested)):
            raise ValueError(f"{name} filter contains duplicates.")
    if args.seeds is not None and any(seed not in ALLOWED_SEEDS for seed in args.seeds):
        raise ValueError(f"seeds must be selected from {list(ALLOWED_SEEDS)}.")
    selected = [
        spec
        for spec in specs
        if (args.methods is None or spec.method in args.methods)
        and (args.structures is None or spec.structure in args.structures)
        and (args.seeds is None or spec.seed in args.seeds)
    ]
    dataset_path, split_path, _results_root, split = preflight(
        config, selected, PROJECT_ROOT
    )
    if args.dry_run:
        payload = [
            {
                "method": spec.method,
                "structure": spec.structure,
                "seed": spec.seed,
                "output_dir": str(run_output_dir(spec, config)),
                "command": command_for(spec, options),
            }
            for spec in selected
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    _reject_occupied_outputs(selected, config, PROJECT_ROOT)
    if options.limit is None and options.max_epochs is None:
        audit_report = audit_standardized_caches(
            project_root=PROJECT_ROOT,
            dataset_path=dataset_path,
            split_path=split_path,
            structures=sorted({spec.structure for spec in selected}),
        )
        print(json.dumps({"cache_audit": audit_report}, sort_keys=True))
    git_commit, git_dirty = collect_git_state(PROJECT_ROOT)
    runtime_metadata = {}
    for spec in selected:
        if spec.method not in runtime_metadata:
            runtime_metadata[spec.method] = collect_runtime_metadata(
                spec, config["methods"][spec.method], PROJECT_ROOT
            )
    failures = 0
    canonical_fold_counts = _fold_counts(split)
    effective_fold_counts = _effective_fold_counts(canonical_fold_counts, options.limit)
    for spec in selected:
        command = command_for(spec, options)
        device, package_versions = runtime_metadata[spec.method]
        reservation = _reserve_run_output(spec, config, PROJECT_ROOT)
        run_dir = reservation.run_dir
        manifest: Optional[RunManifest] = None
        try:
            try:
                manifest = RunManifest.start(
                    run_dir / "run_manifest.json",
                    spec=spec,
                    dataset_path=dataset_path,
                    split_path=split_path,
                    topology_parameters=config["topology_parameters"][spec.structure],
                    canonical_fold_counts=canonical_fold_counts,
                    effective_fold_counts=effective_fold_counts,
                    class_ordering=list(split["classes"]),
                    parameter_count=None,
                    device=device,
                    package_versions=package_versions,
                    git_commit=git_commit,
                    git_dirty=git_dirty,
                    command=command,
                    run_dir_fd=reservation.run_fd,
                )
            except BaseException:
                reservation.release_unstarted()
                raise
            reservation.mark_started()
            try:
                result = subprocess.run(
                    command, cwd=PROJECT_ROOT, shell=False, check=False
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"{spec.method}/{spec.structure}/seed_{spec.seed} exited with "
                        f"exit code {result.returncode}."
                    )
                method_config = config["methods"][spec.method]
                manifest.complete(
                    run_dir / method_config["metrics_path"],
                    run_dir / method_config["checkpoint_path"],
                )
            except BaseException as exc:
                if manifest.status == "running":
                    message = f"{type(exc).__name__}: {exc}".strip()
                    manifest.fail(message)
                if not isinstance(exc, Exception):
                    raise
                failures += 1
                if not args.continue_on_error:
                    return 1
        finally:
            if manifest is not None:
                manifest.close()
            reservation.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
