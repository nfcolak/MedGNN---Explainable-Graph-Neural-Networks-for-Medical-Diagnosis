"""Atomic, auditable lifecycle records for standardized benchmark runs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Mapping, Optional

from shared.lib.benchmark_contract import BenchmarkSpec, file_sha256


SCHEMA_VERSION = 1
_REQUIRED_FOLD_COUNTS = {"train": 59607, "validation": 7448, "test": 7456}
STANDARDIZED_METRICS = (
    "accuracy",
    "balanced_acc",
    "macro_f1",
    "micro_f1",
    "top3_acc",
    "top5_acc",
)


def topology_policy_fingerprint(parameters: Mapping[str, Any]) -> str:
    """Hash topology policy parameters using one canonical JSON encoding."""
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("topology_parameters must be a nonempty mapping.")
    try:
        canonical = json.dumps(
            dict(parameters),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("topology_parameters must be canonical JSON values.") from exc
    return hashlib.sha256(canonical).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _require_descriptor_safety() -> None:
    """Fail closed when the host cannot provide openat/no-follow primitives."""
    if not getattr(os, "O_NOFOLLOW", 0) or os.open not in os.supports_dir_fd:
        raise RuntimeError(
            "Descriptor-safe benchmark paths require O_NOFOLLOW and dir_fd support."
        )


def _open_directory_no_follow(path: Path) -> int:
    """Open an existing directory one component at a time without following links."""
    _require_descriptor_safety()
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _single_name(path: Path, label: str) -> str:
    if path.name != os.fspath(path) or path.name in {"", ".", ".."}:
        raise ValueError(f"{label} must be a single path component.")
    return path.name


def _atomic_json_write(
    path: Path,
    payload: Mapping[str, Any],
    *,
    directory_fd: Optional[int] = None,
) -> None:
    """Durably replace ``path`` relative to one already-validated directory."""
    path = Path(path)
    name = _single_name(Path(path.name), "manifest path")
    active_directory_fd = (
        _open_directory_no_follow(path.parent)
        if directory_fd is None
        else os.dup(directory_fd)
    )
    temp_name = f".{name}.{secrets.token_hex(8)}.tmp"
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
            0o600,
            dir_fd=active_directory_fd,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temp_name,
            name,
            src_dir_fd=active_directory_fd,
            dst_dir_fd=active_directory_fd,
        )
        os.fsync(active_directory_fd)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temp_name, dir_fd=active_directory_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(active_directory_fd)


def _reserve_manifest_owner(directory_fd: int, run_dir: Path) -> tuple[int, int]:
    """Claim a lifecycle with an exclusive marker anchored to the run descriptor."""
    owner_name = ".run_manifest.owner"
    try:
        descriptor = os.open(
            owner_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
    except FileExistsError as exc:
        raise FileExistsError(f"Run cell is already reserved: {run_dir}") from exc
    owner_stat = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
    except BaseException:
        try:
            current = os.stat(owner_name, dir_fd=directory_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) == (owner_stat.st_dev, owner_stat.st_ino):
                os.unlink(owner_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise
    return owner_stat.st_dev, owner_stat.st_ino


@dataclass(frozen=True)
class _ValidatedArtifact:
    relative: str
    content: Optional[bytes]


def _artifact_parts(path: Any, run_dir: Path, label: str) -> tuple[str, ...]:
    artifact = Path(path)
    if ".." in artifact.parts:
        raise ValueError(
            f"{label} path must be contained within the run directory without escapes."
        )
    if artifact.is_absolute():
        run_absolute = Path(os.path.abspath(os.fspath(run_dir)))
        artifact_absolute = Path(os.path.abspath(os.fspath(artifact)))
        try:
            artifact = artifact_absolute.relative_to(run_absolute)
        except ValueError as exc:
            raise ValueError(
                f"{label} path must be contained within the run directory."
            ) from exc
    parts = artifact.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            f"{label} path must be contained within the run directory without escapes."
        )
    return parts


def _artifact_relative_to_run(
    path: Any,
    run_dir: Path,
    label: str,
    *,
    directory_fd: Optional[int] = None,
) -> _ValidatedArtifact:
    """Open and consume an artifact through one no-follow descriptor chain."""
    parts = _artifact_parts(path, run_dir, label)
    active_directory_fd = (
        _open_directory_no_follow(run_dir)
        if directory_fd is None
        else os.dup(directory_fd)
    )
    artifact_fd: Optional[int] = None
    try:
        for component in parts[:-1]:
            next_descriptor = os.open(
                component, _DIRECTORY_FLAGS, dir_fd=active_directory_fd
            )
            os.close(active_directory_fd)
            active_directory_fd = next_descriptor
        artifact_fd = os.open(
            parts[-1], os.O_RDONLY | _FILE_NOFOLLOW, dir_fd=active_directory_fd
        )
        artifact_stat = os.fstat(artifact_fd)
        if not stat.S_ISREG(artifact_stat.st_mode):
            raise ValueError(f"{label} artifact must be a regular file.")
        if label == "metrics":
            chunks = []
            while True:
                chunk = os.read(artifact_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            content: Optional[bytes] = b"".join(chunks)
            try:
                payload = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("metrics artifact must be valid UTF-8 JSON.") from exc
            if not isinstance(payload, dict):
                raise ValueError("metrics artifact must contain a JSON object.")
        else:
            if artifact_stat.st_size == 0 or not os.read(artifact_fd, 1):
                raise ValueError("checkpoint artifact must be nonempty.")
            content = None
        return _ValidatedArtifact("/".join(parts), content)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} artifact does not exist: {path}") from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                f"{label} path must be contained within the run directory and may "
                "not contain symlinks or non-directories."
            ) from exc
        raise
    finally:
        if artifact_fd is not None:
            os.close(artifact_fd)
        os.close(active_directory_fd)


def _validated_metrics(content: bytes, method: str) -> int:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("metrics artifact must be valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("metrics artifact must contain a JSON object.")
    if method == "protgnn":
        if "test" in payload:
            raise ValueError("ProtGNN metrics schema must be a flat JSON object.")
        metrics = payload
    else:
        metrics = payload.get("test")
        if not isinstance(metrics, dict):
            raise ValueError(f"{method} metrics schema must contain a test JSON object.")
    missing = [name for name in STANDARDIZED_METRICS if name not in metrics]
    if missing:
        raise ValueError(
            "metrics artifact must contain the exact six standardized metrics; "
            f"missing: {', '.join(missing)}."
        )
    for name in STANDARDIZED_METRICS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"metrics {name} must be a finite numeric non-boolean value.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"metrics {name} must be a finite numeric non-boolean value.")
        if not 0.0 <= numeric_value <= 1.0:
            raise ValueError(f"metrics {name} must be in [0, 1].")

    if "parameter_count" not in payload:
        raise ValueError("metrics parameter_count is required at the documented top level.")
    top_level_count = payload["parameter_count"]
    if type(top_level_count) is not int or top_level_count < 0:
        raise ValueError("metrics parameter_count must be a nonnegative exact integer.")
    nested_count = metrics.get("parameter_count")
    if nested_count is not None and top_level_count != nested_count:
        raise ValueError("metrics parameter_count values must agree across the output schema.")
    return top_level_count


class RunManifest:
    """One run manifest with atomic start/complete/fail transitions."""

    def __init__(
        self,
        path: Any,
        data: Mapping[str, Any],
        directory_fd: int,
        owner_identity: tuple[int, int],
    ):
        self.path = Path(path)
        self.run_dir = self.path.parent
        self._data = deepcopy(dict(data))
        self._directory_fd = directory_fd
        self._run_identity = self._identity(os.fstat(directory_fd))
        self._owner_identity = owner_identity

    @staticmethod
    def _identity(metadata: os.stat_result) -> tuple[int, int]:
        return metadata.st_dev, metadata.st_ino

    @classmethod
    def start(
        cls,
        path: Any,
        *,
        spec: BenchmarkSpec,
        dataset_path: Any,
        split_path: Any,
        topology_parameters: Mapping[str, Any],
        canonical_fold_counts: Mapping[str, int],
        effective_fold_counts: Mapping[str, int],
        class_ordering: list[str],
        parameter_count: Optional[int],
        device: Mapping[str, Any],
        package_versions: Mapping[str, str],
        git_commit: str,
        git_dirty: bool,
        command: list[str],
        run_dir_fd: Optional[int] = None,
    ) -> "RunManifest":
        """Create a running manifest; an existing manifest is never overwritten."""
        manifest_path = Path(path)
        _single_name(Path(manifest_path.name), "manifest path")
        if not isinstance(spec, BenchmarkSpec):
            raise ValueError("spec must be a validated BenchmarkSpec.")
        dataset = Path(dataset_path)
        split = Path(split_path)
        for label, source in (("dataset", dataset), ("split", split)):
            if not source.exists() or not source.is_file() or source.stat().st_size == 0:
                raise ValueError(f"{label} path must identify a nonempty regular file.")
        canonical_counts = dict(canonical_fold_counts)
        if canonical_counts != _REQUIRED_FOLD_COUNTS:
            raise ValueError(
                f"canonical_fold_counts must exactly equal {_REQUIRED_FOLD_COUNTS}."
            )
        effective_counts = dict(effective_fold_counts)
        if set(effective_counts) != set(canonical_counts) or any(
            type(value) is not int
            or value < 0
            or value > canonical_counts[fold]
            for fold, value in effective_counts.items()
        ):
            raise ValueError(
                "effective_fold_counts must contain exact nonnegative integers no "
                "larger than canonical_fold_counts."
            )
        if (
            not isinstance(class_ordering, list)
            or len(class_ordering) != 30
            or len(set(class_ordering)) != 30
            or any(type(name) is not str or not name for name in class_ordering)
        ):
            raise ValueError("class_ordering must contain exactly 30 unique nonempty strings.")
        if parameter_count is not None and (
            type(parameter_count) is not int or parameter_count < 0
        ):
            raise ValueError("parameter_count must be a nonnegative exact integer or null.")
        if not isinstance(topology_parameters, Mapping) or not topology_parameters:
            raise ValueError("topology_parameters must be a nonempty mapping.")
        topology_fingerprint = topology_policy_fingerprint(topology_parameters)
        if not isinstance(device, Mapping) or not device:
            raise ValueError("device must be a nonempty mapping.")
        if not isinstance(package_versions, Mapping) or not package_versions:
            raise ValueError("package_versions must be a nonempty mapping.")
        if type(git_commit) is not str or not git_commit:
            raise ValueError("git_commit must be a nonempty string.")
        if type(git_dirty) is not bool:
            raise ValueError("git_dirty must be a boolean.")
        if (
            not isinstance(command, list)
            or not command
            or any(type(part) is not str or not part for part in command)
        ):
            raise ValueError("command must be a nonempty argv list of strings.")

        started_at = _utc_now()
        data = {
            "schema_version": SCHEMA_VERSION,
            "method": spec.method,
            "topology": spec.structure,
            "seed": spec.seed,
            "dataset": {"path": str(dataset), "sha256": file_sha256(dataset)},
            "split": {"path": str(split), "sha256": file_sha256(split)},
            "topology_parameters": deepcopy(dict(topology_parameters)),
            "topology_policy_fingerprint": topology_fingerprint,
            "canonical_fold_counts": canonical_counts,
            "effective_fold_counts": effective_counts,
            "class_ordering": list(class_ordering),
            "parameter_count": parameter_count,
            "device": deepcopy(dict(device)),
            "package_versions": deepcopy(dict(package_versions)),
            "git": {"commit": git_commit, "dirty": git_dirty},
            "command": list(command),
            "timestamps": {"started_at": started_at, "ended_at": None},
            "status": "running",
            "metrics_path": None,
            "checkpoint_path": None,
            "error": None,
        }
        instance: Optional[RunManifest] = None
        directory_fd = (
            _open_directory_no_follow(manifest_path.parent)
            if run_dir_fd is None
            else os.dup(run_dir_fd)
        )
        try:
            directory_stat = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_stat.st_mode):
                raise ValueError("Run directory descriptor must identify a directory.")
            try:
                os.stat(
                    manifest_path.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(f"Run manifest already exists: {manifest_path}")
            owner_identity = _reserve_manifest_owner(directory_fd, manifest_path.parent)
            instance = cls(manifest_path, data, directory_fd, owner_identity)
            directory_fd = -1
            _atomic_json_write(
                instance.path,
                instance._data,
                directory_fd=instance._directory_fd,
            )
        except BaseException:
            if directory_fd >= 0:
                os.close(directory_fd)
            elif instance is not None:
                instance._remove_owned_marker()
                instance.close()
            raise
        return instance

    def complete(self, metrics_path: Any, checkpoint_path: Any) -> None:
        """Atomically mark a run complete after validating both contained artifacts."""
        self._require_running()
        try:
            metrics = _artifact_relative_to_run(
                metrics_path,
                self.run_dir,
                "metrics",
                directory_fd=self._directory_fd,
            )
            checkpoint = _artifact_relative_to_run(
                checkpoint_path,
                self.run_dir,
                "checkpoint",
                directory_fd=self._directory_fd,
            )
            assert metrics.content is not None
            parameter_count = _validated_metrics(metrics.content, self._data["method"])
        except Exception as exc:
            self.fail(f"Completion validation failed: {exc}")
            raise
        updated = deepcopy(self._data)
        updated["status"] = "completed"
        updated["metrics_path"] = metrics.relative
        updated["checkpoint_path"] = checkpoint.relative
        updated["parameter_count"] = parameter_count
        updated["timestamps"]["ended_at"] = _utc_now()
        updated["error"] = None
        self._validate_owner()
        _atomic_json_write(self.path, updated, directory_fd=self._directory_fd)
        self._data = updated

    def fail(self, error: Any) -> None:
        """Atomically mark a running manifest failed with a nonempty error."""
        self._require_running()
        message = str(error).strip()
        if not message:
            raise ValueError("error must be nonempty.")
        updated = deepcopy(self._data)
        updated["status"] = "failed"
        updated["timestamps"]["ended_at"] = _utc_now()
        updated["error"] = message
        self._validate_owner()
        _atomic_json_write(self.path, updated, directory_fd=self._directory_fd)
        self._data = updated

    def _validate_owner(self) -> None:
        if self._identity(os.fstat(self._directory_fd)) != self._run_identity:
            raise RuntimeError("Run directory ownership changed during the lifecycle.")
        try:
            owner = os.stat(
                ".run_manifest.owner",
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Run manifest ownership marker is missing.") from exc
        if not stat.S_ISREG(owner.st_mode) or self._identity(owner) != self._owner_identity:
            raise RuntimeError("Run manifest ownership marker changed.")

    def _remove_owned_marker(self) -> None:
        try:
            owner = os.stat(
                ".run_manifest.owner",
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            if self._identity(owner) == self._owner_identity:
                os.unlink(".run_manifest.owner", dir_fd=self._directory_fd)
                os.fsync(self._directory_fd)
        except FileNotFoundError:
            pass

    def close(self) -> None:
        """Release this process's descriptor without removing retained ownership."""
        if self._directory_fd >= 0:
            os.close(self._directory_fd)
            self._directory_fd = -1

    def _require_running(self) -> None:
        if self._data.get("status") != "running":
            raise RuntimeError("Only a running manifest may transition state.")

    @property
    def status(self) -> str:
        """Return the in-memory lifecycle state for exception-safe callers."""
        return self._data["status"]
