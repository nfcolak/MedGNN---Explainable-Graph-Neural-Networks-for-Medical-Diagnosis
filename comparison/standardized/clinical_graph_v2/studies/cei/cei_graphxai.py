"""Exact fixed-graph GraphXAI bridge for the clinical CEI adapter."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from ...paths import PACKAGE_ROOT
from typing import Any, Mapping, Sequence

import torch
from torch import nn


ADAPTATION_VERSION = "clinical_graph_v2_cei_gnn_v1"
REQUIRED_ALGORITHMS = ("GradExplainer", "IntegratedGradExplainer", "GNNExplainer")
EXPLANATION_BOUNDARIES = {
    "node_gradients": "continuous numeric and embedding channels, conditional on fixed edge metadata",
    "feature_zeroing": "continuous-representation intervention; not clinical event deletion",
    "causal_claim": False,
}


class ClinicalGraphXAIWrapper(nn.Module):
    """Expose the adapter's exact continuous-input core on one immutable graph."""

    def __init__(self, adapter: Any, graph: Any):
        super().__init__()
        self.adapter = adapter
        self.graph = copy.deepcopy(graph)
        for key in self.graph.keys():
            value = getattr(self.graph, key)
            if isinstance(value, torch.Tensor):
                setattr(self.graph, key, value.detach().clone())
        self._edge_index = self.graph.edge_index.detach().clone()
        self._batch = _batch_of(self.graph)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor, batch=None):
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("features must be a rank-2 continuous tensor")
        if features.size(0) != int(self.graph.num_nodes):
            raise ValueError("feature node count does not match bound graph")
        if edge_index.shape != self._edge_index.shape or not torch.equal(edge_index, self._edge_index.to(edge_index.device)):
            raise ValueError("edge list/order differs from bound graph")
        expected_batch = self._batch
        if batch is None:
            batch = expected_batch
        if batch.shape != expected_batch.shape or not torch.equal(batch, expected_batch.to(batch.device)):
            raise ValueError("batch identity differs from bound graph")
        return self.adapter.forward_continuous(features, edge_index, self.graph, return_parts=False)


def _batch_of(graph):
    batch = getattr(graph, "batch", None)
    if batch is None:
        return torch.zeros(int(graph.num_nodes), dtype=torch.long, device=graph.edge_index.device)
    return batch.detach().clone()


class GraphXAIExplanationError(RuntimeError):
    """Failed explanation with a machine-readable record that cannot pass export."""

    def __init__(self, message: str):
        super().__init__(message)
        self.record = {"status": "failed", "error": str(message)}
        self.partial_records = [self.record]


def explain_graph(adapter, graph, *, steps=32, epochs=50) -> dict:
    """Run the repository's actual vendored GraphXAI algorithms on one graph."""
    features = adapter.continuous_inputs(graph).detach()
    wrapper = ClinicalGraphXAIWrapper(adapter, graph).eval()
    expected = adapter(graph, epoch=0).logits
    actual = wrapper(features, graph.edge_index, batch=_batch_of(graph))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    from shared.lib.graphxai_standardized import explain_algorithms

    try:
        result = explain_algorithms(
            wrapper, features, graph.edge_index, batch=_batch_of(graph), steps=steps, epochs=epochs
        )
    except Exception as exc:
        raise GraphXAIExplanationError(str(exc)) from exc
    if set(result) != set(REQUIRED_ALGORITHMS):
        raise GraphXAIExplanationError("incomplete GraphXAI algorithm set")
    for name, item in result.items():
        if item.get("status") != "success":
            raise GraphXAIExplanationError(f"GraphXAI algorithm failed: {name}")
        importance = item.get("node_explanation", {}).get("node_importance")
        if importance is None or not torch.isfinite(torch.as_tensor(importance)).all():
            raise GraphXAIExplanationError(f"GraphXAI algorithm produced nonfinite/incomplete record: {name}")
        item["provenance"].update(EXPLANATION_BOUNDARIES)
    return result


def validate_dev_cohort(*, requested_ids, frozen_dev_ids, validation_ids, test_ids):
    """Return exact visit/sample keys only when every ID is frozen-dev-only."""
    requested = _unique_ids(requested_ids, "requested cohort")
    dev = set(_unique_ids(frozen_dev_ids, "frozen dev roster"))
    validation = set(_unique_ids(validation_ids, "validation roster"))
    test = set(_unique_ids(test_ids, "test roster"))
    if validation & test or validation & dev or test & dev:
        raise ValueError("fold rosters overlap")
    if not requested:
        raise ValueError("requested cohort must not be empty")
    if not set(requested) <= dev:
        raise ValueError("cohort contains non-dev or unknown sample IDs")
    # Subject-only IDs do not satisfy a frozen sample-ID membership check.
    return tuple(requested)


def _unique_ids(values, label):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a sequence of sample IDs")
    ids = tuple(values)
    if any(type(value) is not str or not value.strip() for value in ids):
        raise ValueError(f"{label} contains invalid sample IDs")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate sample IDs")
    return ids


def binding_hashes(*, package_root, graph_path, membership_path, checkpoint_path, cohort_ids):
    """Compute runner-compatible provenance hashes from source and artifact files."""
    from ...core.contracts import recursive_source_hashes

    def file_hash(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    source = recursive_source_hashes(package_root)
    cohort_digest = hashlib.sha256(json.dumps(list(cohort_ids), separators=(",", ":")).encode()).hexdigest()
    return {
        "source_sha256": source,
        "graph_sha256": file_hash(graph_path),
        "membership_sha256": file_hash(membership_path),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "cohort_ids_sha256": cohort_digest,
    }


def reconstruct_adapter(binding: Mapping[str, Any], checkpoint_path, *, expected_checkpoint_sha256):
    """Strictly rebuild CEI from the runner binding and state_dict checkpoint."""
    if binding.get("method") != "cei_gnn" or binding.get("adaptation_version") != ADAPTATION_VERSION:
        raise ValueError("incompatible method/adaptation binding")
    method_config = binding.get("method_config")
    if not isinstance(method_config, Mapping) or method_config.get("adaptation_version") != ADAPTATION_VERSION:
        raise ValueError("incompatible method configuration")
    architecture = method_config.get("architecture")
    effective = method_config.get("effective_settings")
    if not isinstance(architecture, Mapping) or not isinstance(effective, Mapping):
        raise ValueError("missing architecture/effective settings")
    runner = binding.get("runner_settings", binding)
    if not isinstance(runner, Mapping):
        raise ValueError("runner settings are missing")
    dims = dict(architecture)
    required_architecture = {
        "num_tokens", "node_dim", "edge_dim", "num_triples", "num_relations",
        "num_node_types", "num_classes", "hidden", "layers", "dropout",
        "token_dim", "interaction_rank", "parameter_count",
        "active_parameter_count", "inactive_parameter_count",
    }
    if not required_architecture <= set(dims):
        raise ValueError("method architecture is missing required dimensions or parameter counts")
    top_level = {
        "num_tokens": binding.get("vocabulary_size"),
        "node_dim": binding.get("node_dim"), "edge_dim": binding.get("edge_dim"),
        "num_triples": binding.get("num_meta_relations"),
        "num_classes": binding.get("num_classes"), "hidden": binding.get("hidden"),
        "layers": binding.get("layers"), "dropout": binding.get("dropout"),
        "token_dim": binding.get("token_dim"), "num_relations": binding.get("num_relations"),
    }
    for key, value in top_level.items():
        # The current runner records token_dim only inside method_config.architecture.
        # If a future runner also emits a top-level value, it must agree.
        if key == "token_dim" and value is None:
            continue
        if value is None or dims.get(key) != value:
            raise ValueError(f"runner binding {key} differs from method architecture")
    constructor_keys = ("num_tokens", "node_dim", "edge_dim", "num_classes", "hidden",
                        "layers", "dropout", "token_dim", "num_triples")
    if any(type(dims.get(key)) not in (int, float) or isinstance(dims.get(key), bool)
           for key in constructor_keys):
        raise ValueError("runner binding is missing constructor dimensions")
    if effective.get("interaction_rank") != dims["interaction_rank"] or dims["interaction_rank"] != 16:
        raise ValueError("unsupported CEI interaction rank")
    if effective.get("use_interactions") not in (True, False) or dims["layers"] != 1:
        raise ValueError("unsupported CEI effective settings or architecture depth")
    from ...core.contracts import recursive_source_hashes
    source_binding = binding.get("source_code")
    current_sources = recursive_source_hashes(PACKAGE_ROOT)
    if not isinstance(source_binding, Mapping) or dict(source_binding) != current_sources:
        raise ValueError("source binding differs from current package-relative source hashes")
    expected_checkpoint_sha256 = str(expected_checkpoint_sha256)
    if (len(expected_checkpoint_sha256) != 64 or
            any(char not in "0123456789abcdef" for char in expected_checkpoint_sha256)):
        raise ValueError("explicit expected checkpoint digest must be a lowercase SHA-256")
    actual_checkpoint_sha256 = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("checkpoint digest does not match explicit expected digest")
    from ...methods.plugin_cei_gnn import EvidenceInteractionAdapter
    from types import SimpleNamespace

    args_values = dict(runner)
    args_values["method_options"] = {
        "interaction_rank": effective["interaction_rank"],
        "use_interactions": effective["use_interactions"],
    }
    args = SimpleNamespace(**args_values)
    adapter = EvidenceInteractionAdapter(**{key: dims[key] for key in constructor_keys}, args=args)
    config = adapter.run_config()
    if config.get("adaptation_version") != ADAPTATION_VERSION:
        raise ValueError("reconstructed adapter version mismatch")
    for key in ("architecture", "effective_settings"):
        if config.get(key) != method_config.get(key):
            raise ValueError(f"reconstructed {key} mismatch")
    state = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if not isinstance(state, Mapping) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("checkpoint must be a nonempty state_dict")
    try:
        adapter.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("checkpoint state_dict is incompatible") from exc
    return adapter.cpu().eval()


def _finite_json_tree(value):
    import math
    from numbers import Real

    if isinstance(value, Mapping):
        return all(_finite_json_tree(key) and _finite_json_tree(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_json_tree(item) for item in value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return value is None or isinstance(value, (str, bool, int))


def _validate_algorithm_record(name, item):
    if not isinstance(item, Mapping) or item.get("status") != "success":
        raise RuntimeError(f"incomplete explanation export: {name} did not succeed")
    explanation = item.get("node_explanation")
    expected_keys = {"target", "node_importance", "top_nodes", "top_k", "fidelity_plus",
                     "fidelity_minus", "sparsity"}
    if not isinstance(explanation, Mapping) or set(explanation) != expected_keys:
        raise ValueError(f"{name} explanation does not match the genuine node explanation schema")
    importance = explanation.get("node_importance")
    if not isinstance(importance, (list, tuple)) or not importance or not _finite_json_tree(importance):
        raise ValueError(f"{name} node explanation is missing or nonfinite")
    target = explanation.get("target")
    if (not isinstance(target, Mapping) or target.get("provenance") != "model_prediction"
            or type(target.get("class_id")) is not int):
        raise ValueError(f"{name} explanation target provenance is invalid")
    if not _finite_json_tree(explanation):
        raise ValueError(f"{name} explanation contains nonfinite metrics")
    provenance = item.get("provenance")
    if not isinstance(provenance, Mapping) or not {
        "implementation", "source_sha256", "target", "node_gradients", "feature_zeroing", "causal_claim"
    } <= set(provenance):
        raise ValueError(f"{name} explanation provenance is missing")
    hashes = provenance.get("source_sha256")
    if (not isinstance(hashes, Mapping) or not hashes or
            any(not isinstance(digest, str) or len(digest) != 64 or
                any(char not in "0123456789abcdef" for char in digest) for digest in hashes.values())):
        raise ValueError(f"{name} explanation source provenance is invalid")
    if any(provenance.get(key) != value for key, value in EXPLANATION_BOUNDARIES.items()):
        raise ValueError(f"{name} explanation interpretation boundaries are invalid")


def export_explanations(
    output_dir, *, records, manifest, binding=None, frozen_dev_ids=None,
    graph_path=None, membership_path=None, checkpoint_path=None,
    expected_checkpoint_sha256=None,
):
    """Journal records beside a fresh output; publish manifest only when complete."""
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    # mkdir(exist_ok=False) is the ownership boundary: a concurrent creator wins
    # or this call does, and this function never removes the requested path.
    output.mkdir(mode=0o700, exist_ok=False)
    fd, journal_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".journal", dir=output.parent)
    journal = Path(journal_name)
    published = False
    record_values = []
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                record_values.append(record)
            if not record_values:
                raise ValueError("incomplete explanation export: no records")
        sample_ids = []
        for record in record_values:
            if not isinstance(record, Mapping) or type(record.get("sample_id")) is not str or not record["sample_id"]:
                raise ValueError("incomplete explanation export: missing sample-keyed record")
            sample_ids.append(record["sample_id"])
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("incomplete explanation export: duplicate sample IDs")
        if not isinstance(binding, Mapping) or not isinstance(frozen_dev_ids, Sequence):
            raise ValueError("export requires the runner binding and frozen dev sample-ID roster")
        dev_ids = _unique_ids(frozen_dev_ids, "frozen dev roster")
        split_hashes = binding.get("split_sample_ids_sha256")
        if not isinstance(split_hashes, Mapping) or not isinstance(split_hashes.get("dev"), str):
            raise ValueError("runner binding is missing its dev sample-ID hash")
        roster_hash = hashlib.sha256(json.dumps(list(dev_ids), separators=(",", ":"),
                                               ensure_ascii=True).encode("utf-8")).hexdigest()
        if split_hashes["dev"] != roster_hash:
            raise ValueError("frozen dev roster hash does not match runner binding")
        if not set(sample_ids) <= set(dev_ids):
            raise ValueError("export records contain IDs outside the bound frozen dev roster")
        for record in record_values:
            graphxai = record.get("graphxai")
            if not isinstance(graphxai, Mapping) or set(graphxai) != set(REQUIRED_ALGORITHMS):
                raise ValueError("incomplete explanation export: required algorithms absent")
            for name in REQUIRED_ALGORITHMS:
                _validate_algorithm_record(name, graphxai[name])
        label_order = binding.get("label_order")
        if not isinstance(label_order, (list, tuple)) or not label_order:
            raise ValueError("runner binding is missing ordered class labels")
        if not isinstance(manifest, Mapping):
            raise ValueError("manifest must be an object")
        required_hashes = {"source_sha256", "graph_sha256", "membership_sha256", "checkpoint_sha256", "cohort_ids_sha256"}
        required_manifest = required_hashes | {"fold", "seed", "class_order", "cohort_identity"}
        if not required_manifest <= set(manifest):
            raise ValueError("manifest missing fold, seed, class order, cohort identity or provenance hashes")
        if manifest.get("fold") != "dev" or type(manifest.get("seed")) is not int:
            raise ValueError("manifest must explicitly bind the dev fold and integer seed")
        if not isinstance(manifest.get("class_order"), (list, tuple)) or not manifest["class_order"]:
            raise ValueError("manifest must contain a nonempty ordered class list")
        if list(manifest["class_order"]) != list(label_order):
            raise ValueError("manifest class order differs from the runner binding")
        if not isinstance(manifest.get("cohort_identity"), str) or not manifest["cohort_identity"].strip():
            raise ValueError("manifest must bind exact cohort identity")
        if dict(manifest.get("interpretation_boundaries", {})) != EXPLANATION_BOUNDARIES:
            raise ValueError("manifest interpretation boundaries do not match the fixed contract")
        for key in ("graph_sha256", "membership_sha256", "checkpoint_sha256", "cohort_ids_sha256"):
            value = manifest[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"manifest has invalid {key}")
        if not isinstance(manifest["source_sha256"], Mapping) or not manifest["source_sha256"]:
            raise ValueError("manifest must contain source file hashes")
        actual_cohort_hash = hashlib.sha256(json.dumps(sample_ids, separators=(",", ":")).encode()).hexdigest()
        if manifest["cohort_ids_sha256"] != actual_cohort_hash:
            raise ValueError("manifest cohort hash does not match sample-keyed records")
        if any(path is None for path in (graph_path, membership_path, checkpoint_path)):
            raise ValueError("export requires real graph, membership, and checkpoint file paths")
        if (not isinstance(expected_checkpoint_sha256, str) or len(expected_checkpoint_sha256) != 64
                or any(char not in "0123456789abcdef" for char in expected_checkpoint_sha256)):
            raise ValueError("export requires an explicit lowercase checkpoint SHA-256")
        derived = binding_hashes(
            package_root=PACKAGE_ROOT, graph_path=graph_path,
            membership_path=membership_path, checkpoint_path=checkpoint_path, cohort_ids=sample_ids,
        )
        if dict(binding.get("source_code", {})) != derived["source_sha256"]:
            raise ValueError("runner source binding differs from current package source files")
        if dict(manifest.get("source_sha256", {})) != derived["source_sha256"]:
            raise ValueError("manifest source hashes differ from current package source files")
        if expected_checkpoint_sha256 != derived["checkpoint_sha256"]:
            raise ValueError("checkpoint file digest differs from explicit expected digest")
        runner_input_hashes = {
            "artifact_graphs_sha256": derived["graph_sha256"],
            "artifact_visit_membership_sha256": derived["membership_sha256"],
        }
        for binding_key, actual_hash in runner_input_hashes.items():
            if binding.get(binding_key) != actual_hash:
                raise ValueError(f"runner binding {binding_key} differs from the actual file")
        if type(binding.get("seed")) is not int or manifest["seed"] != binding["seed"]:
            raise ValueError("manifest seed differs from the runner binding")
        for key in ("graph_sha256", "membership_sha256", "checkpoint_sha256"):
            if manifest[key] != derived[key]:
                raise ValueError(f"manifest {key} differs from the actual file")
        (output / "records.jsonl").write_text(journal.read_text(encoding="utf-8"), encoding="utf-8")
        payload = dict(manifest)
        payload.update(derived)
        payload["status"] = "completed"
        payload["record_count"] = len(record_values)
        (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        published = True
        return output
    except Exception:
        raise
    finally:
        if published:
            journal.unlink(missing_ok=True)
        elif journal.exists():
            journal.rename(journal.with_suffix(".failed.journal"))
