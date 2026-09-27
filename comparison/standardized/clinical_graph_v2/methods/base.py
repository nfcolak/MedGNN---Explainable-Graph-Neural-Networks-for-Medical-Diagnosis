"""Shared method-adapter outputs and training hooks.

The runner owns data loading, loss composition, evaluation, and run artifacts. An
adapter owns only its forward mechanics, method-native objective, optimizer groups,
and epoch-start schedule.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from .. import INFORMATIVE_RELATIONS, NODE_KINDS, STRUCTURAL_RELATIONS
from ..tensorize import relation_vocabulary

# Relation ids of the historical forward edge view. A bidirectional run appends typed
# reverse relations after these ids; adapters size their relation tables from the
# edge view the runner actually tensorized (`relation_count`), never from this alone.
NUM_RELATIONS = len(STRUCTURAL_RELATIONS) + len(INFORMATIVE_RELATIONS)


def relation_count(args) -> int:
    """Relation vocabulary size of the runner's edge view; forward when unspecified."""
    if isinstance(args, Mapping):
        direction = args.get("edge_direction")
    else:
        direction = getattr(args, "edge_direction", None) if args is not None else None
    return len(relation_vocabulary(direction or "forward"))


@dataclass
class MethodOutput:
    """Prediction logits, differentiable method loss, and JSON-safe diagnostics."""

    logits: torch.Tensor
    auxiliary_loss: torch.Tensor
    diagnostics: dict[str, float]


class ClinicalMethodAdapter(nn.Module, ABC):
    """Exact shared API implemented by each clinical method adapter."""

    @abstractmethod
    def forward(self, batch, *, epoch: int) -> MethodOutput:
        raise NotImplementedError

    @abstractmethod
    def on_epoch_start(self, epoch: int, train_loader) -> None:
        raise NotImplementedError

    @abstractmethod
    def optimizer_groups(self, args) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def run_config(self) -> dict:
        raise NotImplementedError


def method_setting(args, names, default, cast, *, minimum=None, maximum=None):
    """Return the first supplied, non-None override (Namespace or mapping), validated."""
    value = default
    for name in names:
        if isinstance(args, Mapping):
            supplied = args.get(name)
        else:
            supplied = getattr(args, name, None) if args is not None else None
        if supplied is not None:
            value = supplied
            break
    value = cast(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{names[0]} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{names[0]} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{names[0]} must be <= {maximum}")
    return value


def epoch_index(epoch) -> int:
    """Validate the runner's 0-based epoch index."""
    if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    return int(epoch)


def diagnostic_float(value, method: str) -> float:
    """JSON-safe finite scalar; a non-finite diagnostic fails the run closed."""
    result = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{method} produced a non-finite diagnostic")
    return result


@dataclass(frozen=True)
class ClinicalBatch:
    """Validated shared clinical tensors of one PyG batch or single graph."""

    x: torch.Tensor
    token: torch.Tensor
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_relation: torch.Tensor
    edge_triple: torch.Tensor
    batch_index: torch.Tensor
    graph_count: int


def read_clinical_batch(batch, *, method, node_dim, edge_dim, num_tokens,
                        num_triples, num_relations=NUM_RELATIONS) -> ClinicalBatch:
    """Fetch and validate the shared tensor contract every adapter consumes."""
    tensors = {}
    for name in ("x", "token", "node_type", "edge_index", "edge_attr",
                 "edge_relation", "edge_triple"):
        value = getattr(batch, name, None)
        if value is None:
            raise ValueError(f"{method} requires {name}")
        tensors[name] = value
    x = tensors["x"]
    if x.ndim != 2 or x.size(1) != node_dim:
        raise ValueError("clinical node feature width differs from adapter configuration")
    node_count = x.size(0)
    if node_count == 0:
        raise ValueError(f"{method} does not accept a graph batch with no nodes")
    token, node_type = tensors["token"].long(), tensors["node_type"].long()
    if token.ndim != 1 or token.numel() != node_count:
        raise ValueError("token indices must align with graph nodes")
    if node_type.ndim != 1 or node_type.numel() != node_count:
        raise ValueError("node_type indices must align with graph nodes")
    edge_index = tensors["edge_index"].long()
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    edge_count = edge_index.size(1)
    edge_attr = tensors["edge_attr"]
    if edge_attr.ndim != 2 or tuple(edge_attr.shape) != (edge_count, edge_dim):
        raise ValueError("edge_attr width differs from adapter configuration")
    edge_relation = tensors["edge_relation"].long()
    edge_triple = tensors["edge_triple"].long()
    if edge_relation.ndim != 1 or edge_relation.numel() != edge_count:
        raise ValueError("edge_relation indices must align with graph edges")
    if edge_triple.ndim != 1 or edge_triple.numel() != edge_count:
        raise ValueError("edge_triple indices must align with graph edges")
    if token.min() < 0 or token.max() >= num_tokens:
        raise ValueError("token index is outside the fitted vocabulary")
    if node_type.min() < 0 or node_type.max() >= len(NODE_KINDS):
        raise ValueError("node_type index is outside the clinical node-kind vocabulary")
    if edge_count:
        if edge_index.min() < 0 or edge_index.max() >= node_count:
            raise ValueError("edge_index refers to a node outside the batch")
        if edge_relation.min() < 0 or edge_relation.max() >= num_relations:
            raise ValueError("edge_relation index is outside the clinical relation vocabulary")
        if edge_triple.min() < 0 or edge_triple.max() >= num_triples:
            raise ValueError("edge_triple index is outside the fitted meta-relation vocabulary")

    batch_index = getattr(batch, "batch", None)
    if batch_index is None:
        batch_index = torch.zeros(node_count, dtype=torch.long, device=x.device)
        graph_count = 1
    else:
        batch_index = batch_index.to(device=x.device, dtype=torch.long)
        if batch_index.ndim != 1 or batch_index.numel() != node_count:
            raise ValueError("batch assignment must align with graph nodes")
        graph_count = int(batch_index.max().item()) + 1
        declared_count = getattr(batch, "num_graphs", None)
        if declared_count is not None:
            graph_count = max(graph_count, int(declared_count))
    return ClinicalBatch(x=x.float(), token=token, node_type=node_type,
                         edge_index=edge_index, edge_attr=edge_attr.float(),
                         edge_relation=edge_relation, edge_triple=edge_triple,
                         batch_index=batch_index, graph_count=graph_count)


def graph_mean(values, batch_index, graph_count, mask=None):
    """Per-graph mean of node rows; a graph with no selected rows gets zeros."""
    if mask is not None:
        values, batch_index = values[mask], batch_index[mask]
    total = values.new_zeros((graph_count, values.size(-1))).index_add(0, batch_index, values)
    count = values.new_zeros((graph_count, 1)).index_add(
        0, batch_index, values.new_ones((batch_index.numel(), 1)))
    return total / count.clamp_min(1.0)


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _method_options(args) -> dict:
    if isinstance(args, Mapping):
        options = args.get("method_options")
    else:
        options = getattr(args, "method_options", None) if args is not None else None
    return dict(options or {})


def method_option(args, key, default, cast, *, minimum=None, maximum=None, choices=None):
    """Read one `--method-option key=value` setting of a plugin method, validated."""
    options = _method_options(args)
    value = cast(options[key]) if key in options else default
    if cast is bool and key in options:
        value = str(options[key]).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"method option {key} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"method option {key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"method option {key} must be <= {maximum}")
    if choices is not None and value not in choices:
        raise ValueError(f"method option {key} must be one of {list(choices)}")
    return value


def reject_unknown_options(args, known) -> None:
    """Fail closed on a misspelled plugin option instead of silently ignoring it."""
    unknown = sorted(set(_method_options(args)) - set(known))
    if unknown:
        raise ValueError(f"unknown method option(s): {unknown}; known: {sorted(known)}")
