"""Clinical graph method-adapter registry.

Tasks adding a method should register its adapter here; shared loading and
validation remain in the clinical_graph_v2 runner.
"""
from __future__ import annotations

from .base import ClinicalMethodAdapter, MethodOutput
from .graphcare import GraphCareAdapter
from .gsat import GSATAdapter
from .protgnn import ProtGNNAdapter

METHOD_REGISTRY: dict[str, type[ClinicalMethodAdapter]] = {
    "graphcare": GraphCareAdapter,
    "gsat": GSATAdapter,
    "protgnn": ProtGNNAdapter,
}


def build_method(name, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args) -> ClinicalMethodAdapter:
    """Construct one registered clinical adapter using the shared dimensions."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("method name must be a nonempty string")
    key = name.strip().lower()
    adapter_type = METHOD_REGISTRY.get(key)
    if adapter_type is None:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"unknown clinical method {name!r}; registered methods: {available}")
    return adapter_type(
        num_tokens=num_tokens,
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_classes=num_classes,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        token_dim=token_dim,
        num_triples=num_triples,
        args=args,
    )


__all__ = [
    "ClinicalMethodAdapter",
    "MethodOutput",
    "METHOD_REGISTRY",
    "GraphCareAdapter",
    "GSATAdapter",
    "ProtGNNAdapter",
    "build_method",
]
