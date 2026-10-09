"""Temporal patient/event/knowledge graph tensorization and classification.

Fit EventGraphTensorizer only on the training split, serialize its state with the
model checkpoint, and use the same frozen adapter for validation/test inference.
The package requires the repository's existing torch and torch-geometric deps.
"""
from .tensorize import (
    EDGE_FEATURES,
    KIND_TO_ID,
    NODE_FEATURES,
    NODE_KINDS,
    SCHEMA_VERSION,
    EventGraphTensorizer,
)
from .model import EventGraphGNN, TemporalRelationalLayer
from .data import EventGraphArtifact, EventGraphDataset

__all__ = [
    "EventGraphTensorizer", "EventGraphGNN", "TemporalRelationalLayer",
    "EventGraphArtifact", "EventGraphDataset",
    "SCHEMA_VERSION", "NODE_KINDS", "KIND_TO_ID", "NODE_FEATURES", "EDGE_FEATURES",
]
