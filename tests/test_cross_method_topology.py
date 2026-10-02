"""Production-cache cross-method canonical topology invariant tests."""


import pytest
import torch
from torch_geometric.data import Data

from shared.lib.canonical_graph import (
    CanonicalGraph,
    canonical_graph_from_graphcare_record,
    canonical_graph_from_pyg_record,
    validate_cross_method_fingerprints,
)


@pytest.mark.parametrize("adapter", ["pyg", "graphcare"])
def test_production_adapters_fail_closed_when_canonical_metadata_is_missing(adapter):
    if adapter == "pyg":
        record = Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            y=torch.tensor([0]),
            subject_id=torch.tensor([1]),
        )
        with pytest.raises(ValueError, match="(?i)canonical.*metadata"):
            canonical_graph_from_pyg_record(record)
    else:
        record = {
            "node_ids": torch.tensor([0]),
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "rel_ids": torch.zeros(0, dtype=torch.long),
            "y": 0,
            "subject_id": "1",
        }
        with pytest.raises(ValueError, match="(?i)canonical.*metadata"):
            canonical_graph_from_graphcare_record(record, {"num_nodes": 1})
