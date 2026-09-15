"""Faithful deterministic node attribution for standardized GraphCare runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

try:
    from torch.func import functional_call
except ImportError:  # torch 1.x GraphCare environment
    from torch.nn.utils.stateless import functional_call

from graphcare_analysis.adapter import _collate, build_standardized_dataset
from graphcare_analysis.build_kg import validate_kg_schema, validate_training_provenance
from graphcare_analysis.config import cfg
from graphcare_analysis.run import build_graphcare_model
from shared.lib.benchmark_contract import BenchmarkSpec
from shared.lib.config_base import set_seed
from shared.lib.explanation_contract import (
    COHORT_SIZE,
    EXPLANATION_SCHEMA,
    EXPLANATION_SCHEMA_VERSION,
    build_node_explanation,
    load_explanation_cohort,
    select_exact_subject_records,
    subject_id_of,
    write_standardized_explanations,
)
from shared.lib.graph_structures import structure_dir


class GraphCareGraphXAIWrapper(nn.Module):
    """Expose GraphCare's embedding inputs as differentiable local node masks.

    At an all-ones mask this calls the unchanged model with its unchanged
    checkpoint parameters and is numerically identical to production forward.
    Zeroing a row masks that global node's embedding in both the GNN lookup and
    GraphCare's direct-EHR pooling path, matching the shared node-feature masking
    fidelity contract without inventing attention scores.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._graph: dict[str, torch.Tensor] | None = None
        self._reference_logits: torch.Tensor | None = None

    def set_context(self, graph: Mapping[str, Any]) -> torch.Tensor:
        required = {
            "node_ids", "rel_ids", "edge_index", "batch", "visit_node", "ehr_nodes"
        }
        missing = required - set(graph)
        if missing:
            raise ValueError(f"GraphCare explanation graph missing {sorted(missing)}.")
        node_ids = graph["node_ids"]
        if not torch.is_tensor(node_ids) or node_ids.ndim != 1 or node_ids.numel() < 1:
            raise ValueError("GraphCare node_ids must be a non-empty vector.")
        visit_node = graph["visit_node"]
        ehr_nodes = graph["ehr_nodes"]
        if visit_node.ndim == 1:
            visit_node = visit_node.unsqueeze(0).unsqueeze(0)
        elif visit_node.ndim == 2:
            visit_node = visit_node.unsqueeze(1)
        if ehr_nodes.ndim == 1:
            ehr_nodes = ehr_nodes.unsqueeze(0)
        self._graph = {
            "node_ids": node_ids,
            "rel_ids": graph["rel_ids"],
            "edge_index": graph["edge_index"],
            "batch": graph["batch"],
            "visit_node": visit_node,
            "ehr_nodes": ehr_nodes,
        }
        self.model.eval()
        with torch.no_grad():
            self._reference_logits = self.model(
                node_ids,
                self._graph["rel_ids"],
                self._graph["edge_index"],
                self._graph["batch"],
                visit_node,
                ehr_nodes,
            ).detach()
        return torch.ones(
            (node_ids.numel(), 1),
            dtype=self.model.node_emb.weight.dtype,
            device=node_ids.device,
        )

    def forward(self, x, edge_index, batch=None):
        if self._graph is None:
            raise RuntimeError("call set_context before GraphCare wrapper forward.")
        graph = self._graph
        if x.ndim != 2 or x.shape != (graph["node_ids"].numel(), 1):
            raise ValueError("GraphCare explanation x must be one scalar mask per local node.")
        if not torch.equal(edge_index, graph["edge_index"]):
            raise ValueError("GraphCare explanation cannot mutate the checkpoint topology.")
        resolved_batch = graph["batch"] if batch is None else batch
        if not torch.equal(resolved_batch, graph["batch"]):
            raise ValueError("GraphCare explanation batch vector differs from context.")

        global_mask = torch.ones(
            self.model.node_emb.weight.size(0),
            dtype=x.dtype,
            device=x.device,
        ).scatter(0, graph["node_ids"], x[:, 0])
        masked_weight = self.model.node_emb.weight * global_mask.unsqueeze(1)
        return functional_call(
            self.model,
            {"node_emb.weight": masked_weight},
            (
                graph["node_ids"],
                graph["rel_ids"],
                graph["edge_index"],
                resolved_batch,
                graph["visit_node"],
                graph["ehr_nodes"],
            ),
        )

    @torch.no_grad()
    def verify(self, atol: float = 1e-6) -> tuple[bool, float]:
        if self._graph is None or self._reference_logits is None:
            raise RuntimeError("call set_context before GraphCare wrapper verify.")
        x = torch.ones(
            (self._graph["node_ids"].numel(), 1),
            dtype=self.model.node_emb.weight.dtype,
            device=self._graph["node_ids"].device,
        )
        got = self.forward(x, self._graph["edge_index"], self._graph["batch"])
        difference = float((got - self._reference_logits).abs().max().item())
        return bool(torch.allclose(got, self._reference_logits, atol=atol)), difference


def explain_graphcare_record(model: nn.Module, graph: Mapping[str, Any], *, graphxai=False) -> dict[str, Any]:
    """Explain one real GraphCare record with deterministic input×gradient."""
    wrapper = GraphCareGraphXAIWrapper(model)
    x = wrapper.set_context(graph)
    faithful, difference = wrapper.verify()
    if not faithful:
        raise RuntimeError(
            f"GraphCare explanation wrapper is not faithful; max_abs_diff={difference:.3e}."
        )
    explanation = build_node_explanation(
        wrapper,
        x,
        graph["edge_index"],
        batch=graph["batch"],
    )
    record = {
        "schema": EXPLANATION_SCHEMA,
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "method": "graphcare",
        "subject_id": subject_id_of(graph),
        "topology": str(graph.get("topology", "")),
        "seed": int(graph.get("seed", -1)),
        "true_class_id": int(graph["y"].view(-1)[0].item()),
        "prediction_class_id": explanation["target"]["class_id"],
        "attribution_method": "deterministic_input_x_gradient",
        "node_explanation": explanation,
    }

    if graphxai:
        from shared.lib.graphxai_standardized import explain_algorithms
        record["schema_version"] = 2
        record["graphxai"] = explain_algorithms(wrapper, x, graph["edge_index"], batch=graph["batch"])
    return record


def select_standardized_records(records, requested_subject_ids):
    return select_exact_subject_records(records, requested_subject_ids)


def main(*, checkpoint, graph_structure, canonical_split, cohort, dataset, seed, out_dir, cohort_size=COHORT_SIZE):
    spec = BenchmarkSpec("graphcare", graph_structure, seed)
    cohort_contract = load_explanation_cohort(
        cohort, split_path=canonical_split, dataset_path=dataset, expected_count=cohort_size
    )
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GraphCare checkpoint missing: {checkpoint}")
    kg_path = structure_dir(cfg.data_dir, spec.structure, "graphcare") / "kg.pt"
    if not kg_path.is_file():
        raise FileNotFoundError(f"GraphCare standardized KG cache missing: {kg_path}")
    kg = torch.load(kg_path, map_location="cpu")
    validate_kg_schema(kg)
    validate_training_provenance(kg, canonical_split, dataset_path=dataset)
    dataset_object, _, _, test_indices, class_count, _ = build_standardized_dataset(
        kg,
        split_json=canonical_split,
        structure=spec.structure,
        dataset_path=dataset,
    )
    test_records = [dataset_object[index] for index in test_indices]
    selected = select_standardized_records(test_records, cohort_contract.subject_ids)

    saved = torch.load(checkpoint, map_location="cpu")
    if saved.get("structure") != spec.structure or saved.get("seed") != spec.seed:
        raise ValueError("GraphCare checkpoint topology/seed does not match request.")
    if saved.get("num_classes") != class_count:
        raise ValueError("GraphCare checkpoint class count does not match canonical split.")
    model = build_graphcare_model(kg, class_count, torch.device("cpu"))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    set_seed(spec.seed)

    records = []
    for item in selected:
        graph = _collate([item])
        graph["subject_id"] = subject_id_of(item)
        graph["topology"] = spec.structure
        graph["seed"] = spec.seed
        records.append(explain_graphcare_record(model, graph, graphxai=True))
    return write_standardized_explanations(
        output_dir=out_dir,
        records=records,
        method="graphcare",
        topology=spec.structure,
        seed=spec.seed,
        cohort_path=cohort,
        split_path=canonical_split,
        dataset_path=dataset,
        checkpoint_path=checkpoint,
        expected_count=cohort_size,
    )


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--graph-structure", "--graph_structure", dest="graph_structure", required=True)
    parser.add_argument("--canonical-split", "--canonical_split", dest="canonical_split", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--cohort-size", type=int, default=COHORT_SIZE, help="Expected test subjects (default: 500).")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", required=True, type=Path)
    args = parser.parse_args(argv)
    return main(**vars(args))


if __name__ == "__main__":
    cli()
