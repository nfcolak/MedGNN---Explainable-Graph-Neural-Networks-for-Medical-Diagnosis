"""Fixed-cohort deterministic explanations for a standardized ProtGNN checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protgnn_analysis.config import STANDARDIZED_DATASET_NAME, data_args, model_args
from protgnn_analysis.explainability.graphxai_wrapper import ProtGNNWrapper
from protgnn_analysis.load_dataset import get_dataloader, get_dataset
from protgnn_analysis.models import GnnNets
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


def select_standardized_records(records, requested_subject_ids):
    """Select exact cohort records by subject ID, never loader position."""
    return select_exact_subject_records(records, requested_subject_ids)


def explain_protgnn_record(model, graph, *, topology, seed, graphxai=False):
    model.eval()
    wrapper = ProtGNNWrapper(model).eval()
    x = graph.x.detach().to(torch.device("cpu"))
    edge_index = graph.edge_index.detach().to(torch.device("cpu"))
    batch = torch.zeros(x.size(0), dtype=torch.long)
    explanation = build_node_explanation(
        wrapper, x, edge_index, batch=batch
    )
    record = {
        "schema": EXPLANATION_SCHEMA,
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "method": "protgnn",
        "subject_id": subject_id_of(graph),
        "topology": topology,
        "seed": seed,
        "true_class_id": int(graph.y.view(-1)[0].item()),
        "prediction_class_id": explanation["target"]["class_id"],
        "attribution_method": "deterministic_input_x_gradient",
        "node_explanation": explanation,
    }

    if graphxai:
        from shared.lib.graphxai_standardized import explain_algorithms
        record["schema_version"] = 2
        record["graphxai"] = explain_algorithms(wrapper, x, edge_index, batch=batch)
    return record


def checkpoint_model_args(config):
    import copy
    args = copy.deepcopy(model_args)
    mapping = {'model':'model_name', 'latent_dim':'latent_dim', 'mlp_hidden':'mlp_hidden',
               'readout':'readout','dropout':'dropout','adj_normalize':'adj_normlize',
               'emb_normalize':'emb_normlize','enable_prototypes':'enable_prot',
               'num_prototypes_per_class':'num_prototypes_per_class'}
    for source, target in mapping.items():
        setattr(args, target, config[source])
    args.device = 'cpu'
    return args


def main(*, checkpoint, graph_structure, canonical_split, cohort, dataset, seed, out_dir, cohort_size=COHORT_SIZE):
    spec = BenchmarkSpec("protgnn", graph_structure, seed)
    cohort_contract = load_explanation_cohort(
        cohort, split_path=canonical_split, dataset_path=dataset, expected_count=cohort_size
    )
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ProtGNN checkpoint missing: {checkpoint}")
    set_seed(spec.seed)
    dataset_object = get_dataset(
        data_args.dataset_dir,
        STANDARDIZED_DATASET_NAME,
        task=data_args.task,
        graph_structure=spec.structure,
        canonical_split=canonical_split,
    )
    loaders = get_dataloader(
        dataset_object,
        batch_size=1,
        random_split_flag=data_args.random_split,
        data_split_ratio=data_args.data_split_ratio,
        seed=spec.seed,
        canonical_split=canonical_split,
    )
    selected = select_standardized_records(
        loaders["test"].dataset, cohort_contract.subject_ids
    )
    output_dim = len(cohort_contract.classes)
    import json
    config_path = checkpoint.parents[2] / 'model_config.json'
    reconstructed = checkpoint_model_args(json.loads(config_path.read_text()))
    model = GnnNets(dataset_object.num_node_features, output_dim, reconstructed)
    saved = torch.load(checkpoint, map_location="cpu")
    if "net" not in saved:
        raise ValueError("ProtGNN standardized checkpoint must contain 'net'.")
    model.load_state_dict(saved["net"], strict=True)
    model.to(torch.device("cpu")).eval()
    records = [
        explain_protgnn_record(model, graph, topology=spec.structure, seed=spec.seed, graphxai=True)
        for graph in selected
    ]
    return write_standardized_explanations(
        output_dir=out_dir,
        records=records,
        method="protgnn",
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
