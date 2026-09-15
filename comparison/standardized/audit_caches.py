"""Fail-closed audit of real standardized graph-cache records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from graphcare_analysis.adapter import build_standardized_dataset
from graphcare_analysis.build_kg import validate_training_provenance
from protgnn_analysis.load_dataset import (
    IntraPatientHeteroDataset,
    standardized_cache_paths,
)
from shared.lib.benchmark_contract import load_canonical_split
from shared.lib.canonical_graph import (
    canonical_graph_from_graphcare_record,
    canonical_graph_from_pyg_record,
    validate_cross_method_fingerprints,
)
from shared.lib.graph_structures import structure_dir


PROJECT_ROOT = _PROJECT_ROOT


def _torch_load(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:  # torch 1.12 GraphCare environment
        return torch.load(path)


def _require_cache_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(
            f"Missing production {label} cache: {path}. Regenerate standardized caches first."
        )


def audit_standardized_caches(
    *,
    project_root: Path = PROJECT_ROOT,
    dataset_path: Optional[Path] = None,
    split_path: Optional[Path] = None,
    structures: Sequence[str] = ("star", "cooccur"),
    protgnn_parameters: Optional[Mapping[str, float]] = None,
):
    """Audit every subject in real PyG and GraphCare production-shaped records."""
    root = Path(project_root).resolve()
    source = Path(dataset_path or (root / "data" / "merged_ed.csv")).resolve()
    split_file = Path(
        split_path or (root / "comparison" / "canonical_split.json")
    ).resolve()
    split = load_canonical_split(split_file)
    if source != (root / "data" / "merged_ed.csv").resolve():
        # Tests may supply a temporary root but the relative production contract remains exact.
        if source.parent != (root / "data").resolve() or source.name != "merged_ed.csv":
            raise ValueError("Cache audit dataset must be <project_root>/data/merged_ed.csv.")
    parameters = dict(protgnn_parameters or {})
    med_min_prev = float(parameters.get("med_min_prev", 0.01))
    pmi_threshold = float(parameters.get("pmi_threshold", 2.0))
    unknown_parameters = set(parameters) - {"med_min_prev", "pmi_threshold"}
    if unknown_parameters:
        raise ValueError("Unknown ProtGNN cache audit parameters.")

    requested = tuple(structures)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Cache audit structures must be nonempty and unique.")
    if any(structure not in {"star", "cooccur"} for structure in requested):
        raise ValueError("Cache audit supports only primary star and cooccur structures.")

    report = {}
    expected_subjects = set(split["fold"])
    for structure in requested:
        pyg_cache, pyg_metadata = standardized_cache_paths(
            source.parent,
            split_file,
            structure,
            med_min_prev=med_min_prev,
            pmi_threshold=pmi_threshold,
        )
        kg_path = structure_dir(source.parent, structure, "graphcare") / "kg.pt"
        _require_cache_file(pyg_cache, f"ProtGNN/GSAT {structure}")
        _require_cache_file(pyg_metadata, f"ProtGNN/GSAT {structure} metadata")
        _require_cache_file(kg_path, f"GraphCare {structure}")

        pyg_dataset = IntraPatientHeteroDataset(
            root=source.parent,
            name="mimic_intra_patient_disease",
            csv_filename=source.name,
            med_min_prev=med_min_prev,
            pmi_threshold=pmi_threshold,
            icd_min_prev=0.0,
            target="disease",
            graph_structure=structure,
            canonical_split=split_file,
        )
        if Path(pyg_dataset.processed_paths[0]).resolve() != pyg_cache.resolve():
            raise ValueError("Loaded PyG cache path does not match the audited recipe path.")
        kg = _torch_load(kg_path)
        validate_training_provenance(kg, split_file, dataset_path=source)
        graphcare_dataset, *_ = build_standardized_dataset(
            kg,
            split_json=split_file,
            structure=structure,
            dataset_path=source,
        )

        if len(pyg_dataset) != len(expected_subjects) or len(graphcare_dataset) != len(
            expected_subjects
        ):
            raise ValueError(
                "Production cache subject membership does not match the canonical split: "
                f"expected={len(expected_subjects)}, pyg={len(pyg_dataset)}, "
                f"graphcare={len(graphcare_dataset)}."
            )

        subject_fingerprints = []
        seen_subjects = set()
        for index in range(len(pyg_dataset)):
            pyg_graph = canonical_graph_from_pyg_record(pyg_dataset[index])
            graphcare_graph = canonical_graph_from_graphcare_record(
                graphcare_dataset[index], kg
            )
            if pyg_graph.subject_id != graphcare_graph.subject_id:
                raise ValueError(
                    "Production cache subject ordering differs between PyG and GraphCare."
                )
            subject_id = pyg_graph.subject_id
            if subject_id in seen_subjects:
                raise ValueError(
                    f"Production cache contains duplicate canonical subject {subject_id!r}."
                )
            seen_subjects.add(subject_id)
            fingerprint = validate_cross_method_fingerprints(
                {
                    "protgnn": pyg_graph,
                    "gsat": pyg_graph,
                    "graphcare": graphcare_graph,
                }
            )
            subject_fingerprints.append([subject_id, fingerprint])
        if seen_subjects != expected_subjects:
            raise ValueError(
                "Production cache subject identities do not match the canonical split."
            )
        subject_fingerprints.sort()
        parity_payload = json.dumps(
            subject_fingerprints,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        report[structure] = {
            "subject_count": len(subject_fingerprints),
            "parity_sha256": hashlib.sha256(parity_payload).hexdigest(),
            "protgnn_cache": str(pyg_cache.relative_to(root)),
            "graphcare_cache": str(kg_path.relative_to(root)),
        }
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structures", nargs="+", choices=("star", "cooccur"),
        default=["star", "cooccur"],
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/merged_ed.csv")
    )
    parser.add_argument(
        "--canonical-split", type=Path,
        default=Path("comparison/canonical_split.json"),
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    report = audit_standardized_caches(
        project_root=PROJECT_ROOT,
        dataset_path=PROJECT_ROOT / args.dataset,
        split_path=PROJECT_ROOT / args.canonical_split,
        structures=args.structures,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
