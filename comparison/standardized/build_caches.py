"""Build missing split-aware caches; default is a read-only plan, not execution.

Existing GraphCare caches must validate against the current source and recipe.
Stale/partial artifacts are never replaced: preserve them elsewhere explicitly
before retrying. This command performs preprocessing, never training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--canonical-split", type=Path, default=ROOT / "comparison/canonical_split.json")
    parser.add_argument("--structures", nargs="+", choices=("star", "cooccur"), default=["star", "cooccur"])
    args = parser.parse_args(argv)
    if len(set(args.structures)) != len(args.structures):
        parser.error("structures must be unique")
    print(json.dumps({"structures": args.structures, "dataset_dir": str(args.dataset_dir),
                      "canonical_split": str(args.canonical_split), "execute": args.execute}))
    if not args.execute:
        return 0

    import torch
    from graphcare_analysis.build_kg import build_global_kg, validate_training_provenance
    from protgnn_analysis.load_dataset import get_dataset, standardized_cache_paths
    from shared.lib.graph_structures import structure_dir

    source = args.dataset_dir / "merged_ed.csv"
    kg_paths = [structure_dir(args.dataset_dir, t, "graphcare") / "kg.pt" for t in args.structures]
    # Check all occupied destinations before starting any expensive work.
    for path in kg_paths:
        if path.exists() or path.is_symlink():
            try:
                if path.is_symlink():
                    raise ValueError("cache is a symlink")
                try:
                    kg = torch.load(path, weights_only=False)
                except TypeError:
                    kg = torch.load(path)
                validate_training_provenance(kg, args.canonical_split, dataset_path=source)
            except Exception as exc:
                raise ValueError(f"Preserve stale/invalid cache {path} elsewhere before rebuilding: {exc}") from exc
    for topology in args.structures:
        cache, metadata = standardized_cache_paths(args.dataset_dir, args.canonical_split, topology)
        if cache.exists() != metadata.exists():
            raise ValueError(f"Preserve partial cache directory {cache.parent} elsewhere before rebuilding.")
    for topology in args.structures:
        get_dataset(args.dataset_dir, "mimic_intra_patient_disease",
                    graph_structure=topology, canonical_split=args.canonical_split)
    missing = [path for path in kg_paths if not path.exists()]
    if missing:
        kg = build_global_kg(save=False, split_json=args.canonical_split, dataset_path=source)
        for path in missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation preserves another invocation's output.
            with path.open("xb") as handle:
                torch.save(kg, handle)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
