#!/usr/bin/env python3
"""Export the ProtGNN intra-patient PyG dataset for the browser visualizer."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from protgnn_analysis.config import data_args
from protgnn_analysis.load_dataset import get_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "visualizer" / "public" / "graphs"
TYPE_ALIASES = {
    "patient": "patient",
    "vital": "vital",
    "med": "medication",
    "icd": "icd",
    "symptom": "symptom",
    "chiefcomplaint": "chief_complaint",
}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def tensor_scalar(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if torch.is_tensor(value):
        return int(value.view(-1)[0].item())
    return int(value)


def clean_label(name: str) -> str:
    name = str(name)
    name = re.sub(r"^(med_|pyx_)", "", name)
    return name.replace("_", " ").strip() or name


def block(feature_cols: list[str], prefix: str) -> list[tuple[int, str]]:
    start = f"{prefix}["
    return [
        (idx, col[len(start) : -1])
        for idx, col in enumerate(feature_cols)
        if col.startswith(start) and col.endswith("]")
    ]


def active_name(row: np.ndarray, entries: list[tuple[int, str]]) -> tuple[int | None, str | None]:
    for idx, name in entries:
        if row[idx] > 0.5:
            return idx, name
    return None, None


def raw_vital_value(z: float, stats: dict[str, Any], name: str, missing: bool) -> float | None:
    if missing:
        return None
    meta = stats.get(name) or {}
    std = float(meta.get("std") or 0.0)
    mean = float(meta.get("mean") or 0.0)
    if std <= 0:
        return None
    return round(mean + z * std, 3)


def decode_node(
    local_idx: int,
    row: np.ndarray,
    graph_idx: int,
    graph_id: int,
    subject_id: int | None,
    feature_cols: list[str],
    blocks: dict[str, list[tuple[int, str]]],
    value_idx: int,
    abnormal_idx: int,
    missing_idx: int,
    vital_stats: dict[str, Any],
) -> dict[str, Any]:
    type_slots = blocks["type"]
    type_pos = int(np.argmax(row[: len(type_slots)]))
    raw_type = type_slots[type_pos][1]
    node_type = TYPE_ALIASES.get(raw_type, raw_type)

    if node_type == "patient":
        node = {
            "id": "patient",
            "label": "PATIENT",
            "type": "patient",
            "original_feature_name": "encounter_root",
            "graph_index": graph_idx,
        }
        if subject_id is not None:
            node["subject_id"] = subject_id
        return node

    prefix_by_type = {
        "vital": "vital",
        "medication": "med",
        "icd": "icd",
        "symptom": "sym",
        "chief_complaint": "cc",
    }
    block_key = prefix_by_type[node_type]
    _, original_name = active_name(row, blocks[block_key])
    original_name = original_name or f"{node_type}_{local_idx}"
    label = clean_label(original_name)
    value = float(row[value_idx]) if value_idx < len(row) else None
    abnormal = bool(row[abnormal_idx] > 0.5) if abnormal_idx < len(row) else False
    missing = bool(row[missing_idx] > 0.5) if missing_idx < len(row) else False

    node_id = f"{node_type}_{local_idx}_{re.sub(r'[^a-zA-Z0-9]+', '_', original_name).strip('_')}"
    node = {
        "id": node_id,
        "label": label,
        "type": node_type,
        "original_feature_name": original_name,
        "graph_index": graph_idx,
    }
    if node_type == "vital":
        node["value"] = round(value, 3)
        node["unit"] = "z-score"
        node["abnormal"] = abnormal
        node["missing"] = missing
        raw_value = raw_vital_value(value, vital_stats, original_name, missing)
        if raw_value is not None:
            node["raw_value"] = raw_value
    elif node_type in {"medication", "icd", "symptom", "chief_complaint"}:
        node["value"] = 1
    return node


def edge_type(nodes: list[dict[str, Any]], a: int, b: int) -> str:
    ta = nodes[a]["type"]
    tb = nodes[b]["type"]
    if ta == "medication" and tb == "medication":
        return "pmi_bundle"
    return "patient_feature"


def graph_to_json(ds: Any, idx: int, feature_cols: list[str], blocks: dict[str, Any], label_mapping: dict[str, str]) -> dict[str, Any]:
    graph = ds[idx]
    x = graph.x.detach().cpu().numpy()
    graph_id = tensor_scalar(getattr(graph, "dataset_index", None), idx)
    subject_id = tensor_scalar(getattr(graph, "subject_id", None), None)
    y = tensor_scalar(graph.y, 0)
    diagnosis = label_mapping.get(str(y), f"class_{y}")
    value_idx = feature_cols.index("value")
    abnormal_idx = feature_cols.index("abnormal_flag")
    missing_idx = feature_cols.index("missing_flag") if "missing_flag" in feature_cols else abnormal_idx + 1
    vital_stats = getattr(ds, "feature_metadata", {}) or {}

    nodes = [
        decode_node(
            local_idx=n,
            row=x[n],
            graph_idx=idx,
            graph_id=graph_id,
            subject_id=subject_id,
            feature_cols=feature_cols,
            blocks=blocks,
            value_idx=value_idx,
            abnormal_idx=abnormal_idx,
            missing_idx=missing_idx,
            vital_stats=vital_stats,
        )
        for n in range(x.shape[0])
    ]

    edges = []
    seen = set()
    edge_index = graph.edge_index.detach().cpu().numpy()
    for a, b in zip(edge_index[0], edge_index[1]):
        a = int(a)
        b = int(b)
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "source": nodes[a]["id"],
            "target": nodes[b]["id"],
            "type": edge_type(nodes, a, b),
        })

    return {
        "graph_id": graph_id,
        "dataset_index": idx,
        "subject_id": subject_id,
        "diagnosis": diagnosis,
        "nodes": nodes,
        "edges": edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="mimic_intra_patient_disease")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    ds = get_dataset(data_args.dataset_dir, args.dataset_name)
    total = len(ds) if args.limit is None else min(args.limit, len(ds))
    feature_cols = ds.feature_cols
    blocks = {
        "type": block(feature_cols, "type"),
        "vital": block(feature_cols, "vital_id"),
        "med": block(feature_cols, "med_id"),
        "icd": block(feature_cols, "icd_id"),
        "sym": block(feature_cols, "sym_id"),
        "cc": block(feature_cols, "cc_id"),
    }
    label_mapping = getattr(ds, "label_mapping", {}) or {}

    shard_dir = args.out_dir / "dataset"
    if shard_dir.exists() and not args.keep_existing:
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    shard_count = math.ceil(total / args.shard_size)
    for shard_idx in range(shard_count):
        start = shard_idx * args.shard_size
        end = min(start + args.shard_size, total)
        graphs = []
        for idx in range(start, end):
            graph_json = graph_to_json(ds, idx, feature_cols, blocks, label_mapping)
            offset = len(graphs)
            graphs.append(graph_json)
            subject = graph_json.get("subject_id")
            graph_id = graph_json["graph_id"]
            diagnosis = graph_json["diagnosis"]
            subject_label = f"Patient {subject}" if subject is not None else f"Graph {graph_id}"
            manifest.append({
                "id": f"graph-{graph_id}",
                "label": f"{subject_label} · Graph {graph_id} · {diagnosis}",
                "file": f"dataset/graphs_{shard_idx:04d}.json::{offset}",
                "graph_id": graph_id,
                "subject_id": subject,
                "diagnosis": diagnosis,
                "node_count": len(graph_json["nodes"]),
                "edge_count": len(graph_json["edges"]),
            })
        shard_path = shard_dir / f"graphs_{shard_idx:04d}.json"
        with shard_path.open("w") as f:
            json.dump({"graphs": graphs}, f, separators=(",", ":"))
        print(f"wrote {display_path(shard_path)} ({start}-{end - 1})")

    with (args.out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, separators=(",", ":"))
    with (args.out_dir / "dataset_summary.json").open("w") as f:
        json.dump({
            "dataset_name": args.dataset_name,
            "graph_count": total,
            "shard_size": args.shard_size,
            "shard_count": shard_count,
            "label_count": len(label_mapping),
        }, f, indent=2)
    print(f"exported {total} graphs to {display_path(args.out_dir)}")


if __name__ == "__main__":
    main()
