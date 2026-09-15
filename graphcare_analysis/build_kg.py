"""Global knowledge graph over the code vocabulary for GraphCare.  (Phase 2.)

Model-direct Path B: produces the global KG that the adapter expands into
per-patient subgraphs, plus the sizes the GraphCare model is built with
(num_nodes, num_rels).

KG (ontology + training-fold PMI, no LLM):
    nodes      : training-fitted non-leaky code vocab plus reserved patient:hub,
    relation 0 : "co_occurs"  — code pairs with training-fold PMI > threshold,
    relation 1 : "same_class" — meds sharing a therapeutic class (MED_CLASS),
    relation 2 : "patient_has_concept" — dynamic bidirectional patient spokes.

Output (torch.save to cfg.kg_path, plain types so it loads in any torch):
    ent2id     : entity string ("med:<g>" / "sym:<v>" / "cc:<v>" /
                 "patient:hub") -> node id
    rel2id     : relation names -> deterministic IDs
    neighbours : node id -> list of (relation_id, neighbour_node_id)
                 (concept relations only; patient spokes are record-specific)
    num_nodes, num_rels, preprocessing provenance

Run (either env has pandas/numpy/torch):
    PYTHONPATH=. .venv-graphcare/bin/python3 graphcare_analysis/build_kg.py
"""
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graphcare_analysis.config import cfg
from shared.data_prep.med_classes import MED_CLASS
from shared.lib.benchmark_contract import file_sha256, load_canonical_split
from shared.lib.canonical_graph import STANDARDIZED_GRAPH_MEMBERSHIP_CONTRACT


PREPROCESSING_REL2ID = {
    "co_occurs": 0,
    "same_class": 1,
    "patient_has_concept": 2,
}
PREPROCESSING_SCHEMA_VERSION = 4
PREPROCESSING_RECIPE_VERSION = "graphcare-kg-v4"
MED_CLASS_MAPPING_VERSION = "med-classes-v1"


def _json_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _preprocessing_recipe():
    return {
        "version": PREPROCESSING_RECIPE_VERSION,
        "symptom_input_policy": "exclude_diagnosis_derived",
        "graph_membership_contract": STANDARDIZED_GRAPH_MEMBERSHIP_CONTRACT,
        "med_min_prev": cfg.med_min_prev,
        "pmi_threshold": cfg.pmi_threshold,
        "med_class_mapping_version": MED_CLASS_MAPPING_VERSION,
        "med_class_mapping_sha256": _json_sha256(MED_CLASS),
    }


def _training_identity_sha256(df, fit_indices):
    subjects = df["subject_id"].values
    identity = [
        [int(row_index), str(int(subjects[int(row_index)]))]
        for row_index in fit_indices
    ]
    return _json_sha256(identity)


def typed_adjacency_sha256(kg):
    """Fingerprint the complete sorted static typed adjacency multiset."""
    edges = sorted(
        [int(source), int(relation_id), int(target)]
        for source, adjacency in kg["neighbours"].items()
        for relation_id, target in adjacency
    )
    return _json_sha256(edges)


def _canonical_type(entity_name):
    if entity_name == "patient:hub":
        return "patient"
    if entity_name.startswith("med:"):
        return "medication"
    if entity_name.startswith("cc:"):
        return "chiefcomplaint"
    if entity_name.startswith("sym:"):
        return "symptom"
    raise ValueError(f"GraphCare entity has no canonical type: {entity_name!r}.")


def _has_valid_canonical_metadata(kg):
    names = kg.get("canonical_node_ids_by_global_id")
    types = kg.get("canonical_node_types_by_global_id")
    relations = kg.get("canonical_edge_types_by_relation_id")
    expected_names = [
        name for name, _ in sorted(kg["ent2id"].items(), key=lambda item: item[1])
    ]
    return (
        names == expected_names
        and types == [_canonical_type(name) for name in expected_names]
        and relations == ["cooccur", "ontology", "patient_concept"]
    )


def _kg_schema_error(detail):
    raise ValueError(f"GraphCare KG schema is invalid: {detail}")


def validate_kg_schema(kg):
    """Validate all static entity, relation, and typed-adjacency domains."""
    if not isinstance(kg, dict):
        _kg_schema_error("cache payload must be a dictionary.")
    num_nodes = kg.get("num_nodes")
    num_rels = kg.get("num_rels")
    if type(num_nodes) is not int or num_nodes < 1:
        _kg_schema_error("num_nodes must be a positive exact integer.")
    if type(num_rels) is not int or num_rels != len(PREPROCESSING_REL2ID):
        _kg_schema_error("num_rels must exactly match the relation mapping.")

    rel2id = kg.get("rel2id")
    if (
        not isinstance(rel2id, dict)
        or set(rel2id) != set(PREPROCESSING_REL2ID)
        or any(type(relation_id) is not int for relation_id in rel2id.values())
        or rel2id != PREPROCESSING_REL2ID
    ):
        _kg_schema_error(f"rel2id must equal {PREPROCESSING_REL2ID!r}.")
    if sorted(rel2id.values()) != list(range(num_rels)):
        _kg_schema_error("relation IDs must be contiguous and in range.")

    ent2id = kg.get("ent2id")
    if (
        not isinstance(ent2id, dict)
        or any(type(name) is not str or type(node_id) is not int
               for name, node_id in ent2id.items())
        or sorted(ent2id.values()) != list(range(num_nodes))
    ):
        _kg_schema_error("entity IDs must be unique contiguous exact integers.")
    hub_id = kg.get("patient_hub_id")
    patient_relation_id = kg.get("patient_relation_id")
    if (
        type(hub_id) is not int
        or hub_id != num_nodes - 1
        or ent2id.get("patient:hub") != hub_id
    ):
        _kg_schema_error("patient:hub must be the final entity ID.")
    if (
        type(patient_relation_id) is not int
        or patient_relation_id != rel2id["patient_has_concept"]
        or patient_relation_id != num_rels - 1
    ):
        _kg_schema_error("patient_has_concept must be the final relation ID.")

    neighbours = kg.get("neighbours")
    if (
        not isinstance(neighbours, dict)
        or any(type(node_id) is not int for node_id in neighbours)
        or set(neighbours) != set(range(num_nodes))
    ):
        _kg_schema_error("neighbours must contain every and only valid node ID.")
    if neighbours[hub_id] != []:
        _kg_schema_error("reserved patient hub adjacency must be empty.")
    for source, edges in neighbours.items():
        if not isinstance(edges, list):
            _kg_schema_error(f"adjacency for node {source} must be a list.")
        for edge in edges:
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                _kg_schema_error(f"adjacency entry at node {source} must be a pair.")
            relation_id, target = edge
            if type(relation_id) is not int or not 0 <= relation_id < num_rels:
                _kg_schema_error(f"relation ID at node {source} is out of range.")
            if type(target) is not int or not 0 <= target < num_nodes:
                _kg_schema_error(f"neighbour node ID at node {source} is out of range.")
            if relation_id == patient_relation_id or target == hub_id:
                _kg_schema_error("reserved patient relation/hub cannot appear in static adjacency.")
    return rel2id


def _validated_fit_indices(fit_indices, row_count):
    """Return exact, unique, in-bounds one-dimensional row indices."""
    if fit_indices is None:
        values = list(range(row_count))
    else:
        try:
            values = list(fit_indices)
        except TypeError as exc:
            raise ValueError("fit_indices must be a one-dimensional sequence of integers.") from exc
    if not values:
        raise ValueError("fit_indices must contain at least one training row.")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in values
    ):
        raise ValueError("fit_indices must contain exact integers, without coercion.")
    indices = np.asarray(values, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("fit_indices must be one-dimensional.")
    if len(set(indices.tolist())) != indices.size:
        raise ValueError("fit_indices must contain unique row indices.")
    if np.any(indices < 0) or np.any(indices >= row_count):
        raise ValueError(f"fit_indices must satisfy 0 <= index < {row_count}.")
    return indices


def canonical_row_indices(df, split):
    """Map every canonical subject to exactly one dataset row, fail closed."""
    if "subject_id" not in df.columns:
        raise ValueError("Canonical GraphCare subject/row identity requires subject_id.")
    fold = split["fold"]
    subject_keys = []
    for row_index, subject_id in enumerate(df["subject_id"].values):
        if isinstance(subject_id, (bool, np.bool_)) or not isinstance(
            subject_id, Integral
        ):
            raise ValueError(
                f"Canonical GraphCare subject/row identity has a non-integer "
                f"subject_id at row {row_index}."
            )
        subject_keys.append(str(int(subject_id)))
    counts = {}
    for subject_key in subject_keys:
        if subject_key in fold:
            counts[subject_key] = counts.get(subject_key, 0) + 1
    missing = sorted(set(fold) - set(counts))
    duplicate = sorted(key for key, count in counts.items() if count != 1)
    if missing or duplicate:
        raise ValueError(
            "Canonical GraphCare subject/row identity mismatch: "
            f"missing_subjects={missing[:10]}, duplicate_subjects={duplicate[:10]}."
        )
    keep_indices = [
        row_index
        for row_index, subject_key in enumerate(subject_keys)
        if subject_key in fold
    ]
    fit_indices = _validated_fit_indices(
        [row for row in keep_indices if fold[subject_keys[row]] == 0], len(df)
    )
    return keep_indices, fit_indices, subject_keys


def _multihot_vocab(df, cols, fit_indices=None):
    """Fit cell-value vocabulary on selected rows, then transform every row."""
    counter = {}
    if not cols:
        return [], np.zeros((len(df), 0), dtype=bool)
    rows = df[cols].fillna("").astype(str).values.tolist()
    fit_rows = range(len(rows)) if fit_indices is None else fit_indices
    for i in fit_rows:
        for s in rows[int(i)]:
            s = s.strip()
            if s:
                counter[s] = counter.get(s, 0) + 1
    vocab = sorted(counter)
    index = {v: j for j, v in enumerate(vocab)}
    present = np.zeros((len(df), len(vocab)), dtype=bool)
    for i, row in enumerate(rows):
        for s in row:
            s = s.strip()
            if s in index:
                present[i, index[s]] = True
    return vocab, present


def vocab_and_presence(df, fit_indices=None, include_symptoms=True):
    """Fit a concept vocabulary on selected rows and transform the full frame.

    Returns: ent2id, ent_names, M (N x concept_count bool presence), med_cols.
    ``fit_indices`` must identify the training rows for standardized runs.
    """
    fit_indices = _validated_fit_indices(fit_indices, len(df))

    med_all = sorted(c for c in df.columns if c.startswith(("med_", "pyx_")))
    med_bin = (df[med_all].apply(pd.to_numeric, errors="coerce").fillna(0).values > 0)
    keep = med_bin[fit_indices].mean(0) >= cfg.med_min_prev
    med_cols = [c for c, k in zip(med_all, keep) if k]
    med_present = med_bin[:, keep]

    sym_cols = (
        sorted(c for c in df.columns if c.startswith("symptom_"))
        if include_symptoms else []
    )
    cc_cols = sorted(c for c in df.columns if c.startswith("chiefcomplaint_"))
    sym_vocab, sym_present = _multihot_vocab(df, sym_cols, fit_indices)
    cc_vocab, cc_present = _multihot_vocab(df, cc_cols, fit_indices)

    ent_names = (
        [f"med:{c[4:]}" for c in med_cols]
        + [f"sym:{v}" for v in sym_vocab]
        + [f"cc:{v}" for v in cc_vocab]
    )
    ent2id = {name: i for i, name in enumerate(ent_names)}
    M = np.concatenate([med_present, sym_present, cc_present], axis=1).astype(bool)
    return ent2id, ent_names, M, med_cols


def validate_training_provenance(kg, split_json, dataset_path=None):
    """Reject a standardized KG unless all fitting and topology identities match."""
    split = load_canonical_split(split_json)
    validate_kg_schema(kg)
    source_path = Path(dataset_path or (cfg.data_dir / cfg.csv_filename))
    if not source_path.is_file():
        raise ValueError(
            f"GraphCare standardized dataset is unavailable for provenance: {source_path}."
        )
    df = pd.read_csv(source_path, low_memory=False)
    if "subject_id" not in df.columns:
        raise ValueError("GraphCare standardized provenance requires subject_id.")
    _, fit_indices, _ = canonical_row_indices(df, split)
    provenance = kg.get("preprocessing")
    valid = (
        isinstance(provenance, dict)
        and provenance.get("schema_version") == PREPROCESSING_SCHEMA_VERSION
        and provenance.get("fit_scope") == "training_fold"
        and provenance.get("fit_fold") == 0
        and type(provenance.get("fit_row_count")) is int
        and provenance["fit_row_count"] == int(fit_indices.size)
        and provenance.get("split_sha256") == file_sha256(split_json)
        and provenance.get("dataset_sha256") == file_sha256(source_path)
        and provenance.get("recipe") == _preprocessing_recipe()
        and provenance.get("fit_identity_sha256")
        == _training_identity_sha256(df, fit_indices)
        and provenance.get("typed_adjacency_sha256")
        == typed_adjacency_sha256(kg)
        and _has_valid_canonical_metadata(kg)
    )
    if not valid:
        raise ValueError(
            "GraphCare standardized preprocessing requires matching training-fold "
            "provenance for the exact dataset, recipe, training rows, and typed "
            "adjacency; regenerate this KG with the canonical split."
        )
    return provenance


def build_global_kg(
    save=True, save_path=None, df=None, split_json=None, dataset_path=None
):
    source_path = Path(dataset_path or (cfg.data_dir / cfg.csv_filename))
    if df is None:
        df = pd.read_csv(source_path, low_memory=False)
    elif split_json is not None:
        if not source_path.is_file():
            raise ValueError(
                "Split-aware GraphCare preprocessing requires the exact source dataset file."
            )
        source_df = pd.read_csv(source_path, low_memory=False)
        if not df.equals(source_df):
            raise ValueError(
                "Split-aware GraphCare dataframe does not match the exact source dataset file."
            )

    if split_json is None:
        fit_indices = np.arange(len(df), dtype=np.int64)
        preprocessing = {
            "schema_version": 1,
            "fit_scope": "all_rows_legacy",
            "fit_fold": None,
            "fit_row_count": len(df),
            "split_sha256": None,
        }
    else:
        if "subject_id" not in df.columns:
            raise ValueError("Split-aware GraphCare preprocessing requires subject_id.")
        if not source_path.is_file():
            raise ValueError(
                "Split-aware GraphCare preprocessing requires the exact source dataset file."
            )
        split = load_canonical_split(split_json)
        _, fit_indices, _ = canonical_row_indices(df, split)
        preprocessing = {
            "schema_version": PREPROCESSING_SCHEMA_VERSION,
            "fit_scope": "training_fold",
            "fit_fold": 0,
            "fit_row_count": int(fit_indices.size),
            "split_sha256": file_sha256(split_json),
            "dataset_sha256": file_sha256(source_path),
            "recipe": _preprocessing_recipe(),
            "fit_identity_sha256": _training_identity_sha256(df, fit_indices),
        }

    n = int(fit_indices.size)
    ent2id, ent_names, M, med_cols = vocab_and_presence(
        df,
        fit_indices=fit_indices,
        include_symptoms=split_json is None,
    )
    concept_count = len(ent_names)
    n_med = len(med_cols)

    Mf = M[fit_indices].astype(np.float64)

    # --- relation 0: co_occurs (PMI) ---
    eps = 1e-9
    p = Mf.mean(0)
    co = (Mf.T @ Mf) / n
    pmi = np.log((co + eps) / (np.outer(p, p) + eps))
    np.fill_diagonal(pmi, 0.0)
    co_pairs = [(int(a), int(b)) for a, b in np.argwhere(pmi > cfg.pmi_threshold) if a < b]

    # --- relation 1: same_class (meds sharing a therapeutic class) ---
    by_class = {}
    for j, c in enumerate(med_cols):                       # med node id == j
        gen = c.split("_", 1)[1] if "_" in c else c
        cls = MED_CLASS.get(gen) or MED_CLASS.get(gen.lower())
        if cls:
            by_class.setdefault(cls, []).append(j)
    same_pairs = []
    for members in by_class.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                same_pairs.append((members[x], members[y]))

    patient_hub_id = concept_count
    ent_names = [*ent_names, "patient:hub"]
    ent2id = {name: i for i, name in enumerate(ent_names)}
    num_nodes = len(ent_names)
    rel2id = dict(PREPROCESSING_REL2ID)
    neighbours = {i: [] for i in range(num_nodes)}
    for a, b in co_pairs:
        neighbours[a].append((0, b)); neighbours[b].append((0, a))
    for a, b in same_pairs:
        neighbours[a].append((1, b)); neighbours[b].append((1, a))

    kg = {"ent2id": ent2id, "rel2id": rel2id, "neighbours": neighbours,
          "num_nodes": num_nodes, "num_rels": len(rel2id), "n_med": n_med,
          "patient_hub_id": patient_hub_id,
          "patient_relation_id": rel2id["patient_has_concept"],
          "canonical_node_ids_by_global_id": ent_names,
          "canonical_node_types_by_global_id": [
              _canonical_type(name) for name in ent_names
          ],
          "canonical_edge_types_by_relation_id": [
              "cooccur", "ontology", "patient_concept"
          ],
          "preprocessing": preprocessing}
    validate_kg_schema(kg)
    if split_json is not None:
        preprocessing["typed_adjacency_sha256"] = typed_adjacency_sha256(kg)

    degs = [len(v) for v in neighbours.values()]
    print(f"  KG built: nodes={num_nodes} (med={n_med}, sym={M.shape[1]-n_med-len([e for e in ent_names if e.startswith('cc:')])}, cc={len([e for e in ent_names if e.startswith('cc:')])})")
    print(f"  edges: co_occurs={len(co_pairs)}, same_class={len(same_pairs)} | rels={len(rel2id)}")
    print(f"  degree mean={np.mean(degs):.1f} max={max(degs)} | isolated nodes={sum(d == 0 for d in degs)}")

    if save:
        path = save_path or cfg.kg_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(kg, path)
        print(f"  saved -> {path}")
    return kg


if __name__ == "__main__":
    build_global_kg()
