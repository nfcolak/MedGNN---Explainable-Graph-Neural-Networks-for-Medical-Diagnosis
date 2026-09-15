"""Adapter: merged_ed.csv  ->  GraphCare model-input tensors.  (Phase 1.)

Model-direct Path B (validated in test_model_smoke.py). Per patient (our ED
cohort is SINGLE-VISIT, so max_visit = 1) we build a personalized subgraph with
an explicit patient hub, the patient's codes, and (only for full_kg_expanded)
their record-external 1-hop KG neighbours. We emit the tensors the upstream
GraphCare model consumes:

    node_ids   [N]      global node id of each node in the subgraph
    edge_index [2, E]   LOCAL node indices of subgraph edges
    rel_ids    [E]      relation id of each edge
    visit_node [V]      membership of the (single) visit over the V global nodes
    ehr_nodes  [V]      indicator of the patient's OWN EHR code nodes
    y          scalar   disease_1 class (0..29)

Batching (collate): node_ids/edge_index/batch PyG-style; visit_node/ehr_nodes
stack along the graph dim. Split: shared.lib.splits (subject-aware, same seed),
class ids identical to protgnn (sorted unique disease_1).
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from shared.lib.splits import stratified_split_indices, subject_aware_split_indices
from graphcare_analysis.config import cfg
from graphcare_analysis.build_kg import (
    PREPROCESSING_REL2ID,
    canonical_row_indices,
    validate_kg_schema,
    validate_training_provenance,
    vocab_and_presence,
)
from shared.lib.benchmark_contract import load_canonical_split


# Which KG relations each graph structure keeps, and whether it expands the
# subgraph to KG neighbours that are NOT among the patient's own codes. Mirrors
# the ProtGNN structures so the two methods compare on the SAME topology.
#   star     patient hub spokes only
#   cooccur  hub spokes + co_occurs among present codes
#   ontology hub spokes + same_class among present codes
#   full     hub spokes + both relations among present codes
#   full_kg_expanded  full plus record-external 1-hop KG neighbours
_STRUCT_REL_NAMES = {
    "star": set(),
    "cooccur": {"co_occurs"},
    "ontology": {"same_class"},
    "full": {"co_occurs", "same_class"},
    "full_kg_expanded": {"co_occurs", "same_class"},
}
_STRUCT_EXPAND = {
    "star": False,
    "cooccur": False,
    "ontology": False,
    "full": False,
    "full_kg_expanded": True,
}


def _subgraph(code_ids, neighbours, num_nodes, structure="full",
              patient_hub_id=None, patient_relation_id=2, rel2id=None):
    """Patient codes -> model-input tensors, wired per `structure`.

    `structure` selects which KG relations become edges and whether the
    subgraph expands to 1-hop KG neighbours outside the patient's own codes.
    The reserved patient hub defaults to the final global node ID and connects
    bidirectionally to every observed concept.
    """
    relation_schema = PREPROCESSING_REL2ID if rel2id is None else rel2id
    allow_rel = {relation_schema[name] for name in _STRUCT_REL_NAMES[structure]}
    expand = _STRUCT_EXPAND[structure]
    patient_hub_id = num_nodes - 1 if patient_hub_id is None else int(patient_hub_id)
    code_set = set(int(c) for c in code_ids)
    node_set = set(code_set)
    node_set.add(patient_hub_id)
    src, rel, dst = [], [], []
    for c in sorted(code_set):
        src.extend((patient_hub_id, c))
        rel.extend((patient_relation_id, patient_relation_id))
        dst.extend((c, patient_hub_id))
        for r, nbr in neighbours[c]:
            if r not in allow_rel:
                continue
            if not expand and nbr not in code_set:
                continue
            node_set.add(nbr)
            src.append(c); rel.append(r); dst.append(nbr)
    sub = sorted(node_set)
    g2l = {g: l for l, g in enumerate(sub)}
    node_ids = torch.tensor(sub, dtype=torch.long)
    if src:
        edge_index = torch.tensor([[g2l[a] for a in src], [g2l[b] for b in dst]], dtype=torch.long)
        rel_ids = torch.tensor(rel, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        rel_ids = torch.zeros(0, dtype=torch.long)
    visit_node = torch.zeros(num_nodes)
    visit_node[sub] = 1.0
    ehr_nodes = torch.zeros(num_nodes)
    ehr_nodes[[int(c) for c in code_ids]] = 1.0
    return {"node_ids": node_ids, "edge_index": edge_index, "rel_ids": rel_ids,
            "visit_node": visit_node, "ehr_nodes": ehr_nodes}


def _collate(batch):
    sizes = [b["node_ids"].numel() for b in batch]
    offs = torch.tensor([0] + sizes[:-1]).cumsum(0)
    return {
        "node_ids":   torch.cat([b["node_ids"] for b in batch]),
        "rel_ids":    torch.cat([b["rel_ids"] for b in batch]),
        "edge_index": torch.cat([b["edge_index"] + offs[i] for i, b in enumerate(batch)], dim=1),
        "batch":      torch.cat([torch.full((s,), i, dtype=torch.long) for i, s in enumerate(sizes)]),
        "visit_node": torch.stack([b["visit_node"] for b in batch]).unsqueeze(1),  # [B,1,V]
        "ehr_nodes":  torch.stack([b["ehr_nodes"] for b in batch]),                # [B,V]
        "y":          torch.tensor([b["y"] for b in batch], dtype=torch.long),
    }


class GraphCareDataset(Dataset):
    def __init__(self, code_lists, labels, neighbours, num_nodes, structure="full",
                 patient_hub_id=None, patient_relation_id=2, rel2id=None,
                 subject_ids=None):
        self.code_lists = code_lists
        self.labels = labels
        self.neighbours = neighbours
        self.num_nodes = num_nodes
        self.structure = structure
        self.patient_hub_id = (num_nodes - 1 if patient_hub_id is None
                               else int(patient_hub_id))
        self.patient_relation_id = int(patient_relation_id)
        self.rel2id = PREPROCESSING_REL2ID if rel2id is None else rel2id
        self.subject_ids = subject_ids
        if subject_ids is not None and len(subject_ids) != len(labels):
            raise ValueError("GraphCare subject_ids must align one-to-one with labels.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        s = _subgraph(
            self.code_lists[i], self.neighbours, self.num_nodes, self.structure,
            patient_hub_id=self.patient_hub_id,
            patient_relation_id=self.patient_relation_id,
            rel2id=self.rel2id,
        )
        s["y"] = int(self.labels[i])
        if self.subject_ids is not None:
            s["subject_id"] = str(self.subject_ids[i])
        return s


def build_standardized_dataset(
    kg, *, split_json, structure, dataset_path=None
):
    """Build the exact split-aware GraphCare records used by training and audits."""
    rel2id = validate_kg_schema(kg)
    source_path = (
        cfg.data_dir / cfg.csv_filename
        if dataset_path is None else dataset_path
    )
    df = pd.read_csv(source_path, low_memory=False)
    dis = df["disease_1"].fillna("").astype(str)
    meta = load_canonical_split(split_json)
    provenance = validate_training_provenance(
        kg, split_json, dataset_path=source_path
    )
    fold = meta["fold"]
    keep, fit_indices, subject_keys = canonical_row_indices(df, meta)
    if provenance["fit_row_count"] != len(fit_indices):
        raise ValueError(
            "GraphCare standardized preprocessing training-fold provenance "
            "does not match the dataset's training row count."
        )
    expected_ent2id, _, presence, _ = vocab_and_presence(
        df, fit_indices=fit_indices, include_symptoms=False
    )
    classes = meta["classes"]
    class_to_id = {disease: index for index, disease in enumerate(classes)}
    labels = dis.map(class_to_id).fillna(-1).astype(int).values
    invalid_label_rows = [row for row in keep if labels[row] < 0]
    if invalid_label_rows:
        diagnostics = "; ".join(
            f"subject_id={subject_keys[row]}, fold={fold[subject_keys[row]]}, "
            f"row={row}, label={dis.iloc[row]!r}"
            for row in invalid_label_rows[:10]
        )
        raise ValueError(
            "GraphCare standardized canonical row has a label outside the "
            f"canonical class mapping: {diagnostics}."
        )
    # Retain every canonical subject, including clinical-empty records. _subgraph
    # represents these by the patient hub alone, with zero EHR membership.
    # graphcare_analysis.model defines their raw EHR mean as zero; no fake code.

    patient_hub_id = kg.get("patient_hub_id")
    patient_relation_id = kg.get("patient_relation_id")
    concept_ent2id = {
        name: node_id
        for name, node_id in kg.get("ent2id", {}).items()
        if name != "patient:hub"
    }
    if (
        patient_hub_id != kg.get("ent2id", {}).get("patient:hub")
        or patient_relation_id
        != kg.get("rel2id", {}).get("patient_has_concept")
        or patient_hub_id != presence.shape[1]
        or kg.get("num_nodes") != presence.shape[1] + 1
        or concept_ent2id != expected_ent2id
    ):
        raise ValueError(
            "GraphCare KG vocabulary or reserved patient hub metadata does not "
            "match the fitted dataset representation."
        )
    code_lists = [np.nonzero(presence[row])[0] for row in keep]
    dataset = GraphCareDataset(
        code_lists,
        labels[keep],
        kg["neighbours"],
        kg["num_nodes"],
        structure=structure,
        patient_hub_id=patient_hub_id,
        patient_relation_id=patient_relation_id,
        rel2id=rel2id,
        subject_ids=[subject_keys[row] for row in keep],
    )
    folds = np.array([fold[subject_keys[row]] for row in keep])
    train = np.where(folds == 0)[0].tolist()
    validation = np.where(folds == 1)[0].tolist()
    test = np.where(folds == 2)[0].tolist()
    return dataset, train, validation, test, len(classes), class_to_id


def build_loaders(kg, limit=None, batch_size=None, split_json=None, structure="full"):
    """Build GraphCare loaders with frozen training-fold preprocessing.

    If ``split_json`` is supplied, both the cached KG provenance and the fitted
    vocabulary must match that split's training fold. Legacy runs without an
    explicit split retain their historical all-row fitting behavior.
    """
    batch_size = batch_size or cfg.batch_size
    if split_json:
        ds, tr, va, te, class_count, class_to_id = build_standardized_dataset(
            kg,
            split_json=split_json,
            structure=structure,
            dataset_path=cfg.data_dir / cfg.csv_filename,
        )
        if limit is not None:
            if type(limit) is not int or limit < 1:
                raise ValueError("GraphCare standardized limit must be a positive exact integer.")
            tr = tr[:limit]
            va = va[:limit]
            te = te[:limit]
    else:
        rel2id = validate_kg_schema(kg)
        df = pd.read_csv(cfg.data_dir / cfg.csv_filename, low_memory=False)
        dis = df["disease_1"].fillna("").astype(str)
        subj = (df["subject_id"].values if "subject_id" in df.columns
                else np.arange(len(df), dtype=np.int64))
        expected_ent2id, _, M, _ = vocab_and_presence(df)
        classes = sorted(d for d in dis.unique() if d.strip())
        class_to_id = {d: i for i, d in enumerate(classes)}
        y = dis.map(class_to_id).fillna(-1).astype(int).values
        keep = [i for i in range(len(df)) if y[i] >= 0 and M[i].any()]
        if limit:
            keep = keep[:limit]
        labels_k, groups_k = y[keep], subj[keep]
        if len(np.unique(groups_k)) < len(groups_k):
            tr, va, te = subject_aware_split_indices(
                labels_k, groups_k, cfg.split_ratio, cfg.seed
            )
        else:
            tr, va, te = stratified_split_indices(
                labels_k, cfg.split_ratio, cfg.seed
            )

        patient_hub_id = kg.get("patient_hub_id")
        patient_relation_id = kg.get("patient_relation_id")
        concept_ent2id = {
            name: node_id
            for name, node_id in kg.get("ent2id", {}).items()
            if name != "patient:hub"
        }
        if (
            patient_hub_id != kg.get("ent2id", {}).get("patient:hub")
            or patient_relation_id
            != kg.get("rel2id", {}).get("patient_has_concept")
            or patient_hub_id != M.shape[1]
            or kg.get("num_nodes") != M.shape[1] + 1
            or concept_ent2id != expected_ent2id
        ):
            raise ValueError(
                "GraphCare KG vocabulary or reserved patient hub metadata does not "
                "match the fitted dataset representation."
            )
        code_lists = [np.nonzero(M[i])[0] for i in range(len(df))]
        ds = GraphCareDataset(
            [code_lists[i] for i in keep],
            y[keep],
            kg["neighbours"],
            kg["num_nodes"],
            structure=structure,
            patient_hub_id=patient_hub_id,
            patient_relation_id=patient_relation_id,
            rel2id=rel2id,
        )
        class_count = len(classes)

    def loader(idx, shuffle):
        return DataLoader(Subset(ds, idx), batch_size=batch_size,
                          shuffle=shuffle, collate_fn=_collate)

    print(f"  patients kept={len(ds)} | train/val/test={len(tr)}/{len(va)}/{len(te)} | classes={class_count}")
    return loader(tr, True), loader(va, False), loader(te, False), class_count, class_to_id
