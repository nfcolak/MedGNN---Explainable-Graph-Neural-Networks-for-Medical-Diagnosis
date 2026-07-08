"""Adapter: merged_ed.csv  ->  GraphCare model-input tensors.  (Phase 1.)

Model-direct Path B (validated in test_model_smoke.py). Per patient (our ED
cohort is SINGLE-VISIT, so max_visit = 1) we build a personalized subgraph =
the patient's codes + their 1-hop KG neighbours, and emit the tensors the
upstream GraphCare model consumes:

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
from graphcare_analysis.build_kg import vocab_and_presence


# Which KG relations each graph structure keeps, and whether it expands the
# subgraph to KG neighbours that are NOT among the patient's own codes. Mirrors
# the ProtGNN structures so the two methods compare on the SAME topology.
#   star     no edges (codes only)          cooccur  co_occurs, present codes
#   ontology same_class, present codes      full     both rels + 1-hop expansion
_STRUCT_RELS = {"star": set(), "cooccur": {0}, "ontology": {1}, "full": {0, 1}}
_STRUCT_EXPAND = {"star": False, "cooccur": False, "ontology": False, "full": True}


def _subgraph(code_ids, neighbours, num_nodes, structure="full"):
    """Patient codes -> model-input tensors, wired per `structure`.

    `structure` selects which KG relations become edges and whether the
    subgraph expands to 1-hop KG neighbours outside the patient's own codes.
    """
    allow_rel = _STRUCT_RELS[structure]
    expand = _STRUCT_EXPAND[structure]
    code_set = set(int(c) for c in code_ids)
    node_set = set(code_set)
    src, rel, dst = [], [], []
    for c in code_ids:
        c = int(c)
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
    def __init__(self, code_lists, labels, neighbours, num_nodes, structure="full"):
        self.code_lists = code_lists
        self.labels = labels
        self.neighbours = neighbours
        self.num_nodes = num_nodes
        self.structure = structure

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        s = _subgraph(self.code_lists[i], self.neighbours, self.num_nodes, self.structure)
        s["y"] = int(self.labels[i])
        return s


def build_loaders(kg, limit=None, batch_size=None, split_json=None, structure="full"):
    """merged_ed.csv + global KG -> (train, val, test) DataLoaders, num_classes, class_to_id.

    If `split_json` (comparison/canonical_split.json) is given, the patient set,
    the class ids, and the train/val/test folds are taken from it verbatim, so
    GraphCare and ProtGNN evaluate on the IDENTICAL test patients.
    """
    batch_size = batch_size or cfg.batch_size
    df = pd.read_csv(cfg.data_dir / cfg.csv_filename, low_memory=False)

    # vocab/presence on the FULL data so node ids match the KG exactly
    _, _, M, _ = vocab_and_presence(df)
    assert M.shape[1] == kg["num_nodes"], (M.shape[1], kg["num_nodes"])

    dis = df["disease_1"].fillna("").astype(str)
    subj = (df["subject_id"].values if "subject_id" in df.columns
            else np.arange(len(df), dtype=np.int64))
    code_lists = [np.nonzero(M[i])[0] for i in range(len(df))]

    if split_json:
        import json
        meta = json.load(open(split_json))
        fold = meta["fold"]
        classes = meta["classes"]
        class_to_id = {d: i for i, d in enumerate(classes)}
        y = dis.map(class_to_id).fillna(-1).astype(int).values
        keep = [i for i in range(len(df)) if str(int(subj[i])) in fold]
        folds = np.array([fold[str(int(subj[i]))] for i in keep])
        tr = np.where(folds == 0)[0].tolist()
        va = np.where(folds == 1)[0].tolist()
        te = np.where(folds == 2)[0].tolist()
    else:
        classes = sorted(d for d in dis.unique() if d.strip())
        class_to_id = {d: i for i, d in enumerate(classes)}
        y = dis.map(class_to_id).fillna(-1).astype(int).values
        keep = [i for i in range(len(df)) if y[i] >= 0 and len(code_lists[i]) > 0]
        if limit:
            keep = keep[:limit]
        labels_k, groups_k = y[keep], subj[keep]
        if len(np.unique(groups_k)) < len(groups_k):
            tr, va, te = subject_aware_split_indices(labels_k, groups_k, cfg.split_ratio, cfg.seed)
        else:
            tr, va, te = stratified_split_indices(labels_k, cfg.split_ratio, cfg.seed)

    ds = GraphCareDataset([code_lists[i] for i in keep],
                          y[keep], kg["neighbours"], kg["num_nodes"],
                          structure=structure)

    def loader(idx, shuffle):
        return DataLoader(Subset(ds, idx), batch_size=batch_size,
                          shuffle=shuffle, collate_fn=_collate)

    print(f"  patients kept={len(keep)} | train/val/test={len(tr)}/{len(va)}/{len(te)} | classes={len(classes)}")
    return loader(tr, True), loader(va, False), loader(te, False), len(classes), class_to_id
