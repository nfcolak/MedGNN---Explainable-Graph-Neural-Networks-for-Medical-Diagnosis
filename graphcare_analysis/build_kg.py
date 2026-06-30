"""Global knowledge graph over the code vocabulary for GraphCare.  (Phase 2.)

Model-direct Path B: produces the global KG that the adapter expands into
per-patient subgraphs, plus the sizes the GraphCare model is built with
(num_nodes, num_rels).

KG (ontology + PMI, no LLM — and the FAIREST vs ProtGNN, same KG source):
    nodes      : the non-leaky code vocab (meds >=1% prevalence + symptoms + cc),
                 mirroring protgnn's IntraPatientHeteroDataset vocab.
    relation 0 : "co_occurs"  — code pairs with population PMI > threshold.
    relation 1 : "same_class" — meds sharing a therapeutic class (MED_CLASS).

Output (torch.save to cfg.kg_path, plain types so it loads in any torch):
    ent2id     : entity string ("med:<g>" / "sym:<v>" / "cc:<v>") -> node id
    rel2id     : {"co_occurs": 0, "same_class": 1}
    neighbours : node id -> list of (relation_id, neighbour_node_id)   (bidirectional)
    num_nodes, num_rels

Run (either env has pandas/numpy/torch):
    PYTHONPATH=. .venv-graphcare/bin/python3 graphcare_analysis/build_kg.py
"""
import numpy as np
import pandas as pd
import torch

from graphcare_analysis.config import cfg

try:
    from shared.data_prep.med_classes import MED_CLASS
except Exception:
    MED_CLASS = {}


def _multihot_vocab(df, cols):
    """Unique non-empty cell values across `cols` (mirrors protgnn _build_vocab)."""
    counter = {}
    if not cols:
        return [], np.zeros((len(df), 0), dtype=bool)
    rows = df[cols].fillna("").astype(str).values.tolist()
    for row in rows:
        for s in row:
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


def vocab_and_presence(df):
    """Shared by build_kg and the adapter so the vocab/ids are guaranteed identical.

    Returns: ent2id, ent_names, M (N x num_nodes bool presence), med_cols
    (node order: meds, then symptoms, then chief complaints).
    """
    med_all = sorted(c for c in df.columns if c.startswith(("med_", "pyx_")))
    med_bin = (df[med_all].apply(pd.to_numeric, errors="coerce").fillna(0).values > 0)
    keep = med_bin.mean(0) >= cfg.med_min_prev
    med_cols = [c for c, k in zip(med_all, keep) if k]
    med_present = med_bin[:, keep]

    sym_cols = sorted(c for c in df.columns if c.startswith("symptom_"))
    cc_cols = sorted(c for c in df.columns if c.startswith("chiefcomplaint_"))
    sym_vocab, sym_present = _multihot_vocab(df, sym_cols)
    cc_vocab, cc_present = _multihot_vocab(df, cc_cols)

    ent_names = (
        [f"med:{c[4:]}" for c in med_cols]
        + [f"sym:{v}" for v in sym_vocab]
        + [f"cc:{v}" for v in cc_vocab]
    )
    ent2id = {name: i for i, name in enumerate(ent_names)}
    M = np.concatenate([med_present, sym_present, cc_present], axis=1).astype(bool)
    return ent2id, ent_names, M, med_cols


def build_global_kg(save=True):
    df = pd.read_csv(cfg.data_dir / cfg.csv_filename, low_memory=False)
    n = len(df)
    ent2id, ent_names, M, med_cols = vocab_and_presence(df)
    num_nodes = len(ent_names)
    n_med = len(med_cols)

    Mf = M.astype(np.float64)

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

    rel2id = {"co_occurs": 0, "same_class": 1}
    neighbours = {i: [] for i in range(num_nodes)}
    for a, b in co_pairs:
        neighbours[a].append((0, b)); neighbours[b].append((0, a))
    for a, b in same_pairs:
        neighbours[a].append((1, b)); neighbours[b].append((1, a))

    kg = {"ent2id": ent2id, "rel2id": rel2id, "neighbours": neighbours,
          "num_nodes": num_nodes, "num_rels": len(rel2id), "n_med": n_med}

    degs = [len(v) for v in neighbours.values()]
    print(f"  KG built: nodes={num_nodes} (med={n_med}, sym={M.shape[1]-n_med-len([e for e in ent_names if e.startswith('cc:')])}, cc={len([e for e in ent_names if e.startswith('cc:')])})")
    print(f"  edges: co_occurs={len(co_pairs)}, same_class={len(same_pairs)} | rels={len(rel2id)}")
    print(f"  degree mean={np.mean(degs):.1f} max={max(degs)} | isolated nodes={sum(d == 0 for d in degs)}")

    if save:
        cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(kg, cfg.kg_path)
        print(f"  saved -> {cfg.kg_path}")
    return kg


if __name__ == "__main__":
    build_global_kg()
