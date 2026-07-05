"""
extract_protgnn_subjects.py  —  recover the subject_ids ProtGNN explained,
read straight from the PROCESSED CACHE (drift-proof).
==============================================================================
ProtGNN's explanation JSONs store `dataset_index`. The source CSV has since
drifted (rows added/removed/reordered), so `dataset_index` no longer aligns to
merged_ed.csv. But the processed cache the explanations were computed on is
still on disk and stores, per graph:  dataset_index, subject_id, y.

We build {dataset_index -> (subject_id, y)} from the cache, then for each
explained graph look up its subject_id by dataset_index, and SELF-VERIFY by
checking the cache's y matches the JSON's true_label (must be N/N — same data).

RUN IN THE PROTGNN ENV (needs torch + torch_geometric to unpickle the cache):
    python protgnn_analysis/explainability/extract_protgnn_subjects.py

Writes comparison/protgnn_explained_subjects.json:
    {"subjects":[...], "per_graph":{"<graph_idx>":<subject_id>},
     "source":"processed_cache", "label_check":"N/N", "in_test_fold":K}
"""
import os
import sys
import glob
import json

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, os.pardir, os.pardir))

CACHE = "data/processed_hetero_merged_ed_noLOS_prev10_pmi2_miss_disease/data.pt"
EXPL_DIR = "protgnn_analysis/outputs/results/2026-07-03_11-01_disease_gcn/explanations"
CANONICAL_SPLIT = "comparison/canonical_split.json"


def _grab(data, name):
    try:
        if name in data:
            return data[name]
    except Exception:
        pass
    return getattr(data, name, None)


def main():
    cache_path = os.path.join(_ROOT, CACHE)
    if not os.path.isfile(cache_path):
        sys.exit(f"Cache not found: {cache_path}")
    store = torch.load(cache_path, weights_only=False)
    data = store[0] if isinstance(store, tuple) else store

    di = _grab(data, "dataset_index").view(-1).tolist()
    sid = _grab(data, "subject_id").view(-1).tolist()
    yv = _grab(data, "y").view(-1).tolist()
    # dataset_index -> (subject_id, y). dataset_index values are the stable key.
    idx2subj = {int(di[k]): int(sid[k]) for k in range(len(di))}
    idx2y = {int(di[k]): int(yv[k]) for k in range(len(di))}
    print(f"Cache: {len(di)} graphs, {len(set(sid))} unique subjects")

    files = sorted(glob.glob(os.path.join(_ROOT, EXPL_DIR, "graph_*.json")),
                   key=lambda p: int(os.path.basename(p)[6:-5]))
    if not files:
        sys.exit(f"No explanation JSONs in {EXPL_DIR}")

    per_graph, label_ok, missing = {}, 0, []
    for fp in files:
        d = json.load(open(fp))
        gidx = int(d["graph_idx"]); dsi = int(d["dataset_index"]); tl = int(d["true_label"])
        if dsi not in idx2subj:
            missing.append(dsi); continue
        per_graph[str(gidx)] = idx2subj[dsi]
        label_ok += int(idx2y[dsi] == tl)

    n = len(files)
    print(f"Explained graphs: {n} | label self-check (cache y == json true_label): {label_ok}/{n}")
    if missing:
        print(f"  [warn] {len(missing)} dataset_index not found in cache: {missing[:10]}")
    if label_ok != n:
        print("  [warn] label self-check imperfect — the cache on disk may not be the exact "
              "one used for these explanations. Investigate before trusting the mapping.")
    else:
        print("  label self-check PERFECT — this is the right cache; subjects are exact.")

    subjects = sorted(set(per_graph.values()))
    in_test = None
    sp = os.path.join(_ROOT, CANONICAL_SPLIT)
    if os.path.isfile(sp):
        fold = json.load(open(sp))["fold"]
        in_test = sum(1 for s in subjects if fold.get(str(s)) == 2)
        print(f"  canonical split: {in_test}/{len(subjects)} recovered subjects in TEST fold")
        if in_test != len(subjects):
            print("  [note] some subjects aren't in GraphCare's test fold — those patients "
                  "can't be matched on the GraphCare side; the matched set will be the "
                  "intersection. This reflects a real split/cohort difference to report.")

    out_dir = os.path.join(_ROOT, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "protgnn_explained_subjects.json")
    json.dump({"subjects": subjects, "per_graph": per_graph, "source": "processed_cache",
               "n_explained": n, "label_check": f"{label_ok}/{n}", "in_test_fold": in_test},
              open(out_path, "w"), indent=2)
    print(f"\n  wrote {out_path}  ({len(subjects)} unique subjects)")
    print(f"  -> GraphCare:  --match_subjects comparison/protgnn_explained_subjects.json")


if __name__ == "__main__":
    main()
    