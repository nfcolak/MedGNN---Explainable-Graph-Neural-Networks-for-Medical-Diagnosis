"""Build ONE canonical train/val/test split that BOTH methods must use, so the
comparison is on the IDENTICAL test patients (no train/test leakage).

Common evaluable set = patients with a disease_1 label AND >=1 code
(med/symptom/cc) — this matches GraphCare's drop policy; ProtGNN is restricted
to the same subjects. Split = shared subject-aware logic + seed 1234.

Saves comparison/canonical_split.json:
    fold     : {subject_id: 0|1|2}   (0=train, 1=val, 2=test)
    classes  : sorted disease_1 classes (identical class ids for both methods)

Run:  PYTHONPATH=. python3 comparison/build_split.py
"""
import json

import numpy as np
import pandas as pd

from shared.lib.config_base import DATA_DIR, SEED, SPLIT_RATIO
from shared.lib.splits import stratified_split_indices, subject_aware_split_indices
from graphcare_analysis.build_kg import vocab_and_presence

OUT = "comparison/canonical_split.json"


def main():
    df = pd.read_csv(DATA_DIR / "merged_ed.csv", low_memory=False)
    _, _, M, _ = vocab_and_presence(df)
    n_codes = M.sum(1)

    dis = df["disease_1"].fillna("").astype(str)
    classes = sorted(d for d in dis.unique() if d.strip())
    c2i = {d: i for i, d in enumerate(classes)}
    y = dis.map(c2i).fillna(-1).astype(int).values
    subj = (df["subject_id"].values if "subject_id" in df.columns
            else np.arange(len(df), dtype=np.int64))

    keep = [i for i in range(len(df)) if y[i] >= 0 and n_codes[i] > 0]
    labels_k = y[keep]
    groups_k = subj[keep]
    if len(np.unique(groups_k)) < len(groups_k):
        tr, va, te = subject_aware_split_indices(labels_k, groups_k, SPLIT_RATIO, SEED)
        how = "subject-aware"
    else:
        tr, va, te = stratified_split_indices(labels_k, SPLIT_RATIO, SEED)
        how = "stratified (subjects unique = single-visit)"

    fold = {}
    for local in tr:
        fold[str(int(subj[keep[local]]))] = 0
    for local in va:
        fold[str(int(subj[keep[local]]))] = 1
    for local in te:
        fold[str(int(subj[keep[local]]))] = 2

    json.dump({"fold": fold, "classes": classes, "n_kept": len(keep),
               "split": how, "seed": SEED, "ratio": SPLIT_RATIO},
              open(OUT, "w"))
    print(f"  common evaluable patients : {len(keep)} / {len(df)}")
    print(f"  split ({how}): train={len(tr)} val={len(va)} test={len(te)} | classes={len(classes)}")
    print(f"  saved -> {OUT}")


if __name__ == "__main__":
    main()
