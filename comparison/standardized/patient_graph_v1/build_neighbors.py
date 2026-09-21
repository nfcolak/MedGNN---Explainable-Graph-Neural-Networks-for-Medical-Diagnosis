"""Build the patient-patient neighbour artifact for the transductive experiment.

Run with ``python -m comparison.standardized.patient_graph_v1.build_neighbors --execute``.

Why this artifact exists
------------------------
The native star graph is a lossless redrawing of the tabular feature vector: with
zero repeated concepts and zero non-star topologies across all 74,511 patients,
the concept set determines the edge list exactly, and the 192-dim presence vector
XGBoost consumes reconstructs every graph byte-for-byte. A GNN on that input
therefore reads no information the tree model lacks, which is consistent with
message passing being worth only ~+0.009 macro-F1 over a plain additive model.

A patient-patient neighbourhood is different in kind. Measured on this cohort:

    neighbour label homophily (k=25)      0.412   (random-pair baseline 0.063)
    true label present in 25-neighbourhood 0.870
    recovered by plain majority vote       0.525
    XGBoost alone                          0.639
    blending XGBoost with neighbour votes  +0.0082 on a held-out half
    the same blend with RANDOM neighbours  -0.0010

The random-neighbour control is the important line: an equal-degree random graph
gives nothing, so the gain is attributable to structure rather than to smoothing.
The gap between 0.525 (majority vote) and 0.870 (label is present) is the room a
learned aggregator has, and a learned aggregator over a graph is exactly a GNN.

This is information a row-independent tree model cannot use in principle: other
patients' labels are not features of this patient.

Leakage contract
----------------
- Neighbours are ALWAYS drawn from the training fold. Only training labels are
  ever read, mirroring the standard transductive node-classification setup.
- A training row excludes itself from its own neighbourhood; otherwise every
  training patient would see its own label at distance zero and the aggregator
  would learn to copy it.
- The scaler and the neighbour index are fitted on training rows only.
- Fold 2 is never indexed, queried, or written. The artifact stores neighbours
  for folds 0 and 1 only, and a consumer that asks for a test row fails closed.

Setting disclosure
------------------
This changes the problem from "predict from one patient's record" to transductive
prediction with access to a labelled patient population. That is a legitimate and
standard setting, but it is NOT the same task as the star-graph benchmark, and
results must be reported beside a baseline evaluated in the same setting rather
than against the existing single-patient numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from comparison.standardized.native_reference_v1.data import (
    DEFAULT, PINNED_CONTRACT, Reference, digest, sha, save,
)
from comparison.standardized.xgboost_native_baseline import build_tabular_matrix

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPO / "comparison/standardized/native_inputs/patient_knn_v1"
K = 50
CLASSES = 30


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(json.dumps([a.dtype.str, list(a.shape)]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def standardize(X, fit_rows):
    """Train-fold z-scaling so the 192 binary concept columns do not dominate.

    Without this the concept block, being 0/1 with small variance, contributes
    far less to Euclidean distance than the already-z-scored hub block, and the
    neighbourhood degenerates towards hub-only similarity.
    """
    mu = X[fit_rows].mean(0)
    sd = X[fit_rows].std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32), mu, sd


def build(output=DEFAULT_OUTPUT, artifact=DEFAULT, k=K, execute=False):
    from sklearn.neighbors import NearestNeighbors

    X, y, folds, feature_names, meta = build_tabular_matrix(artifact)
    require(meta["contract_sha256"] == PINNED_CONTRACT, "Wrong native contract")
    train_idx = np.flatnonzero(folds == 0)
    val_idx = np.flatnonzero(folds == 1)
    Z, mu, sd = standardize(X, train_idx)

    report = {
        "version": "patient-knn-v1",
        "contract_sha256": meta["contract_sha256"],
        "input_sha256": meta["input_sha256"],
        "k": k,
        "metric": "euclidean on train-fold z-scaled 324-dim tabular features",
        "counts": {"train": int(len(train_idx)), "validation": int(len(val_idx))},
        "train_ordinals_sha256": digest(train_idx.tolist()),
        "validation_ordinals_sha256": digest(val_idx.tolist()),
        "leakage_policy": (
            "neighbours drawn from the training fold only; training rows exclude "
            "themselves; scaler fitted on training rows; fold 2 never indexed"),
        "setting": "transductive node classification with labelled training population",
        "test_evaluated": False,
    }
    if not execute:
        return {**report, "status": "dry_run"}
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Occupied neighbour output: {output}")

    start = time.monotonic()
    index = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Z[train_idx])

    # Training rows: ask for k+1 and drop the self-match. The self-match is the
    # zero-distance hit; assert it is actually present rather than assuming it,
    # because duplicate feature rows could otherwise silently shift the drop.
    d_tr, i_tr = index.kneighbors(Z[train_idx], n_neighbors=k + 1)
    self_hit = i_tr[:, 0] == np.arange(len(train_idx))
    require(self_hit.mean() > 0.98, "Self-match missing for most training rows")
    rows = np.arange(len(train_idx))
    keep = np.empty((len(train_idx), k), dtype=np.int64)
    keepd = np.empty((len(train_idx), k), dtype=np.float32)
    for r in range(len(train_idx)):
        cand = i_tr[r]
        mask = cand != r
        take = np.flatnonzero(mask)[:k]
        require(len(take) == k, "Not enough non-self neighbours")
        keep[r] = cand[take]
        keepd[r] = d_tr[r][take]
    train_nb, train_d = keep, keepd

    # Validation rows: all train neighbours are admissible, no self-exclusion.
    val_d, val_nb = index.kneighbors(Z[val_idx], n_neighbors=k)

    require(not np.any(train_nb == np.arange(len(train_idx))[:, None]),
            "A training row kept itself as a neighbour")
    require(train_nb.max() < len(train_idx) and val_nb.max() < len(train_idx),
            "Neighbour index outside the training fold")

    ytr = y[train_idx]
    hom_tr = float((ytr[train_nb[:, :25]] == ytr[:, None]).mean())
    hom_va = float((ytr[val_nb[:, :25]] == y[val_idx][:, None]).mean())
    base = float(sum((np.bincount(ytr, minlength=CLASSES) / len(ytr)) ** 2))

    output.mkdir(parents=True)
    train_nb = train_nb.astype(np.int32)
    val_nb = val_nb.astype(np.int32)
    np.savez_compressed(
        output / "neighbours.npz",
        train_neighbours=train_nb,
        train_distances=train_d.astype(np.float32),
        val_neighbours=val_nb,
        val_distances=val_d.astype(np.float32),
        train_ordinals=train_idx, validation_ordinals=val_idx,
        scaler_mean=mu.astype(np.float32), scaler_std=sd.astype(np.float32),
    )
    manifest = {
        **report, "status": "completed",
        "neighbours_sha256": array_sha(train_nb, val_nb, train_idx, val_idx),
        "homophily_k25": {"train": hom_tr, "validation": hom_va, "random_pair_baseline": base},
        "elapsed_seconds": round(time.monotonic() - start, 1),
        "artifact_files": {"neighbours.npz": sha(output / "neighbours.npz")},
    }
    save(output / "neighbour_manifest.json", manifest)
    return manifest


def load_neighbours(root, ref, k=None):
    """Load and re-verify the neighbour artifact against the reference split."""
    root = Path(root)
    m = json.loads((root / "neighbour_manifest.json").read_text())
    require(m["status"] == "completed", "Incomplete neighbour artifact")
    require(m["contract_sha256"] == ref.fingerprint == PINNED_CONTRACT, "Contract mismatch")
    require(m["test_evaluated"] is False, "Neighbour artifact touched test")
    require(sha(root / "neighbours.npz") == m["artifact_files"]["neighbours.npz"],
            "Neighbour checksum mismatch")
    with np.load(root / "neighbours.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    train_idx, val_idx = data["train_ordinals"], data["validation_ordinals"]
    require(np.array_equal(train_idx, ref.fold(0)), "Neighbour train cohort mismatch")
    require(np.array_equal(val_idx, ref.fold(1)), "Neighbour validation cohort mismatch")
    require(array_sha(data["train_neighbours"], data["val_neighbours"],
                      train_idx, val_idx) == m["neighbours_sha256"], "Neighbour hash mismatch")
    require(not np.any(data["train_neighbours"] == np.arange(len(train_idx))[:, None]),
            "Self-neighbour present in the artifact")
    if k is not None:
        require(k <= data["train_neighbours"].shape[1], "Requested k exceeds the artifact")
    return data, m


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--artifact", type=Path, default=DEFAULT)
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    r = build(args.output, args.artifact, args.k, args.execute)
    print(json.dumps({key: r.get(key) for key in
                      ("status", "k", "counts", "homophily_k25", "neighbours_sha256",
                       "elapsed_seconds")}, indent=2))
    return r


if __name__ == "__main__":
    main()
