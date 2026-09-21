"""Read-only diagnosis: why does the new event-graph input score far below the old native input?

Validation-only. The test fold is never loaded into any matrix. Nothing here overwrites an
existing run directory; every cell appends one JSON line to results.jsonl.

Two artifacts are compared under ONE shared tabular learner (XGBoost) so that the difference
attributable to the INPUT is isolated from architecture, optimizer and training budget:

  old  = comparison/standardized/native_inputs/protgsat_snapshot_v1   (324 tabular columns)
  new  = comparison/standardized/event_inputs/..._features_v1         (3483 tabular columns)

Block ablations answer where the old artifact's signal actually lives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD_ROOT = REPO / 'comparison/standardized/native_inputs/protgsat_snapshot_v1'
NEW_FEATURES = REPO / 'comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_features_v1'
OLD_CONTRACT_SHA = '2a2566e662bf87b10d97e58700f868d8d6d6b036582ec046dd5453e6dcd9f7c0'
NEW_FEATURES_SHA = 'b5988ae72e77e162cc1c66dcfc1940a8305e4feb0ed4e95f5d44bae30fdf36b4'

# One shared budget for every ablation cell. Deliberately smaller than the frozen
# matched protocol (1200 rounds @ lr 0.03); reported as its own budget, never merged
# into the frozen matched_gchm_xgb_v1 tables.
PARAMS = dict(objective='multi:softprob', num_class=30, tree_method='hist',
              max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
              reg_lambda=2.0, min_child_weight=3.0, nthread=16,
              disable_default_eval_metric=1)
ROUNDS = 400


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def metrics(y_true, proba):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 top_k_accuracy_score)
    pred = proba.argmax(1)
    return {
        'accuracy': round(float(accuracy_score(y_true, pred)), 6),
        'balanced_acc': round(float(balanced_accuracy_score(y_true, pred)), 6),
        'macro_f1': round(float(f1_score(y_true, pred, average='macro', zero_division=0)), 6),
        'micro_f1': round(float(f1_score(y_true, pred, average='micro', zero_division=0)), 6),
        'top3_acc': round(float(top_k_accuracy_score(y_true, proba, k=3, labels=np.arange(30))), 6),
        'top5_acc': round(float(top_k_accuracy_score(y_true, proba, k=5, labels=np.arange(30))), 6),
    }


def class_weights(labels, policy):
    counts = np.bincount(labels, minlength=30)
    if np.any(counts == 0):
        raise ValueError('All 30 classes must occur in training')
    w = len(labels) / (30 * counts)
    if policy == 'sqrt_inverse':
        return np.sqrt(w)
    if policy == 'none':
        return np.ones(30)
    raise ValueError('Unknown policy')


# --------------------------------------------------------------------------- old
def load_old():
    contract = json.loads((OLD_ROOT / 'contract.json').read_text())
    binding = contract.copy()
    recorded = binding.pop('contract_sha256')
    canon = json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if hashlib.sha256(canon.encode()).hexdigest() != recorded or recorded != OLD_CONTRACT_SHA:
        raise ValueError('Old contract fingerprint mismatch')
    if sha(OLD_ROOT / 'inputs.npz') != contract['input_sha256']:
        raise ValueError('Old inputs.npz fingerprint mismatch')
    a = np.load(OLD_ROOT / 'inputs.npz', allow_pickle=False)
    n = int(a['y'].shape[0])
    concept = np.zeros((n, 192), dtype=np.float32)
    node_ids, node_ptr = a['node_ids'], a['node_ptr']
    for i in range(n):
        ids = node_ids[node_ptr[i]:node_ptr[i + 1]]
        concept[i, ids[ids < 192]] = 1.0
    X = np.concatenate([concept, a['hub'].astype(np.float32)], axis=1)
    names = [f'concept[{i}]' for i in range(192)] + list(contract['hub_fields'])
    return X, a['y'].astype(np.int64), a['folds'].astype(np.int64), names, recorded


def old_blocks(names):
    hub = names[192:]
    idx = lambda pred: np.array([192 + i for i, f in enumerate(hub) if pred(f)], dtype=np.int64)
    # Native concept slots 0:192 are 104 medication nodes then 88 chief-complaint nodes
    # (order fixed by xgboost_native_baseline._concept_names).
    concepts = np.arange(192)
    meds = np.arange(104)
    complaints = np.arange(104, 192)
    demo = idx(lambda f: f.startswith(('gender_', 'race_', 'transport_')) or f in ('age', 'bmi'))
    hx = idx(lambda f: f.startswith('hx_'))
    labs = idx(lambda f: f.startswith('lab_'))
    medclass = idx(lambda f: f.startswith('medclass_'))
    counts = idx(lambda f: f in ('n_ed_visits', 'n_medications'))
    vitals = idx(lambda f: f.startswith('vs_'))
    return {'concepts': concepts, 'meds': meds, 'complaints': complaints, 'demo': demo,
            'hx': hx, 'labs': labs, 'medclass': medclass, 'counts': counts, 'vitals': vitals}


# --------------------------------------------------------------------------- new
def load_new():
    if sha(NEW_FEATURES / 'features.npz') != NEW_FEATURES_SHA:
        raise ValueError('New features.npz fingerprint mismatch')
    d = np.load(NEW_FEATURES / 'features.npz', allow_pickle=False)
    names = json.loads((NEW_FEATURES / 'feature_names.json').read_text())
    if d['X'].shape[1] != len(names):
        raise ValueError('New feature name/width mismatch')
    return (d['X'].astype(np.float32, copy=False), d['y'].astype(np.int64),
            d['folds'].astype(np.int64), names, NEW_FEATURES_SHA)


def new_blocks(names):
    arr = np.asarray(names)
    pick = lambda p: np.flatnonzero(np.char.startswith(arr, p)).astype(np.int64)
    return {'concept_present': pick('concept_present::'),
            'event_stats': pick('event::'),
            'edge_stats': pick('edge::'),
            'node_counts': pick('node::'),
            'coverage': pick('coverage::'),
            'knowledge': pick('knowledge')}


# --------------------------------------------------------------------------- cell
def run_cell(tag, X, y, folds, cols, policy, train_limit, seed, rounds, out):
    t0 = time.monotonic()
    tr = np.flatnonzero(folds == 0)
    va = np.flatnonzero(folds == 1)
    if train_limit is not None:
        rng = np.random.default_rng(seed)
        tr = np.sort(rng.choice(tr, size=train_limit, replace=False))
    cols = np.sort(np.asarray(cols, dtype=np.int64))
    Xtr, Xva = X[np.ix_(tr, cols)], X[np.ix_(va, cols)]
    ytr, yva = y[tr], y[va]
    w = class_weights(y[folds == 0], policy)  # full-train counts, training rows only
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=w[ytr])
    dva = xgb.DMatrix(Xva, label=yva)
    booster = xgb.train({**PARAMS, 'seed': seed}, dtr, num_boost_round=rounds)
    proba = booster.predict(dva)
    row = {'cell': tag, 'policy': policy, 'seed': seed, 'rounds': rounds,
           'n_features': int(len(cols)), 'n_train': int(len(tr)), 'n_val': int(len(va)),
           'train_limit': train_limit, 'metrics': metrics(yva, proba),
           'seconds': round(time.monotonic() - t0, 1),
           'proba_sha256': hashlib.sha256(np.ascontiguousarray(proba).tobytes()).hexdigest()}
    with out.open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', choices=['old', 'new'], required=True)
    ap.add_argument('--cells', default='all')
    ap.add_argument('--rounds', type=int, default=ROUNDS)
    ap.add_argument('--policy', default='sqrt_inverse')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', default=str(HERE / 'results.jsonl'))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)['cell'] for l in out.read_text().splitlines() if l.strip()}

    if args.family == 'old':
        X, y, folds, names, fp = load_old()
        b = old_blocks(names)
        allcols = np.arange(X.shape[1])
        plan = [
            ('old/full', allcols, None),
            ('old/no_hx', np.setdiff1d(allcols, b['hx']), None),
            ('old/no_hx_no_medclass', np.setdiff1d(allcols, np.union1d(b['hx'], b['medclass'])), None),
            ('old/hx_only', b['hx'], None),
            ('old/labs_only', b['labs'], None),
            ('old/concepts_only', b['concepts'], None),
            ('old/complaints_only', b['complaints'], None),
            ('old/meds_only', b['meds'], None),
            ('old/no_complaints', np.setdiff1d(allcols, b['complaints']), None),
            ('old/labs_only_train6000', b['labs'], 6000),
            ('old/demo_labs_vitals', np.concatenate([b['demo'], b['labs'], b['vitals']]), None),
            ('old/full_train6000', allcols, 6000),
        ]
    else:
        X, y, folds, names, fp = load_new()
        b = new_blocks(names)
        allcols = np.arange(X.shape[1])
        struct = np.concatenate([b['edge_stats'], b['node_counts'], b['coverage'], b['knowledge']])
        plan = [
            ('new/full', allcols, None),
            ('new/concept_present_only', b['concept_present'], None),
            ('new/event_stats_only', b['event_stats'], None),
            ('new/structure_only', struct, None),
            ('new/full_train6000', allcols, 6000),
        ]
    print(json.dumps({'family': args.family, 'artifact_fingerprint': fp,
                      'X': list(X.shape), 'blocks': {k: int(len(v)) for k, v in b.items()},
                      'fold_counts': np.bincount(folds, minlength=3).tolist()}), flush=True)
    wanted = None if args.cells == 'all' else set(args.cells.split(','))
    for tag, cols, limit in plan:
        if wanted is not None and tag not in wanted:
            continue
        if tag in done:
            print(json.dumps({'cell': tag, 'status': 'already_present_skipped'}), flush=True)
            continue
        run_cell(tag, X, y, folds, cols, args.policy, limit, args.seed, args.rounds, out)


if __name__ == '__main__':
    main()
