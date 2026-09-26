"""Tabular control on the SAME v3 artifact and split.

If a gradient-boosted model on a flat feature row matches the GNN, the graph
machinery is not paying for itself and that is the finding. Features are built only
from information the graphs already contain: which tokens are present, their value
summaries, and counts of each relation -- i.e. the graph flattened.
"""
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

from .tensorize import ALL_RELATIONS, iter_graphs

NUM_CLASSES = 30


def load_targets(path):
    t = {}
    for r in csv.DictReader(Path(path).open()):
        if r['target'] and r['target'] != '-1':
            t[r['sample_id']] = (int(r['target']), r['split'])
    return t


def fit_vocab(artifact, targets, min_count=20):
    c = Counter()
    for g in iter_graphs(Path(artifact) / 'graphs.jsonl'):
        e = targets.get(g['sample_id'])
        if e is None or e[1] != 'train':
            continue
        for n in g['nodes']:
            c[n['token']] += 1
    return sorted(t for t, k in c.items() if k >= min_count)


def build(artifact, targets, tokens):
    tix = {t: i for i, t in enumerate(tokens)}
    rix = {r: i for i, r in enumerate(ALL_RELATIONS)}
    n_tok, n_rel = len(tokens), len(ALL_RELATIONS)
    # presence | mean value | has value | relation counts | coverage
    width = n_tok * 3 + n_rel + 6
    X, y, folds = [], [], []
    for g in iter_graphs(Path(artifact) / 'graphs.jsonl'):
        e = targets.get(g['sample_id'])
        if e is None or e[1] == 'test':
            continue
        row = np.zeros(width, dtype=np.float32)
        vals = defaultdict(list)
        for nd in g['nodes']:
            i = tix.get(nd['token'])
            if i is None:
                continue
            row[i] = 1.0
            if nd.get('value') is not None:
                vals[i].append(nd['value'])
        for i, v in vals.items():
            row[n_tok + i] = float(np.mean(v))
            row[2 * n_tok + i] = 1.0
        for ed in g['edges']:
            row[3 * n_tok + rix[ed['relation']]] += 1.0
        cov = g['coverage']
        row[3 * n_tok + n_rel:] = (
            cov['complaints'], cov['index_measurements'], cov['prior_visits'],
            cov.get('prior_diagnoses', 0), cov['informative_edges'], len(g['nodes']))
        X.append(row)
        y.append(e[0])
        folds.append(0 if e[1] == 'train' else 1)
    return np.asarray(X), np.asarray(y), np.asarray(folds)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', required=True)
    p.add_argument('--targets', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rounds', type=int, default=300)
    p.add_argument('--seed', type=int, default=1234)
    a = p.parse_args()

    t0 = time.monotonic()
    targets = load_targets(a.targets)
    tokens = fit_vocab(a.artifact, targets)
    X, y, folds = build(a.artifact, targets, tokens)
    tr, va = folds == 0, folds == 1
    print(json.dumps({'stage': 'built', 'X': list(X.shape),
                      'train': int(tr.sum()), 'val': int(va.sum()),
                      'seconds': round(time.monotonic() - t0, 1)}), flush=True)

    counts = np.bincount(y[tr], minlength=NUM_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    w = np.sqrt(len(y[tr]) / (NUM_CLASSES * counts))

    dtr = xgb.DMatrix(X[tr], label=y[tr], weight=w[y[tr]])
    dva = xgb.DMatrix(X[va], label=y[va])
    params = dict(objective='multi:softprob', num_class=NUM_CLASSES, tree_method='hist',
                  max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8,
                  reg_lambda=2.0, min_child_weight=3.0, nthread=8, seed=a.seed,
                  disable_default_eval_metric=1)
    booster = xgb.train(params, dtr, num_boost_round=a.rounds)
    proba = booster.predict(dva)

    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 top_k_accuracy_score)
    pred = proba.argmax(1)
    m = {'accuracy': round(float(accuracy_score(y[va], pred)), 6),
         'balanced_acc': round(float(balanced_accuracy_score(y[va], pred)), 6),
         'macro_f1': round(float(f1_score(y[va], pred, average='macro', zero_division=0)), 6),
         'top3_acc': round(float(top_k_accuracy_score(y[va], proba, k=3,
                                                      labels=np.arange(NUM_CLASSES))), 6)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'result.json').write_text(json.dumps(
        {'method': 'xgboost_tabular_control', 'metrics': m, 'features': int(X.shape[1]),
         'rounds': a.rounds, 'seed': a.seed, 'test_evaluated': False,
         'counts': {'train': int(tr.sum()), 'validation': int(va.sum())},
         'seconds': round(time.monotonic() - t0, 1)}, indent=2) + '\n')
    np.savez_compressed(out / 'validation.npz', proba=proba, y=y[va])
    print(json.dumps({'stage': 'completed', **m}), flush=True)


if __name__ == '__main__':
    main()
