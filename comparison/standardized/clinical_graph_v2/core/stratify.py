"""Stratify validation predictions by graph shape, on the SAME fold and order.

The question this answers: are richer graphs -- multi-visit patients, more prior
diagnoses, more baseline links -- predicted better, and if so is that the GRAPH
paying off or just an easier cohort?

Two rules make the answer trustworthy:

1. **The row order is verified, never assumed.** Match stored visit sample_ids,
   subjects and targets after replaying the run's label/history filters. Legacy
   repeated-patient arrays without sample_ids are ambiguous and are refused.

2. **Every stratum is quoted against its own baselines.** A stratum where the GNN
   scores higher may simply hold easier classes, so each row reports that stratum's
   own majority-class accuracy and, when available, the tabular control's accuracy
   on identical, verified rows. These are descriptive associations, not causal
   evidence for graph structure, even when the GNN-minus-XGBoost gap is positive.

Usage:
    python3 -m comparison.standardized.clinical_graph_v2.stratify \\
        --run comparison/standardized/clinical_runs_v3/main_seed1234 \\
        --control comparison/standardized/clinical_runs_v3/xgb_control
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .tensorize import iter_graphs
from .schema import sha256
from .train import load_targets, select_top_labels

CACHE_NAME = 'graph_shape_cache.json'


def extract_shapes(artifact, targets, cache_dir):
    """Cache by graph bytes and target mapping; never reuse an unbound legacy cache."""
    graph_path = Path(artifact) / 'graphs.jsonl'
    binding = {'schema_version': 2, 'graphs_sha256': sha256(graph_path),
               'targets_sha256': hashlib.sha256(json.dumps(
                   targets, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / f'{Path(CACHE_NAME).stem}_{key}.json'
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get('binding') != binding:
            raise ValueError('Shape cache binding mismatch')
        return cached
    order, shapes = [], {}
    for graph in iter_graphs(Path(artifact) / 'graphs.jsonl'):
        sid = graph['sample_id']
        entry = targets.get(sid)
        if entry is None or entry[1] != 'validation':
            continue
        if sid in shapes:
            raise ValueError('Duplicate sample_id in validation graph artifact')
        cov = graph['coverage']
        order.append(sid)
        shapes[sid] = {
            'subject': entry[2],
            'target': entry[0],
            'nodes': len(graph['nodes']),
            'edges': len(graph['edges']),
            'prior_visits': cov['prior_visits'],
            'complaints': cov['complaints'],
            'index_measurements': cov['index_measurements'],
            'baseline_links': cov['baseline_links'],
            'prior_diagnoses': cov.get('prior_diagnoses', 0),
            'informative_edges': cov['informative_edges'],
        }
    out = {'binding': binding, 'order': order, 'shapes': shapes}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open('x') as stream:
        json.dump(out, stream)
    return out


def verify_alignment(order, shapes, stored_subjects, stored_sample_ids=None, stored_y=None):
    """Require exact visit identity; subject-only legacy rows must be unique."""
    if len(set(order)) != len(order):
        raise ValueError('Duplicate sample_ids in replayed cohort')
    if len(order) != len(stored_subjects):
        raise ValueError(f'row count mismatch: replayed {len(order)} vs stored '
                         f'{len(stored_subjects)} -- alignment unsafe')
    if stored_sample_ids is None:
        if len(set(map(str, stored_subjects))) != len(stored_subjects):
            raise ValueError('Legacy repeated-patient predictions have no sample_ids; '
                             'visit alignment cannot be proven')
    elif list(map(str, stored_sample_ids)) != list(order):
        raise ValueError('sample_ids do not match the replayed visit sequence')
    replayed = [shapes[s]['subject'] for s in order]
    bad = [i for i, (a, b) in enumerate(zip(replayed, stored_subjects)) if a != str(b)]
    if bad:
        raise ValueError(f'subject mismatch at {len(bad)} rows -- '
                         'alignment unsafe, refusing to report')
    if stored_y is not None and not np.array_equal(
            [shapes[s]['target'] for s in order], stored_y):
        raise ValueError('Targets do not match the replayed visit sequence')
    return True


def validate_partition(bins, size):
    masks = np.asarray([mask for _, mask in bins], dtype=bool)
    if masks.shape != (len(bins), size) or not np.all(masks.sum(axis=0) == 1):
        raise ValueError('Strata must cover every visit exactly once')


def count_bins(key, values, boundaries):
    """Zero plus disjoint half-open positive intervals; boundaries are starts."""
    values = np.asarray(values)
    if (values.ndim != 1 or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values != np.floor(values))):
        raise ValueError('Graph counts must be finite nonnegative integers')
    starts = sorted({1, *(int(v) for v in boundaries if v > 0)})
    bins = [(f'{key}=0', values == 0)]
    for lo, hi in zip(starts, starts[1:] + [None]):
        label = f'{key}={lo}-{hi - 1}' if hi is not None else f'{key}>={lo}'
        bins.append((label, (values >= lo) & (values < hi) if hi is not None
                     else values >= lo))
    validate_partition(bins, len(values))
    return bins


def stratum_row(name, mask, y, pred, control_pred, num_classes=30):
    from sklearn.metrics import f1_score
    n = int(mask.sum())
    if n == 0:
        return None
    ys, ps = y[mask], pred[mask]
    counts = np.bincount(ys, minlength=num_classes)
    row = {
        'stratum': name, 'n': n,
        'acc': float((ps == ys).mean()),
        'macro_f1': float(f1_score(ys, ps, labels=np.arange(num_classes),
                                  average='macro', zero_division=0)),
        'classes_present': int((counts > 0).sum()),
        # A stratum holding easier classes scores higher for free; this is the
        # floor that difference has to clear.
        'majority_acc': float(counts.max() / n),
    }
    row['lift'] = row['acc'] - row['majority_acc']
    if control_pred is not None:
        row['xgb_acc'] = float((control_pred[mask] == ys).mean())
        row['gnn_minus_xgb'] = row['acc'] - row['xgb_acc']
    return row


def load_run_binding(run):
    path = Path(run) / 'binding.json'
    if path.exists():
        return json.loads(path.read_text())
    return json.loads((Path(run) / 'result.json').read_text())['binding']


def correlation(values, outcome):
    if len(values) < 2 or np.std(values) == 0 or np.std(outcome) == 0:
        return float('nan')
    return float(np.corrcoef(values, outcome)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, help='run dir holding validation.npz')
    p.add_argument('--control', help='tabular control run dir (same fold/order)')
    p.add_argument('--artifact', help='defaults to the run-bound artifact')
    p.add_argument('--targets', help='defaults to the run-bound targets_path')
    p.add_argument('--cache-dir', help='defaults to a versioned cache in the run directory')
    a = p.parse_args()

    run = Path(a.run)
    binding = load_run_binding(run)
    artifact = a.artifact or binding.get('artifact')
    targets_path = a.targets or binding.get('targets_path')
    if not artifact or not targets_path:
        raise ValueError('Specify --artifact/--targets when missing from the run binding')
    if binding.get('targets_sha256') != sha256(targets_path):
        raise ValueError('Target sidecar differs from the run binding')
    d = np.load(run / 'validation.npz', allow_pickle=False)
    y, proba = d['y'], d['proba']
    pred = proba.argmax(1)

    targets = load_targets(targets_path)
    if binding.get('top_k_labels') is not None:
        targets, kept, _ = select_top_labels(targets, binding['top_k_labels'])
        if kept != binding.get('kept_label_indices'):
            raise ValueError('Replayed label order differs from the run binding')
    cached = extract_shapes(artifact, targets, a.cache_dir or run / 'analysis_cache')
    if cached['binding']['graphs_sha256'] != binding.get('artifact_graphs_sha256'):
        raise ValueError('Graph bytes differ from the run binding')
    order, shapes = cached['order'], cached['shapes']
    order = [sid for sid in order if shapes[sid]['prior_visits']
             >= binding.get('min_prior_visits', 0)]
    verify_alignment(order, shapes, d['subjects'], d.get('sample_ids'), y)
    print(f'alignment verified: {len(order)} validation rows, '
          f'visit identity, subject and target sequence match\n')

    control_pred = None
    if a.control:
        control_binding = load_run_binding(a.control)
        for key in ('artifact_graphs_sha256', 'targets_sha256', 'num_classes',
                    'kept_label_indices', 'min_prior_visits', 'evaluation_version',
                    'input_contract_version', 'preprocessing_sha256'):
            if (binding.get(key) != control_binding.get(key)
                    or (key == 'preprocessing_sha256' and not binding.get(key))):
                raise ValueError(f'Control binding differs on {key}; comparison refused')
        c = np.load(Path(a.control) / 'validation.npz', allow_pickle=False)
        verify_alignment(order, shapes, c['subjects'], c.get('sample_ids'), c['y'])
        if c['proba'].shape != proba.shape or not np.array_equal(c['y'], y):
            raise ValueError('Control probability shape or target sequence differs')
        control_pred = c['proba'].argmax(1)

    feats = {k: np.array([shapes[s][k] for s in order])
             for k in ('prior_visits', 'nodes', 'edges', 'complaints',
                       'index_measurements', 'baseline_links', 'prior_diagnoses',
                       'informative_edges')}

    def table(title, bins):
        validate_partition(bins, len(y))
        print(f'=== {title} ===')
        head = f"{'katman':28}{'n':>7}{'acc':>8}{'mF1':>8}{'cogunluk':>10}{'kazanc':>8}"
        if control_pred is not None:
            head += f"{'xgb':>8}{'GNN-XGB':>9}"
        print(head)
        for name, mask in bins:
            r = stratum_row(name, mask, y, pred, control_pred, proba.shape[1])
            if r is None:
                continue
            line = (f"{r['stratum']:28}{r['n']:>7}{r['acc']:>8.4f}{r['macro_f1']:>8.4f}"
                    f"{r['majority_acc']:>10.4f}{r['lift']:>8.4f}")
            if control_pred is not None:
                line += f"{r['xgb_acc']:>8.4f}{r['gnn_minus_xgb']:>+9.4f}"
            print(line)
        print()

    pv = feats['prior_visits']
    table('GECMIS ZIYARET SAYISI (asil soru: multi-visit daha mi iyi)',
          [('0 (tek ziyaret)', pv == 0), ('1', pv == 1), ('2', pv == 2),
           ('3-5', (pv >= 3) & (pv <= 5)), ('6+', pv >= 6)])
    table('TEK vs COKLU ZIYARET',
          [('tek ziyaret', pv == 0), ('coklu ziyaret (1+)', pv >= 1)])

    for key, title, edges in (
        ('baseline_links', 'BASELINE_OF BAGLANTISI (delta tasiyan kenar)', [0, 1, 4, 9]),
        ('prior_diagnoses', 'GECMIS TANI SAYISI', [0, 1, 3, 8]),
        ('complaints', 'SIKAYET SAYISI', [0, 1, 2, 3]),
        ('index_measurements', 'INDEKS OLCUM SAYISI', [0, 5, 9, 15]),
    ):
        table(title, count_bins(key, feats[key], edges))

    v = feats['nodes']
    q = np.quantile(v, [0.25, 0.5, 0.75])
    table('GRAF BUYUKLUGU (dugum sayisi ceyrekleri)',
          [(f'Q1 <={q[0]:.0f}', v <= q[0]),
           (f'Q2 {q[0]:.0f}-{q[1]:.0f}', (v > q[0]) & (v <= q[1])),
           (f'Q3 {q[1]:.0f}-{q[2]:.0f}', (v > q[1]) & (v <= q[2])),
           (f'Q4 >{q[2]:.0f}', v > q[2])])

    print('=== KORELASYON: graf ozelligi <-> dogru tahmin ===')
    correct = (pred == y).astype(float)
    print(f"{'ozellik':22}{'r (dogruluk)':>14}", end='')
    if control_pred is not None:
        print(f"{'r (GNN-XGB farki)':>20}", end='')
    print()
    cdiff = correct - (control_pred == y).astype(float) if control_pred is not None else None
    for k, v in feats.items():
        r = correlation(v, correct)
        line = f'{k:22}{r:>14.3f}'
        if cdiff is not None:
            line += f'{correlation(v, cdiff):>20.3f}'
        print(line)
    print('\n  Korelasyonlar betimseldir; pozitif veya negatif GNN-XGB korelasyonu '
          'graf yapisinin nedensel katkisini kanitlamaz. Eslenmis kontroller ve '
          'hasta-kumeli belirsizlik analizi gerekir.')


if __name__ == '__main__':
    main()
