"""Fair-protocol comparison: any number of methods, ranked by mean selected DEV macro-F1.

Only runs of the `fair_v1` protocol (see `core.train.FAIR_PROTOCOL`) with no protocol
overrides are accepted. The validation fold is read from each run's single validation
evaluation: every method's mean +- SD is reported and the patient-cluster paired bootstrap
covers all pairs among the top 3 methods by dev.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from core.contracts import sample_ids_sha256
from core.train import FAIR_PROTOCOL, PROTOCOL_NAME

K = 10
N_BOOT = 1000
SEED = 2026
TOP_N = 3
FOLDS = ('train', 'dev', 'validation')
CAVEAT = ('Fair protocol fair_v1: one fixed 10k train sample and 5k dev draw; ranking uses the '
          'dev macro-F1 selected per run, CI is conditional on these training seeds; '
          'validation fold only (single evaluation per run); test fold never evaluated.')


class Refusal(Exception):
    pass


def macro_f1_from_conf(conf):
    tp = np.diagonal(conf, axis1=-2, axis2=-1).astype(np.float64)
    fp = conf.sum(axis=-2) - tp
    fn = conf.sum(axis=-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return f1.mean(axis=-1)


def point_metrics(proba, y):
    pred = proba.argmax(1)
    conf = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    recall = np.diagonal(conf) / np.maximum(conf.sum(1), 1)
    present = conf.sum(1) > 0
    top3 = np.argsort(-proba, axis=1)[:, :3]
    return {
        'macro_f1': float(macro_f1_from_conf(conf)),
        'accuracy': float((pred == y).mean()),
        'balanced_accuracy': float(recall[present].mean()),
        'top3_accuracy': float((top3 == y[:, None]).any(1).mean()),
    }


def load_run(path):
    path = Path(path)
    for name in ('result.json', 'binding.json', 'validation.npz'):
        if not (path / name).is_file():
            raise Refusal(f'{path}: missing {name}')
    result = json.loads((path / 'result.json').read_text())
    binding = json.loads((path / 'binding.json').read_text())
    if result.get('status') != 'completed':
        raise Refusal(f"{path}: status is {result.get('status')!r}, not 'completed'")
    if binding.get('test_evaluated') is not False:
        raise Refusal(f'{path}: binding test_evaluated is not false')
    if binding.get('protocol') != PROTOCOL_NAME:
        raise Refusal(f"{path}: protocol is {binding.get('protocol')!r}, not {PROTOCOL_NAME!r}")
    if binding.get('protocol_overrides') != []:
        raise Refusal(f"{path}: protocol_overrides is {binding.get('protocol_overrides')!r}, "
                      'not an empty list')
    if binding.get('protocol_resolved') != FAIR_PROTOCOL:
        raise Refusal(f"{path}: protocol_resolved {binding.get('protocol_resolved')!r} "
                      f'differs from the fair protocol {FAIR_PROTOCOL!r}')
    hashes = binding.get('split_sample_ids_sha256')
    if not isinstance(hashes, dict) or any(not hashes.get(f) for f in FOLDS):
        raise Refusal(f'{path}: split_sample_ids_sha256 lacks train, dev or validation')
    dev = (result.get('dev_metrics') or {}).get('macro_f1')
    if result.get('selection_fold') != 'dev' or not isinstance(dev, (int, float)):
        raise Refusal(f'{path}: result.json has no dev-selected macro_f1')
    method = binding['method']
    if method == 'clinical_gnn':
        method = f"{method}:{binding.get('conv')}"
    z = np.load(path / 'validation.npz')
    return {'path': str(path), 'method': method, 'seed': binding['seed'], 'binding': binding,
            'dev_macro_f1': float(dev),
            'result': result, 'proba': z['proba'].astype(np.float64), 'y': z['y'].astype(np.int64),
            'subjects': z['subjects'], 'sample_ids': z['sample_ids']}


def check_and_align(runs):
    ref = runs[0]
    for key in ('split_sample_ids_sha256', 'kept_label_indices', 'artifact_graphs_sha256',
                'targets_sha256'):
        for r in runs:
            if r['binding'].get(key) != ref['binding'].get(key):
                raise Refusal(f"{key} differs between {ref['path']} and {r['path']}")
    for r in runs:
        if len(set(r['sample_ids'])) != len(r['sample_ids']):
            raise Refusal(f"{r['path']}: duplicate validation sample_ids")
    seen = set()
    for r in runs:
        ident = (r['method'], r['seed'])
        if ident in seen:
            raise Refusal(f'duplicate method/seed {ident}')
        seen.add(ident)
    if all(np.array_equal(r['sample_ids'], ref['sample_ids']) for r in runs):
        return
    if any(set(r['sample_ids']) != set(ref['sample_ids']) for r in runs):
        raise Refusal('validation sample_id sets differ between runs')
    order = {s: i for i, s in enumerate(ref['sample_ids'])}
    for r in runs:
        idx = np.argsort([order[s] for s in r['sample_ids']])
        for k in ('proba', 'y', 'subjects', 'sample_ids'):
            r[k] = r[k][idx]
    for r in runs:
        if not (np.array_equal(r['y'], ref['y']) and np.array_equal(r['subjects'], ref['subjects'])):
            raise Refusal('y/subjects disagree after alignment by sample_id')


def rank_by_dev(runs, methods):
    """Methods by mean selected dev macro-F1 over their seeds, best first."""
    mean = {m: float(np.mean([r['dev_macro_f1'] for r in runs if r['method'] == m]))
            for m in methods}
    return sorted(methods, key=lambda m: (-mean[m], m)), mean


def bootstrap(runs, methods, ref):
    subj, inv = np.unique(ref['subjects'], return_inverse=True)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(subj), size=(N_BOOT, len(subj)))
    counts = np.stack([np.bincount(d, minlength=len(subj)) for d in draws])  # [B, S]
    row_w = counts[:, inv].astype(np.float64)  # [B, N]
    y = ref['y']
    per_method = {m: [] for m in methods}
    for r in runs:
        idx = y * K + r['proba'].argmax(1)
        conf = np.stack([np.bincount(idx, weights=w, minlength=K * K) for w in row_w])
        per_method[r['method']].append(macro_f1_from_conf(conf.reshape(N_BOOT, K, K)))
    return {m: np.mean(v, axis=0) for m, v in per_method.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog='comparison.top3.compare')
    ap.add_argument('--run', action='append', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    try:
        out = Path(args.out)
        if out.exists():
            raise Refusal(f'--out already exists: {out}')
        runs = [load_run(p) for p in args.run]
        check_and_align(runs)
        for r in runs:
            r['metrics'] = point_metrics(r['proba'], r['y'])
            rep = r['result']['metrics']['macro_f1']
            if abs(r['metrics']['macro_f1'] - rep) > 1e-6:
                raise Refusal(f"{r['path']}: recomputed macro_f1 {r['metrics']['macro_f1']:.6f} "
                              f'!= result.json {rep}')
    except Refusal as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return 2

    methods = list(dict.fromkeys(r['method'] for r in runs))
    names = ('macro_f1', 'accuracy', 'balanced_accuracy', 'top3_accuracy')
    summary = {}
    for m in methods:
        rs = [r for r in runs if r['method'] == m]
        summary[m] = {'n_seeds': len(rs), 'seeds': [r['seed'] for r in rs]}
        for n in names:
            v = np.array([r['metrics'][n] for r in rs])
            summary[m][n] = {'mean': float(v.mean()),
                             'sd': float(v.std(ddof=1)) if len(v) > 1 else None}

    ranking, dev_mean = rank_by_dev(runs, methods)
    for m in methods:
        dev = np.array([r['dev_macro_f1'] for r in runs if r['method'] == m])
        summary[m]['dev_macro_f1'] = {'mean': dev_mean[m],
                                      'sd': float(dev.std(ddof=1)) if len(dev) > 1 else None}
    top = ranking[:TOP_N]
    ref = runs[0]
    boot = bootstrap([r for r in runs if r['method'] in top], top, ref)
    pairs = []
    for a, b in itertools.combinations(top, 2):
        d = boot[a] - boot[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        pairs.append({'a': a, 'b': b,
                      'observed_delta_macro_f1': summary[a]['macro_f1']['mean'] - summary[b]['macro_f1']['mean'],
                      'ci95': [float(lo), float(hi)], 'share_delta_gt_0': float((d > 0).mean())})

    doc = {
        'inputs': [{'run': r['path'], 'method': r['method'], 'seed': r['seed'],
                    **{k: r['binding'].get(k) for k in (
                        'split_sample_ids_sha256', 'kept_label_indices', 'artifact_graphs_sha256',
                        'targets_sha256')}} for r in runs],
        'validation_sample_ids_sha256': sample_ids_sha256(ref['sample_ids'].tolist()),
        'n_validation_rows': int(len(ref['y'])),
        'n_validation_subjects': int(len(np.unique(ref['subjects']))),
        'per_run': [{'run': r['path'], 'method': r['method'], 'seed': r['seed'], **r['metrics']}
                    for r in runs],
        'protocol': PROTOCOL_NAME,
        'protocol_resolved': FAIR_PROTOCOL,
        'ranking_by_dev_macro_f1': [{'rank': i + 1, 'method': m, 'dev_macro_f1_mean': dev_mean[m]}
                                    for i, m in enumerate(ranking)],
        'bootstrap_methods': top,
        'per_method': summary,
        'pairs': pairs,
        'settings': {'bootstrap_resamples': N_BOOT, 'rng': f'numpy default_rng({SEED})',
                     'resampling_unit': 'subjects (all rows, with multiplicity)',
                     'ranking': 'mean selected dev macro-F1 over seeds, descending',
                     'bootstrap_scope': f'all pairs among the top {TOP_N} methods by dev',
                     'statistic': 'validation macro-F1 averaged over each method\'s seeds; delta = A - B',
                     'ci': '95% percentile', 'sd': 'sample SD over seeds (ddof=1)'},
        'caveat': CAVEAT,
    }
    out.mkdir(parents=True)
    (out / 'comparison.json').write_text(json.dumps(doc, indent=2))

    print('Ranking by mean selected DEV macro-F1')
    for i, m in enumerate(ranking):
        sd = summary[m]['dev_macro_f1']['sd']
        print(f"{i + 1:>3}. {m:<24}{dev_mean[m]:.4f} ± {sd if sd is not None else float('nan'):.4f}"
              + ('   (bootstrap)' if m in top else ''))
    print()
    print('Validation (single evaluation per run)')
    print(f"{'method':<24}{'seeds':>6}  {'macro-F1':>16}  {'accuracy':>16}  {'top-3':>16}")
    for m in ranking:
        s = summary[m]
        cell = lambda n: f"{s[n]['mean']:.4f} ± {s[n]['sd'] if s[n]['sd'] is not None else float('nan'):.4f}"
        print(f"{m:<24}{s['n_seeds']:>6}  {cell('macro_f1'):>16}  {cell('accuracy'):>16}  {cell('top3_accuracy'):>16}")
    print()
    print(f'Paired validation bootstrap, all pairs among the top {len(top)} by dev')
    print(f"{'A vs B':<38}{'delta':>9}  {'95% CI':>20}  {'P(d>0)':>7}")
    for p in pairs:
        print(f"{p['a'] + ' - ' + p['b']:<38}{p['observed_delta_macro_f1']:>+9.4f}  "
              f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]  {p['share_delta_gt_0']:>7.3f}")
    print('\n' + CAVEAT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
