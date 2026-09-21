"""Aggregate the v3 arms and decide which differences survive seed noise.

A single-seed table whose entries sit within a few thousandths of each other is an
ordering of noise. This script quotes every claimed gain as a multiple of the
measured seed standard deviation, and refuses to name a winner when the delta is
smaller than the spread.
"""
import glob
import json
import os
import statistics
from collections import defaultdict

import numpy as np

RUNS = 'comparison/standardized/clinical_runs_v3'

rows = {}
for path in sorted(glob.glob(f'{RUNS}/*/result.json')):
    name = os.path.basename(os.path.dirname(path))
    rows[name] = json.load(open(path))

arms = defaultdict(dict)
for name, r in rows.items():
    arm, seed = name.rsplit('_seed', 1)
    arms[arm][seed] = r

print('=== TEK KOSU SONUCLARI (validation, test ACILMADI) ===')
print(f"{'arm':12}{'seed':>6}{'param':>9}{'ep':>4}{'acc':>9}{'bal':>9}{'mF1':>9}{'top3':>9}")
for arm in sorted(arms):
    for seed in sorted(arms[arm]):
        r = arms[arm][seed]
        m, b = r['metrics'], r['binding']
        print(f"{arm:12}{seed:>6}{b['parameter_count']:>9}{r['selected_epoch']:>4}"
              f"{m['accuracy']:9.4f}{m['balanced_acc']:9.4f}{m['macro_f1']:9.4f}{m['top3_acc']:9.4f}")

params = {r['binding']['parameter_count'] for r in rows.values()}
print(f"\nparametre sayisi tum arm'larda esit mi: {len(params) == 1} ({params})")

print('\n=== SEED YAYILIMI ===')
spread = {}
for arm in sorted(arms):
    vals = [arms[arm][s]['metrics']['macro_f1'] for s in sorted(arms[arm])]
    if len(vals) < 2:
        print(f'  {arm:12} n={len(vals)} tek seed -- yayilim olculemedi')
        continue
    sd = statistics.stdev(vals)
    spread[arm] = sd
    print(f'  {arm:12} n={len(vals)} macro_f1 ort={statistics.mean(vals):.4f} '
          f'SS={sd:.4f} min={min(vals):.4f} max={max(vals):.4f}')

if spread:
    ref = max(spread.values())
    print(f'\n  olculen en buyuk seed SS = {ref:.4f}  -> bundan kucuk farklar GURULTU')

print('\n=== ARM KARSILASTIRMASI (macro_f1 ortalamasi) ===')
means = {a: statistics.mean([arms[a][s]['metrics']['macro_f1'] for s in arms[a]])
         for a in arms}
base = 'nomp'
if base in means:
    for arm in sorted(means):
        if arm == base:
            continue
        d = means[arm] - means[base]
        mult = abs(d) / ref if spread else float('nan')
        verdict = 'GURULTU' if mult < 1 else f'{mult:.1f}x SS'
        print(f'  {arm:12} - {base:10} = {d:+.4f}   {verdict}')

print('\n=== EN IYI ARM: SINIF BAZINDA ===')
best_name = max(rows, key=lambda n: rows[n]['metrics']['macro_f1'])
r = rows[best_name]
print(f'  {best_name} (mF1={r["metrics"]["macro_f1"]:.4f}, epoch {r["selected_epoch"]})')
pc = sorted(r['per_class'], key=lambda x: -x['f1'])
print(f'\n  {"en iyi 6 sinif":44}{"f1":>7}{"recall":>8}{"n":>7}')
for c in pc[:6]:
    print(f'    {c["label"][:42]:44}{c["f1"]:7.3f}{c["recall"]:8.3f}{c["support"]:7d}')
zero = [c for c in pc if c['f1'] == 0.0]
print(f'\n  F1 = 0 olan sinif: {len(zero)}/30')
for c in zero:
    print(f'    {c["label"][:42]:44}{"":7}{"":8}{c["support"]:7d}')
print(f'\n  cogunluk tabani accuracy: {r["majority_baseline_accuracy"]:.4f}')

# support vs f1 relationship
sup = np.array([c['support'] for c in r['per_class']])
f1 = np.array([c['f1'] for c in r['per_class']])
print(f'  destek-F1 korelasyonu (Pearson): {np.corrcoef(sup, f1)[0,1]:.3f}')
