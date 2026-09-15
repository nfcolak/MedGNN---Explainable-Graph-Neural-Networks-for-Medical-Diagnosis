"""Build/load a NEW enriched source-snapshot NPZ; legacy inputs remain immutable."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from pna_analysis.data import load_common_input, train_degree_histogram
from shared.lib.benchmark_contract import file_sha256
from .features import align_source, fit_transform
from .audit import audit_source
from .spec import SPEC, HISTORY

PROJECT = Path(__file__).resolve().parents[3]
SCHEMA = 'medgnn.enriched-source-snapshot.v1'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def code_hashes():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [PROJECT / n for n in (
        'pna_analysis/model.py', 'pna_analysis/data.py', 'pna_analysis/train.py',
        'shared/lib/metrics.py', 'shared/lib/benchmark_contract.py',
        'shared/lib/graph_structures.py', 'shared/data_prep/merge_ed.py',
        'shared/data_prep/extract_ed_labs.py', 'graphcare_analysis/adapter.py',
        'comparison/standardized/common_input_improvement.py',
        'comparison/standardized/performance_review.py', 'shared/lib/graphxai_standardized.py',
        'shared/lib/explanation_contract.py', 'shared/lib/fidelity.py',
        'gsat_analysis/train.py', 'gsat_analysis/config.py',
        'protgnn_analysis/config.py', 'protgnn_analysis/load_dataset.py')]
    files += list((PROJECT / 'protgnn_analysis/models').glob('*.py'))
    files += list((PROJECT / 'gsat_analysis/models').glob('*.py'))
    files += [PROJECT / 'external/GraphXAI-main/graphxai/explainers' / n
              for n in ['grad.py', 'integrated_grad.py', 'gnn_explainer.py', '_base.py']]
    return {str(p.relative_to(PROJECT)): file_sha256(p) for p in sorted(files)}


def snapshot_code(output, hashes):
    for name, digest in hashes.items():
        destination = output / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, destination)
        if file_sha256(destination) != digest:
            raise ValueError('Source changed during snapshot')


def build_input(legacy_root, source, split_path, raw_root, output, execute=False):
    output, raw_root = Path(output), Path(raw_root)
    if execute and output.exists():
        raise FileExistsError('Enriched output must be new; overwrite is forbidden')
    legacy = load_common_input(legacy_root, source, split_path)
    frame = pd.read_csv(source, low_memory=False)
    split = json.loads(Path(split_path).read_text())
    frame, row_hash = align_source(frame, split, legacy)
    # Independently bind each original concept to these source rows, not just fold totals.
    cc = [c for c in frame if c.startswith('chiefcomplaint_')]
    for j, name in enumerate(legacy['contract']['names']):
        if name.startswith('med:'):
            values = pd.to_numeric(frame['med_' + name[4:]], errors='coerce').fillna(0).to_numpy() > 0
        else:
            values = frame[cc].eq(name[3:]).any(axis=1).to_numpy()
        if not np.array_equal(values, legacy['presence'][:, j]):
            raise ValueError('Canonical concept/source row mismatch')
    absent = [f['source'] for f in SPEC if f['source'] not in frame]
    for name in absent + [c for c in ['gender_F', 'gender_M'] if c not in frame]:
        frame[name] = np.nan
    frame, timing = audit_source(frame, raw_root, ['hx_' + n for n in HISTORY])
    numeric, recipe, coverage = fit_transform(frame, legacy['folds'], SPEC)
    recipe.update({'spec': SPEC, 'sex_source_columns': ['gender_F', 'gender_M'],
                   'history_vocabulary': 'fixed audited source columns; no target-conditioned features',
                   'numeric_imputation': 'observed training mean, all-missing train => 0',
                   'scale': 'observed training population std; constant/all-missing => 1'})
    preservation = dict(legacy['preservation'])
    for name, digest in timing.get('raw_hashes', {}).items():
        preservation[str((raw_root / name).resolve())] = digest
    # Read-only binding of previous results and checkpoints, not just the old NPZ.
    for root in [Path(legacy_root), PROJECT / 'comparison/standardized/pna_experiments']:
        if root.exists():
            for p in root.rglob('*'):
                if p.is_file() and p.suffix in {'.pt', '.pth', '.npz', '.json'} and 'source_snapshot' not in p.parts:
                    preservation[str(p.resolve())] = file_sha256(p)
    train = legacy['folds'] == 0
    manifest = {'schema': SCHEMA, 'status': 'dry_run',
                'view': 'medication-history/labs-available SOURCE-SNAPSHOT diagnostic',
                'temporal_clean': False, 'raw_to_model_train_only': False,
                'early_triage_eligible': False, 'test_evaluated': False,
                'counts': legacy['contract']['counts'], 'rows': len(frame),
                'ordered_subject_sha256': row_hash,
                'train_subject_sha256': hashlib.sha256('\n'.join(sorted(frame.loc[train, 'subject_id'].astype(str))).encode()).hexdigest(),
                'base_contract': legacy['contract'], 'recipe': recipe,
                'recipe_sha256': digest_json(recipe), 'coverage': coverage,
                'timing_audit': timing, 'absent_source_fields': absent,
                'numeric_channels': numeric.shape[1], 'constant_hub_channels': 1,
                'hub_only_by_fold': [int(((legacy['presence'].sum(1) == 0) & (legacy['folds'] == f)).sum()) for f in range(3)],
                'fit_rows': int(train.sum()), 'all_values_finite': bool(np.isfinite(numeric).all()),
                'preprocessing_fit_train_only': True,
                'eligibility': {'initial_measurements': list(['temperature', 'heartrate', 'o2sat', 'sbp', 'dbp']),
                                'age': 'anchor age; exact index age not reconstructed',
                                'sex': 'source sex coding F/M/unknown, not gender identity inference',
                                'labs': 'later_or_unknown; no first/early-result claim; source units unverified',
                                'history': 'prior-ended-stay corroborated positives, documentation time unverified',
                                'concepts': 'existing medication/complaint snapshot eligibility unchanged'},
                'integration': {'pna': 'opt-in numerical hub encoder and bounded train/replay',
                                'protgnn': 'opt-in shared PyG numeric adapter; forward/backward verified, full run pending',
                                'gsat': 'opt-in shared PyG numeric adapter; forward/backward verified, full run pending',
                                'graphcare': 'categorical baseline unchanged; numeric consumer pending'},
                'source_paths': {'legacy_root': str(Path(legacy_root).resolve()), 'source': str(Path(source).resolve()),
                                 'split': str(Path(split_path).resolve()), 'raw_root': str(raw_root.resolve())},
                'preservation': preservation, 'code_sha256': code_hashes()}
    if not execute:
        return manifest
    output.mkdir(parents=True, exist_ok=False)
    manifest['status'] = 'building'
    save_json(output / 'manifest.json', manifest)
    try:
        np.savez_compressed(output / 'inputs.npz', **{k: legacy[k] for k in ['presence', 'y', 'folds']}, hub_numeric=numeric)
        manifest['npz_sha256'] = file_sha256(output / 'inputs.npz')
        snapshot_code(output, manifest['code_sha256'])
        after = {p: file_sha256(p) for p in preservation}
        if after != preservation:
            raise ValueError('Protected input/result changed during build')
        manifest['preservation_verified'] = True
        manifest['status'] = 'completed'
        manifest['binding_sha256'] = digest_json(manifest)
        save_json(output / 'manifest.json', manifest)
        load_enriched(output)
    except BaseException:
        manifest['status'] = 'failed'
        save_json(output / 'manifest.json', manifest)
        raise
    return manifest


def load_enriched(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    binding = manifest.pop('binding_sha256', None)
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'completed' or digest_json(manifest) != binding:
        raise ValueError('Enriched manifest binding mismatch')
    manifest['binding_sha256'] = binding
    if digest_json(manifest['recipe']) != manifest['recipe_sha256'] or manifest['recipe']['spec'] != SPEC:
        raise ValueError('Feature recipe/spec mismatch')
    expected_names = [name for f in SPEC for name in [f['source'], f['source'] + '__missing']]
    expected_names += ['sex_F', 'sex_M', 'sex_unknown']
    if manifest['recipe']['names'] != expected_names:
        raise ValueError('Ordered feature allowlist mismatch')
    if file_sha256(root / 'inputs.npz') != manifest['npz_sha256']:
        raise ValueError('Enriched NPZ checksum mismatch')
    for name, digest in manifest['code_sha256'].items():
        if file_sha256(root / 'source_snapshot' / name) != digest:
            raise ValueError('Input source snapshot checksum mismatch')
    paths = manifest['source_paths']
    legacy = load_common_input(paths['legacy_root'], paths['source'], paths['split'])
    with np.load(root / 'inputs.npz', allow_pickle=False) as data:
        if set(data.files) != {'presence', 'y', 'folds', 'hub_numeric'}:
            raise ValueError('Unexpected enriched payload')
        result = {k: data[k].copy() for k in data.files}
    for key in ['presence', 'y', 'folds']:
        if not np.array_equal(result[key], legacy[key]):
            raise ValueError('Enriched canonical input mismatch')
    if result['hub_numeric'].shape != (len(result['y']), len(expected_names)) or not np.isfinite(result['hub_numeric']).all():
        raise ValueError('Invalid numeric payload')
    result.update({'manifest': manifest, 'contract': manifest['base_contract'],
                   'degree_histogram': train_degree_histogram(result['presence'], result['folds'])})
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root', type=Path, default=PROJECT / 'comparison/standardized/common_input_20260913')
    p.add_argument('--source', type=Path, default=PROJECT / 'data/merged_ed.csv')
    p.add_argument('--split', type=Path, default=PROJECT / 'comparison/canonical_split.json')
    p.add_argument('--raw-root', type=Path, default=PROJECT / 'data/Original CSVs')
    p.add_argument('--output-dir', type=Path, required=True)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    m = build_input(args.legacy_root, args.source, args.split, args.raw_root, args.output_dir, args.execute)
    print(json.dumps({k: m[k] for k in ['status', 'counts', 'numeric_channels', 'all_values_finite', 'temporal_clean']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
