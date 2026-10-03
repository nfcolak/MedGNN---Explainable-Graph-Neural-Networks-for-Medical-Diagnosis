"""One approved validation pass of the twelve frozen dev-selected checkpoints.

No flag is a read-only plan. --execute requires a fresh, ignored output root.
No training, selection, test deserialisation, or inference retries are available.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from . import cei_pilot as pilot
from . import cei_v3_screen as screen
from . import cei_v3_study as study
from comparisons.cei_vs_protgnn import cei_v3_vs_protgnn as comparison
from core import train
from core.contracts import code_source_hashes, sample_ids_sha256
from .cei_v3_ext.validation_scoring import (
    APPROVAL_FILENAME, approval_record_sha256, validate_approval_record,
    write_approval_record,
)
from core.registry import build_method
from core.schema import sha256

from . import cei_v3_paths as io_paths

REPO = io_paths.REPO
V3_ROOT = io_paths.DEFAULT_RESULTS / 'core'
P_ROOT = io_paths.DEFAULT_RESULTS / 'protgnn'
ARTIFACT = REPO / io_paths.ARTIFACT_REL
TARGETS = REPO / io_paths.TARGETS_REL
CANONICAL = REPO / 'comparison/canonical_split.json'
OUTPUT = REPO / 'comparison/standardized/clinical_runs_cei_v3_validation_20261001'
PATHS = None


def configure(paths, output=None):
    global PATHS, V3_ROOT, P_ROOT, ARTIFACT, TARGETS, CANONICAL, OUTPUT
    PATHS = paths
    comparison.configure(paths)
    V3_ROOT, P_ROOT = paths.v3_root, paths.protgnn_root
    ARTIFACT, TARGETS, CANONICAL = paths.artifact, paths.targets, paths.canonical
    if output is not None:
        OUTPUT = io_paths.check_output(paths, output)

APPROVAL_REFERENCE = ('user chat 2026-09-30: validation setini acip tekrar dene; '
                      'scope v3 core A/B/C_K4 + ProtGNN P, 3 seeds, one pass')
CONTRASTS = [('C', 'P'), ('A', 'P'), ('C', 'A'), ('B', 'A')]
# These are top-level producer fields, read only after checking the split text.
SPLIT_RE = re.compile(r'"split"\s*:\s*"([^"\\]*)"')
ID_RE = re.compile(r'"sample_id"\s*:\s*"([0-9a-f]{64})"')


def read(path):
    return comparison.read(path)


def equal(label, actual, expected):
    comparison.require_equal(label, actual, expected)


def stages():
    return [(arm, seed, (P_ROOT if arm == 'P' else V3_ROOT) /
             (f'C_K4_seed{seed}' if arm == 'C' else f'{arm}_seed{seed}'))
            for arm in ('A', 'B', 'C', 'P') for seed in study.SEEDS]


def snapshot_inputs():
    return {str(root): {p.relative_to(root).as_posix(): sha256(p)
                       for p in sorted(root.rglob('*')) if p.is_file()}
            for root in (V3_ROOT, P_ROOT)}


def validation_targets():
    """Never retain or interpret a test target; Top-10 is ranked on TRAIN only."""
    targets, seen, patient_splits = {}, set(), {}
    with TARGETS.open() as stream:
        for row in csv.DictReader(stream):
            split = row['split']
            if split == 'test':
                continue
            if split not in ('train', 'validation'):
                raise ValueError(f'Unknown target split {split!r}')
            sid, subject = row['sample_id'], row['subject_id']
            if not sid or not subject or sid in seen:
                raise ValueError('Missing/duplicate target identity')
            seen.add(sid)
            if subject in patient_splits and patient_splits[subject] != split:
                raise ValueError('Patient occurs in train and validation')
            patient_splits[subject] = split
            if row['target'] and row['target'] != '-1':
                target = int(row['target'])
                if not 0 <= target < train.NUM_CLASSES:
                    raise ValueError('Target outside canonical class order')
                targets[sid] = (target, split, subject)
    filtered, kept, _ = train.select_top_labels(targets, study.TOP_K_LABELS)
    return filtered, list(kept)


def raw_validation_lines(targets):
    """Stream shared bytes, skipping non-validation graphs BEFORE JSON parsing.

    No graph or membership object from test (or train) is loaded. Aligned raw
    validation rows are copied into a new validation-only encoder input view.
    """
    with (ARTIFACT / 'graphs.jsonl').open() as graphs, \
            (ARTIFACT / 'visit_membership.jsonl').open() as memberships:
        for graph_line, membership_line in zip_longest(graphs, memberships):
            if graph_line is None or membership_line is None:
                raise ValueError('Graph/membership line counts differ')
            splits = SPLIT_RE.findall(graph_line)
            if len(splits) != 1:
                raise ValueError('Graph split cannot be safely identified before parsing')
            if splits[0] != 'validation':
                continue
            ids = ID_RE.findall(graph_line)
            if len(ids) != 1:
                raise ValueError('Validation sample_id cannot be safely identified')
            sid = ids[0]
            entry = targets.get(sid)
            if entry is None:
                continue
            equal('raw validation target split', entry[1], 'validation')
            yield sid, graph_line, membership_line


def preflight(root):
    if root.exists():
        raise FileExistsError(f'Refusing existing output root: {root}')
    if root.parent != OUTPUT.parent or not root.name.startswith(OUTPUT.name):
        raise ValueError('Output must be the approved dated root (or a fresh retry suffix)')
    ignored = subprocess.run(['git', 'check-ignore', str(root) + '/'], cwd=REPO,
                             capture_output=True, text=True)
    equal('output root git-ignored', ignored.returncode, 0)
    frozen = read(V3_ROOT / 'k_selection.json')
    study._validate_k_selection_record(frozen)
    equal('frozen K*', frozen['k_selected'], 4)
    equal('freeze file hash', sha256(V3_ROOT / 'k_selection.json'), study.k_selection_sha256(frozen))
    for parent in (V3_ROOT, P_ROOT):
        decision = read(parent / 'decision.json')
        equal(f'{parent}: completed screen', decision['status'], 'completed')
        equal(f'{parent}: validation unopened', decision['validation_evaluated'], False)
        equal(f'{parent}: test closed', decision['test_evaluated'], False)
    targets, kept = validation_targets()
    ids = [sid for sid, _, _ in raw_validation_lines(targets)]
    expected_ids = {sid for sid, entry in targets.items() if entry[1] == 'validation'}
    equal('full unique validation ids', len(set(ids)), len(ids))
    equal('complete validation coverage', set(ids), expected_ids)
    equal('expected validation row count', len(ids), 4254)
    ids_hash = sample_ids_sha256(ids)
    source = code_source_hashes()
    bindings, checkpoints = {}, {}
    reference = read(V3_ROOT / 'A_seed1234/binding.json')
    for arm, seed, directory in stages():
        name = directory.name
        binding = read(directory / 'binding.json')
        study.check_stage_result(directory, binding, read(directory / 'result.json'))
        comparison.identity(binding, reference, name)
        equal(f'{name}: label indices', binding['kept_label_indices'], kept)
        equal(f'{name}: validation row count', binding['counts']['validation'], len(ids))
        equal(f'{name}: validation ordered id hash', binding['split_sample_ids_sha256']['validation'], ids_hash)
        equal(f'{name}: method', binding['method'], 'protgnn' if arm == 'P' else study.STUDY_METHOD)
        equal(f'{name}: seed', binding['seed'], seed)
        equal(f'{name}: CPU', binding['device'], 'cpu')
        equal(f'{name}: preprocessing', sha256(directory / 'preprocessing.json'), binding['preprocessing_sha256'])
        for key, value in binding['source_code'].items():
            equal(f'{name}: source {key}', source.get(key), value)
        if arm != 'P':
            successor = read(directory / study.STUDY_BINDING_FILENAME)
            stage = study._stage_of_binding(directory, binding, successor)
            study.validate_v3_binding(binding, stage, k_selection=frozen, study_binding=successor)
            equal(f'{name}: arm', stage.arm, arm)
            equal(f'{name}: K', stage.k, 4)
            equal(f'{name}: bound v3 state', sha256((PATHS or io_paths.resolve()).path_map.resolve(
                      binding['method_config']['v3_state_path'], binding['method_config']['v3_state_sha256'])),
                  binding['method_config']['v3_state_sha256'])
        bindings[name] = binding
        checkpoints[name] = sha256(directory / 'best.pt')
    for path, key in ((ARTIFACT / 'graphs.jsonl', 'artifact_graphs_sha256'),
                      (ARTIFACT / 'visit_membership.jsonl', 'artifact_visit_membership_sha256'),
                      (TARGETS, 'targets_sha256')):
        equal(f'parent input bytes: {key}', sha256(path), reference[key])
    return targets, ids, ids_hash, bindings, checkpoints, source


def load_model(binding, checkpoint_path):
    """CEI factory; ProtGNN reconstruction exactly as score_protgnn's loader."""
    import torch
    if binding['method'] == study.STUDY_METHOD:
        model = io_paths.mapped_model_factory(binding, checkpoint_path,
                    paths=PATHS or io_paths.resolve(), factory=study._default_model_factory)
    else:
        config = binding['method_config']
        architecture = config['architecture']
        args = dict(binding)
        args.update(config['effective_settings'])
        args['num_relations'] = architecture['num_relations']
        model = build_method('protgnn', num_tokens=binding['vocabulary_size'],
                             node_dim=binding['node_dim'], edge_dim=binding['edge_dim'],
                             num_classes=binding['num_classes'], hidden=binding['hidden'],
                             layers=binding['layers'], dropout=binding['dropout'],
                             token_dim=architecture['token_dim'], num_triples=binding['num_meta_relations'],
                             args=SimpleNamespace(**args))
        equal('rebuilt ProtGNN method_config', model.run_config(), config)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)
    equal('rebuilt parameter_count', sum(p.numel() for p in model.parameters()), binding['parameter_count'])
    equal('rebuilt method_config', model.run_config(), binding['method_config'])
    model.eval()
    return model


def load_saved(directory, ids, reference_rows=None):
    result = read(directory / 'validation_result.json')
    equal('validation result hash', result['result_sha256'],
          pilot._sha_json({k: v for k, v in result.items() if k != 'result_sha256'}))
    arrays = {}
    rows = None
    for key in ('logits', 'proba'):
        path = directory / f'{key}.npz'
        equal(f'{directory.name}: {key} file hash', sha256(path), result[f'{key}_file_sha256'])
        with np.load(path, allow_pickle=False) as saved:
            values = saved[key].copy()
            identity = tuple(saved[k].copy() for k in ('y', 'subjects', 'sample_ids'))
        if rows is None:
            rows = identity
        for label, actual, expected in zip(('y', 'subjects', 'sample_ids'), identity, rows):
            equal(f'{directory.name}: saved {label}', np.array_equal(actual, expected), True)
        if values.dtype != np.float32 or values.shape != (len(ids), study.NUM_CLASSES) or not np.isfinite(values).all():
            raise ValueError(f'{directory}: invalid {key} dtype/shape/values')
        equal(f'{directory.name}: {key} array hash', comparison.array_hash(values), result[f'{key}_sha256'])
        arrays[key] = values
    equal('stored ordered validation ids', rows[2].astype(str).tolist(), ids)
    if reference_rows is not None:
        for key, actual, expected in zip(('y', 'subjects', 'sample_ids'), rows, reference_rows):
            equal(f'{directory.name}: paired {key}', np.array_equal(actual, expected), True)
    equal('validation evaluated', result['validation_evaluated'], True)
    equal('test closed', result['test_evaluated'], False)
    score = study.weighted_macro_f1(rows[0], arrays['logits'].argmax(1))
    equal('stored macro_f1', score, result['macro_f1'])
    return result, rows, (rows[0], arrays['logits'].argmax(1), rows[1])


def score_checkpoint(arm, seed, directory, binding, model, view, targets, ids,
                     approval, root, reference_rows):
    import torch
    from torch_geometric.loader import DataLoader
    start = time.monotonic()
    validate_approval_record(approval)
    checkpoint_hash = sha256(directory / 'best.pt')
    if checkpoint_hash not in approval['checkpoint_sha256']:
        raise ValueError('Checkpoint absent from approval record')
    equal('approved validation hash', sample_ids_sha256(ids), approval['validation_sample_ids_sha256'])
    encoded = list(screen.encode_rows(view, directory / 'preprocessing.json', ids,
                   fold='validation', allow_validation=True, approval_record=approval,
                   edge_direction=binding['edge_direction'], targets=targets,
                   preprocessing_sha256=binding['preprocessing_sha256']))
    by_id = {row.sample_id: row for row in encoded}
    equal('encoded unique row count', len(by_id), len(encoded))
    equal('encoded validation coverage', set(by_id), set(ids))
    rows = [by_id[sid] for sid in ids]
    identity = (np.asarray([int(row.y.item()) for row in rows], dtype=np.int64),
                np.asarray([str(row.subject) for row in rows]), np.asarray(ids))
    if reference_rows is not None:
        for key, actual, expected in zip(('y', 'subjects', 'sample_ids'), identity, reference_rows):
            equal(f'{directory.name}: encoded paired {key}', np.array_equal(actual, expected), True)
    chunks = []
    with torch.inference_mode():
        for batch in DataLoader(rows, batch_size=binding['batch_size'], shuffle=False):
            if arm == 'P':
                logits = model(batch, epoch=binding['selected_dev']['epoch_index']).logits
            else:
                logits = model.forward_continuous(model.continuous_inputs(batch), batch.edge_index, batch)
            chunks.append(logits.detach().cpu().to(torch.float32))
    logits = torch.cat(chunks).numpy().astype(np.float32, copy=False)
    proba = torch.softmax(torch.from_numpy(logits), dim=1).numpy().astype(np.float32, copy=False)
    if logits.shape != (len(ids), study.NUM_CLASSES) or not np.isfinite(logits).all() or not np.isfinite(proba).all():
        raise ValueError(f'{directory.name}: invalid validation predictions')
    out = root / directory.name
    out.mkdir(exist_ok=False)
    y, subjects, sample_ids = identity
    for key, values in (('logits', logits), ('proba', proba)):
        np.savez_compressed(out / f'{key}.npz', **{key: values}, y=y, subjects=subjects, sample_ids=sample_ids)
    result = {'arm': arm, 'seed': seed, 'stage': directory.name, 'fold': 'validation',
              'k': 4 if arm != 'P' else None, 'row_count': len(ids),
              'checkpoint_sha256': checkpoint_hash, 'binding_sha256': sha256(directory / 'binding.json'),
              'preprocessing_sha256': binding['preprocessing_sha256'],
              'parameter_count': binding['parameter_count'], 'parameter_count_verified': True,
              'selected_dev': binding['selected_dev'], 'inference_passes': 1,
              'approval_record_path': str(root / APPROVAL_FILENAME),
              'approval_record_sha256': approval_record_sha256(approval),
              'validation_sample_ids_sha256': sample_ids_sha256(ids),
              'k_selection_sha256': sha256(V3_ROOT / 'k_selection.json'),
              'macro_f1': study.weighted_macro_f1(y, logits.argmax(1)),
              'row_identity_verified': True, 'validation_evaluated': True, 'test_evaluated': False,
              'wall_seconds': time.monotonic() - start}
    for key, values in (('logits', logits), ('proba', proba)):
        result.update({f'{key}_sha256': comparison.array_hash(values),
                       f'{key}_file_sha256': sha256(out / f'{key}.npz'), f'{key}_path': str(out / f'{key}.npz')})
    result['result_sha256'] = pilot._sha_json(result)
    study._write_new_json(out / 'validation_result.json', result)
    saved, stored_rows, predictions = load_saved(out, ids, reference_rows)
    print(f'{directory.name}: validation_macro_f1={saved["macro_f1"]:.12f}; wall_seconds={saved["wall_seconds"]:.3f}', flush=True)
    return saved, stored_rows, predictions


def execute(root):
    start = time.monotonic()
    before = snapshot_inputs()
    targets, ids, ids_hash, bindings, checkpoints, source = preflight(root)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    equal('committed clean tree', subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=REPO, text=True), '')
    import torch
    torch.set_num_threads(comparison.THREADS)
    root.mkdir(parents=True, exist_ok=False)
    approval = {'allow_validation': True, 'approval_reference': APPROVAL_REFERENCE,
                'checkpoint_sha256': list(checkpoints.values()), 'checkpoints': checkpoints,
                'validation_sample_ids_sha256': ids_hash,
                'timestamp': datetime.now(timezone.utc).isoformat()}
    # The approval is the FIRST written artifact, before validation parsing/encoding.
    write_approval_record(root / APPROVAL_FILENAME, approval)
    equal('approval file hash', sha256(root / APPROVAL_FILENAME), approval_record_sha256(approval))
    study._write_new_json(root / 'input_snapshot_before.json', before)
    try:
        # Rebuild all twelve before the first inference, never from validation scores.
        models = {directory.name: load_model(bindings[directory.name], directory / 'best.pt')
                  for _, _, directory in stages()}
        view = root / 'validation_input_view'
        view.mkdir(exist_ok=False)
        view_ids = []
        with (view / 'graphs.jsonl').open('x') as graphs, (view / 'visit_membership.jsonl').open('x') as memberships:
            for sid, graph_line, membership_line in raw_validation_lines(targets):
                graphs.write(graph_line)
                memberships.write(membership_line)
                view_ids.append(sid)
        equal('validation-only view ordered ids', view_ids, ids)
        study._write_new_json(view / 'binding.json', {
            'parent_artifact': str(ARTIFACT), 'fold': 'validation', 'row_count': len(ids),
            'validation_sample_ids_sha256': ids_hash, 'graphs_sha256': sha256(view / 'graphs.jsonl'),
            'visit_membership_sha256': sha256(view / 'visit_membership.jsonl'),
            'non_validation_graphs_deserialized': 0, 'test_evaluated': False,
            'approval_record_sha256': approval_record_sha256(approval)})
        arms = {arm: {} for arm in ('A', 'B', 'C', 'P')}
        scores = {arm: {} for arm in arms}
        result_hashes, reference_rows = {}, None
        for arm, seed, directory in stages():
            result, rows, prediction = score_checkpoint(arm, seed, directory, bindings[directory.name],
                models.pop(directory.name), view, targets, ids, approval, root, reference_rows)
            reference_rows = rows if reference_rows is None else reference_rows
            arms[arm][seed], scores[arm][seed] = prediction, result['macro_f1']
            result_hashes[directory.name] = result['result_sha256']
        deltas = study.paired_bootstrap(arms, CONTRASTS, resamples=1000, seed=2026)
        decisions = {}
        for treatment, control in CONTRASTS:
            decision = study.decide_v3(deltas, scores, treatment=treatment, control=control,
                                       resamples=1000, bootstrap_seed=2026)
            decision.update(validation_evaluated=True, test_evaluated=False,
                            decisive=(treatment, control) == ('C', 'P'),
                            statement=(f'{treatment} benefit over {control} demonstrated on this validation fold'
                                       if decision['v3_beats_control'] else
                                       f'{treatment} benefit over {control} not demonstrated on this validation fold'))
            decisions[f'{treatment}_vs_{control}'] = decision
        equal('source unchanged', code_source_hashes(), source)
    finally:
        after = snapshot_inputs()
        unchanged = after == before
        study._write_new_json(root / 'input_unchanged.json', {
            'verified': unchanged, 'before': before, 'after': after,
            'file_counts': {key: len(value) for key, value in before.items()}})
        equal('every file under both input roots unchanged', unchanged, True)
    report = {'status': 'completed', 'scope': 'frozen_v3_core_and_protgnn_full_validation',
              'git_revision': revision, 'output_root': str(root), 'k_selected': 4,
              'validation_row_count': len(ids), 'validation_distinct_subject_count': len(set(reference_rows[1])),
              'validation_sample_ids_sha256': ids_hash, 'validation_hash_matches_all_12_bindings': True,
              'approval_record_path': str(root / APPROVAL_FILENAME),
              'approval_record_sha256': approval_record_sha256(approval),
              'canonical_sha256': sha256(CANONICAL),
              'k_selection_sha256': sha256(V3_ROOT / 'k_selection.json'),
              'validation_macro_f1': {a: {str(s): scores[a][s] for s in study.SEEDS} for a in scores},
              'seed_means': {a: float(np.mean(list(scores[a].values()))) for a in scores},
              'decisions': decisions, 'decision': decisions['C_vs_P'],
              'secondary_A_vs_P': decisions['A_vs_P'], 'secondary_C_vs_A': decisions['C_vs_A'],
              'secondary_B_vs_A': decisions['B_vs_A'], 'validation_result_hashes': result_hashes,
              'validation_row_identity_verified': True, 'input_roots_unchanged': True,
              'input_file_counts': {key: len(value) for key, value in before.items()},
              'validation_evaluated': True, 'test_evaluated': False,
              'non_validation_graphs_deserialized': 0, 'retraining': False, 'reselection': False,
              'temporal_clean': False, 'total_wall_seconds': time.monotonic() - start}
    report['result_sha256'] = pilot._sha_json(report)
    study._write_new_json(root / 'decision.json', report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-root', type=Path, default=None)
    io_paths.add_arguments(parser)
    args = parser.parse_args(argv)
    if args.plan and args.execute:
        parser.error('--plan and --execute are mutually exclusive')
    configure(io_paths.resolve(args), args.output_root or REPO / 'comparison/standardized/clinical_runs_cei_v3_validation_20261001')
    root = OUTPUT
    if not args.execute:
        report = io_paths.metadata_plan(PATHS, root)
        report.update(inference_passes_per_checkpoint=1, contrasts=CONTRASTS,
                      resamples=1000, bootstrap_seed=2026, quantile_method=study.QUANTILE_METHOD,
                      retraining=False, reselection=False, test_graphs_will_be_deserialized=False)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    return execute(root)


if __name__ == '__main__':
    raise SystemExit(main())
