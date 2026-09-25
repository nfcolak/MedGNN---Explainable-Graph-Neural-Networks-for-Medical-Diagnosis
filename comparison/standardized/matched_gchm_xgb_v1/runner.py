"""Isolated matched-weight XGBoost experiment; validation only, no test inference.

Six explicitly authorized full-cohort cells use a frozen protocol. Completed cells
can be verified/reused with --resume. Incomplete cells are preserved, never silently
restarted: choose a fresh output root for an explicitly authorized retry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score

from comparison.standardized.native_reference_v1.data import DEFAULT, REPO, Reference, digest, save, sha
from comparison.standardized.performance_review import class_weights
from comparison.standardized.xgboost_native_baseline import build_tabular_matrix, split_indices
from shared.lib.metrics import multiclass_metrics

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'protocol.json'
SELECTION = 'validation macro_f1; first maximum rounded to 6 decimals; full fixed budget'
SOURCE_PATHS = [
    Path(__file__).relative_to(REPO),
    Path('comparison/standardized/xgboost_native_baseline.py'),
    Path('comparison/standardized/native_reference_v1/data.py'),
    Path('comparison/standardized/performance_review.py'),
    Path('shared/lib/metrics.py'),
]


def selection_metric(predictions, data):
    """Same rounded sklearn macro-F1 as the shared benchmark metric."""
    probs = np.asarray(predictions).reshape(-1, 30)
    if not np.isfinite(probs).all():
        raise ValueError('Nonfinite validation probabilities')
    score = f1_score(data.get_label().astype(np.int64), probs.argmax(1),
                     average='macro', zero_division=0)
    return 'macro_f1', round(float(score), 6)


def first_best(history):
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty finite selection history required')
    return int(np.argmax(values))


def check_probabilities(probs, count):
    if probs.shape != (count, 30) or not np.isfinite(probs).all():
        raise ValueError('Invalid probability shape or nonfinite values')
    if np.any(probs < 0) or not np.allclose(probs.sum(1), 1., rtol=0, atol=1e-5):
        raise ValueError('Invalid probability distribution')


class Progress(xgb.callback.TrainingCallback):
    def __init__(self, output, rounds):
        self.output = output
        self.rounds = rounds
        self.started = time.monotonic()

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % 50 == 0 or epoch + 1 == self.rounds:
            elapsed = time.monotonic() - self.started
            row = {'rounds_completed': epoch + 1, 'rounds_total': self.rounds,
                   'elapsed_seconds': elapsed,
                   'estimated_remaining_seconds': elapsed / (epoch + 1) * (self.rounds - epoch - 1),
                   'validation_macro_f1': evals_log['validation']['macro_f1'][-1]}
            save(self.output / 'progress.json', row)
            print(json.dumps({'cell': str(self.output), **row}), flush=True)
        return False


def verify_completed(output, binding, X, y, val_idx):
    state = json.loads((output / 'run_manifest.json').read_text())
    if state['status'] != 'completed' or state['binding'] != binding:
        raise ValueError('Occupied cell is incomplete or has a different binding; preserved without overwrite')
    for name, fingerprint in state['artifact_files'].items():
        if sha(output / name) != fingerprint:
            raise ValueError('Artifact checksum mismatch: ' + name)
    stored = np.load(output / 'validation.npz', allow_pickle=False)
    if not np.array_equal(stored['y'], y[val_idx]) or not np.array_equal(stored['ordinals'], val_idx):
        raise ValueError('Saved validation cohort differs')
    booster = xgb.Booster()
    booster.load_model(output / 'model.ubj')
    proba = booster.predict(xgb.DMatrix(X[val_idx]))
    check_probabilities(proba, len(val_idx))
    if not np.array_equal(proba, stored['proba']):
        raise ValueError('Completed model replay differs')
    if multiclass_metrics(y[val_idx], proba.argmax(1), proba) != state['metrics']:
        raise ValueError('Completed model metrics differ')
    return state


def run_cell(output, artifact=DEFAULT, seed=1234, weight_policy='none',
             limit=None, smoke_rounds=None, execute=False, resume=False):
    protocol = json.loads(PROTOCOL.read_text())
    if seed not in protocol['seeds'] or weight_policy not in protocol['weight_policies']:
        raise ValueError('Seed/weight policy outside frozen protocol')
    if smoke_rounds is not None and (limit is None or smoke_rounds < 1):
        raise ValueError('Round override is allowed only in explicit bounded wiring scope')
    X, y, folds, feature_names, meta = build_tabular_matrix(artifact)
    train_idx, val_idx = split_indices(folds, limit)
    weights = class_weights(y[folds == 0], 30, weight_policy).astype(np.float32)
    protocol_params = dict(protocol['xgboost'])
    rounds = smoke_rounds if smoke_rounds is not None else protocol_params.pop('num_boost_round')
    protocol_params.pop('num_boost_round', None)
    params = {**protocol_params, 'seed': seed, 'disable_default_eval_metric': 1}
    code = {str(p): sha(REPO / p) for p in SOURCE_PATHS}
    binding = {
        'contract_sha256': meta['contract_sha256'], 'input_sha256': meta['input_sha256'],
        'protocol_sha256': sha(PROTOCOL), 'source_code': code,
        'seed': seed, 'weight_policy': weight_policy, 'class_weights': weights.tolist(),
        'train_ordinals_sha256': digest(train_idx.tolist()),
        'validation_ordinals_sha256': digest(val_idx.tolist()),
        'labels_sha256': digest(meta['labels']), 'params': params, 'rounds': rounds,
        'limit': limit, 'selection': SELECTION, 'xgboost_version': xgb.__version__,
        'numpy_version': np.__version__,
    }
    report = {
        'method': 'xgboost', 'seed': seed, 'weight_policy': weight_policy,
        'scope': 'full_cohort' if limit is None else 'bounded_wiring_not_benchmark',
        'counts': meta['counts'], 'selected_counts': [len(train_idx), len(val_idx)],
        'contract_sha256': meta['contract_sha256'], 'input_sha256': meta['input_sha256'],
        'protocol_sha256': binding['protocol_sha256'], 'binding': binding,
        'selection': SELECTION, 'test_evaluated': False, 'feature_count': X.shape[1],
        'temporal_clean': False, 'clinical_early_prediction': False,
    }
    if not execute:
        return {**report, 'status': 'dry_run'}
    output = Path(output).resolve()
    if output.exists():
        if not resume:
            raise FileExistsError('Occupied output; explicit --resume verifies completed cells only')
        return verify_completed(output, binding, X, y, val_idx)
    output.mkdir(parents=True)
    state = {**report, 'status': 'running'}
    save(output / 'run_manifest.json', state)
    start = time.monotonic()
    try:
        for path in code:
            target = output / 'source_snapshot' / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / path, target)
        shutil.copyfile(PROTOCOL, output / 'protocol.json')
        save(output / 'feature_names.json', feature_names)
        save(output / 'labels.json', meta['labels'])
        np.savez_compressed(output / 'cohort.npz', train_ordinals=train_idx, validation_ordinals=val_idx)
        # Only training rows receive weights. Validation is never weighted/resampled.
        train = xgb.DMatrix(X[train_idx], label=y[train_idx], weight=weights[y[train_idx]])
        validation = xgb.DMatrix(X[val_idx], label=y[val_idx])
        history = {}
        booster = xgb.train(
            params, train, num_boost_round=rounds, evals=[(validation, 'validation')],
            custom_metric=selection_metric, maximize=True, evals_result=history,
            verbose_eval=False, callbacks=[Progress(output, rounds)],
        )
        values = history['validation']['macro_f1']
        if len(values) != rounds or booster.num_boosted_rounds() != rounds:
            raise ValueError('Full fixed training budget did not complete')
        best = first_best(values)
        final_proba = booster.predict(validation)
        chosen = booster[:best + 1]
        proba = chosen.predict(validation)
        check_probabilities(proba, len(val_idx))
        check_probabilities(final_proba, len(val_idx))
        metrics = multiclass_metrics(y[val_idx], proba.argmax(1), proba)
        if metrics['macro_f1'] != float(values[best]):
            raise ValueError('Selected checkpoint disagrees with shared selection metric')
        chosen.save_model(output / 'model.ubj')
        booster.save_model(output / 'final_model.ubj')
        np.savez_compressed(output / 'validation.npz', proba=proba, y=y[val_idx], ordinals=val_idx)
        np.savez_compressed(output / 'final_validation.npz', proba=final_proba, y=y[val_idx], ordinals=val_idx)
        save(output / 'history.json', [{'iteration': i, 'validation_macro_f1': float(v)}
                                      for i, v in enumerate(values)])
        loaded = xgb.Booster()
        loaded.load_model(output / 'model.ubj')
        replay = loaded.predict(validation)
        if not np.array_equal(proba, replay):
            raise ValueError('Reloaded selected checkpoint did not replay exactly')
        save(output / 'replay.json', {'exact_proba': True, 'max_abs_diff': 0.,
             'validation_count': len(val_idx), 'test_evaluated': False,
             'selected_iteration': best, 'checkpoint_sha256': sha(output / 'model.ubj')})
        # Fail closed if input, protocol or implementation changed during training.
        Reference(artifact, expected=meta['contract_sha256'])
        if sha(PROTOCOL) != binding['protocol_sha256'] or any(sha(REPO / p) != h for p, h in code.items()):
            raise ValueError('Protocol or training source changed during execution')
        artifacts = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*'))
                     if p.is_file() and p.name != 'run_manifest.json'}
        state.update(status='completed', selected_iteration=best, metrics=metrics,
                     final_metrics=multiclass_metrics(y[val_idx], final_proba.argmax(1), final_proba),
                     artifact_files=artifacts, checkpoint_path='model.ubj', metrics_path='history.json',
                     elapsed_seconds=time.monotonic() - start)
        save(output / 'run_manifest.json', state)
        verify_completed(output, binding, X, y, val_idx)
        return state
    except BaseException as exc:
        state.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        save(output / 'run_manifest.json', state)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=HERE / 'runs')
    p.add_argument('--artifact', type=Path, default=DEFAULT)
    p.add_argument('--seed', type=int, choices=[1234, 1235, 1236])
    p.add_argument('--weight-policy', choices=['none', 'sqrt_inverse'])
    p.add_argument('--all', action='store_true', help='All six authorized cells')
    p.add_argument('--limit', type=int)
    p.add_argument('--smoke-rounds', type=int)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true', help='Verify/reuse completed cells; never overwrite incomplete ones')
    args = p.parse_args()
    if args.all and (args.seed is not None or args.weight_policy is not None):
        p.error('--all cannot be combined with single-cell selectors')
    if not args.all and (args.seed is None or args.weight_policy is None):
        p.error('Supply --all or both --seed and --weight-policy')
    if args.limit is not None and args.output_root.resolve() == (HERE / 'runs').resolve():
        p.error('Bounded wiring must use a separate --output-root')
    cells = [(w, s) for w in ('none', 'sqrt_inverse') for s in (1234, 1235, 1236)] if args.all else [(args.weight_policy, args.seed)]
    for w, s in cells:
        result = run_cell(args.output_root / w / f'seed{s}', args.artifact, s, w,
                          args.limit, args.smoke_rounds, args.execute, args.resume)
        print(json.dumps({k: result.get(k) for k in ('status', 'scope', 'seed', 'weight_policy',
              'selected_counts', 'selected_iteration', 'metrics', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
