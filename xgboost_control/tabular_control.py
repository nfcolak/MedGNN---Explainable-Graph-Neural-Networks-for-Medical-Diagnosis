"""Tabular summaries of the SAME encoded clinical primitives as the GNN.

Preprocessing is shared, not merely the source JSON. Raw values and coverage are
never predictor shortcuts. Aggregation loses node pairing/order and therefore is
not an assertion that these two representations have equal inductive biases.
"""
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import xgboost as xgb

from core import NODE_KINDS
from core.contracts import (VISIT_MEMBERSHIP_CONTRACT_VERSION, VISIT_MEMBERSHIP_FILENAME,
                        iter_graphs_with_membership)
from core.tensorize import (ALL_RELATIONS, PREPROCESSING_VERSION, encode_graph,
                        fit_preprocessing, node_feature_layout,
                        preprocessing_state, relation_vocabulary, triple_count)

NUM_CLASSES = 30
PROJECTION_VERSION = 'encoded_graph_summary_v2'
# Legacy booster settings; `--max-depth`/`--learning-rate` default to these values.
XGB_DEFAULTS = {'max_depth': 6, 'learning_rate': 0.1}


def load_targets(path):
    from core.train import load_targets as shared_load_targets
    return shared_load_targets(path)


def apply_top_k(targets, k):
    """Same selection rule as train.py: rank on TRAIN counts, drop the rest.

    Accepts both the older 2-tuples and subject-bearing target tuples. The
    ranking key `(-count, class)` and the drop-not-merge policy are identical, so
    the two runs model the same label set. The check in `main` verifies that
    equality against the GNN run instead of trusting this comment.
    """
    counts = Counter(entry[0] for entry in targets.values() if entry[1] == 'train')
    if k < 1 or k > len(counts):
        raise ValueError('--top-k-labels must be within the number of TRAIN classes')
    kept = sorted(sorted(counts, key=lambda c: (-counts[c], c))[:k])
    remap = {c: i for i, c in enumerate(kept)}
    filtered = {sid: (remap[entry[0]], *entry[1:]) for sid, entry in targets.items()
                if entry[0] in remap}
    return filtered, kept


def subsample_train(targets, limit, seed):
    """Reproduce train.py's deterministic TRAIN subsample, sample id for sample id.

    The GNN arm trains on a seeded 20k subset. Giving XGBoost the full 52k training
    rows would make it win on data volume rather than on model class, so the same
    `random.Random(seed).sample(sorted(ids), limit)` draw is replayed here. Because
    the draw is over sorted sample ids with the same seed, the two models see the
    SAME patients, which `main` asserts before training.
    """
    if limit is None:
        return targets
    train_ids = sorted(sid for sid, entry in targets.items() if entry[1] == 'train')
    keep = set(random.Random(seed).sample(train_ids, min(limit, len(train_ids))))
    return {sid: v for sid, v in targets.items()
            if v[1] != 'train' or sid in keep}


def fit_features(artifact, targets, min_count=20, min_prior_visits=0):
    """Fit exactly the GNN preprocessing on the selected TRAIN sample IDs."""
    path = Path(artifact) / 'graphs.jsonl'
    membership_path = Path(artifact) / VISIT_MEMBERSHIP_FILENAME
    train_ids = {sid for sid, entry in targets.items() if entry[1] == 'train'}
    if min_prior_visits:
        eligible = {graph['sample_id'] for graph, _ in
                    iter_graphs_with_membership(path, membership_path)
                    if graph['coverage']['prior_visits'] >= min_prior_visits}
        train_ids &= eligible
    return fit_preprocessing(path, train_ids, token_min_count=min_count,
                             membership_path=membership_path)


def fit_vocab(artifact, targets, min_count=20, min_prior_visits=0):
    """Compatibility accessor for tokens; build now requires fit_features state."""
    return fit_features(artifact, targets, min_count, min_prior_visits)['vocabulary'].tokens


def projection_layout(prep, edge_direction='forward'):
    """Ordered, serializable columns for summaries of encoded tensors only."""
    node_layout = node_feature_layout(prep)
    relations = relation_vocabulary(edge_direction)
    names = []
    for i in range(len(prep['vocabulary'])):  # includes the GNN's unknown bucket
        names.append(f'token[{i}].count')
        names.extend(f'token[{i}].mean.{name}' for name in node_layout[len(NODE_KINDS):len(NODE_KINDS) + 6])
    names.extend(f'node_kind.{kind}.count' for kind in NODE_KINDS)
    names.extend('sum.' + name for name in node_layout[len(NODE_KINDS) + 6:])
    payload_names = preprocessing_state(prep)['edge_feature_layout'][len(ALL_RELATIONS):]
    for relation in relations:
        names.append(f'relation.{relation}.count')
        names.extend(f'relation.{relation}.mean.{name}' for name in payload_names)
    names.extend(f'triple[{i}].count' for i in range(triple_count(prep, edge_direction)))
    return names


def project_encoded(data, prep, use_edge_payload=True, edge_direction='forward'):
    """Project the actual GNN primitives, never graph coverage or raw payloads.

    Counts and means lose topology/order; they do not add information unavailable
    to the GNN. Masks are averaged too, so missing and observed zero stay distinct.
    A bidirectional edge view only duplicates relation/triple columns (a reverse
    edge repeats its forward edge's payload), so it adds no information here.
    """
    x = data.x.detach().cpu().numpy()
    token = data.token.detach().cpu().numpy()
    n_tok = len(prep['vocabulary'])
    n_rel = len(relation_vocabulary(edge_direction))
    counts = np.bincount(token, minlength=n_tok).astype(np.float32)
    summaries = np.zeros((n_tok, 7), dtype=np.float32)
    summaries[:, 0] = counts
    np.add.at(summaries[:, 1:], token, x[:, len(NODE_KINDS):len(NODE_KINDS) + 6])
    summaries[:, 1:] /= np.maximum(counts[:, None], 1)
    relation = data.edge_relation.detach().cpu().numpy()
    edge_counts = np.bincount(relation, minlength=n_rel).astype(np.float32)
    edge_summaries = np.zeros((n_rel, 8), dtype=np.float32)
    edge_summaries[:, 0] = edge_counts
    if use_edge_payload:
        np.add.at(edge_summaries[:, 1:], relation, data.edge_payload.detach().cpu().numpy())
        edge_summaries[:, 1:] /= np.maximum(edge_counts[:, None], 1)
    triple_counts = np.bincount(data.edge_triple.detach().cpu().numpy(),
                               minlength=triple_count(prep, edge_direction))
    return np.concatenate([summaries.ravel(), x[:, :len(NODE_KINDS)].sum(0),
                           x[:, len(NODE_KINDS) + 6:].sum(0), edge_summaries.ravel(),
                           triple_counts]).astype(np.float32)


def build(artifact, targets, prep, min_prior_visits=0, *, return_metadata=False,
          edge_mode='all', drop_relations=(), rewire_relations=(), rewire_seed=0,
          use_edge_payload=True, edge_direction='forward'):
    """Build shared-input rows; old bare-token lists are deliberately incompatible.

    The default return remains (X, y, folds). Metadata is an opt-in fourth result
    containing aligned sample_ids and subjects, never predictor columns.
    Fold codes: 0 train, 1 validation, 2 dev (only when targets carry 'dev' rows).
    """
    if not isinstance(prep, dict) or prep.get('preprocessing_version') != PREPROCESSING_VERSION:
        raise ValueError('build requires versioned preprocessing from fit_features, not a token list')
    fold_code = {'train': 0, 'validation': 1, 'dev': 2}
    X, y, folds, sample_ids, subjects = [], [], [], [], []
    rewired_edge_counts = []
    graphs_path = Path(artifact) / 'graphs.jsonl'
    membership_path = Path(artifact) / VISIT_MEMBERSHIP_FILENAME
    for g, visit_membership in iter_graphs_with_membership(graphs_path, membership_path):
        entry = targets.get(g['sample_id'])
        if entry is None or entry[1] == 'test':
            continue
        if g['coverage']['prior_visits'] < min_prior_visits:
            continue
        if entry[1] not in fold_code:
            raise ValueError(f'Unknown target split: {entry[1]}')
        data = encode_graph(g, prep, visit_membership, edge_mode, drop_relations,
                            rewire_relations, rewire_seed, edge_direction=edge_direction)
        X.append(project_encoded(data, prep, use_edge_payload, edge_direction))
        y.append(entry[0])
        folds.append(fold_code[entry[1]])
        sample_ids.append(g['sample_id'])
        subjects.append(entry[2] if len(entry) > 2 else '')
        rewired_edge_counts.append(data.rewired_edge_count)
    matrix = np.asarray(X, dtype=np.float32).reshape(
        len(X), len(projection_layout(prep, edge_direction)))
    result = (matrix, np.asarray(y, dtype=np.int64), np.asarray(folds, dtype=np.int64))
    if return_metadata:
        return (*result, {'sample_ids': np.asarray(sample_ids, dtype=str),
                         'subjects': np.asarray(subjects, dtype=str),
                         'rewired_edge_counts': np.asarray(rewired_edge_counts, dtype=np.int64)})
    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', required=True)
    p.add_argument('--targets', required=True)
    p.add_argument('--canonical', default='comparison/canonical_split.json')
    p.add_argument('--out', required=True)
    p.add_argument('--rounds', type=int, default=300)
    p.add_argument('--max-depth', type=int, default=XGB_DEFAULTS['max_depth'])
    p.add_argument('--learning-rate', type=float, default=XGB_DEFAULTS['learning_rate'])
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--sample-seed', type=int, default=None,
                   help='seed of the TRAIN sample and dev draw (default: --seed); '
                        'identical to train.py --sample-seed.')
    p.add_argument('--selection-fold', choices=['none', 'dev'], default='none',
                   help="'dev' selects the boosting-round count on the patient-disjoint "
                        'dev split (train.py --dev-limit policy); default keeps the '
                        'legacy fixed budget with no selection.')
    p.add_argument('--final-eval', choices=['validation', 'none'], default='validation')
    p.add_argument('--dev-limit', type=int, default=None)
    p.add_argument('--round-step', type=int, default=25,
                   help='dev selection evaluates every N rounds up to --rounds.')
    p.add_argument('--token-min-count', type=int, default=20)
    p.add_argument('--edges', choices=['all', 'informative', 'structural'], default='all')
    p.add_argument('--edge-direction', choices=['forward', 'bidirectional'], default='forward')
    p.add_argument('--drop-relation', action='append', default=[])
    p.add_argument('--rewire-relation', action='append', default=[])
    p.add_argument('--no-edge-payload', action='store_true')
    p.add_argument('--weights', choices=['none', 'sqrt_inverse', 'inverse'], default='sqrt_inverse')
    p.add_argument('--min-prior-visits', type=int, default=0, metavar='N',
                   help='keep only visits with >= N completed earlier encounters, '
                        'matching train.py --min-prior-visits.')
    p.add_argument('--top-k-labels', type=int, metavar='K',
                   help='model only the K most frequent TRAIN labels, matching '
                        'train.py --top-k-labels.')
    p.add_argument('--train-limit', type=int, metavar='N',
                   help='train on the same seeded N-row TRAIN subsample train.py uses.')
    p.add_argument('--match-run', metavar='DIR',
                   help='a GNN run directory; its binding.json must agree with this '
                        'run on classes, kept labels and row counts, or abort.')
    a = p.parse_args()

    t0 = time.monotonic()
    from core.contracts import (code_source_hashes, sample_ids_sha256,
                            validate_artifact_manifest, validate_control_configuration,
                            verify_graph_file, verify_target_binding,
                            verify_visit_membership_file)
    from core.schema import sha256
    from core.rewiring import REWIRING_POLICY
    from core.train import (DEV_POLICY, EVALUATION_VERSION, class_weights, metrics,
                        patient_equal_metrics, sample_train_ids, select_dev_ids)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError('Occupied output directory; choose a fresh path')
    if a.selection_fold == 'dev' and a.dev_limit is None:
        raise ValueError('--selection-fold dev requires --dev-limit')
    if a.selection_fold != 'dev' and (a.dev_limit is not None or a.final_eval != 'validation'):
        raise ValueError('--dev-limit and --final-eval none require --selection-fold dev')
    if a.max_depth < 1 or a.learning_rate <= 0 or a.round_step < 1:
        raise ValueError('Invalid booster settings')
    sample_seed = a.seed if a.sample_seed is None else a.sample_seed
    artifact = Path(a.artifact)
    manifest = json.loads((artifact / 'manifest.json').read_text())
    validate_artifact_manifest(manifest)
    verify_graph_file(artifact, manifest)
    verify_visit_membership_file(artifact, manifest)
    labels = json.loads(Path(a.canonical).read_text())['classes']
    target_binding_sha256 = verify_target_binding(manifest, a.targets, labels)
    validate_control_configuration(a.rewire_relation, not a.no_edge_payload)
    if (set(a.drop_relation) | set(a.rewire_relation)) - set(ALL_RELATIONS):
        raise ValueError('Unknown relation')
    if set(a.drop_relation) & set(a.rewire_relation):
        raise ValueError('Cannot drop and rewire the same relation')
    if a.rounds < 1 or a.token_min_count < 1 or a.min_prior_visits < 0:
        raise ValueError('Invalid training/preprocessing limits')
    targets = load_targets(a.targets)
    num_classes = NUM_CLASSES
    kept = None
    if a.top_k_labels is not None:
        targets, kept = apply_top_k(targets, a.top_k_labels)
        num_classes = len(kept)
    if a.dev_limit is None:
        targets = subsample_train(targets, a.train_limit, sample_seed)
    else:
        # Same sample and dev draw as train.build_dataset, via the shared helpers.
        train_ids = sample_train_ids(targets, a.train_limit, sample_seed)
        eligible = None
        if a.min_prior_visits:
            eligible = {graph['sample_id'] for graph, _ in iter_graphs_with_membership(
                Path(a.artifact) / 'graphs.jsonl', Path(a.artifact) / VISIT_MEMBERSHIP_FILENAME)
                if graph['coverage']['prior_visits'] >= a.min_prior_visits}
        dev_ids = select_dev_ids(targets, train_ids, a.dev_limit, sample_seed, eligible)
        targets = {sid: ((entry[0], 'dev', *entry[2:]) if sid in dev_ids else entry)
                   for sid, entry in targets.items()
                   if entry[1] != 'train' or sid in train_ids or sid in dev_ids}
    prep = fit_features(a.artifact, targets, a.token_min_count, a.min_prior_visits)
    X, y, folds, metadata = build(
        a.artifact, targets, prep, a.min_prior_visits, return_metadata=True,
        edge_mode=a.edges, drop_relations=a.drop_relation, rewire_relations=a.rewire_relation,
        rewire_seed=a.seed, use_edge_payload=not a.no_edge_payload,
        edge_direction=a.edge_direction)
    tr, va, dv = folds == 0, folds == 1, folds == 2
    if not tr.any() or not va.any():
        raise ValueError('Empty train or validation split')
    if a.selection_fold == 'dev' and int(dv.sum()) != a.dev_limit:
        raise ValueError(f'{int(dv.sum())} dev rows encoded for --dev-limit {a.dev_limit}; '
                         'the target sidecar is not artifact-local')
    state = preprocessing_state(prep)

    if a.match_run:
        # Fail closed rather than publish a table whose two rows ran different tasks.
        b = json.loads((Path(a.match_run) / 'binding.json').read_text())
        problems = []
        expected = {'input_contract_version': PREPROCESSING_VERSION,
                    'evaluation_version': EVALUATION_VERSION,
                    'target_binding_sha256': target_binding_sha256,
                    'artifact_graphs_sha256': manifest['graphs_sha256'],
                    'artifact_visit_membership_file': VISIT_MEMBERSHIP_FILENAME,
                    'artifact_visit_membership_sha256': manifest['visit_membership_sha256'],
                    'visit_membership_contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION,
                    'targets_sha256': sha256(a.targets), 'seed': a.seed,
                    'train_limit': a.train_limit, 'edges': a.edges,
                    'edge_payload': not a.no_edge_payload, 'weights': a.weights,
                    'dropped_relations': sorted(a.drop_relation),
                    'rewired_relations': sorted(a.rewire_relation)}
        if a.dev_limit is not None or a.sample_seed is not None:
            expected.update({'sample_seed': sample_seed, 'dev_limit': a.dev_limit})
        for key, value in expected.items():
            if b.get(key) != value:
                problems.append(f'{key} differs')
        source_code = code_source_hashes()
        if b.get('source_code') != source_code:
            problems.append('source_code differs')
        peer_state = json.loads((Path(a.match_run) / 'preprocessing.json').read_text())
        if peer_state != state:
            problems.append('shared preprocessing state differs')
        with np.load(Path(a.match_run) / 'validation.npz') as peer_validation:
            if ('sample_ids' not in peer_validation or not np.array_equal(
                    peer_validation['sample_ids'], metadata['sample_ids'][va])):
                problems.append('validation sample IDs differ or are missing')
        if a.dev_limit is not None:
            dev_path = Path(a.match_run) / 'dev.npz'
            if not dev_path.exists():
                problems.append('peer dev predictions are missing')
            else:
                with np.load(dev_path) as peer_dev:
                    if ('sample_ids' not in peer_dev or not np.array_equal(
                            peer_dev['sample_ids'], metadata['sample_ids'][dv])):
                        problems.append('dev sample IDs differ or are missing')
        if b.get('min_prior_visits', 0) != a.min_prior_visits:
            problems.append(f"min_prior_visits {b.get('min_prior_visits')} "
                            f"vs {a.min_prior_visits}")
        if b.get('num_classes', NUM_CLASSES) != num_classes:
            problems.append(f"classes {b.get('num_classes')} vs {num_classes}")
        if kept is not None and b.get('kept_label_indices') != kept:
            problems.append('kept label set differs')
        for fold, n in (('train', int(tr.sum())), ('validation', int(va.sum())),
                        ('dev', int(dv.sum()))):
            if b['counts'].get(fold, 0) != n:
                problems.append(f"{fold} rows {b['counts'].get(fold, 0)} vs {n}")
        if problems:
            raise SystemExit('protocol mismatch vs ' + a.match_run + ': '
                             + '; '.join(problems))
        print(json.dumps({'stage': 'matched', 'against': a.match_run,
                          'classes': num_classes,
                          'counts': {'train': int(tr.sum()), 'validation': int(va.sum())}}),
              flush=True)
    print(json.dumps({'stage': 'built', 'X': list(X.shape),
                      'train': int(tr.sum()), 'val': int(va.sum()),
                      'seconds': round(time.monotonic() - t0, 1)}), flush=True)

    w = class_weights(y[tr], a.weights, num_classes)

    dtr = xgb.DMatrix(X[tr], label=y[tr], weight=w[y[tr]])
    dva = xgb.DMatrix(X[va], label=y[va])
    params = dict(objective='multi:softprob', num_class=num_classes, tree_method='hist',
                  max_depth=a.max_depth, learning_rate=a.learning_rate,
                  subsample=0.9, colsample_bytree=0.8,
                  reg_lambda=2.0, min_child_weight=3.0, nthread=8, seed=a.seed,
                  disable_default_eval_metric=1)
    booster = xgb.train(params, dtr, num_boost_round=a.rounds)
    selected_rounds, dev_curve, dev_metrics, dev_proba = a.rounds, None, None, None
    if a.selection_fold == 'dev':
        # Round-count selection is the booster's analogue of GNN epoch selection.
        ddev = xgb.DMatrix(X[dv], label=y[dv])
        grid = list(range(a.round_step, a.rounds + 1, a.round_step))
        if not grid or grid[-1] != a.rounds:
            grid.append(a.rounds)
        dev_curve = [{'rounds': r, **metrics(y[dv], booster.predict(
            ddev, iteration_range=(0, r)), num_classes)} for r in grid]
        # Highest dev macro-F1; ties go to the smaller round count.
        selected_rounds = max(dev_curve, key=lambda c: (c['macro_f1'], -c['rounds']))['rounds']
        dev_proba = booster.predict(ddev, iteration_range=(0, selected_rounds))
        dev_metrics = metrics(y[dv], dev_proba, num_classes)
    def predict(model, matrix):
        # The historical fixed-budget path keeps its exact full-model call.
        if a.selection_fold == 'dev':
            return model.predict(matrix, iteration_range=(0, selected_rounds))
        return model.predict(matrix)

    proba = predict(booster, dva) if a.final_eval == 'validation' else None

    m = metrics(y[va], proba, num_classes) if proba is not None else None
    patient_equal = (patient_equal_metrics(y[va], proba, metadata['subjects'][va], num_classes)
                     if proba is not None else None)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    (out / 'preprocessing.json').write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    (out / 'projection.json').write_text(json.dumps(
        {'projection_version': PROJECTION_VERSION,
         'features': projection_layout(prep, a.edge_direction),
         'input': 'encode_graph tensors only', 'edge_payload': not a.no_edge_payload}, indent=2) + '\n')
    fold_masks = [('train', tr), ('validation', va)] + ([('dev', dv)] if dv.any() else [])
    if a.selection_fold == 'dev':
        selection = ('boosting rounds selected on the patient-disjoint dev split; '
                     'validation ' + ('evaluated once at the selected round count'
                                      if proba is not None else 'not evaluated'))
    else:
        selection = 'fixed boosting rounds; validation not used for selection'
    binding = {'preprocessing_version': PREPROCESSING_VERSION,
               'input_contract_version': PREPROCESSING_VERSION,
               'preprocessing_schema_version': PREPROCESSING_VERSION,
               'preprocessing_sha256': sha256(out / 'preprocessing.json'),
               'evaluation_version': EVALUATION_VERSION,
               'projection_version': PROJECTION_VERSION,
               'logic_contract_version': manifest['logic_contract_version'],
               'artifact': str(artifact.resolve()),
               'temporal_clean': manifest['temporal_clean'],
               'temporal_limitations': manifest.get('limitations', []),
               'source_code': code_source_hashes(),
               'label_order': [labels[i] for i in kept] if kept is not None else labels,
               'selection': selection,
               'rounds': a.rounds, 'test_evaluated': False,
               'max_depth': a.max_depth, 'learning_rate': a.learning_rate,
               'round_step': a.round_step if a.selection_fold == 'dev' else None,
               'selected_rounds': selected_rounds,
               'sample_seed': sample_seed, 'selection_fold': a.selection_fold,
               'final_eval': a.final_eval, 'dev_limit': a.dev_limit,
               'dev_policy': DEV_POLICY if a.dev_limit is not None else None,
               'edge_direction': a.edge_direction,
               'num_relations': len(relation_vocabulary(a.edge_direction)),
               'artifact_graphs_sha256': manifest['graphs_sha256'],
               'artifact_visit_membership_file': VISIT_MEMBERSHIP_FILENAME,
               'artifact_visit_membership_sha256': manifest['visit_membership_sha256'],
               'visit_membership_contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION,
               'targets_path': str(Path(a.targets).resolve()),
               'target_binding_sha256': target_binding_sha256,
               'targets_sha256': sha256(a.targets), 'seed': a.seed, 'edges': a.edges,
               'edge_payload': not a.no_edge_payload, 'weights': a.weights,
               'dropped_relations': sorted(a.drop_relation),
               'rewired_relations': sorted(a.rewire_relation),
               'rewiring_policy': REWIRING_POLICY if a.rewire_relation else None,
               'rewiring': {fold: {
                   'changed_edges': int(metadata['rewired_edge_counts'][mask].sum()),
                   'changed_graphs': int((metadata['rewired_edge_counts'][mask] > 0).sum()),
                   'unchanged_graphs': int((metadata['rewired_edge_counts'][mask] == 0).sum())}
                   for fold, mask in (('train', tr), ('validation', va))},
               'num_classes': num_classes, 'kept_label_indices': kept,
               'train_limit': a.train_limit, 'min_prior_visits': a.min_prior_visits,
               'token_min_count': a.token_min_count,
               'split_sample_ids_sha256': {
                   fold: sample_ids_sha256(metadata['sample_ids'][mask].tolist())
                   for fold, mask in fold_masks},
               'counts': {fold: int(mask.sum()) for fold, mask in fold_masks}}
    (out / 'binding.json').write_text(json.dumps(binding, indent=2, sort_keys=True) + '\n')
    booster.save_model(out / 'model.ubj')
    reloaded = xgb.Booster()
    reloaded.load_model(out / 'model.ubj')
    if proba is not None:
        np.testing.assert_array_equal(proba, predict(reloaded, dva))
    (out / 'result.json').write_text(json.dumps(
        {'status': 'completed', 'method': 'xgboost_tabular_control', 'metrics': m, 'patient_equal': patient_equal,
         'binding': binding, 'features': int(X.shape[1]),
         'rounds': a.rounds, 'selected_rounds': selected_rounds, 'seed': a.seed,
         'test_evaluated': False,
         'dev_metrics': dev_metrics, 'dev_curve': dev_curve,
         'validation_evaluations': 0 if proba is None else 1,
         'num_classes': num_classes, 'top_k_labels': a.top_k_labels,
         'min_prior_visits': a.min_prior_visits,
         'kept_label_indices': kept, 'train_limit': a.train_limit,
         'matched_against': a.match_run,
         'counts': {fold: int(mask.sum()) for fold, mask in fold_masks},
         'seconds': round(time.monotonic() - t0, 1)}, indent=2) + '\n')
    if proba is not None:
        np.savez_compressed(out / 'validation.npz', proba=proba, y=y[va],
                            subjects=metadata['subjects'][va],
                            sample_ids=metadata['sample_ids'][va])
    if dev_proba is not None:
        np.savez_compressed(out / 'dev.npz', proba=dev_proba, y=y[dv],
                            subjects=metadata['subjects'][dv],
                            sample_ids=metadata['sample_ids'][dv])
    print(json.dumps({'stage': 'completed', 'selected_rounds': selected_rounds,
                      'dev_macro_f1': dev_metrics['macro_f1'] if dev_metrics else None,
                      **(m or {})}), flush=True)


if __name__ == '__main__':
    main()
