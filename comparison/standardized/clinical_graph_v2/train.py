"""Train the small clinical GNN and report honest, fold-correct metrics.

Scope rules enforced here, not left to discipline:

  * The TEST fold is never loaded. Selection and reporting both use validation.
  * Preprocessing (vocabulary, scalers) is fitted on TRAIN graphs only.
  * Class weights come from full TRAIN counts and are applied to training samples
    only; validation loss and metrics are unweighted.
  * Every run writes a manifest with artifact hashes, source hashes, seed, parameter
    count and the exact arm configuration, plus stored validation probabilities so a
    result can be replayed without retraining.

Arms share a class and explicit flags; bypassed layers change active capacity:

    --edges all|informative|structural   which relations survive tensorization
    --edge-direction forward|bidirectional
                                         producer edges only, or plus typed reverse
                                         edges so evidence can reach the visit hub
    --no-edge-payload                    blank the delta/interval/recency payload
    --no-message-passing                 skip propagation entirely (node-only control)
    --conv edge_conditioned|hgt|gchm|gchm_v2|gchm_v3
                                         shared message MLP, HGT typed attention,
                                         GCHM-PNA, hub-gated compact GCHM-PNA v2, or
                                         v3 (v2 + wide token path, label-wise readout,
                                         jumping knowledge, edge dropout)
    --selection-fold validation|dev      epoch selection on validation (historical)
                                         or on a patient-disjoint dev split drawn from
                                         unused TRAIN patients (--dev-limit), after
                                         which validation is evaluated exactly once

`--conv hgt` is not capacity-matched to the others by construction: typed projections
cost parameters. `--hidden` is therefore reported alongside every result, and a fair
HGT-vs-baseline claim needs a width-matched baseline run, not just this flag.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .contracts import (VISIT_MEMBERSHIP_CONTRACT_VERSION, VISIT_MEMBERSHIP_FILENAME,
                        iter_graphs_with_membership, sample_ids_sha256,
                        recursive_source_hashes,
                        validate_artifact_manifest, validate_control_configuration,
                        verify_graph_file, verify_target_binding,
                        verify_visit_membership_file)
from .methods import ClinicalMethodAdapter, build_method
from .gchm_v2 import AGGREGATIONS, GCHMv2, READOUTS
from .gchm_v3 import GCHMv3
from .gchm_v3 import READOUTS as V3_READOUTS
from .model import ClinicalGNN
from .schema import sha256
from .rewiring import REWIRING_POLICY
from .tensorize import (ALL_RELATIONS, EDGE_DIRECTIONS, PAYLOAD_WIDTH, PREPROCESSING_VERSION,
                        degree_histogram, edge_feature_layout, encode_graph,
                        fit_preprocessing, preprocessing_state, relation_vocabulary,
                        triple_count)

NUM_CLASSES = 30
EVALUATION_VERSION = 'visit_targets_patient_equal_v2'
METHOD_DEFAULTS = {
    'clinical_gnn': dict(hidden=96, layers=3, dropout=0.1, lr=1e-3,
                         weight_decay=1e-5, batch_size=64, epochs=12,
                         patience=None, min_delta=None),
    'protgnn': dict(hidden=128, layers=3, dropout=0.51, lr=1.79e-3,
                    weight_decay=4.3e-5, batch_size=128, epochs=300,
                    patience=10, min_delta=0.005),
    'protonode': dict(hidden=128, layers=3, dropout=0.51, lr=1.79e-3,
                      weight_decay=4.3e-5, batch_size=128, epochs=300,
                      patience=10, min_delta=0.005),
    'gsat': dict(hidden=128, layers=3, dropout=0.3, lr=1e-3,
                 weight_decay=0.0, batch_size=128, epochs=100,
                 patience=10, min_delta=0.005),
    'graphcare': dict(hidden=128, layers=2, dropout=0.3, lr=1e-3,
                      weight_decay=1e-5, batch_size=32, epochs=100,
                      patience=10, min_delta=0.005),
}
# GCHM-PNA v2 is a new arm, so it carries its own profile rather than inheriting the
# v1 clinical_gnn defaults. hidden=92 keeps it below ProtGNN's 399,884 parameters on
# the bidirectional view; dropout/weight decay answer v1's measured overfitting.
GCHM_V2_DEFAULTS = dict(hidden=92, layers=3, dropout=0.3, lr=1e-3,
                        weight_decay=1e-4, batch_size=64, epochs=40,
                        patience=None, min_delta=None)
# v3 keeps v2's optimiser profile; hidden 88 keeps it below the frozen ProtGNN count
# (mechanism_check.v3_parameters_below_protgnn: 391,448 < 399,884 bidirectional).
GCHM_V3_DEFAULTS = dict(GCHM_V2_DEFAULTS, hidden=88)
SELECTION_FOLDS = ('validation', 'dev')
FINAL_EVALS = ('validation', 'none')
DEV_POLICY = ('labelled TRAIN-fold rows whose patient has no row in the drawn training '
              'sample; min-prior-visits eligibility applied; sampled with '
              'random.Random("dev-<sample_seed>") over sorted sample ids')
METHOD_NATIVE_ARGUMENTS = {
    'clinical_gnn': (),
    'protgnn': (
        'warm_epochs', 'proj_epochs', 'proj_interval', 'nearest_graphs',
        'prototypes_per_class', 'cluster_weight', 'separation_weight', 'margin',
        'rollout', 'min_atoms', 'max_atoms', 'expand_atoms', 'c_puct',
    ),
    'protonode': ('protonode_readout', 'protonode_no_wide', 'protonode_wide_l1',
        'protonode_warm_epochs', 'protonode_proj_epochs', 'protonode_proj_interval',
        'protonode_nearest_graphs', 'protonode_prototypes_per_class',
        'protonode_cluster_weight', 'protonode_separation_weight', 'protonode_margin'),
    'gsat': (
        'gsat_temperature', 'gsat_info_loss_coef', 'gsat_init_r', 'gsat_final_r',
        'gsat_decay_interval', 'gsat_decay_r', 'gsat_extractor_dropout',
    ),
    'graphcare': ('graphcare_decay_rate', 'graphcare_message_dropout'),
}


def method_defaults(method, conv=None):
    """Native defaults of one arm; GCHM-PNA v2 carries its own profile."""
    if method == 'clinical_gnn' and conv == 'gchm_v2':
        return dict(GCHM_V2_DEFAULTS)
    if method == 'clinical_gnn' and conv == 'gchm_v3':
        return dict(GCHM_V3_DEFAULTS)
    return dict(METHOD_DEFAULTS[method])


def normalize_method_args(args, parser=None):
    """Fill method profiles while retaining legacy direct-call Namespaces."""
    if getattr(args, '_method_args_normalized', False):
        return args

    def fail(message):
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    if not hasattr(args, 'method') or args.method is None:
        args.method = 'clinical_gnn'
    if args.method not in METHOD_DEFAULTS:
        raise ValueError('Unknown clinical method: ' + str(args.method))
    if args.method != 'clinical_gnn' and getattr(args, 'conv', None) is not None:
        fail('--conv is only valid with --method clinical_gnn')
    if args.method == 'clinical_gnn' and getattr(args, 'conv', None) is None:
        args.conv = 'edge_conditioned'
    is_gchm_v2 = args.method == 'clinical_gnn' and args.conv == 'gchm_v2'
    v2_only = [flag for flag, name in (('--aggregation', 'aggregation'),
                                       ('--readout', 'readout'))
               if getattr(args, name, None) is not None]
    if getattr(args, 'no_hub_gate', False):
        v2_only.append('--no-hub-gate')
    if v2_only and not is_gchm_v2:
        fail(', '.join(v2_only) + ' only valid with --conv gchm_v2')
    if is_gchm_v2:
        args.aggregation = getattr(args, 'aggregation', None) or 'pna'
        args.readout = getattr(args, 'readout', None) or 'hub'
        if args.aggregation not in AGGREGATIONS or args.readout not in READOUTS:
            fail('unknown gchm_v2 aggregation or readout')
    is_gchm_v3 = args.method == 'clinical_gnn' and args.conv == 'gchm_v3'
    v3_only = [flag for flag, name in (('--v3-readout', 'v3_readout'),
                                       ('--edge-dropout', 'edge_dropout'))
               if getattr(args, name, None) is not None]
    v3_only += [flag for flag, name in (('--no-wide', 'no_wide'), ('--no-jk', 'no_jk'))
                if getattr(args, name, False)]
    if v3_only and not is_gchm_v3:
        fail(', '.join(v3_only) + ' only valid with --conv gchm_v3')
    if is_gchm_v3:
        args.v3_readout = getattr(args, 'v3_readout', None) or 'labelwise'
        if args.v3_readout not in V3_READOUTS:
            fail('unknown gchm_v3 readout')
        edge_dropout = getattr(args, 'edge_dropout', None)
        args.edge_dropout = 0.1 if edge_dropout is None else float(edge_dropout)
        if not 0.0 <= args.edge_dropout < 1.0:
            fail('--edge-dropout must be in [0, 1)')
    for name, default in (('edge_direction', 'forward'), ('selection_fold', 'validation'),
                          ('final_eval', 'validation'), ('dev_limit', None),
                          ('sample_seed', None), ('no_hub_gate', False),
                          ('no_wide', False), ('no_jk', False), ('v3_readout', None),
                          ('edge_dropout', None)):
        if getattr(args, name, None) is None:
            setattr(args, name, default)
    if args.edge_direction not in EDGE_DIRECTIONS:
        fail('--edge-direction must be one of ' + ', '.join(EDGE_DIRECTIONS))
    if args.selection_fold not in SELECTION_FOLDS or args.final_eval not in FINAL_EVALS:
        fail('unknown --selection-fold or --final-eval')
    if args.selection_fold == 'validation' and args.final_eval != 'validation':
        fail('--final-eval none requires --selection-fold dev')
    if args.selection_fold == 'dev' and args.dev_limit is None:
        fail('--selection-fold dev requires --dev-limit')
    if args.selection_fold != 'dev' and args.dev_limit is not None:
        fail('--dev-limit requires --selection-fold dev')
    if args.dev_limit is not None and (isinstance(args.dev_limit, bool)
                                       or int(args.dev_limit) < 1):
        fail('--dev-limit must be a positive integer')
    method_overrides = []
    for method, names in METHOD_NATIVE_ARGUMENTS.items():
        for name in names:
            if getattr(args, name, None) is None:
                continue
            if method != args.method:
                option_name = name.removeprefix(method + '_')
                option = '--' + method + '-' + option_name.replace('_', '-')
                fail(f'{option} is only valid with --method {method}')
            method_overrides.append(name)
    defaults = method_defaults(args.method, args.conv)
    supplied = []
    for name, default in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, default)
        else:
            supplied.append(name)
    for name, default in (
        ('patience', defaults['patience']),
        ('min_delta', defaults['min_delta']),
        ('weight_decay', defaults['weight_decay']),
    ):
        if not hasattr(args, name):
            setattr(args, name, default)
    if any(getattr(args, name) < 1 for name in ('hidden', 'layers', 'batch_size', 'epochs')):
        raise ValueError('hidden, layers, batch-size, and epochs must be positive')
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError('dropout must be in [0, 1)')
    if args.lr < 0.0 or args.weight_decay < 0.0:
        raise ValueError('learning rate and weight decay must be nonnegative')
    if args.patience is not None and args.patience < 1:
        raise ValueError('patience must be positive when configured')
    if args.min_delta is not None and args.min_delta < 0.0:
        raise ValueError('min-delta must be nonnegative when configured')
    args._common_overrides = sorted(set(supplied))
    args._method_overrides = sorted(set(method_overrides))
    args._method_args_normalized = True
    return args


def select_top_labels(targets, k):
    """Keep the k most frequent TRAIN labels; return (kept ids, remap, dropped counts).

    Frequency is counted on the TRAIN fold only. Ranking classes by their overall
    frequency would let the validation fold decide which classes exist, which is a
    label-distribution leak even though no individual validation row is read.

    Rows outside the kept set are DROPPED, not merged into an "other" class. Merging
    would invent a heterogeneous class that no clinician would recognise and would
    quietly re-inflate the difficulty this option is removing. Ties at the boundary
    are broken by class index so the selection is deterministic.

    **This changes the task, not the model.** A k-class score is not comparable with
    a 30-class score: fewer classes means a higher random floor, a higher majority
    baseline, and the hardest rare classes removed from the denominator of macro-F1.
    Every run records `num_classes` and `dropped_rows` so the two can never be put
    in one table by accident.
    """
    counts = Counter(t for t, split, _ in targets.values() if split == 'train')
    if not counts:
        raise ValueError('No training rows to rank labels by')
    if k < 1 or k > len(counts):
        raise ValueError(f'--top-k-labels {k} exceeds the {len(counts)} training classes')
    ranked = sorted(counts, key=lambda c: (-counts[c], c))[:k]
    kept = sorted(ranked)
    remap = {c: i for i, c in enumerate(kept)}
    dropped = Counter()
    filtered = {}
    for sid, (t, split, subject) in targets.items():
        if t in remap:
            filtered[sid] = (remap[t], split, subject)
        else:
            dropped[split] += 1
    return filtered, kept, dict(dropped)


def sample_train_ids(targets, limit, sample_seed):
    """The seeded TRAIN subsample shared by every arm, including tabular_control."""
    train_ids = {sid for sid, entry in targets.items() if entry[1] == 'train'}
    if limit is None:
        return train_ids
    rng = random.Random(sample_seed)
    return set(rng.sample(sorted(train_ids), min(limit, len(train_ids))))


def select_dev_ids(targets, train_ids, dev_limit, sample_seed, eligible=None):
    """Patient-disjoint development rows from TRAIN-fold patients outside the sample.

    Epoch and hyper-parameter selection happen on these rows, so the validation fold
    is read once per final run and the test fold stays closed. Excluding every patient
    of the drawn sample stops one patient from informing both fitting and selection.
    """
    if dev_limit is None:
        return frozenset()
    if isinstance(dev_limit, bool) or int(dev_limit) < 1:
        raise ValueError('dev_limit must be a positive integer')
    sampled_subjects = {targets[sid][2] for sid in train_ids}
    candidates = sorted(sid for sid, entry in targets.items()
                        if entry[1] == 'train' and sid not in train_ids
                        and entry[2] not in sampled_subjects
                        and (eligible is None or sid in eligible))
    if len(candidates) < int(dev_limit):
        raise ValueError(f'only {len(candidates)} patient-disjoint TRAIN rows remain for '
                         f'a {int(dev_limit)}-row dev split; lower --dev-limit or '
                         '--train-limit')
    rng = random.Random(f'dev-{sample_seed}')
    return frozenset(rng.sample(candidates, int(dev_limit)))


def metrics(y_true, proba, num_classes=NUM_CLASSES, sample_weight=None):
    """Visit-level metrics, with a fixed label universe for macro-F1.

    Balanced accuracy averages recalls of observed target classes, as sklearn does.
    Top-k ties prefer the larger class index; k is capped at the class count.
    """
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    if (y_true.ndim != 1 or not len(y_true) or num_classes < 1
            or proba.shape != (len(y_true), num_classes)
            or not np.issubdtype(y_true.dtype, np.integer)
            or np.any((y_true < 0) | (y_true >= num_classes))
            or not np.isfinite(proba).all()):
        raise ValueError('Metrics require aligned, finite probabilities and class indices')
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if (sample_weight.shape != y_true.shape or not np.isfinite(sample_weight).all()
                or np.any(sample_weight < 0) or sample_weight.sum() <= 0):
            raise ValueError('Invalid evaluation sample weights')
    pred = proba.argmax(1)
    ranking = np.argsort(proba, axis=1, kind='stable')
    def top_k(k):
        correct = np.any(ranking[:, -min(k, num_classes):] == y_true[:, None], axis=1)
        return round(float(np.average(correct, weights=sample_weight)), 6)
    return {
        'accuracy': round(float(accuracy_score(y_true, pred, sample_weight=sample_weight)), 6),
        'balanced_acc': round(float(recall_score(
            y_true, pred, labels=np.unique(y_true), average='macro',
            sample_weight=sample_weight, zero_division=0)), 6),
        'macro_f1': round(float(f1_score(
            y_true, pred, labels=np.arange(num_classes), average='macro',
            sample_weight=sample_weight, zero_division=0)), 6),
        'micro_f1': round(float(f1_score(
            y_true, pred, labels=np.arange(num_classes), average='micro',
            sample_weight=sample_weight, zero_division=0)), 6),
        'top3_acc': top_k(3),
        'top5_acc': top_k(5),
    }


def patient_equal_metrics(y_true, proba, subjects, num_classes=NUM_CLASSES):
    """Keep visit targets/predictions; give each evaluated patient total weight 1.

    A patient can have different diagnoses at different visits. Averaging their
    predictions and choosing a majority diagnosis would define a different task.
    Counts refer only to visits in this evaluation cohort, not their raw history.
    """
    y, proba = np.asarray(y_true), np.asarray(proba)
    subjects = np.asarray(subjects).astype(str)
    if (y.ndim != 1 or subjects.ndim != 1 or not len(y)
            or proba.shape != (len(y), num_classes) or len(subjects) != len(y)):
        raise ValueError('Patient-equal metrics require aligned, nonempty visit arrays')
    counts = Counter(subjects)
    weights = np.asarray([1.0 / counts[s] for s in subjects])
    return {'patients': len(counts), 'visits': len(y),
            'weight_policy': 'inverse_evaluated_visits_per_patient',
            **metrics(y, proba, num_classes, sample_weight=weights)}


def per_class_table(y_true, proba, labels, num_classes=NUM_CLASSES):
    from sklearn.metrics import precision_recall_fscore_support
    pred = proba.argmax(1)
    p, r, f, s = precision_recall_fscore_support(
        y_true, pred, labels=np.arange(num_classes), zero_division=0)
    return [{'index': i, 'label': labels[i], 'precision': round(float(p[i]), 6),
             'recall': round(float(r[i]), 6), 'f1': round(float(f[i]), 6),
             'support': int(s[i])} for i in range(num_classes)]


def load_targets(path):
    targets = {}
    seen, patient_splits = set(), {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            sid, subject, split = row['sample_id'], row['subject_id'], row['split']
            if not sid or not subject or split not in {'train', 'validation', 'test'}:
                raise ValueError('Target row has missing identity or invalid fold')
            if sid in seen:
                raise ValueError('Duplicate sample_id in target sidecar')
            seen.add(sid)
            if subject in patient_splits and patient_splits[subject] != split:
                raise ValueError('A patient occurs in more than one fold')
            patient_splits[subject] = split
            if row['target'] and row['target'] != '-1':
                target = int(row['target'])
                if not 0 <= target < NUM_CLASSES:
                    raise ValueError('Target index is outside the canonical class order')
                targets[sid] = (target, split, subject)
    if not targets:
        raise ValueError('No labelled samples in the target sidecar')
    return targets


def build_dataset(artifact, targets, edge_mode, limit, token_min_count, seed,
                  drop_relations=(), rewire_relations=(), min_prior_visits=0, *,
                  edge_direction='forward', dev_limit=None, sample_seed=None):
    """Encode labelled graphs. Test-fold rows are dropped before any tensor work.

    `min_prior_visits` keeps only visits with at least that many COMPLETED earlier
    encounters, i.e. the multi-visit cohort. The filter is applied to every fold
    alike, so train and validation describe the same population; filtering only one
    side would train on one task and score on another.

    Selecting on prior-visit count concentrates repeated patients: they contribute
    one row per visit, so a patient with 25 eligible visits outweighs 25 single-visit
    patients. Subject-level fold integrity still holds (a patient never spans folds),
    so this is a weighting distortion, not leakage -- and `patient_equal_metrics`
    reports inverse-visit-weighted metrics with each visit target retained.

    `sample_seed` (default: `seed`) draws the TRAIN sample; `dev_limit` adds a
    `dev` split (see `select_dev_ids`). Preprocessing is fitted on the sample only.
    """
    graphs_path = Path(artifact) / 'graphs.jsonl'
    membership_path = Path(artifact) / VISIT_MEMBERSHIP_FILENAME
    sample_seed = seed if sample_seed is None else sample_seed
    # Deterministic subsample of TRAIN ids; validation always stays complete.
    train_ids = sample_train_ids(targets, limit, sample_seed)
    eligible = None
    if min_prior_visits:
        eligible = {graph['sample_id'] for graph, _ in
                    iter_graphs_with_membership(graphs_path, membership_path)
                    if graph['coverage']['prior_visits'] >= min_prior_visits}
    dev_ids = select_dev_ids(targets, train_ids, dev_limit, sample_seed, eligible)
    if eligible is not None:
        train_ids &= eligible
        if not train_ids:
            raise ValueError('No training visits meet --min-prior-visits')
    prep = fit_preprocessing(graphs_path, train_ids, token_min_count,
                             membership_path=membership_path)
    splits = {'train': [], 'validation': []}
    if dev_ids:
        splits['dev'] = []
    labels_seen = Counter()
    for graph, visit_membership in iter_graphs_with_membership(graphs_path, membership_path):
        entry = targets.get(graph['sample_id'])
        if entry is None:
            continue
        y, split, subject = entry
        if split == 'test':
            continue  # held out; never loaded
        if graph['sample_id'] in dev_ids:
            split = 'dev'
        elif split == 'train' and graph['sample_id'] not in train_ids:
            continue
        if graph['coverage']['prior_visits'] < min_prior_visits:
            continue
        data = encode_graph(graph, prep, visit_membership, edge_mode, drop_relations,
                            rewire_relations, seed, edge_direction=edge_direction)
        data.y = torch.tensor([y], dtype=torch.long)
        data.subject = subject
        splits[split].append(data)
        labels_seen[split] += 1
    if dev_ids and len(splits['dev']) != len(dev_ids):
        raise ValueError(f"{len(dev_ids) - len(splits['dev'])} dev rows are absent from the "
                         'artifact; the target sidecar is not artifact-local')
    if not all(splits.values()):
        raise ValueError('Empty train, validation or dev split')
    return splits, prep


def class_weights(labels, policy, num_classes=NUM_CLASSES):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if (counts == 0).any():
        # A class absent from this training subsample gets weight 1 rather than inf.
        counts = np.where(counts == 0, 1.0, counts)
    w = len(labels) / (num_classes * counts)
    if policy == 'sqrt_inverse':
        return np.sqrt(w)
    if policy == 'inverse':
        return w
    if policy == 'none':
        return np.ones(num_classes)
    raise ValueError('Unknown weight policy')


@torch.no_grad()
def evaluate(model, loader, device, *, epoch=0):
    model.eval()
    logits, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        if isinstance(model, ClinicalMethodAdapter):
            output = model(batch, epoch=epoch)
            logits.append(output.logits.cpu())
        else:
            logits.append(model(batch).cpu())
        ys.append(batch.y.cpu())
    logits = torch.cat(logits)
    return torch.softmax(logits, dim=1).numpy(), torch.cat(ys).numpy()


def early_stopping_start_epoch(method, model):
    """Return the first zero-based epoch where patience may be consumed."""
    if method in ('protgnn', 'protonode'):
        return int(model.proj_epochs)
    return 0


def clip_gradients(method, model):
    """Method-native gradient clipping; prototype arms share ProtGNN's value clip."""
    if method == 'clinical_gnn':
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    elif method in ('protgnn', 'protonode'):
        torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)


def run(args):
    args = normalize_method_args(args)
    if args.method != 'clinical_gnn':
        if getattr(args, 'no_edge_payload', False):
            raise ValueError('--no-edge-payload is only valid with --method clinical_gnn')
        if getattr(args, 'no_message_passing', False):
            raise ValueError('--no-message-passing is only valid with --method clinical_gnn')
    t0 = time.monotonic()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError('Occupied output directory; choose a fresh path')


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(False)

    artifact = Path(args.artifact).resolve()
    manifest = json.loads((artifact / 'manifest.json').read_text())
    validate_artifact_manifest(manifest)
    verify_graph_file(artifact, manifest)
    verify_visit_membership_file(artifact, manifest)
    labels = json.loads(Path(args.canonical).read_text())['classes']
    if len(labels) != NUM_CLASSES:
        raise ValueError('Canonical class order has an unexpected length')
    recorded_labels = manifest.get('reference_class_order_only')
    if recorded_labels is not None and recorded_labels != labels:
        raise ValueError('Artifact label order differs from the canonical class order')
    target_binding_sha256 = verify_target_binding(manifest, args.targets, labels)

    targets = load_targets(args.targets)
    kept_labels, dropped_rows = None, {}
    num_classes = NUM_CLASSES
    if args.top_k_labels is not None:
        targets, kept_labels, dropped_rows = select_top_labels(targets, args.top_k_labels)
        num_classes = len(kept_labels)
    drop_relations = tuple(sorted(args.drop_relation or ()))
    rewire_relations = tuple(sorted(args.rewire_relation or ()))
    unknown = (set(drop_relations) | set(rewire_relations)) - set(ALL_RELATIONS)
    if unknown:
        raise ValueError(f'unknown relation(s): {sorted(unknown)}')
    both = set(drop_relations) & set(rewire_relations)
    if both:
        raise ValueError(f'cannot drop and rewire the same relation: {sorted(both)}')
    validate_control_configuration(rewire_relations, not args.no_edge_payload)
    edge_direction = args.edge_direction
    sample_seed = args.seed if args.sample_seed is None else args.sample_seed
    selection_fold, final_eval = args.selection_fold, args.final_eval
    is_gchm_v2 = args.method == 'clinical_gnn' and args.conv == 'gchm_v2'
    is_gchm_v3 = args.method == 'clinical_gnn' and args.conv == 'gchm_v3'
    out.mkdir(parents=True, mode=0o700)
    splits, prep = build_dataset(artifact, targets, args.edges, args.train_limit,
                                 args.token_min_count, args.seed, drop_relations,
                                 rewire_relations, args.min_prior_visits,
                                 edge_direction=edge_direction, dev_limit=args.dev_limit,
                                 sample_seed=sample_seed)

    device = torch.device(args.device)
    train_loader = DataLoader(splits['train'], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(splits['validation'], batch_size=args.batch_size)
    select_loader = (DataLoader(splits['dev'], batch_size=args.batch_size)
                     if selection_fold == 'dev' else val_loader)

    node_dim = splits['train'][0].x.size(1)
    edge_dim = splits['train'][0].edge_attr.size(1)
    # index 0 = meta-relation unseen in train; reverse edges reuse the fitted triples.
    num_triples = triple_count(prep, edge_direction)
    num_relations = len(relation_vocabulary(edge_direction))
    # PNA needs the in-degree distribution. Fit it on TRAIN graphs only, exactly as
    # the vocabulary and scalers are, so no validation structure reaches the model.
    hist = (degree_histogram(splits['train'])
            if args.method == 'clinical_gnn' and args.conv in ('gchm', 'gchm_v2', 'gchm_v3')
            else None)
    if is_gchm_v3:
        model = GCHMv3(num_tokens=len(prep['vocabulary']), node_dim=node_dim,
                       edge_dim=edge_dim, num_classes=num_classes,
                       num_relations=num_relations, degree_histogram=hist,
                       hidden=args.hidden, layers=args.layers, dropout=args.dropout,
                       token_dim=args.token_dim,
                       use_edge_payload=not args.no_edge_payload,
                       use_message_passing=not args.no_message_passing,
                       readout=args.v3_readout, wide=not args.no_wide, jk=not args.no_jk,
                       edge_dropout=args.edge_dropout).to(device)
        method_config = {
            'method': 'clinical_gnn',
            'adaptation_version': 'clinical_graph_v2_gchm_pna_v3',
            'native_defaults': dict(GCHM_V3_DEFAULTS),
            'effective_settings': dict(model.settings),
            'architecture': {
                'parameter_count': model.parameter_count(),
                'active_parameter_count': model.active_parameter_count(),
            },
        }
    elif is_gchm_v2:
        model = GCHMv2(num_tokens=len(prep['vocabulary']), node_dim=node_dim,
                       edge_dim=edge_dim, num_classes=num_classes,
                       num_relations=num_relations, degree_histogram=hist,
                       hidden=args.hidden, layers=args.layers, dropout=args.dropout,
                       token_dim=args.token_dim,
                       use_edge_payload=not args.no_edge_payload,
                       use_message_passing=not args.no_message_passing,
                       modulation=args.modulation, aggregation=args.aggregation,
                       readout=args.readout, hub_gate=not args.no_hub_gate).to(device)
        method_config = {
            'method': 'clinical_gnn',
            'adaptation_version': 'clinical_graph_v2_gchm_pna_v2',
            'native_defaults': dict(GCHM_V2_DEFAULTS),
            'effective_settings': dict(model.settings),
            'architecture': {
                'parameter_count': model.parameter_count(),
                'active_parameter_count': model.active_parameter_count(),
            },
        }
    elif args.method == 'clinical_gnn':
        model = ClinicalGNN(num_classes=num_classes,
                            num_tokens=len(prep['vocabulary']), node_dim=node_dim,
                            edge_dim=edge_dim, hidden=args.hidden, layers=args.layers,
                            dropout=args.dropout, token_dim=args.token_dim,
                            use_edge_payload=not args.no_edge_payload,
                            use_message_passing=not args.no_message_passing,
                            conv=args.conv, num_triples=num_triples,
                            payload_dim=PAYLOAD_WIDTH, heads=args.heads,
                            degree_histogram=hist,
                            modulation=args.modulation).to(device)
        method_config = {
            'method': 'clinical_gnn',
            'adaptation_version': None,
            'native_defaults': dict(METHOD_DEFAULTS['clinical_gnn']),
            'architecture': {
                'parameter_count': model.parameter_count(),
                'active_parameter_count': model.active_parameter_count(),
            },
        }
    else:
        model = build_method(
            args.method,
            num_tokens=len(prep['vocabulary']),
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_classes=num_classes,
            hidden=args.hidden,
            layers=args.layers,
            dropout=args.dropout,
            token_dim=args.token_dim,
            num_triples=num_triples,
            args=args,
        ).to(device)
        method_config = model.run_config()

    y_train = np.array([int(d.y) for d in splits['train']])
    weights = torch.tensor(class_weights(y_train, args.weights, num_classes),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    if isinstance(model, ClinicalMethodAdapter):
        optimizer = torch.optim.Adam(
            model.optimizer_groups(args),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    if kept_labels is not None:
        # Keep the ORIGINAL class names against the new contiguous indices, so a
        # k-class report still names real diagnoses instead of 0..k-1.
        labels = [labels[c] for c in kept_labels]

    state = preprocessing_state(prep)
    if state.get('preprocessing_version') != PREPROCESSING_VERSION:
        raise ValueError('Incompatible preprocessing state')
    (out / 'preprocessing.json').write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    architecture = method_config['architecture']
    parameter_count = int(architecture['parameter_count'])
    active_parameter_count = int(
        architecture.get('joint_active_parameter_count',
                         architecture.get('active_parameter_count', parameter_count)))
    early_stop_start = (early_stopping_start_epoch(args.method, model)
                        if isinstance(model, ClinicalMethodAdapter) else 0)
    source_code = recursive_source_hashes(Path(__file__).parent)
    native_defaults = method_defaults(args.method, args.conv)
    if selection_fold == 'dev':
        selection = ('best dev macro_f1 on a patient-disjoint split of unused TRAIN '
                     'patients; validation '
                     + ('evaluated once at the selected checkpoint'
                        if final_eval == 'validation' else 'not evaluated')
                     + ('; adapter early stopping on dev with configured '
                        'patience/min_delta after method schedule gate'
                        if isinstance(model, ClinicalMethodAdapter)
                        else '; full fixed budget; no early stop'))
    else:
        selection = ('best validation macro_f1; adapter early stopping on configured '
                     'patience/min_delta after method schedule gate'
                     if isinstance(model, ClinicalMethodAdapter)
                     else 'best validation macro_f1; full fixed budget; no early stop')
    binding = {
        'method': args.method,
        'adaptation_version': method_config.get('adaptation_version'),
        'method_config': method_config,
        'method_native_defaults': method_config.get('native_defaults', native_defaults),
        'common_overrides': args._common_overrides,
        'method_overrides': args._method_overrides,
        'artifact': str(artifact),
        'artifact_graphs_sha256': manifest['graphs_sha256'],
        'artifact_visit_membership_file': VISIT_MEMBERSHIP_FILENAME,
        'artifact_visit_membership_sha256': manifest['visit_membership_sha256'],
        'visit_membership_contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION,
        'artifact_schema': manifest['schema_version'],
        'logic_contract_version': manifest['logic_contract_version'],
        'evaluation_version': EVALUATION_VERSION,
        'input_contract_version': PREPROCESSING_VERSION,
        'preprocessing_schema_version': PREPROCESSING_VERSION,
        'preprocessing_sha256': sha256(out / 'preprocessing.json'),
        'token_min_count': args.token_min_count,
        'temporal_clean': manifest['temporal_clean'],
        'temporal_limitations': manifest.get('limitations', []),
        'diagnosis_tier': manifest.get('diagnosis_tier', False),
        'targets_sha256': sha256(args.targets),
        'targets_path': str(Path(args.targets).resolve()),
        'target_binding_sha256': target_binding_sha256,
        'label_order': labels,
        'source_code': source_code,
        'seed': args.seed, 'edges': args.edges,
        'edge_direction': edge_direction,
        'num_relations': num_relations,
        'edge_feature_layout': edge_feature_layout(prep, edge_direction),
        'sample_seed': sample_seed,
        'selection_fold': selection_fold,
        'final_eval': final_eval,
        'dev_limit': args.dev_limit,
        'dev_policy': DEV_POLICY if args.dev_limit is not None else None,
        'num_classes': num_classes,
        'min_prior_visits': args.min_prior_visits,
        'top_k_labels': args.top_k_labels,
        'kept_label_indices': kept_labels,
        'dropped_rows': dropped_rows,
        'edge_payload': not args.no_edge_payload,
        'message_passing': not args.no_message_passing,
        'conv': args.conv if args.method == 'clinical_gnn' else None,
        'heads': args.heads if args.method == 'clinical_gnn' and args.conv == 'hgt' else None,
        'modulation': (args.modulation
                       if args.method == 'clinical_gnn' and args.conv in ('gchm', 'gchm_v2')
                       else None),
        'aggregation': args.aggregation if is_gchm_v2 else None,
        'readout': args.readout if is_gchm_v2 else None,
        'hub_gate': (not args.no_hub_gate) if is_gchm_v2 else None,
        'v3_readout': args.v3_readout if is_gchm_v3 else None,
        'wide': (not args.no_wide) if is_gchm_v3 else None,
        'jk': (not args.no_jk) if is_gchm_v3 else None,
        'edge_dropout': args.edge_dropout if is_gchm_v3 else None,
        'degree_histogram_fit': 'train fold only' if hist is not None else None,
        'dropped_relations': list(drop_relations),
        'rewired_relations': list(rewire_relations),
        'rewire_seed': args.seed if rewire_relations else None,
        'rewiring_policy': REWIRING_POLICY if rewire_relations else None,
        'rewiring': {fold: {
            'changed_edges': sum(d.rewired_edge_count for d in data),
            'changed_graphs': sum(d.rewired_edge_count > 0 for d in data),
            'unchanged_graphs': sum(d.rewired_edge_count == 0 for d in data)}
            for fold, data in splits.items()},
        'num_meta_relations': num_triples,
        'meta_relations': (['<unseen>'] + prep['triples']
                           + (['rev:' + triple for triple in prep['triples']]
                              if edge_direction == 'bidirectional' else [])),
        'weights': args.weights,
        'class_weight_values': [float(value) for value in weights.detach().cpu().tolist()],
        'class_counts_train': np.bincount(y_train, minlength=num_classes).tolist(),
        'hidden': args.hidden, 'layers': args.layers,
        'dropout': args.dropout, 'lr': args.lr, 'weight_decay': args.weight_decay,
        'batch_size': args.batch_size, 'epochs': args.epochs,
        'patience': args.patience, 'min_delta': args.min_delta,
        'early_stopping_start_epoch_index': (early_stop_start
                                             if isinstance(model, ClinicalMethodAdapter)
                                             else None),
        'optimizer': 'Adam' if isinstance(model, ClinicalMethodAdapter) else 'AdamW',
        'train_limit': args.train_limit,
        'parameter_count': parameter_count,
        'active_parameter_count': active_parameter_count,
        'split_sample_ids_sha256': {fold: sample_ids_sha256(d.sample_id for d in data)
                                    for fold, data in splits.items()},
        'counts': {fold: len(data) for fold, data in splits.items()},
        'node_dim': node_dim, 'edge_dim': edge_dim,
        'vocabulary_size': len(prep['vocabulary']),
        'device': str(device), 'torch': torch.__version__,
        'selection': selection,
        'test_evaluated': False,
    }
    (out / 'binding.json').write_text(json.dumps(binding, indent=2, sort_keys=True) + '\n')

    print(json.dumps({'stage': 'bound', **{k: binding[k] for k in
                                           ('method', 'parameter_count', 'counts', 'edges',
                                            'edge_direction', 'selection_fold',
                                            'edge_payload', 'message_passing',
                                            'conv', 'num_meta_relations')}}), flush=True)

    # BG-HGNN's relation-collapse check, measured at init so the trained value has
    # something to be compared against rather than being read in isolation.
    separation_at_init = (model.relation_separation()
                          if not isinstance(model, ClinicalMethodAdapter) else None)

    history, best = [], None
    early_stop_best, epochs_without_gain = None, 0
    for epoch_index in range(args.epochs):
        model.train()
        if isinstance(model, ClinicalMethodAdapter):
            model.on_epoch_start(epoch_index, train_loader)
        total, auxiliary_total, seen = 0.0, 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, ClinicalMethodAdapter):
                output = model(batch, epoch=epoch_index)
                loss = criterion(output.logits, batch.y.view(-1)) + output.auxiliary_loss
                auxiliary_total += float(output.auxiliary_loss.detach()) * batch.num_graphs
            else:
                loss = criterion(model(batch), batch.y.view(-1))
            loss.backward()
            clip_gradients(args.method, model)
            optimizer.step()
            total += float(loss.detach()) * batch.num_graphs
            seen += batch.num_graphs
        proba, y_select = evaluate(model, select_loader, device, epoch=epoch_index)
        m = metrics(y_select, proba, num_classes)
        display_epoch = epoch_index + 1
        row = {'epoch': display_epoch, 'train_loss': round(total / seen, 6), **m,
               'seconds': round(time.monotonic() - t0, 1)}
        if selection_fold == 'dev':
            row['selection_fold'] = 'dev'
        if isinstance(model, ClinicalMethodAdapter):
            row['train_auxiliary_loss'] = round(auxiliary_total / seen, 6)
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or m['macro_f1'] > best['metrics']['macro_f1']:
            best = {'epoch': display_epoch, 'epoch_index': epoch_index,
                    'metrics': m, 'proba': proba, 'y': y_select}
            torch.save(model.state_dict(), out / 'best.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        if (isinstance(model, ClinicalMethodAdapter)
                and epoch_index >= early_stop_start):
            if (early_stop_best is None
                    or m['macro_f1'] > early_stop_best + args.min_delta):
                early_stop_best = m['macro_f1']
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                break

    if best is None:
        raise RuntimeError('Training produced no validation checkpoint')

    def save_fold(path, fold, proba, y):
        np.savez_compressed(path, proba=proba, y=y,
                            subjects=np.array([d.subject for d in splits[fold]]),
                            sample_ids=np.array([d.sample_id for d in splits[fold]]))

    def proba_digest(proba):
        return hashlib.sha256(np.ascontiguousarray(proba).tobytes()).hexdigest()

    final = best
    if selection_fold == 'dev':
        save_fold(out / 'dev.npz', 'dev', best['proba'], best['y'])
        binding['selected_dev'] = {
            'epoch': best['epoch'],
            'epoch_index': best['epoch_index'],
            'metric': 'macro_f1',
            'metric_value': best['metrics']['macro_f1'],
            'prediction_sha256': proba_digest(best['proba']),
            'sample_ids_sha256': binding['split_sample_ids_sha256']['dev'],
        }
        final = None
        if final_eval == 'validation':
            # The single validation read of this run, at the dev-selected checkpoint.
            model.load_state_dict(torch.load(out / 'best.pt', map_location=device))
            proba_val, y_val = evaluate(model, val_loader, device, epoch=best['epoch_index'])
            final = {'epoch': best['epoch'], 'epoch_index': best['epoch_index'],
                     'metrics': metrics(y_val, proba_val, num_classes),
                     'proba': proba_val, 'y': y_val}
    prediction_sha256 = None
    if final is not None:
        save_fold(out / 'validation.npz', 'validation', final['proba'], final['y'])
        prediction_sha256 = proba_digest(final['proba'])
        binding['selected_validation'] = {
            'epoch': final['epoch'],
            'epoch_index': final['epoch_index'],
            'metric': 'macro_f1',
            'metric_value': final['metrics']['macro_f1'],
            'prediction_sha256': prediction_sha256,
            'sample_ids_sha256': binding['split_sample_ids_sha256']['validation'],
        }
    (out / 'binding.json').write_text(json.dumps(binding, indent=2, sort_keys=True) + '\n')
    validation_subjects = [d.subject for d in splits['validation']]
    result = {
        'status': 'completed', 'binding': binding,
        'selected_epoch': best['epoch'],
        'metrics': final['metrics'] if final is not None else None,
        'per_class': (per_class_table(final['y'], final['proba'], labels, num_classes)
                      if final is not None else None),
        'history': history,
        'patient_equal': (patient_equal_metrics(final['y'], final['proba'],
                                                validation_subjects, num_classes)
                          if final is not None else None),
        'relation_separation': ({'at_init': separation_at_init,
                                 'at_end': model.relation_separation()}
                                if not isinstance(model, ClinicalMethodAdapter) else None),
        'majority_baseline_accuracy': (round(float(
            (final['y'] == np.bincount(y_train, minlength=num_classes).argmax()).mean()), 6)
            if final is not None else None),
        'proba_sha256': prediction_sha256,
        'total_seconds': round(time.monotonic() - t0, 1),
        'caveats': [
            'Single seed; uncertainty is not measured and no significance is established.',
            ('Validation reused for epoch selection; this is a selection score, not a held-out result.'
             if selection_fold == 'validation' else
             'Epoch selected on the patient-disjoint dev split; validation '
             + ('evaluated exactly once at that checkpoint.' if final is not None
                else 'not evaluated (tuning run).')),
            'Test fold never loaded.',
            'Input timing is assumption-bound: untimestamped triage/history and a lab storetime proxy.',
        ] + (['conv=hgt is not parameter-matched to the edge_conditioned arms; compare '
              'only against a width-matched baseline.'] if args.conv == 'hgt' else [])
          + ([f'Only the {num_classes} most frequent TRAIN labels are modelled; '
              f'{sum(dropped_rows.values())} labelled rows were dropped. Scores are NOT '
              'comparable with a 30-class run: the random floor and majority baseline '
              'both rise and the rarest classes leave the macro-F1 denominator.']
             if kept_labels is not None else []),
    }
    if selection_fold == 'dev':
        result['selection_fold'] = 'dev'
        result['dev_metrics'] = best['metrics']
        result['dev_proba_sha256'] = binding['selected_dev']['prediction_sha256']
        result['validation_evaluations'] = 0 if final is None else 1
        if is_gchm_v2 or is_gchm_v3:
            result['gate_mean_last_batch'] = model.gate_report()
    (out / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    summary = {'stage': 'completed', 'selected_epoch': best['epoch']}
    if selection_fold == 'dev':
        summary['dev_macro_f1'] = best['metrics']['macro_f1']
    summary.update(final['metrics'] if final is not None else {})
    print(json.dumps(summary), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', required=True)
    p.add_argument('--targets', required=True)
    p.add_argument('--canonical', default='comparison/canonical_split.json')
    p.add_argument('--output', required=True)
    p.add_argument('--method', choices=sorted(METHOD_DEFAULTS), default='clinical_gnn')
    p.add_argument('--edges', choices=['all', 'informative', 'structural'], default='all')
    p.add_argument('--edge-direction', choices=list(EDGE_DIRECTIONS), default=None,
                   help="'bidirectional' appends a typed reverse edge for every "
                        'visit->evidence relation the producer emits one way only, so '
                        'complaints, vitals and labs can reach the visit hub. '
                        "Default 'forward' is the historical encoding.")
    p.add_argument('--no-edge-payload', action='store_true')
    p.add_argument('--no-message-passing', action='store_true')
    p.add_argument('--conv', choices=['edge_conditioned', 'hgt', 'gchm', 'gchm_v2', 'gchm_v3'],
                   default=None)
    p.add_argument('--aggregation', choices=list(AGGREGATIONS), default=None,
                   help="gchm_v2 only: 'sum' is the no-PNA ablation.")
    p.add_argument('--readout', choices=list(READOUTS), default=None,
                   help="gchm_v2 only: 'pool' drops the hub-state readout term.")
    p.add_argument('--no-hub-gate', action='store_true',
                   help='gchm_v2 only: gate on receiver and relation, not the hub.')
    p.add_argument('--v3-readout', choices=list(V3_READOUTS), default=None,
                   help="gchm_v3 only: 'pool' drops the label-wise attention readout.")
    p.add_argument('--no-wide', action='store_true',
                   help='gchm_v3 only: drop the linear per-token class-vote path.')
    p.add_argument('--no-jk', action='store_true',
                   help='gchm_v3 only: read the last layer instead of jumping knowledge.')
    p.add_argument('--edge-dropout', type=float, default=None,
                   help='gchm_v3 only: training-time edge drop rate (default 0.1).')
    p.add_argument('--selection-fold', choices=list(SELECTION_FOLDS), default=None,
                   help="'dev' selects the epoch on a patient-disjoint split of unused "
                        'TRAIN patients (--dev-limit rows) instead of validation.')
    p.add_argument('--final-eval', choices=list(FINAL_EVALS), default=None,
                   help="with --selection-fold dev: 'none' never reads validation "
                        '(tuning runs); default evaluates it once at the end.')
    p.add_argument('--dev-limit', type=int, default=None)
    p.add_argument('--sample-seed', type=int, default=None,
                   help='seed of the TRAIN sample and dev draw (default: --seed), so '
                        'model-init seeds can vary over one fixed sample.')
    p.add_argument('--min-prior-visits', type=int, default=0, metavar='N',
                   help='keep only visits with >= N completed earlier encounters '
                        '(1 = the multi-visit cohort). Applied to every fold.')
    p.add_argument('--top-k-labels', type=int, metavar='K',
                   help='model only the K most frequent TRAIN labels; other labelled '
                        'rows are dropped (not merged). Changes the task, so scores '
                        'are not comparable with a 30-class run.')
    p.add_argument('--modulation', choices=['multiplicative', 'additive'],
                   default='multiplicative',
                   help="gchm only: 'additive' is the capacity-matched control "
                        'where the receiver gate can shift but not scale.')
    p.add_argument('--drop-relation', action='append', metavar='RELATION',
                   help='remove this relation before tensorization; repeatable. '
                        'Leave-one-relation-out: a score that RISES means the '
                        'relation was hurting the model.')
    p.add_argument('--rewire-relation', action='append', metavar='RELATION',
                   help='degree-preserving same-type edge swaps; repeatable. '
                        'Requires --no-edge-payload and a real-edge/no-payload '
                        'control. Constrained graphs may have no legal swaps.')
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--weights', choices=['none', 'sqrt_inverse', 'inverse'],
                   default='sqrt_inverse')
    p.add_argument('--hidden', type=int, default=None)
    p.add_argument('--layers', type=int, default=None)
    p.add_argument('--token-dim', type=int, default=32)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight-decay', type=float, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--min-delta', type=float, default=None)
    p.add_argument('--protgnn-warm-epochs', dest='warm_epochs', type=int)
    p.add_argument('--protgnn-proj-epochs', dest='proj_epochs', type=int)
    p.add_argument('--protgnn-proj-interval', dest='proj_interval', type=int)
    p.add_argument('--protgnn-nearest-graphs', dest='nearest_graphs', type=int)
    p.add_argument('--protgnn-prototypes-per-class', dest='prototypes_per_class', type=int)
    p.add_argument('--protgnn-cluster-weight', dest='cluster_weight', type=float)
    p.add_argument('--protgnn-separation-weight', dest='separation_weight', type=float)
    p.add_argument('--protgnn-margin', dest='margin', type=float)
    p.add_argument('--protgnn-rollout', dest='rollout', type=int)
    p.add_argument('--protgnn-min-atoms', dest='min_atoms', type=int)
    p.add_argument('--protgnn-max-atoms', dest='max_atoms', type=int)
    p.add_argument('--protgnn-expand-atoms', dest='expand_atoms', type=int)
    p.add_argument('--protgnn-c-puct', dest='c_puct', type=float)
    p.add_argument('--protonode-warm-epochs', dest='protonode_warm_epochs', type=int)
    p.add_argument('--protonode-proj-epochs', dest='protonode_proj_epochs', type=int)
    p.add_argument('--protonode-proj-interval', dest='protonode_proj_interval', type=int)
    p.add_argument('--protonode-nearest-graphs', dest='protonode_nearest_graphs', type=int)
    p.add_argument('--protonode-prototypes-per-class', dest='protonode_prototypes_per_class', type=int)
    p.add_argument('--protonode-cluster-weight', dest='protonode_cluster_weight', type=float)
    p.add_argument('--protonode-separation-weight', dest='protonode_separation_weight', type=float)
    p.add_argument('--protonode-margin', dest='protonode_margin', type=float)
    p.add_argument('--protonode-readout', dest='protonode_readout',
                   choices=['both', 'node_max', 'graph_mean'])
    p.add_argument('--protonode-no-wide', dest='protonode_no_wide',
                   action='store_const', const=True, default=None)
    p.add_argument('--protonode-wide-l1', dest='protonode_wide_l1', type=float)
    p.add_argument('--gsat-temperature', type=float)
    p.add_argument('--gsat-info-loss-coef', type=float)
    p.add_argument('--gsat-init-r', type=float)
    p.add_argument('--gsat-final-r', type=float)
    p.add_argument('--gsat-decay-interval', type=int)
    p.add_argument('--gsat-decay-r', type=float)
    p.add_argument('--gsat-extractor-dropout', type=float)
    p.add_argument('--graphcare-decay-rate', type=float)
    p.add_argument('--graphcare-message-dropout', type=float)
    p.add_argument('--train-limit', type=int, default=None)
    p.add_argument('--token-min-count', type=int, default=20)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cpu')
    p.add_argument('--execute', action='store_true')
    return p


def main():
    p = parser()
    args = normalize_method_args(p.parse_args(), p)
    if args.execute:
        run(args)
    else:
        print(json.dumps({
            'status': 'not_executed',
            'requires': '--execute',
            'method': args.method,
            'conv': args.conv,
            'edge_direction': args.edge_direction,
            'selection_fold': args.selection_fold,
            'effective_defaults': {
                name: getattr(args, name) for name in method_defaults(args.method, args.conv)
            },
        }))


if __name__ == '__main__':
    main()
