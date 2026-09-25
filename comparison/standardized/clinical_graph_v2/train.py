"""Train the small clinical GNN and report honest, fold-correct metrics.

Scope rules enforced here, not left to discipline:

  * The TEST fold is never loaded. Selection and reporting both use validation.
  * Preprocessing (vocabulary, scalers) is fitted on TRAIN graphs only.
  * Class weights come from full TRAIN counts and are applied to training samples
    only; validation loss and metrics are unweighted.
  * Every run writes a manifest with artifact hashes, source hashes, seed, parameter
    count and the exact arm configuration, plus stored validation probabilities so a
    result can be replayed without retraining.

Arms differ by a single flag so the comparison is capacity-matched:

    --edges all|informative|structural   which relations survive tensorization
    --no-edge-payload                    blank the delta/interval/recency payload
    --no-message-passing                 skip propagation entirely (node-only control)
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
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from . import INFORMATIVE_RELATIONS, STRUCTURAL_RELATIONS
from .model import ClinicalGNN
from .schema import sha256
from .tensorize import (ALL_RELATIONS, encode_graph, fit_preprocessing, iter_graphs,
                        preprocessing_state)

NUM_CLASSES = 30


def metrics(y_true, proba):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 top_k_accuracy_score)
    pred = proba.argmax(1)
    return {
        'accuracy': round(float(accuracy_score(y_true, pred)), 6),
        'balanced_acc': round(float(balanced_accuracy_score(y_true, pred)), 6),
        'macro_f1': round(float(f1_score(y_true, pred, average='macro', zero_division=0)), 6),
        'micro_f1': round(float(f1_score(y_true, pred, average='micro', zero_division=0)), 6),
        'top3_acc': round(float(top_k_accuracy_score(y_true, proba, k=3,
                                                     labels=np.arange(NUM_CLASSES))), 6),
        'top5_acc': round(float(top_k_accuracy_score(y_true, proba, k=5,
                                                     labels=np.arange(NUM_CLASSES))), 6),
    }


def per_class_table(y_true, proba, labels):
    from sklearn.metrics import precision_recall_fscore_support
    pred = proba.argmax(1)
    p, r, f, s = precision_recall_fscore_support(
        y_true, pred, labels=np.arange(NUM_CLASSES), zero_division=0)
    return [{'index': i, 'label': labels[i], 'precision': round(float(p[i]), 6),
             'recall': round(float(r[i]), 6), 'f1': round(float(f[i]), 6),
             'support': int(s[i])} for i in range(NUM_CLASSES)]


def load_targets(path):
    targets = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            if row['target'] and row['target'] != '-1':
                targets[row['sample_id']] = (int(row['target']), row['split'],
                                             row['subject_id'])
    if not targets:
        raise ValueError('No labelled samples in the target sidecar')
    return targets


def build_dataset(artifact, targets, edge_mode, limit, token_min_count, seed):
    """Encode labelled graphs. Test-fold rows are dropped before any tensor work."""
    graphs_path = Path(artifact) / 'graphs.jsonl'
    train_ids = {sid for sid, (_, split, _) in targets.items() if split == 'train'}
    if limit is not None:
        # Deterministic subsample of TRAIN ids; validation always stays complete.
        rng = random.Random(seed)
        train_ids = set(rng.sample(sorted(train_ids), min(limit, len(train_ids))))

    prep = fit_preprocessing(graphs_path, train_ids, token_min_count)
    splits = {'train': [], 'validation': []}
    labels_seen = Counter()
    for graph in iter_graphs(graphs_path):
        entry = targets.get(graph['sample_id'])
        if entry is None:
            continue
        y, split, subject = entry
        if split == 'test':
            continue  # held out; never loaded
        if split == 'train' and graph['sample_id'] not in train_ids:
            continue
        data = encode_graph(graph, prep, edge_mode)
        data.y = torch.tensor([y], dtype=torch.long)
        data.subject = subject
        splits[split].append(data)
        labels_seen[split] += 1
    if not splits['train'] or not splits['validation']:
        raise ValueError('Empty train or validation split')
    return splits, prep


def class_weights(labels, policy):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    if (counts == 0).any():
        # A class absent from this training subsample gets weight 1 rather than inf.
        counts = np.where(counts == 0, 1.0, counts)
    w = len(labels) / (NUM_CLASSES * counts)
    if policy == 'sqrt_inverse':
        return np.sqrt(w)
    if policy == 'inverse':
        return w
    if policy == 'none':
        return np.ones(NUM_CLASSES)
    raise ValueError('Unknown weight policy')


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        logits.append(model(batch).cpu())
        ys.append(batch.y.cpu())
    logits = torch.cat(logits)
    return torch.softmax(logits, dim=1).numpy(), torch.cat(ys).numpy()


def run(args):
    t0 = time.monotonic()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError('Occupied output directory; choose a fresh path')
    out.mkdir(parents=True, mode=0o700)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(False)

    artifact = Path(args.artifact).resolve()
    manifest = json.loads((artifact / 'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('Artifact is not completed')

    targets = load_targets(args.targets)
    splits, prep = build_dataset(artifact, targets, args.edges, args.train_limit,
                                 args.token_min_count, args.seed)

    device = torch.device(args.device)
    train_loader = DataLoader(splits['train'], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(splits['validation'], batch_size=args.batch_size)

    node_dim = splits['train'][0].x.size(1)
    edge_dim = splits['train'][0].edge_attr.size(1)
    model = ClinicalGNN(num_tokens=len(prep['vocabulary']), node_dim=node_dim,
                        edge_dim=edge_dim, hidden=args.hidden, layers=args.layers,
                        dropout=args.dropout, token_dim=args.token_dim,
                        use_edge_payload=not args.no_edge_payload,
                        use_message_passing=not args.no_message_passing).to(device)

    y_train = np.array([int(d.y) for d in splits['train']])
    weights = torch.tensor(class_weights(y_train, args.weights),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    labels = manifest.get('reference_class_order_only') or \
        json.loads(Path(args.canonical).read_text())['classes']

    binding = {
        'artifact': str(artifact),
        'artifact_graphs_sha256': manifest['graphs_sha256'],
        'artifact_schema': manifest['schema_version'],
        'diagnosis_tier': manifest.get('diagnosis_tier', False),
        'targets_sha256': sha256(args.targets),
        'source_code': {str(p.name): sha256(p)
                        for p in sorted(Path(__file__).parent.glob('*.py'))},
        'seed': args.seed, 'edges': args.edges,
        'edge_payload': not args.no_edge_payload,
        'message_passing': not args.no_message_passing,
        'weights': args.weights, 'hidden': args.hidden, 'layers': args.layers,
        'dropout': args.dropout, 'lr': args.lr, 'batch_size': args.batch_size,
        'epochs': args.epochs, 'train_limit': args.train_limit,
        'parameter_count': model.parameter_count(),
        'counts': {'train': len(splits['train']), 'validation': len(splits['validation'])},
        'node_dim': node_dim, 'edge_dim': edge_dim,
        'vocabulary_size': len(prep['vocabulary']),
        'device': str(device), 'torch': torch.__version__,
        'selection': 'best validation macro_f1; full fixed budget; no early stop',
        'test_evaluated': False,
    }
    (out / 'binding.json').write_text(json.dumps(binding, indent=2, sort_keys=True) + '\n')
    (out / 'preprocessing.json').write_text(
        json.dumps(preprocessing_state(prep), indent=2, sort_keys=True) + '\n')
    print(json.dumps({'stage': 'bound', **{k: binding[k] for k in
                                           ('parameter_count', 'counts', 'edges',
                                            'edge_payload', 'message_passing')}}), flush=True)

    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss) * batch.num_graphs
            seen += batch.num_graphs
        proba, y_val = evaluate(model, val_loader, device)
        m = metrics(y_val, proba)
        row = {'epoch': epoch, 'train_loss': round(total / seen, 6), **m,
               'seconds': round(time.monotonic() - t0, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or m['macro_f1'] > best['metrics']['macro_f1']:
            best = {'epoch': epoch, 'metrics': m, 'proba': proba, 'y': y_val}
            torch.save(model.state_dict(), out / 'best.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')

    np.savez_compressed(out / 'validation.npz', proba=best['proba'], y=best['y'],
                        subjects=np.array([d.subject for d in splits['validation']]))
    result = {
        'status': 'completed', 'binding': binding,
        'selected_epoch': best['epoch'], 'metrics': best['metrics'],
        'per_class': per_class_table(best['y'], best['proba'], labels),
        'history': history,
        'majority_baseline_accuracy': round(float(
            (best['y'] == np.bincount(y_train, minlength=NUM_CLASSES).argmax()).mean()), 6),
        'proba_sha256': hashlib.sha256(np.ascontiguousarray(best['proba']).tobytes()).hexdigest(),
        'total_seconds': round(time.monotonic() - t0, 1),
        'caveats': [
            'Single seed; seed spread not measured, so deltas below it are noise.',
            'Validation reused for epoch selection; this is a selection score, not a held-out result.',
            'Test fold never loaded.',
            'Input artifact temporal_clean=true but storetime is an availability proxy.',
        ],
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'stage': 'completed', 'selected_epoch': best['epoch'],
                      **best['metrics']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', required=True)
    p.add_argument('--targets', required=True)
    p.add_argument('--canonical', default='comparison/canonical_split.json')
    p.add_argument('--output', required=True)
    p.add_argument('--edges', choices=['all', 'informative', 'structural'], default='all')
    p.add_argument('--no-edge-payload', action='store_true')
    p.add_argument('--no-message-passing', action='store_true')
    p.add_argument('--weights', choices=['none', 'sqrt_inverse', 'inverse'],
                   default='sqrt_inverse')
    p.add_argument('--hidden', type=int, default=96)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--token-dim', type=int, default=32)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--train-limit', type=int, default=None)
    p.add_argument('--token-min-count', type=int, default=20)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cpu')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.execute:
        run(args)
    else:
        print(json.dumps({'status': 'not_executed', 'requires': '--execute'}))


if __name__ == '__main__':
    main()
