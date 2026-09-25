"""Approved single-seed 6000/full-validation local EventGCHM, ten epochs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import json
import random
import signal
import sqlite3
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from comparison.standardized.representative_eventgchm_v1 import audit
from torch_geometric.data import Batch
from comparison.standardized.event_graph_gchm_xgb_v1 import local_train_v2 as native

REPO = native.REPO
FULL = REPO / 'comparison/standardized/event_training/local_first_lab_v2'
ROOT = REPO / 'comparison/standardized/event_training/representative_6000_fullval_v1'
SMALL = REPO / 'comparison/standardized/event_training/sample_500_100_3epochs_v1'
LABELS = REPO / 'comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_local_v2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    if ROOT.exists():
        raise FileExistsError('Refusing to overwrite sample output')
    os.umask(0o077)
    os.nice(19)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    signal.signal(signal.SIGTERM, native.request_stop)
    signal.signal(signal.SIGINT, native.request_stop)
    started = time.monotonic()
    original = json.loads((FULL / 'binding.json').read_text())
    for rel, digest in original['sources'].items():
        if native.sha256(REPO / rel) != digest:
            raise ValueError('Original source binding changed: ' + rel)
    gm = json.loads((native.DEFAULT_GRAPH_ROOT / 'manifest.json').read_text())
    bm = json.loads((LABELS / 'binding_manifest.json').read_text())
    assert bm['status'] == 'completed'
    assert bm['graph_sha256'] == gm['graphs_sha256'] == original['graphs_sha256']
    assert bm['labels'] == gm['reference_class_order_only'] == original['labels']
    assert len(bm['labels']) == 30
    checks = [(native.DEFAULT_GRAPH_ROOT / 'manifest.json', original['graph_manifest_sha256']),
              (native.DEFAULT_GRAPH_ROOT / 'cohort.csv', original['cohort_sha256']),
              (LABELS / 'binding_manifest.json', original['binding_manifest_sha256']),
              (LABELS / 'targets.csv', original['targets_sha256'])]
    for path, digest in checks:
        assert native.sha256(path) == digest, str(path)
    protected = {name: native.sha256(FULL / name) for name in ('binding.json', 'run_manifest.json', 'last.pt', 'best.pt')}
    small_binding = json.loads((SMALL / 'binding.json').read_text())
    for rel, digest in small_binding['sources'].items():
        assert native.sha256(REPO / rel) == digest, 'Small source changed: ' + rel
    small_protected = {str(path.relative_to(REPO)): native.sha256(path) for path in SMALL.rglob('*') if path.is_file()}
    target_rows, folds = {}, {}
    with (LABELS / 'targets.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] not in ('train', 'validation'):
                continue
            previous = folds.setdefault(row['subject_id'], row['split'])
            assert previous == row['split'], 'Inherited patient folds overlap'
            target_rows[row['sample_id']] = row
    db = sqlite3.connect((FULL / 'graph_index.sqlite').as_uri() + '?mode=ro', uri=True)
    allrows = {split: db.execute('SELECT sample,offset,digest,target,nodes,edges FROM graphs WHERE split=? AND target>=0 ORDER BY sample', (split,)).fetchall() for split in ('train', 'validation')}
    db.close()
    assert len(allrows['train']) == 77930 and len(allrows['validation']) == 9582
    chosen, available, allocation = audit.choose(allrows['train'], target_rows)
    selection = {'train': chosen, 'validation': allrows['validation']}
    records = {split: [tuple(r[1:]) for r in rows] for split, rows in selection.items()}
    coverage = {split: {'samples': len(rows), 'available_class_counts': np.bincount([r[3] for r in allrows[split]], minlength=30).tolist(),
                       'selected_class_counts': np.bincount([r[3] for r in rows], minlength=30).tolist(), 'class_coverage': 30} for split, rows in selection.items()}
    subjects = {split: {target_rows[r[0]]['subject_id'] for r in rows} for split, rows in selection.items()}
    assert subjects['train'].isdisjoint(subjects['validation'])
    assert len(subjects['train']) == 6000 and len(subjects['validation']) == 5620
    profile = {'scope': 'Exact full-training index/target metadata; selected train/validation payloads; independent uniform seed81234 1000-train payload reference. No full-corpus payload distribution claim.',
               'source_train': audit.metadata_profile(allrows['train'], target_rows),
               'selected_train': audit.metadata_profile(chosen, target_rows),
               'natural_validation': audit.metadata_profile(allrows['validation'], target_rows)}
    reference_rng = np.random.default_rng(81234)
    reference = [allrows['train'][int(i)] for i in reference_rng.choice(len(allrows['train']), size=1000, replace=False)]
    payload_profiles = {'train': [], 'validation': [], 'independent_train_reference_1000': []}
    payload_digest = hashlib.sha256()
    metadata = []
    with (native.DEFAULT_GRAPH_ROOT / 'graphs.jsonl').open('rb') as stream:
        for split, rows in selection.items():
            for row in rows:
                sample, offset, digest, target, nodes, edges = row
                graph = native.read_graph(stream, (offset, digest))
                label_row = target_rows[sample]
                assert graph['sample_id'] == sample and graph['split'] == split == label_row['split']
                assert graph['schema_version'] == 'event_graph_v1' and 'target' not in graph
                assert int(label_row['target']) == target and label_row['label'] == bm['labels'][target]
                assert len(graph['nodes']) == nodes and len(graph['edges']) == edges
                payload_profiles[split].append(audit.graph_profile(graph))
                stream.seek(offset)
                payload_digest.update(stream.readline())
                metadata.append({'sample_id': sample, 'subject_id': label_row['subject_id'], 'stay_id': label_row['stay_id'],
                                 'split': split, 'target': target, 'offset': offset, 'record_sha256': digest,
                                 'nodes': nodes, 'edges': edges})
        for row in reference:
            payload_profiles['independent_train_reference_1000'].append(audit.graph_profile(native.read_graph(stream, row[1:])))
    profile['payload'] = {k: audit.summarize_profiles(v) for k,v in payload_profiles.items()}
    ROOT.mkdir(parents=True, mode=0o700)
    native.atomic_json(ROOT/'distribution_profile.json', profile)
    native.atomic_json(ROOT/'reference_samples.json', [{'sample_id': r[0], 'offset': r[1], 'record_sha256': r[2]} for r in reference])
    base = {**original, 'epochs': 10, 'threads': 1, 'scope': 'local_representative_6000_train_full_9582_validation',
            'sources': {**original['sources'], str(Path(__file__).resolve().relative_to(REPO)): native.sha256(Path(__file__)), str(Path(audit.__file__).resolve().relative_to(REPO)): native.sha256(Path(audit.__file__))},
            'selected_payload_sha256': payload_digest.hexdigest(), 'sample_counts': {'train': 6000, 'validation': 9582},
            'selection_policy': 'Train seed1234; minimum60 per class then remaining4200 proportional to ORIGINAL class counts, capped largest remainders; scarce classes first, unseen subjects first, one seeded random visit per subject; achieved maximum6000 distinct patients. Validation all9582 natural visits. No size/prediction filtering or truncation. Fixed10epochs seed1234, no early stopping.',
            'audit_scope': 'Original recorded graph SHA binding verified against manifests; 15582 selected raw records plus independent1000-train reference (may overlap) freshly SHA-verified through read-only existing offsets; no fresh whole-18GB SHA scan. Target/cohort and source files freshly hashed. No test graph reads or test evaluation.',
            'original_run_protected_hashes': protected, 'small_run_protected_hashes': small_protected, 'selection_independent_of_predictions': True, 'max_epochs': 10, 'early_stopping': False}
    native.atomic_json(ROOT / 'selected_samples.json', metadata)
    base['selected_metadata_sha256'] = native.sha256(ROOT / 'selected_samples.json')
    native.atomic_json(ROOT / 'binding.json', base)
    native.atomic_json(ROOT / 'sample_coverage.json', coverage)
    for rel in base['sources']:
        dest = ROOT / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / rel).read_bytes())
    native.atomic_json(ROOT / 'process.json', {'pid': os.getpid(), 'argv': sys.argv, 'python': sys.executable,
                      'device': 'cpu', 'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                      'niceness': os.getpriority(os.PRIO_PROCESS, 0), 'environment_threads': {k: os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}})
    state = {'status': 'preparing', 'method': 'EventGCHM', 'seed': 1234, 'epochs': 10, 'completed_epochs': 0,
             'counts': {'train': 6000, 'validation': 9582}, 'device': 'cpu', 'threads': 1, 'microbatch': 1,
             'accumulation': 8, 'test_evaluated': False, 'temporal_clean': False,
             'checkpoint_path': str(ROOT / 'last.pt'), 'selected_checkpoint_path': str(ROOT / 'best.pt'),
             'metrics_path': str(ROOT / 'history.json'), 'selection': 'validation_macro_f1_all_30_classes',
             'class_order': bm['labels'], 'validation_evaluations': 0}
    native.atomic_json(ROOT / 'run_manifest.json', state)
    try:
        adapter, hist = native.prepare_adapter(ROOT, records['train'], base)
        binding = {**base, 'preprocessing_sha256': native.sha256(ROOT / 'preprocessing.pt')}
        model = native.EventGCHM(adapter.num_tokens, adapter.num_relations, 30, degree_histogram=hist,
                                hidden_dim=64, relation_dim=16, num_layers=3, dropout=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001)
        targets = np.array([r[2] for r in records['train']], dtype=np.int64)
        counts = np.bincount(targets, minlength=30)
        assert (counts > 0).all()
        w = 1 / np.sqrt(counts.astype(float)); w /= w.mean()
        weights = torch.tensor(w, dtype=torch.float32)
        native.atomic_json(ROOT / 'train_class_weights.json', {'counts': counts.tolist(), 'weights': weights.tolist(), 'fit_split': 'selected_6000_train_only'})
        history, step, best_score, best_epoch = [], 0, -1.0, -1
        training_started = time.monotonic()
        native.emit(ROOT, 'training_start', parameters=sum(p.numel() for p in model.parameters()), train=6000, validation=9582, epochs=10, threads=1)
        def checkpoint(epoch, cursor, permutation):
            return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'model_config': model.config,
                    'binding': binding, 'epoch': epoch, 'cursor': cursor, 'permutation': permutation,
                    'optimizer_steps': step, 'history': list(history), 'best_score': best_score, 'best_epoch': best_epoch,
                    'class_weights': weights, 'adapter': adapter.to_dict(), 'degree_histogram': hist,
                    'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(), 'rng_torch': torch.get_rng_state(),
                    'gradient_boundary': True}
        with (native.DEFAULT_GRAPH_ROOT / 'graphs.jsonl').open('rb') as stream:
            for epoch in range(1, 11):
                epoch_started = time.monotonic()
                model.train()
                permutation = torch.randperm(6000).tolist()
                epoch_loss, updates = 0.0, 0
                for cursor in range(0, 6000, 8):
                    selected = permutation[cursor:cursor+8]
                    denominator = weights[torch.tensor(targets[selected])].sum()
                    optimizer.zero_grad(set_to_none=True)
                    loss_value = 0.0
                    for index in selected:
                        record = records['train'][index]
                        graph = native.read_graph(stream, record)
                        graph['target'] = int(record[2])
                        batch = Batch.from_data_list([adapter.transform(graph)])
                        logits = model(batch)
                        loss = F.cross_entropy(logits, batch.y.view(-1), weight=weights, reduction='sum') / denominator
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite training loss')
                        loss.backward()
                        loss_value += float(loss.detach())
                        del batch, logits, loss, graph
                    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise ValueError('Nonfinite gradients')
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    step += 1; updates += 1; epoch_loss += loss_value
                    if step == 1 or step % 50 == 0:
                        native.emit(ROOT, 'optimizer_update', epoch=epoch, cursor=min(cursor+8,6000), optimizer_steps=step, loss=loss_value, elapsed_seconds=time.monotonic()-started)
                    if step == 1 or step % 100 == 0:
                        native.atomic_torch(ROOT / 'last.pt', checkpoint(epoch, min(cursor+8,6000), permutation))
                        state.update(status='running', active_epoch=epoch, optimizer_steps=step)
                        native.atomic_json(ROOT / 'run_manifest.json', state)
                    if native.STOP:
                        native.atomic_torch(ROOT/'last.pt', checkpoint(epoch, min(cursor+8,6000), permutation))
                        raise InterruptedError('User stop at optimizer boundary')
                native.emit(ROOT, 'validation_start', epoch=epoch, optimizer_steps=step)
                model.eval()
                logits_list, truth = [], []
                with torch.no_grad():
                    for validation_index, record in enumerate(records['validation']):
                        graph = native.read_graph(stream, record)
                        batch = Batch.from_data_list([adapter.transform(graph)])
                        logits_list.append(model(batch))
                        truth.append(record[2])
                        if (validation_index+1) % 2000 == 0:
                            native.emit(ROOT, 'validation_progress', epoch=epoch, completed=validation_index+1, total=9582)
                        if native.STOP:
                            native.atomic_torch(ROOT/'last.pt', checkpoint(epoch, 6000, permutation))
                            raise InterruptedError('User stop during validation')
                logits = torch.cat(logits_list)
                y = torch.tensor(truth, dtype=torch.long)
                assert torch.isfinite(logits).all()
                proba = logits.softmax(-1).numpy()
                score = float(f1_score(y.numpy(), proba.argmax(1), labels=np.arange(30), average='macro', zero_division=0))
                row = {'epoch': epoch, 'optimizer_steps': step, 'train_loss': epoch_loss/updates,
                       'validation_loss': float(F.cross_entropy(logits, y)), 'validation_macro_f1': score,
                       'train_samples': 6000, 'validation_samples': 9582, 'epoch_seconds': time.monotonic()-epoch_started}
                history.append(row)
                improved = score > best_score
                if improved:
                    best_score, best_epoch = score, epoch
                ck = checkpoint(epoch, 6000, permutation)
                native.atomic_torch(ROOT / 'last.pt', ck)
                if improved:
                    native.atomic_torch(ROOT / 'best.pt', ck)
                    np.savez_compressed(ROOT / 'validation.npz', proba=proba, y=y.numpy(), sample_id=np.array([r[0] for r in selection['validation']]), subject_id=np.array([target_rows[r[0]]['subject_id'] for r in selection['validation']]))
                native.atomic_json(ROOT / 'history.json', history)
                state.update(status='running', completed_epochs=epoch, active_epoch=epoch, optimizer_steps=step,
                             best_epoch=best_epoch, best_validation_macro_f1=best_score, validation_evaluations=epoch)
                native.atomic_json(ROOT / 'run_manifest.json', state)
                native.emit(ROOT, 'epoch_completed', **row)
        best = torch.load(ROOT/'best.pt', map_location='cpu', weights_only=False)
        last = torch.load(ROOT/'last.pt', map_location='cpu', weights_only=False)
        assert last['optimizer_steps'] == step == 7500 and len(last['history']) == 10
        assert best['epoch'] == best_epoch and best['optimizer'] and best['adapter']
        model.load_state_dict(best['model']); model.eval()
        replay = []
        with (native.DEFAULT_GRAPH_ROOT/'graphs.jsonl').open('rb') as stream, torch.no_grad():
            for record in records['validation']:
                batch = Batch.from_data_list([adapter.transform(native.read_graph(stream, record))])
                replay.append(model(batch))
        replay_proba = torch.cat(replay).softmax(-1).numpy()
        saved_proba = np.load(ROOT/'validation.npz')['proba']
        assert np.array_equal(replay_proba, saved_proba), 'Best checkpoint replay mismatch'
        native.atomic_json(ROOT/'replay_verification.json', {'checkpoint_epoch': best_epoch, 'validation_graphs': 9582, 'probabilities_bitwise_equal': True, 'max_absolute_difference': float(np.max(np.abs(replay_proba-saved_proba)))})
        result = audit.report(ROOT, state, native)
        for rel, digest in small_protected.items():
            assert native.sha256(REPO/rel) == digest, 'Small artifact changed: ' + rel
        for rel, digest in small_binding['sources'].items():
            assert native.sha256(REPO/rel) == digest, 'Small source changed: ' + rel
        for rel, digest in base['sources'].items():
            assert native.sha256(REPO/rel) == digest, 'Active source changed: ' + rel
        for name, digest in protected.items():
            assert native.sha256(FULL / name) == digest, 'Original run artifact changed: ' + name
        for rel, digest in original['sources'].items():
            assert native.sha256(REPO / rel) == digest
        state.update(status='completed', elapsed_seconds=time.monotonic()-started,
                     training_and_validation_seconds=time.monotonic()-training_started,
                     original_run_preserved=True, small_run_preserved=True, replay_verified=True, validation_prediction_sha256=native.sha256(ROOT/'validation.npz'), result_metrics_path=str(ROOT/'metrics.json'), result_metrics_sha256=native.sha256(ROOT/'metrics.json'), checkpoint_sha256=native.sha256(ROOT/'last.pt'),
                     selected_checkpoint_sha256=native.sha256(ROOT/'best.pt'), metrics_sha256=native.sha256(ROOT/'history.json'))
        with (ROOT/'report.md').open('a') as f:
            f.write(f"\nTotal elapsed seconds: {state['elapsed_seconds']:.3f}; training/validation/final replay/report seconds: {state['training_and_validation_seconds']:.3f}. Optimizer steps: {step}. Both previous runs preserved by before/after hashes.\n")
        native.atomic_json(ROOT / 'run_manifest.json', state)
        native.emit(ROOT, 'completed', **{k: state[k] for k in ('completed_epochs','optimizer_steps','elapsed_seconds','best_epoch','best_validation_macro_f1')})
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, InterruptedError) else 'failed', error=repr(exc), elapsed_seconds=time.monotonic()-started)
        native.atomic_json(ROOT / 'run_manifest.json', state)
        raise


if __name__ == '__main__':
    main()
