"""Bounded enriched PNA train/validation smoke and exact checkpoint replay.

Not a full-training benchmark runner; old 30-epoch entrypoint is unchanged.
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from pna_analysis.train import predict
from shared.lib.benchmark_contract import file_sha256
from .build import load_enriched, save_json, code_hashes, snapshot_code
from .model import EnrichedPNA, numeric_graph


def graph_subset(bundle, indices):
    return [numeric_graph(np.flatnonzero(bundle['presence'][i]), int(bundle['y'][i]),
                          bundle['contract']['names'], bundle['hub_numeric'][i]) for i in indices]


def make_model(config, bundle):
    return EnrichedPNA(len(bundle['contract']['names']) + 1, len(bundle['contract']['classes']),
                       bundle['degree_histogram'], numeric_dim=bundle['hub_numeric'].shape[1],
                       width=config['width'], layers=config['layers'], rank=config['rank'],
                       interactions=config['interactions'])


def replay(output):
    """Read-only verification of saved input/schema/code, logits and shared metrics."""
    output = Path(output)
    state = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
    bundle = load_enriched(state['input_root'])
    if state['input_binding'] != bundle['manifest']['binding_sha256'] or state['code_sha256'] != code_hashes():
        raise ValueError('Checkpoint input/schema/code binding mismatch')
    for name, digest in state['code_sha256'].items():
        if file_sha256(output / 'source_snapshot' / name) != digest:
            raise ValueError('Checkpoint source snapshot mismatch')
    if not torch.equal(state['degree_histogram'], bundle['degree_histogram']):
        raise ValueError('Train degree histogram mismatch')
    config = state['config']; torch.set_num_threads(config['threads'])
    model = make_model(config, bundle)
    model.load_state_dict(state['state_dict'], strict=True)
    indices = state['evaluation_indices'].numpy()
    if len(indices) != config['val_limit'] or not np.all(bundle['folds'][indices] == 1):
        raise ValueError('Only saved canonical validation rows may be replayed')
    arrays, metrics = predict(model, graph_subset(bundle, indices), config)
    with np.load(output / 'predictions.npz', allow_pickle=False) as saved:
        equal = set(saved.files) == set(arrays) and all(np.array_equal(saved[k], v) for k, v in arrays.items())
        difference = float(np.abs(saved['logits'] - arrays['logits']).max())
    metrics_equal = metrics == json.loads((output / 'metrics.json').read_text())
    if not equal or not metrics_equal:
        raise ValueError('Replay logits/metrics mismatch')
    return {'arrays_equal': equal, 'metrics_equal': metrics_equal, 'logits_max_abs_diff': difference,
            'n_validation_smoke': len(indices), 'checkpoint_sha256': file_sha256(output / 'checkpoint.pt')}


def train_smoke(input_root, output, config):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Run output occupied; never overwrite or silently resume')
    if not (1 <= config['steps'] <= 100 and 1 <= config['train_limit'] <= 4096 and
            1 <= config['val_limit'] <= 1024 and 1 <= config['batch_size'] <= 128):
        raise ValueError('This runner is capped to a bounded wiring smoke')
    bundle = load_enriched(input_root)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
    train_rows = np.flatnonzero(bundle['folds'] == 0)
    validation_rows = np.flatnonzero(bundle['folds'] == 1)
    if config['train_limit'] > len(train_rows) or config['val_limit'] > len(validation_rows):
        raise ValueError('Smoke subset exceeds canonical fold')
    train_indices = np.random.RandomState(config['seed']).permutation(train_rows)[:config['train_limit']]
    eval_indices = validation_rows[:config['val_limit']]
    loader = DataLoader(graph_subset(bundle, train_indices), batch_size=config['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(config['seed']))
    evaluation = graph_subset(bundle, eval_indices)
    model = make_model(config, bundle)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    code = code_hashes()
    output.mkdir(parents=True, exist_ok=False)
    snapshot_code(output, code)
    manifest = {'schema': 'medgnn.enriched-pna-smoke.v1', 'status': 'running', 'config': config,
                'input_root': str(Path(input_root).resolve()), 'input_binding': bundle['manifest']['binding_sha256'],
                'code_sha256': code, 'optimizer_steps': 0, 'test_evaluated': False,
                'scope': 'bounded wiring/optimization proof only; full benchmark pending',
                'temporal_clean': False, 'raw_to_model_train_only': False,
                'selection': 'fixed-step final checkpoint; no validation selection',
                'n_train_smoke': len(train_indices), 'n_validation_smoke': len(eval_indices),
                'degree_fit': 'all canonical training graphs only', 'loss': 'unweighted cross entropy',
                'checkpoint_path': 'checkpoint.pt', 'metrics_path': 'metrics.json',
                'parameter_count': sum(p.numel() for p in model.parameters())}
    save_json(output / 'run_manifest.json', manifest)
    history = []; start = time.monotonic()
    try:
        while len(history) < config['steps']:
            model.train()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                loss = torch.nn.functional.cross_entropy(logits, batch.y)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite optimization loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite optimization gradient')
                gradient = float(model.encoder.numeric.weight.grad.abs().sum())
                optimizer.step()
                history.append({'step': len(history) + 1, 'loss': float(loss.detach()), 'numeric_gradient_l1': gradient})
                save_json(output / 'history.json', history)
                print(json.dumps(history[-1]), flush=True)
                if len(history) >= config['steps']:
                    break
        arrays, metrics = predict(model, evaluation, config)
        zero_numeric = [g.clone() for g in evaluation]
        for graph in zero_numeric:
            graph.x[:, -bundle['hub_numeric'].shape[1]:] = 0.
        zero_arrays, _ = predict(model, zero_numeric, config)
        delta = float(np.abs(arrays['logits'] - zero_arrays['logits']).max())
        gradient = sum(h['numeric_gradient_l1'] for h in history)
        if not gradient > 0 or not delta > 0:
            raise ValueError('Numeric hub path failed gradient/sensitivity proof')
        np.savez_compressed(output / 'predictions.npz', **arrays)
        save_json(output / 'metrics.json', metrics)
        state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config,
                 'input_root': str(Path(input_root).resolve()), 'input_binding': bundle['manifest']['binding_sha256'],
                 'recipe_sha256': bundle['manifest']['recipe_sha256'], 'code_sha256': code,
                 'degree_histogram': bundle['degree_histogram'], 'train_indices': torch.from_numpy(train_indices),
                 'evaluation_indices': torch.from_numpy(eval_indices), 'optimizer_steps': len(history)}
        torch.save(state, output / 'checkpoint.pt')
        verification = replay(output)
        preservation = bundle['manifest']['preservation']
        if any(file_sha256(p) != digest for p, digest in preservation.items()):
            raise ValueError('Legacy source/input/result preservation failed')
        manifest.update({'status': 'completed', 'optimizer_steps': len(history), 'replay': verification,
                         'numeric_gradient_l1': gradient, 'numeric_logit_max_abs_delta': delta,
                         'preservation_verified': True, 'seconds': time.monotonic() - start})
        save_json(output / 'run_manifest.json', manifest)
    except BaseException as exc:
        manifest.update({'status': 'failed', 'error': type(exc).__name__})
        save_json(output / 'run_manifest.json', manifest)
        raise
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path)
    p.add_argument('--output-dir', type=Path)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--replay', type=Path)
    p.add_argument('--architecture', choices=['plain', 'interaction'], default='interaction')
    p.add_argument('--steps', type=int, choices=range(1, 101), default=4)
    p.add_argument('--train-limit', type=int, choices=range(1, 4097), default=128)
    p.add_argument('--val-limit', type=int, choices=range(1, 1025), default=32)
    p.add_argument('--batch-size', type=int, choices=range(1, 129), default=32)
    p.add_argument('--width', type=int, choices=range(8, 129), default=32)
    p.add_argument('--layers', type=int, choices=range(1, 4), default=2)
    p.add_argument('--rank', type=int, choices=range(1, 33), default=8)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args(argv)
    if args.replay:
        print(json.dumps(replay(args.replay))); return 0
    if args.input_root is None:
        p.error('--input-root required')
    if not args.execute:
        bundle = load_enriched(args.input_root)
        print(json.dumps({'status': 'dry_run', 'counts': bundle['manifest']['counts'],
                          'numeric_channels': bundle['hub_numeric'].shape[1], 'scope': 'bounded smoke only'}))
        return 0
    if args.output_dir is None:
        p.error('--execute requires a new --output-dir')
    config = {k: getattr(args, k) for k in ['seed', 'width', 'layers', 'rank', 'steps', 'train_limit', 'val_limit', 'batch_size']}
    config.update({'interactions': args.architecture == 'interaction', 'lr': .001, 'weight_decay': 1e-5,
                   'threads': 4, 'cross_pairs': True, 'no_messages': False})
    result = train_smoke(args.input_root, args.output_dir, config)
    print(json.dumps({k: result[k] for k in ['status', 'optimizer_steps', 'numeric_gradient_l1', 'numeric_logit_max_abs_delta', 'replay']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
