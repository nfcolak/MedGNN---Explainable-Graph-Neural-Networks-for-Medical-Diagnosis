"""Bounded training and deterministic checkpoint/prediction replay, no test evaluation."""
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from comparison.standardized.performance_review import class_weights
from pna_analysis.data import concept_graph
from pna_analysis.model import PNAPredictor
from shared.lib.benchmark_contract import file_sha256
from shared.lib.metrics import multiclass_metrics

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def code_hashes():
    files = sorted((ROOT / 'pna_analysis').glob('*.py'))
    files += [ROOT / p for p in ('shared/lib/metrics.py', 'comparison/standardized/performance_review.py',
              'comparison/standardized/common_input_improvement.py', 'graphcare_analysis/adapter.py')]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in files}


def build_predictor(config, bundle):
    return PNAPredictor(len(bundle['contract']['names']) + 1, len(bundle['contract']['classes']),
                        bundle['degree_histogram'], width=config['width'], layers=config['layers'],
                        rank=config['rank'], interactions=config['interactions'])


def graph_subset(bundle, indices):
    return [concept_graph(np.flatnonzero(bundle['presence'][i]), int(bundle['y'][i]),
                          bundle['contract']['names']) for i in indices]


def predict(model, graphs, config):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=config['batch_size'], shuffle=False):
            out = model(batch, cross_pairs=config['cross_pairs'], no_messages=config['no_messages'])
            logits.append(out.cpu()); labels.append(batch.y.cpu())
    logits = torch.cat(logits)
    labels = torch.cat(labels).numpy()
    probability = logits.softmax(-1).numpy()
    predictions = probability.argmax(-1)
    metrics = multiclass_metrics(labels, predictions, probability)
    return {'logits': logits.numpy(), 'y': labels, 'probability': probability, 'pred': predictions}, metrics


def verify_checkpoint(target, bundle):
    """Read local tensor/state checkpoint, reconstruct, replay exactly saved fold rows.

    No optimizer step, selection, test loader or file write. Current input and code
    must still match the saved binding; portability across Torch versions is not promised.
    """
    target = Path(target)
    checkpoint = torch.load(target / 'checkpoint.pt', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if checkpoint['code_sha256'] != code_hashes():
        raise ValueError('Checkpoint code binding mismatch')
    if checkpoint['input_contract'] != bundle['contract']:
        raise ValueError('Checkpoint input contract mismatch')
    if not torch.equal(checkpoint['degree_histogram'], bundle['degree_histogram']):
        raise ValueError('Checkpoint train-only degree binding mismatch')
    torch.set_num_threads(config['threads'])
    model = build_predictor(config, bundle)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    indices = checkpoint['evaluation_indices'].numpy()
    expected_fold = 1 if config['val_limit'] else 0
    if not np.all(bundle['folds'][indices] == expected_fold):
        raise ValueError('Replay indices must remain in configured train/validation fold')
    arrays, metrics = predict(model, graph_subset(bundle, indices), config)
    with np.load(target / 'predictions.npz', allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError('Prediction schema mismatch')
        difference = float(np.max(np.abs(saved['logits'] - arrays['logits'])))
        equal = all(np.array_equal(saved[k], v) for k, v in arrays.items())
    metrics_equal = metrics == json.loads((target / 'metrics.json').read_text())
    if not equal or not metrics_equal:
        raise ValueError('Checkpoint replay predictions/metrics differ')
    return {'logits_max_abs_diff': difference, 'arrays_equal': equal, 'metrics_equal': metrics_equal,
            'n_evaluated': len(indices), 'fold': 'validation_smoke' if expected_fold else 'train_smoke',
            'checkpoint_sha256': file_sha256(target / 'checkpoint.pt')}


def train_smoke(plan, bundle, target):
    """Fixed bounded update count; all steps train-only, optional small val readout."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    config = plan['config']
    manifest = {**plan, 'status': 'running', 'optimizer_steps': 0,
                'schema': 'medgnn.pna-experimental-smoke-v1',
                'input_contract': bundle['contract'], 'code_sha256': code_hashes(),
                'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                            'pyg': str(torch_geometric.__version__), 'numpy': str(np.__version__)},
                'git_branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'checkpoint_path': 'checkpoint.pt', 'metrics_path': 'metrics.json', 'predictions_path': 'predictions.npz'}
    save_json(target / 'run_manifest.json', manifest)
    save_json(target / 'config.json', config)
    save_json(target / 'preservation_before.json', bundle['preservation'])
    (target / 'git_status_before.txt').write_text(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True))
    start = time.monotonic()
    try:
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        torch.set_num_threads(config['threads'])
        torch.use_deterministic_algorithms(True)
        train_rows = np.flatnonzero(bundle['folds'] == 0)
        rng = np.random.RandomState(config['seed'])
        train_indices = rng.permutation(train_rows)[:config['train_limit']]
        evaluation_indices = (np.flatnonzero(bundle['folds'] == 1)[:config['val_limit']]
                              if config['val_limit'] else train_indices)
        train = graph_subset(bundle, train_indices)
        evaluation = graph_subset(bundle, evaluation_indices)
        generator = torch.Generator().manual_seed(config['seed'])
        loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, generator=generator)
        model = build_predictor(config, bundle)
        initial_pair = {n: p.detach().clone() for n, p in model.named_parameters() if '.pair.' in n}
        weight = torch.tensor(class_weights(bundle['y'][train_rows], len(bundle['contract']['classes']),
                                            'none' if config['loss'] == 'ce' else 'sqrt_inverse'), dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        manifest.update({'class_weights': weight.tolist(),
                         'class_counts_train': np.bincount(bundle['y'][train_rows], minlength=len(weight)).tolist(),
                         'degree_histogram': bundle['degree_histogram'].tolist(),
                         'parameter_count': sum(p.numel() for p in model.parameters()),
                         'n_train_smoke': len(train), 'n_evaluation_smoke': len(evaluation)})
        history = []
        while len(history) < config['steps']:
            model.train()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch, cross_pairs=config['cross_pairs'], no_messages=config['no_messages'])
                loss = torch.nn.functional.cross_entropy(logits, batch.y, weight=weight)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError('Nonfinite training gradient')
                gradients = {n: float(p.grad.abs().sum()) if p.grad is not None else 0.
                             for n, p in model.named_parameters() if n in initial_pair}
                optimizer.step()
                history.append({'step': len(history) + 1, 'loss': float(loss.detach()),
                                'batch_graphs': int(batch.y.numel()), 'pair_gradient_l1': gradients})
                save_json(target / 'history.json', history)
                manifest['optimizer_steps'] = len(history)
                save_json(target / 'run_manifest.json', manifest)
                print(json.dumps({'event': 'train_step', **history[-1]}), flush=True)
                if len(history) == config['steps']:
                    break
        arrays, metrics = predict(model, evaluation, config)
        np.savez_compressed(target / 'predictions.npz', **arrays)
        save_json(target / 'metrics.json', metrics)
        checkpoint = {'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                      'config': config, 'code_sha256': code_hashes(), 'input_contract': bundle['contract'],
                      'degree_histogram': bundle['degree_histogram'], 'class_weights': weight,
                      'optimizer_steps': len(history), 'torch_rng': torch.get_rng_state(),
                      'loader_rng': generator.get_state(), 'train_indices': torch.from_numpy(train_indices),
                      'evaluation_indices': torch.from_numpy(evaluation_indices)}
        torch.save(checkpoint, target / 'checkpoint.pt')
        replay = verify_checkpoint(target, bundle)
        after = {p: file_sha256(p) for p in bundle['preservation']}
        save_json(target / 'preservation_after.json', after)
        if after != bundle['preservation']:
            raise RuntimeError('Protected input/source/split/run binding changed during smoke')
        manifest.update({'status': 'completed', 'optimizer_steps': len(history), 'replay': replay,
                         'preservation_verified': True, 'seconds': time.monotonic() - start,
                         'evaluation_metrics': metrics,
                         'interaction_parameter_deltas': {n: float((p.detach() - initial_pair[n]).abs().sum())
                             for n, p in model.named_parameters() if n in initial_pair}})
        save_json(target / 'run_manifest.json', manifest)
        return manifest
    except BaseException as exc:
        manifest.update({'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)})
        save_json(target / 'run_manifest.json', manifest)
        raise
