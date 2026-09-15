"""Isolated medication-history-available source-snapshot diagnostic.

No early-triage/temporal-clean claim. Never writes production caches/results.
Identical observable inputs before method-specific learned encoders: concept
one-hots plus one constant hub identity; no patient-specific hub payload.
"""
from pathlib import Path
import numpy as np
import torch
import random
import json


def check_attempt(target, binding, execute=False, resume=False):
    if not execute:
        return 'dry_run'
    if not target.exists():
        if resume:
            raise ValueError('Cannot resume missing attempt')
        return 'new'
    if not resume:
        raise FileExistsError('Occupied attempt; explicit --resume required')
    if not (target / 'state.json').exists():
        raise ValueError('Occupied attempt without state')
    state = json.loads((target / 'state.json').read_text())
    if state['binding'] != binding:
        raise ValueError('Resume binding differs')
    return 'completed' if state['status'] == 'completed' else 'resume'


def save_checkpoint(path, model, optimizer, progress):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'progress': progress, 'python_rng': random.getstate(),
             'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state()}
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        state['mps_rng'] = torch.mps.get_rng_state()
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temp); temp.replace(path)


def load_checkpoint(path):
    import inspect
    options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    return torch.load(path, map_location='cpu', **options)


def restore_checkpoint(path, model, optimizer):
    state = load_checkpoint(path)
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if 'mps_rng' in state:
        torch.mps.set_rng_state(state['mps_rng'])
    return state['progress']

ROOT = Path('comparison/standardized/common_input_20260913')


def training_phase(epoch):
    from protgnn_analysis.config import train_args
    return (epoch < train_args.warm_epochs,
            epoch >= train_args.proj_epochs and
            (epoch - train_args.proj_epochs) % train_args.proj_interval == 0)


def build_model(method, concepts, device=None):
    device = torch.device(device or ('cpu' if method == 'graphcare' else
                          'mps' if torch.backends.mps.is_available() else 'cpu'))
    if method == 'graphcare':
        from graphcare_analysis.run import build_graphcare_model
        from graphcare_analysis.config import cfg
        model = build_graphcare_model({'num_nodes': concepts + 1, 'num_rels': 3}, 30, device)
        lr, wd = cfg.lr, cfg.weight_decay
    elif method == 'gsat':
        from gsat_analysis.train import build_model as gsat_model
        from gsat_analysis.config import cfg
        model = gsat_model(concepts + 1, 30, device)
        lr, wd = cfg.lr, cfg.weight_decay
    elif method == 'protgnn':
        from protgnn_analysis.models import GnnNets
        from protgnn_analysis.config import model_args, train_args
        model_args.device = str(device); model_args.enable_prot = True
        model = GnnNets(concepts + 1, 30, model_args); model.to_device()
        lr, wd = train_args.learning_rate, train_args.weight_decay
    else:
        raise ValueError(method)
    return model, lr, wd, device


def forward_loss(model, method, batch, epoch, training, device):
    if method == 'graphcare':
        from graphcare_analysis.run import _move, _forward
        b = _move(batch, device)
        return _forward(model, b), b['y'], 0
    b = batch.to(device); y = b.y.view(-1).long()
    if method == 'gsat':
        out = model(b, epoch=epoch, training=training)
        return out['logits'], y, out['info_loss'] if training else 0
    logits, _, _, _, dist = model(b)
    if not training:
        return logits, y, 0
    identity = model.model.prototype_class_identity.to(device)
    correct = identity[:, y].T.bool()
    count = identity.shape[0] // 30
    cluster = dist[correct].reshape(-1, count).min(1)[0].mean()
    wrong = dist[~correct].reshape(-1, 29 * count).min(1)[0]
    separation = torch.clamp(1. - wrong, min=0).mean()
    l1 = (model.model.last_layer.weight * (1 - identity.T)).norm(p=1)
    return logits, y, .1 * cluster + .1 * separation + 5e-4 * l1


def fit_snapshot(df, train_indices):
    from graphcare_analysis.build_kg import vocab_and_presence
    eligible = [c for c in df if c.startswith(('med_', 'chiefcomplaint_'))]
    _, names, presence, _ = vocab_and_presence(
        df[eligible], fit_indices=train_indices, include_symptoms=False)
    return names, presence


def graphcare_graph(codes, label, concepts):
    from graphcare_analysis.adapter import _subgraph
    g = _subgraph(codes, [[] for _ in range(concepts + 1)], concepts + 1,
                  structure='star', patient_hub_id=concepts)
    g['y'] = int(label)
    return g


def pyg_graph(codes, label, concepts):
    from torch_geometric.data import Data
    g = graphcare_graph(codes, label, concepts)
    return Data(x=torch.nn.functional.one_hot(g['node_ids'], concepts + 1).float(),
                edge_index=g['edge_index'], y=torch.tensor([label], dtype=torch.long))
