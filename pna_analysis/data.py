"""Common-input graph adapter; types are fixed vocabulary metadata, never x.argmax."""
import json
from pathlib import Path

import numpy as np
import torch

from shared.lib.benchmark_contract import file_sha256
from comparison.standardized.common_input_improvement import pyg_graph


def train_degree_histogram(presence, folds):
    """Exact in-degree histogram: k degree-one leaves plus degree-k hub, train only."""
    active = np.asarray(presence)[np.asarray(folds) == 0].sum(axis=1).astype(np.int64)
    if not len(active):
        raise ValueError('Training fold must be nonempty')
    hist = np.bincount(active, minlength=presence.shape[1] + 1)
    hist[1] += active.sum()
    return torch.tensor(hist, dtype=torch.long).detach()


def load_common_input(root, source, split_path):
    """Read only the existing audited NPZ, validate source/split and actual run bindings.

    The NPZ hash binds row order, labels and folds to the existing all-row audit;
    no raw-data preprocessing or cache regeneration is performed here.
    """
    root, source, split_path = Path(root), Path(source), Path(split_path)
    audit = root / 'input_contract_and_audit.json'
    contract = json.loads(audit.read_text())
    paths = {'input_sha256': root / 'inputs_no_identifiers.npz',
             'source_sha256': source, 'split_sha256': split_path}
    hashes = {k: file_sha256(p) for k, p in paths.items()}
    for key, digest in hashes.items():
        if digest != contract[key]:
            raise ValueError(key + ' mismatch against existing audit')
    split = json.loads(split_path.read_text())
    names, classes = contract['names'], contract['classes']
    if classes != split['classes'] or len(set(classes)) != len(classes):
        raise ValueError('Canonical class order mismatch')
    if (len(names) != len(set(names)) or len(names) != contract['concept_channels'] or
            any(not n.startswith(('med:', 'cc:')) for n in names) or
            contract['effective_feature_dim'] != len(names) + 1 or
            contract['patient_specific_hub_channels'] != 0):
        raise ValueError('Common concept/hub input contract mismatch')
    if contract['temporal_clean'] is not False or contract['raw_to_model_train_only'] is not False:
        raise ValueError('This runner is explicitly a source-snapshot diagnostic')
    if contract['fit_scope'] != 'canonical train rows of already-filtered source snapshot only':
        raise ValueError('Train-only vocabulary fit scope required')
    with np.load(paths['input_sha256'], allow_pickle=False) as arrays:
        if set(arrays.files) != {'presence', 'y', 'folds'}:
            raise ValueError('Expected identifier-free common NPZ arrays')
        presence, y, folds = (arrays[k].copy() for k in ('presence', 'y', 'folds'))
    if (presence.shape != (len(y), len(names)) or y.ndim != 1 or folds.shape != y.shape or
            not np.isin(presence, [0, 1]).all() or not np.isin(folds, [0, 1, 2]).all() or
            y.dtype.kind not in 'iu' or folds.dtype.kind not in 'iu' or
            (y < 0).any() or (y >= len(classes)).any()):
        raise ValueError('Invalid common NPZ shapes, binary presence, labels or folds')
    counts = np.bincount(folds, minlength=3).tolist()
    split_counts = [sum(f == i for f in split['fold'].values()) for i in range(3)]
    if counts != contract['counts'] or counts != split_counts or contract['fit_rows'] != counts[0] or contract['audited_graphs'] != len(y):
        raise ValueError('Canonical fold counts mismatch')
    manifests = sorted(root.glob('trials/*/*/run_manifest.json'))
    if not manifests:
        raise ValueError('Existing common-input run binding required')
    preservation = {str(p.resolve()): file_sha256(p) for p in [*paths.values(), audit, *manifests]}
    for path in manifests:
        run = json.loads(path.read_text())
        if (run['status'] != 'completed' or run['topology'] != 'star' or
                run['n_train'] != counts[0] or run['n_validation'] != counts[1] or
                any(run['binding'][k] != hashes[k] for k in ('input_sha256', 'split_sha256'))):
            raise ValueError('Existing run binding mismatch: ' + str(path))
    return {'presence': presence, 'y': y, 'folds': folds, 'contract': contract,
            'degree_histogram': train_degree_histogram(presence, folds),
            'preservation': preservation, 'run_bindings_verified': len(manifests)}


def concept_graph(codes, label, names):
    codes = list(codes)
    if any(not n.startswith(('med:', 'cc:')) for n in names):
        raise ValueError('Only train-vocabulary med:/cc: concepts allowed')
    if len(codes) != len(set(codes)) or any(c < 0 or c >= len(names) for c in codes):
        raise ValueError('Expected unique valid concept indices')
    graph = pyg_graph(codes, label, len(names))
    # Shared builder sorts node IDs. Metadata follows that same order, not x.
    graph.node_type = torch.tensor([0 if names[c].startswith('med:') else 1
                                    for c in sorted(codes)] + [2], dtype=torch.long)
    return graph
