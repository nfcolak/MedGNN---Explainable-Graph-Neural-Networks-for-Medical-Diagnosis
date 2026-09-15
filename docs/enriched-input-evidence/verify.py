"""Aggregate verification of the real enriched artifact; no patient rows emitted.

Run from repository root with python3 docs/enriched-input-evidence/verify.py.
This script is outside frozen runner modules so saving evidence does not break
historical checkpoint source bindings.
"""
import json
from pathlib import Path
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from comparison.standardized.enriched_input_v1.build import load_enriched, save_json
from comparison.standardized.enriched_input_v1.model import numeric_graph, build_comparator, explain_numeric
from comparison.standardized.enriched_input_v1.run import make_model, graph_subset, replay
from comparison.standardized.common_input_improvement import forward_loss
from pna_analysis.data import concept_graph
from shared.lib.benchmark_contract import file_sha256


def main():
    target = Path(__file__).parent
    if (target / 'verification.json').exists():
        raise FileExistsError('Verification report already exists')
    torch.set_num_threads(4)
    artifact = ROOT / 'comparison/standardized/enriched_inputs/source_snapshot_v1'
    bundle = load_enriched(artifact)
    manifest = bundle['manifest']; names = manifest['recipe']['names']
    frame = pd.read_csv(ROOT / 'data/merged_ed.csv', low_memory=False)
    split = json.loads((ROOT / 'comparison/canonical_split.json').read_text())
    frame = frame[frame.subject_id.astype(str).isin(split['fold'])].reset_index(drop=True)
    train = bundle['folds'] == 0
    checked = []
    for spec in manifest['recipe']['spec']:
        if spec['kind'] != 'numeric':
            continue
        name = spec['source']
        v = pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=float)
        v[~np.isfinite(v)] = np.nan
        if 'range' in spec:
            lo, hi = spec['range']; v[(v < lo) | (v > hi)] = np.nan
        fit = v[train & np.isfinite(v)]
        mean = float(fit.mean()) if len(fit) else 0.
        scale = float(fit.std()) if len(fit) else 1.
        scale = scale or 1.
        stat = manifest['recipe']['statistics'][name]
        assert stat['mean'] == mean and stat['scale'] == scale
        expected = np.where(np.isfinite(v), (v - mean) / scale, 0).astype(np.float32)
        np.testing.assert_array_equal(bundle['hub_numeric'][:, names.index(name)], expected)
        np.testing.assert_array_equal(bundle['hub_numeric'][:, names.index(name + '__missing')], ~np.isfinite(v))
        checked.append(name)
    # All-row graph/payload equality: identical effective x for PNA/ProtGNN/GSAT.
    h = hashlib.sha256()
    for i in range(len(bundle['y'])):
        codes = np.flatnonzero(bundle['presence'][i])
        old = concept_graph(codes, int(bundle['y'][i]), bundle['contract']['names'])
        graph = numeric_graph(codes, int(bundle['y'][i]), bundle['contract']['names'], bundle['hub_numeric'][i])
        cut = old.x.shape[1]
        assert torch.equal(graph.x[:, :cut], old.x)
        assert torch.equal(graph.node_type, old.node_type) and torch.equal(graph.edge_index, old.edge_index)
        assert torch.equal(graph.x[-1, cut:], torch.from_numpy(bundle['hub_numeric'][i]))
        assert not graph.x[:-1, cut:].any()
        h.update(graph.x.numpy().tobytes()); h.update(graph.edge_index.numpy().tobytes())
    indices = np.flatnonzero(train)[:8]
    graphs = graph_subset(bundle, indices)
    comparator = {}
    for method in ['protgnn', 'gsat']:
        torch.manual_seed(1234)
        model, device = build_comparator(method, len(bundle['contract']['names']), len(names))
        model.eval()
        batch = Batch.from_data_list(graphs); batch.x.requires_grad_(True)
        logits, labels, _ = forward_loss(model, method, batch, 0, False, device)
        logits.sum().backward()
        grad = batch.x.grad[batch.node_type == 2, -len(names):]
        assert torch.isfinite(logits).all() and torch.isfinite(grad).all() and grad.abs().sum() > 0
        zero = batch.clone(); zero.x = batch.x.detach().clone(); zero.x[:, -len(names):] = 0.
        with torch.no_grad():
            zero_logits, _, _ = forward_loss(model, method, zero, 0, False, device)
        delta = float((logits.detach() - zero_logits).abs().max())
        assert delta > 0
        comparator[method] = {'n_train_records': len(indices), 'numeric_gradient_l1': float(grad.abs().sum()),
                              'numeric_logit_max_abs_delta': delta, 'input_shape': list(batch.x.shape),
                              'scope': 'actual canonical train payload forward/backward only; no full curriculum/projection run'}
    synthetic = {}
    for architecture in ['plain', 'interaction']:
        run = ROOT / 'comparison/standardized/enriched_runs/pna_smoke_v1' / architecture
        state = torch.load(run / 'checkpoint.pt', weights_only=True, map_location='cpu')
        model = make_model(state['config'], bundle); model.load_state_dict(state['state_dict'])
        med = next(i for i, n in enumerate(bundle['contract']['names']) if n.startswith('med:'))
        cc = next(i for i, n in enumerate(bundle['contract']['names']) if n.startswith('cc:'))
        # No real patient features in explanation files; synthetic nonzero vector only.
        for label, codes in [('mixed', [med, cc]), ('hub_only', [])]:
            graph = numeric_graph(codes, 0, bundle['contract']['names'], np.linspace(-.5, .5, len(names)))
            torch.manual_seed(456)
            result = explain_numeric(model, graph, names, steps=4, epochs=3)
            assert all(v['status'] == 'success' for v in result['algorithms'].values())
            save_json(target / ('graphxai_' + architecture + '_' + label + '.json'), result)
            synthetic[architecture + '_' + label] = {k: v['status'] for k, v in result['algorithms'].items()}
    protected = manifest['preservation']
    assert all(file_sha256(p) == digest for p, digest in protected.items())
    lab_columns = [i for i, n in enumerate(names) if n.startswith('lab_') and n.endswith('__missing')]
    hx_columns = [i for i, n in enumerate(names) if n.startswith('hx_') and not n.endswith('__missing')]
    summary = {'status': 'verified', 'rows': len(bundle['y']), 'counts': manifest['counts'],
               'numeric_channels': len(names), 'numeric_train_stats_independently_verified': len(checked),
               'all_row_topology_payload_checks': len(bundle['y']), 'three_pyg_inputs_identical_sha256': h.hexdigest(),
               'any_lab_observed': int((bundle['hub_numeric'][:, lab_columns] == 0).any(axis=1).sum()),
               'any_reconstructed_history_positive': int((bundle['hub_numeric'][:, hx_columns] > 0).any(axis=1).sum()),
               'hub_only_by_fold': manifest['hub_only_by_fold'], 'sex': manifest['coverage']['sex'],
               'age_observed': manifest['coverage']['age']['observed'],
               'lineage': manifest['timing_audit']['lineage'],
               'source_history_positive_values_removed': manifest['timing_audit']['unsupported_positive_values_removed'],
               'history_documentation_time_verified': False, 'temporal_clean': False,
               'comparators': comparator, 'graphcare_numeric_consumer': 'pending',
               'graphxai_synthetic': synthetic, 'test_evaluated': False,
               'protected_file_hashes_unchanged': len(protected),
               'replays': {a: replay(ROOT / 'comparison/standardized/enriched_runs/pna_smoke_v1' / a)
                           for a in ['plain', 'interaction']},
               'artifact_manifest_sha256': file_sha256(artifact / 'manifest.json'),
               'verification_script_sha256': file_sha256(__file__)}
    save_json(target / 'verification.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
