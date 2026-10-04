"""Checkpoint-only GraphXAI comparison on the frozen TRAIN-only screen.

No training, preprocessing fitting, dev/validation/test encoding, or historical
output writes. Feature interventions zero numeric/token/type-vector channels;
CEI's PLE term and absence inputs remain fixed. These are not clinical deletions.
ProtGNN's clinical adapter exposes no native node-importance API: never fabricate
one from its projection/search helpers or relabel a post-hoc score as native.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing

from .cei_graphxai import ClinicalGraphXAIWrapper, _batch_of, EXPLANATION_BOUNDARIES
from . import cei_v3_screen as screen
from . import cei_v3_study as study
from core.contracts import code_source_hashes, sample_ids_sha256
from core.registry import build_method
from core.schema import sha256
from core.explain.explanation_contract import build_node_explanation

from . import cei_v3_paths as io_paths

REPO = io_paths.REPO
MAIN = REPO
V3_ROOT = io_paths.DEFAULT_RESULTS / 'core'
P_ROOT = io_paths.DEFAULT_RESULTS / 'protgnn'
OUTPUT = REPO / 'comparison/standardized/clinical_runs_cei_v3_graphxai_screen500_20261001'
PATHS = None


def configure(paths, output=None):
    global PATHS, MAIN, V3_ROOT, P_ROOT, OUTPUT
    PATHS = paths
    MAIN, V3_ROOT, P_ROOT = paths.data_root, paths.v3_root, paths.protgnn_root
    if output is not None:
        OUTPUT = io_paths.check_output(paths, output)

SEEDS = (1234, 2025, 7)
MODELS = ('C_K4', 'P')
EXPLAINERS = ('GradExplainer', 'IntegratedGradExplainer', 'GNNExplainer', 'Native', 'Random')
METRICS = ('fidelity_plus', 'fidelity_minus', 'sparsity')
NATIVE_UNAVAILABLE = ('Clinical ProtGNNAdapter has no native prototype-similarity node-importance API. '
                      '_initial_coalition/_mcts_project are training projection helpers, not a patient explanation API.')
BOUNDARIES = dict(EXPLANATION_BOUNDARIES, ple_inputs='fixed original PLE term',
                  absence_inputs='fixed original membership/token/type/visit inputs',
                  explained_input='continuous numeric, token embedding and type embedding node channels',
                  edges='fixed topology/order; GNNExplainer optimizes continuous message masks',
                  clinical_correctness_measured=False)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def equal(label, actual, expected):
    if actual != expected:
        raise ValueError(f'{label}: actual={actual!r}; expected={expected!r}')


def stage_dir(model, seed):
    return (V3_ROOT if model == 'C_K4' else P_ROOT) / f'{model}_seed{seed}'


def reconstruct_adapter(binding, checkpoint_path):
    """Strict bound reconstruction; no optimizer or epoch/projection hook called."""
    if binding['method'] == 'cei_gnn_v3':
        model = io_paths.mapped_model_factory(binding, checkpoint_path,
                    paths=PATHS or io_paths.resolve(), factory=study._default_model_factory)
    elif binding['method'] == 'protgnn':
        config = binding['method_config']
        architecture = config['architecture']
        args = dict(binding)
        args.update(config['effective_settings'])
        args['num_relations'] = architecture['num_relations']
        model = build_method('protgnn', num_tokens=binding['vocabulary_size'],
                             node_dim=binding['node_dim'], edge_dim=binding['edge_dim'],
                             num_classes=binding['num_classes'], hidden=binding['hidden'],
                             layers=binding['layers'], dropout=binding['dropout'],
                             token_dim=architecture['token_dim'], num_triples=binding['num_meta_relations'],
                             args=SimpleNamespace(**args))
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)
    else:
        raise ValueError('Unsupported checkpoint method')
    equal('bound parameter count', sum(p.numel() for p in model.parameters()), binding['parameter_count'])
    equal('bound method config', model.run_config(), binding['method_config'])
    return model.cpu().eval().requires_grad_(False)


def _fixed_ple(network, features, metadata):
    return network._graphxai_fixed_ple


class CEIV3GraphXAIWrapper(ClinicalGraphXAIWrapper):
    """Exact v3 continuous path with the original PLE term cached and held fixed."""
    def __init__(self, adapter, graph):
        super().__init__(adapter, graph)
        clinical, membership, visit_graph = adapter._read(self.graph)
        self.clinical, self.membership, self.visit_graph = clinical, membership, visit_graph
        # A private network copy prevents changing the checkpoint adapter's behaviour.
        self.network = copy.deepcopy(adapter.network)
        with torch.no_grad():
            features = adapter.continuous_inputs(self.graph)
            self.network.register_buffer('_graphxai_fixed_ple',
                                         self.network._ple_term(features, clinical).detach(), persistent=False)
        self.network._ple_term = MethodType(_fixed_ple, self.network)

    def forward(self, features, edge_index, batch=None):
        self._validate_fixed(features, edge_index, batch)
        return self.network.forward_continuous(features, edge_index, self.clinical,
                                               self.membership, visit_graph=self.visit_graph)

    def _validate_fixed(self, features, edge_index, batch):
        if features.ndim != 2 or features.size(0) != int(self.graph.num_nodes):
            raise ValueError('Input must match fixed graph nodes')
        if not torch.equal(edge_index, self._edge_index):
            raise ValueError('Fixed edge identity/order changed')
        if batch is not None and not torch.equal(batch, self._batch):
            raise ValueError('Fixed batch identity changed')


class _MessageMask(MessagePassing):
    """PyG mask registration only; retain ProtGNN's exact index_add arithmetic.

    This is deliberately not a replacement encoder. Its only operation is an
    optional message multiplier installed by the shared CompatibleGNNExplainer.
    Unmasked messages are returned unchanged; degree normalization stays fixed.
    """
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, messages):
        mask = self._edge_mask
        if mask is None:
            return messages
        if self._apply_sigmoid:
            mask = mask.sigmoid()
        return messages * mask[:, None]


class ProtGNNGraphXAIWrapper(nn.Module):
    """Thin exact continuous path through the unchanged clinical ProtGNN modules."""
    def __init__(self, adapter, graph):
        super().__init__()
        self.adapter = adapter
        self.graph = copy.deepcopy(graph)
        self._edge_index, self._batch = self.graph.edge_index.clone(), _batch_of(self.graph)
        self.message_masks = nn.ModuleList([_MessageMask() for _ in adapter.layers])
        with torch.no_grad():
            self.edge_context = (adapter.relation_embedding(graph.edge_relation.long())
                                 + adapter.triple_embedding(graph.edge_triple.long())
                                 + adapter.edge_feature_projection(graph.edge_attr)).detach()

    def continuous_inputs(self):
        a, g = self.adapter, self.graph
        return torch.cat([g.x, a.token_embedding(g.token.long()),
                          a.node_type_embedding(g.node_type.long())], dim=-1).detach()

    def forward(self, features, edge_index, batch=None):
        if features.ndim != 2 or features.size(0) != self.graph.num_nodes:
            raise ValueError('Input must match fixed graph nodes')
        if not torch.equal(edge_index, self._edge_index):
            raise ValueError('Fixed edge identity/order changed')
        if batch is not None and not torch.equal(batch, self._batch):
            raise ValueError('Fixed batch identity changed')
        a = self.adapter
        numeric_token, type_vectors = torch.split(features, [a.node_dim + a.token_dim, a.hidden], dim=-1)
        h = a.node_encoder(numeric_token)
        h = F.gelu(a.input_norm(h + type_vectors))
        for layer, mask in zip(a.layers, self.message_masks):
            if edge_index.numel() == 0:
                aggregate = torch.zeros_like(h)
            else:
                source, target = edge_index
                messages = layer.message_mlp(torch.cat([h[source], h[target], self.edge_context], dim=-1))
                messages = mask(messages)
                aggregate = torch.zeros_like(h)
                aggregate.index_add_(0, target, messages)
                degree = h.new_zeros((h.size(0), 1))
                degree.index_add_(0, target, h.new_ones((target.numel(), 1)))
                aggregate = aggregate / degree.clamp_min_(1.0)
            update = layer.update_mlp(torch.cat([h, aggregate], dim=-1))
            h = layer.norm(h + layer.dropout(update))
        graph_sum = h.new_zeros((1, a.hidden))
        graph_sum.index_add_(0, self._batch, h)
        count = h.new_zeros((1, 1))
        count.index_add_(0, self._batch, h.new_ones((self._batch.numel(), 1)))
        graph_embedding = graph_sum / count.clamp_min_(1.0)
        activations, _ = a._prototype_activations(graph_embedding)
        return a.prototype_classifier(activations)


def native_importance(adapter, graph, target):
    """Signed additive accounting, half each edge/pair at each incident node.

    Rank by absolute net contribution for the predicted class. Bias and absence
    evidence are not nodes and are separately exported, not reassigned to nodes.
    """
    if not hasattr(adapter, 'network'):
        return None, {'status': 'unavailable', 'reason': NATIVE_UNAVAILABLE}
    with torch.no_grad():
        parts = adapter.forward_continuous(adapter.continuous_inputs(graph), graph.edge_index,
                                           graph, return_parts=True)
        signed = parts['node_contributions'][:, target].clone()
        for owners, values in ((graph.edge_index, parts['edge_contributions']),
                               (parts['pairs'], parts['pair_contributions'])):
            half = values[:, target] * 0.5
            signed.index_add_(0, owners[0], half)
            signed.index_add_(0, owners[1], half)
        accounting = signed.sum() + parts['bias'][target] + parts['absence_contributions'][:, target].sum()
    return signed.abs(), {'status': 'success', 'implementation': 'v3 return_parts gate-times-vote accounting',
                          'node_reduction': 'absolute signed node plus half incident edge/pair contributions',
                          'signed_node_contributions': signed.tolist(),
                          'fixed_absence_contribution': float(parts['absence_contributions'][:, target].sum()),
                          'bias': float(parts['bias'][target]),
                          'accounting_logit_abs_diff': float((accounting - parts['logits'][0, target]).abs()),
                          **BOUNDARIES}


def explain_graph(adapter, graph, *, random_seed=2026):
    """All three shared GraphXAI algorithms, genuine native score, random baseline."""
    from core.explain.graphxai_standardized import explain_algorithms
    is_cei = hasattr(adapter, 'network')
    wrapper = (CEIV3GraphXAIWrapper(adapter, graph) if is_cei
               else ProtGNNGraphXAIWrapper(adapter, graph)).eval()
    x = adapter.continuous_inputs(graph).detach() if is_cei else wrapper.continuous_inputs()
    batch = _batch_of(graph)
    with torch.no_grad():
        expected = adapter(graph, epoch=0).logits
        actual = wrapper(x, graph.edge_index, batch=batch)
    diff = float((expected - actual).abs().max())
    if not torch.equal(actual, expected):
        raise RuntimeError(f'Wrapper cannot reproduce exact model logits: max_abs_diff={diff}')
    if not torch.isfinite(actual).all():
        raise FloatingPointError('Nonfinite model/wrapper logits')
    start = time.monotonic()
    result = explain_algorithms(wrapper, x, graph.edge_index, batch=batch, steps=32, epochs=50)
    bundle_seconds = time.monotonic() - start
    for item in result.values():
        item['provenance'].update(BOUNDARIES)
        item['runtime_shared_graphxai_bundle_seconds'] = bundle_seconds
    predicted = int(actual.argmax(1).item())
    for name in ('Native', 'Random'):
        start = time.monotonic()
        if name == 'Native':
            importance, provenance = native_importance(adapter, graph, predicted)
        else:
            # Same per-patient draw for both models and every checkpoint seed.
            importance = np.random.default_rng(random_seed).random(int(graph.num_nodes))
            provenance = {'status': 'success', 'implementation': 'numpy uniform random node importance',
                          'random_seed': random_seed, **BOUNDARIES}
        result[name] = {'status': provenance['status'], 'provenance': provenance,
                        'node_explanation': None if importance is None else build_node_explanation(
                            wrapper, x, graph.edge_index, batch=batch, node_importance=importance),
                        'runtime_seconds': time.monotonic() - start}
    return result, actual.detach().numpy()[0], diff


def preflight(limit):
    """Verify frozen screen membership, input/checkpoint identity, and train-only scope."""
    from core import train
    reference = read(stage_dir('C_K4', 1234) / 'binding.json')
    record = read(V3_ROOT / 'screen_record.json')
    equal('screen fold', record['fold'], 'screen')
    targets_path = (PATHS or io_paths.resolve()).targets
    equal('target bytes', sha256(targets_path), record['targets_sha256'])
    targets, kept = screen.load_screen_targets(targets_path, top_k_labels=10)
    equal('class order', kept, reference['kept_label_indices'])
    train_ids = train.sample_train_ids(targets, reference['train_limit'], reference['sample_seed'])
    dev_ids = train.select_dev_ids(targets, train_ids, reference['dev_limit'], reference['sample_seed'])
    screen.assert_screen_replay(record, targets_path, train_ids, dev_ids)
    frozen_ids = screen.select_screen_ids(targets, train_ids, dev_ids,
                                          screen_limit=record['screen_limit'], screen_seed=record['screen_seed'])
    equal('screen ID hash', sample_ids_sha256(frozen_ids), record['screen_sample_ids_sha256'])
    selected = []
    for label in range(10):
        candidates = [sid for sid in frozen_ids if targets[sid][0] == label]
        if len(candidates) < 50:
            raise ValueError(f'class {label}: fewer than 50 screen rows')
        selected.extend(sorted(candidates, key=lambda sid: (hashlib.sha256(sid.encode()).digest(), sid))[:50])
    # Smoke covers distinct true classes; full selection remains exactly 50/class.
    ids = selected if limit == 500 else [selected[c * 50] for c in range(limit)]
    excluded = screen.excluded_subjects(targets, train_ids, dev_ids)
    for sid in ids:
        if targets[sid][1] != 'train' or sid in train_ids or sid in dev_ids or targets[sid][2] in excluded:
            raise ValueError('Requested explanation patient is not unused TRAIN screen-only')
    source = code_source_hashes()
    manifest_stages = []
    identity_keys = ('artifact_graphs_sha256', 'artifact_visit_membership_sha256', 'targets_sha256',
                     'target_binding_sha256', 'kept_label_indices', 'label_order', 'sample_seed',
                     'edges', 'edge_direction', 'token_min_count', 'min_prior_visits', 'weights',
                     'class_weight_values', 'input_contract_version', 'preprocessing_schema_version',
                     'preprocessing_sha256', 'edge_feature_layout')
    reference_rows = None
    for model in MODELS:
        for seed in SEEDS:
            directory = stage_dir(model, seed)
            binding = read(directory / 'binding.json')
            equal('checkpoint seed', binding['seed'], seed)
            equal('checkpoint method', binding['method'], 'cei_gnn_v3' if model == 'C_K4' else 'protgnn')
            equal('validation remains closed', binding['final_eval'], 'none')
            equal('test remains closed', binding['test_evaluated'], False)
            for key in identity_keys:
                equal(f'{model}/{seed} identity {key}', binding[key], reference[key])
            for fold in ('train', 'dev'):
                equal(f'{model}/{seed} {fold} roster hash', binding['split_sample_ids_sha256'][fold],
                      reference['split_sample_ids_sha256'][fold])
            for key, value in binding['source_code'].items():
                equal(f'bound source {key}', source.get(key), value)
            equal('preprocessing bytes', sha256(directory / 'preprocessing.json'), binding['preprocessing_sha256'])
            saved_result = read(directory / 'screen/screen_result.json')
            study.assert_absence_share_replay(
                saved_result, shares_path=directory / 'screen/shares.npz',
                logits_path=directory / 'screen/logits.npz')
            checkpoint_hash = sha256(directory / 'best.pt')
            equal('screen-bound checkpoint', checkpoint_hash, saved_result['checkpoint_sha256'])
            equal('screen-bound binding', sha256(directory / 'binding.json'), saved_result['binding_sha256'])
            equal('screen record identity', saved_result['screen_record_sha256'], study.screen_record_sha256(record))
            with np.load(directory / 'screen/logits.npz', allow_pickle=False) as saved:
                equal('saved screen IDs', saved['sample_ids'].astype(str).tolist(), frozen_ids)
                rows = [saved[k].tolist() for k in ('y', 'subjects', 'sample_ids')]
                if reference_rows is None:
                    reference_rows = rows
                equal('saved true classes/subjects/sample IDs', rows, reference_rows)
                equal('saved logits hash', hashlib.sha256(np.ascontiguousarray(saved['logits']).tobytes()).hexdigest(),
                      saved_result['logits_sha256'])
            manifest_stages.append({'model': model, 'seed': seed, 'directory': str(directory),
                                    'checkpoint_sha256': checkpoint_hash, 'binding_sha256': sha256(directory / 'binding.json'),
                                    'parameter_count': binding['parameter_count']})
    artifact = (PATHS or io_paths.resolve()).artifact
    for filename, key in (('graphs.jsonl', 'artifact_graphs_sha256'),
                          (reference['artifact_visit_membership_file'], 'artifact_visit_membership_sha256')):
        equal(f'artifact bytes {filename}', sha256(artifact / filename), reference[key])
    rows = list(screen.encode_rows(artifact, stage_dir('C_K4', 1234) / 'preprocessing.json', ids,
                fold='screen', edge_direction=reference['edge_direction'], targets=targets,
                preprocessing_sha256=reference['preprocessing_sha256']))
    by_id = {row.sample_id: row for row in rows}
    equal('encoded ID set', sorted(by_id), sorted(ids))
    equal('encoded row count', len(rows), len(ids))
    rows = [by_id[sid] for sid in ids]
    for row in rows:
        equal('encoded true class', int(row.y.item()), targets[row.sample_id][0])
        equal('encoded subject', str(row.subject), str(targets[row.sample_id][2]))
    manifest = {'schema': 'medgnn.cei_v3_graphxai_screen', 'fold': 'screen', 'n': len(ids),
                'selection': 'minimum SHA-256(sample_id) within true class, 50 per class; correctness ignored',
                'class_order': reference['label_order'],
                'class_counts': {str(c): sum(int(r.y.item()) == c for r in rows) for c in range(10)},
                'sample_ids': ids, 'sample_ids_sha256': sample_ids_sha256(ids),
                'subjects': [str(r.subject) for r in rows],
                'distinct_subject_count': len({str(r.subject) for r in rows}),
                'screen_record_sha256': study.screen_record_sha256(record),
                'preprocessing_sha256': reference['preprocessing_sha256'],
                'artifact_graphs_sha256': reference['artifact_graphs_sha256'],
                'stages': manifest_stages, 'source_code': source,
                'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'steps': 32, 'epochs': 50, 'bootstrap_resamples': 1000, 'bootstrap_seed': 2026,
                'random_seed_policy': 'SHA256(2026:sample_id), first 8 bytes modulo 2^63; shared across models/seeds',
                'interpretation_boundaries': BOUNDARIES, 'native_protgnn': NATIVE_UNAVAILABLE,
                'training_performed': False, 'preprocessing_fit_performed': False,
                'validation_evaluated': False, 'test_evaluated': False, 'temporal_clean': reference['temporal_clean']}
    return rows, manifest


def worker(root, model_name, seed):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(root)
    manifest = read(root / 'manifest.json')
    rows = torch.load(root / 'encoded_screen.pt', map_location='cpu', weights_only=False)
    directory = stage_dir(model_name, seed)
    adapter = reconstruct_adapter(read(directory / 'binding.json'), directory / 'best.pt')
    with np.load(directory / 'screen/logits.npz', allow_pickle=False) as saved:
        saved_logits = {str(sid): logits.copy() for sid, logits in zip(saved['sample_ids'], saved['logits'])}
    start = time.monotonic()
    max_diff, max_saved_diff, agreement, edgeless = 0.0, 0.0, 0, 0
    nonfinite = 0
    path = root / f'{model_name}_seed{seed}.jsonl'
    with path.open('x') as stream:
        for i, graph in enumerate(rows):
            sid = graph.sample_id
            patient_start = time.monotonic()
            patient_seed = int.from_bytes(hashlib.sha256(f'2026:{sid}'.encode()).digest()[:8], 'big') % (2**63)
            torch.manual_seed(patient_seed)
            explanations, logits, diff = explain_graph(adapter, graph, random_seed=patient_seed)
            max_diff = max(max_diff, diff)
            saved = saved_logits[sid]
            max_saved_diff = max(max_saved_diff, float(np.abs(logits - saved).max()))
            agreement += int(logits.argmax() == saved.argmax())
            edgeless += int(graph.edge_index.size(1) == 0)
            for name in EXPLAINERS:
                item = explanations[name]
                record = {'model': model_name, 'seed': seed, 'sample_id': sid,
                          'subject': str(graph.subject), 'true_class': int(graph.y.item()),
                          'predicted_class': int(logits.argmax()), 'explainer': name, **item,
                          'wrapper_vs_model_max_abs_logit_diff': diff,
                          'saved_screen_prediction_agrees': bool(logits.argmax() == saved.argmax()),
                          'saved_screen_max_abs_logit_diff': float(np.abs(logits - saved).max()),
                          'node_count': int(graph.num_nodes), 'edge_count': int(graph.edge_index.size(1)),
                          'patient_total_runtime_seconds': time.monotonic() - patient_start,
                          'nonfinite_count': 0}
                exp = item['node_explanation']
                if exp is not None:
                    metrics = [exp['fidelity_plus']['prob'], exp['fidelity_minus']['prob'], exp['sparsity']]
                    nonfinite += int((~np.isfinite(np.asarray(exp['node_importance']))).sum())
                    nonfinite += int((~np.isfinite(metrics)).sum())
                    if nonfinite:
                        raise FloatingPointError('Nonfinite explanation/metric output')
                stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + '\n')
            stream.flush()
            if (i + 1) % 25 == 0 or len(rows) == 5:
                print(f'{model_name}/{seed}: {i+1}/{len(rows)}; seconds={time.monotonic()-start:.2f}', flush=True)
    stats = {'model': model_name, 'seed': seed, 'n': len(rows), 'parameter_count': sum(p.numel() for p in adapter.parameters()),
             'wrapper_vs_model_max_abs_logit_diff': max_diff, 'nonfinite_count': nonfinite,
             'edgeless_graphs': edgeless, 'prediction_agreement_count': agreement,
             'prediction_agreement_fraction': agreement / len(rows), 'saved_screen_max_abs_logit_diff': max_saved_diff,
             'runtime_seconds': time.monotonic() - start}
    write_new(root / f'{model_name}_seed{seed}_stats.json', stats)
    equal('all saved screen predictions agree', agreement, len(rows))
    return 0


def aggregate(root, wall_seconds):
    manifest = read(root / 'manifest.json')
    ids, subjects = manifest['sample_ids'], manifest['subjects']
    records, groups, lookup = [], {}, {}
    for model in MODELS:
        for seed in SEEDS:
            with (root / f'{model}_seed{seed}.jsonl').open() as stream:
                items = [json.loads(line) for line in stream]
            equal('per-stage record count', len(items), len(ids) * len(EXPLAINERS))
            for item in items:
                key = (model, seed, item['explainer'], item['sample_id'])
                if key in lookup:
                    raise ValueError('Duplicate explanation record')
                lookup[key] = item
                groups.setdefault((model, seed, item['explainer']), []).append(item)
            records.extend(items)
    with (root / 'records.jsonl').open('x') as stream:
        for item in records:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + '\n')
    def metric(item, name):
        value = item['node_explanation'][name]
        return value['prob'] if isinstance(value, dict) else value
    csv_rows, matrices = [], {}
    for model in MODELS:
        for seed in SEEDS:
            for name in EXPLAINERS:
                ordered = [lookup[(model, seed, name, sid)] for sid in ids]
                good = [item for item in ordered if item['status'] == 'success']
                row = {'model': model, 'seed': seed, 'explainer': name, 'n': len(good)}
                for m in METRICS:
                    values = np.asarray([metric(item, m) for item in good], dtype=float)
                    row[f'{m}_mean'] = float(values.mean()) if len(values) else None
                    row[f'{m}_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
                    if len(good) == len(ids):
                        matrices[(model, seed, name, m)] = values
                csv_rows.append(row)
    with (root / 'results.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    patients, inverse = np.unique(np.asarray(subjects), return_inverse=True)
    rng = np.random.default_rng(2026)
    weights = np.asarray([np.bincount(rng.integers(len(patients), size=len(patients)),
                                     minlength=len(patients))[inverse] for _ in range(1000)], dtype=float)
    denominator = weights.sum(1)
    contrasts, seed_means = {}, {}
    for name in EXPLAINERS:
        contrasts[name] = {}
        seed_means[name] = {}
        for model in MODELS:
            seed_means[name][model] = {}
            for m in METRICS:
                keys = [(model, s, name, m) for s in SEEDS]
                seed_means[name][model][m] = (float(np.mean([matrices[k].mean() for k in keys]))
                                            if all(k in matrices for k in keys) else None)
        for m in METRICS:
            keys = [(model, s, name, m) for model in MODELS for s in SEEDS]
            if not all(k in matrices for k in keys):
                contrasts[name][m] = {'status': 'unavailable', 'reason': NATIVE_UNAVAILABLE}
                continue
            differences = np.asarray([matrices[('C_K4', s, name, m)] - matrices[('P', s, name, m)] for s in SEEDS])
            averaged = differences.mean(0)
            boot = (weights @ averaged) / denominator
            per_seed = differences.mean(1)
            contrasts[name][m] = {'status': 'success', 'C_minus_P': float(averaged.mean()),
                                  'interval_95': np.quantile(boot, [0.025, 0.975], method='linear').tolist(),
                                  'per_seed_differences': {str(s): float(d) for s, d in zip(SEEDS, per_seed)},
                                  'C_better_seed_count': int(((per_seed < 0) if m == 'fidelity_minus' else (per_seed > 0)).sum()),
                                  'better_direction': 'lower' if m == 'fidelity_minus' else 'higher'}
    stats = [read(root / f'{model}_seed{s}_stats.json') for model in MODELS for s in SEEDS]
    summary = {'status': 'completed', 'n_patients': len(ids), 'distinct_subject_count': len(patients),
               'record_count': len(records), 'fold': 'screen', 'class_counts': manifest['class_counts'],
               'class_order': manifest['class_order'], 'seeds': list(SEEDS), 'seed_means': seed_means,
               'C_minus_P': contrasts, 'per_stage': stats, 'wall_runtime_seconds': wall_seconds,
               'bootstrap': {'unit': 'patient/subject cluster (all visits together)', 'paired_across_models_and_seeds': True,
                             'statistic': 'mean across checkpoint seeds of paired-patient metric differences',
                             'resamples': 1000, 'seed': 2026, 'quantile_method': 'linear',
                             'inference_scope': 'conditional on these three trained checkpoints; not a seed-population interval'},
               'native_protgnn': {'status': 'unavailable', 'reason': NATIVE_UNAVAILABLE},
               'interpretation_boundaries': BOUNDARIES, 'validation_evaluated': False, 'test_evaluated': False,
               'training_performed': False, 'preprocessing_fit_performed': False,
               'input_outputs_unchanged': True, 'causal_claim': False}
    # Read-only input identity recheck of exact checkpoint/binding/preprocessing targets.
    for entry in manifest['stages']:
        directory = Path(entry['directory'])
        mapper = (PATHS or io_paths.resolve()).path_map
        for filename, key in (('best.pt', 'checkpoint_sha256'), ('binding.json', 'binding_sha256')):
            equal(f'unchanged input {directory}/{filename}',
                  sha256(mapper.resolve(directory / filename, entry[key])), entry[key])
        equal('unchanged preprocessing', sha256(mapper.resolve(directory / 'preprocessing.json',
              manifest['preprocessing_sha256'])), manifest['preprocessing_sha256'])
    write_new(root / 'summary.json', summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def execute(root, limit, workers):
    root = Path(root).resolve()
    # Exclusive ownership: neither smoke nor full execution can reuse a filled root.
    root.mkdir(parents=True, exist_ok=False)
    (root / '.gitignore').write_text('*\n')
    start = time.monotonic()
    rows, manifest = preflight(limit)
    manifest['workers'] = workers
    write_new(root / 'manifest.json', manifest)
    torch.save(rows, root / 'encoded_screen.pt')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    pending = [(m, s) for m in MODELS for s in SEEDS]
    active = []
    try:
        while pending or active:
            while pending and len(active) < workers:
                model, seed = pending.pop(0)
                log = (root / f'{model}_seed{seed}.log').open('x')
                command = [sys.executable, '-m', 'cei.studies.cei_v3_graphxai',
                           '--worker', '--output', str(root), '--model', model, '--seed', str(seed)] + \
                          (PATHS or io_paths.resolve()).cli_args()
                process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                active.append((process, log, model, seed))
            finished = []
            for process, log, model, seed in active:
                code = process.poll()
                if code is not None:
                    log.close()
                    if code:
                        raise RuntimeError(f'{model}/{seed} worker exit={code}; see {root}/{model}_seed{seed}.log')
                    print(f'{model}/{seed} complete', flush=True)
                    finished.append((process, log, model, seed))
            active = [item for item in active if item not in finished]
            if active:
                time.sleep(1)
        aggregate(root, time.monotonic() - start)
    finally:
        for process, log, _, _ in active:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', '--output-root', dest='output', type=Path, default=None)
    parser.add_argument('--limit', type=int, choices=(5, 500), default=500)
    parser.add_argument('--workers', type=int, choices=range(1, 7), default=6)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    io_paths.add_arguments(parser)
    args = parser.parse_args(argv)
    if args.plan and (args.execute or args.worker):
        parser.error('--plan is read-only and incompatible with execution/worker mode')
    configure(io_paths.resolve(args), args.output or REPO / 'comparison/standardized/clinical_runs_cei_v3_graphxai_screen500_20261001')
    if args.worker:
        if args.model is None or args.seed is None:
            parser.error('Worker requires model and seed')
        return worker(OUTPUT, args.model, args.seed)
    if not args.execute:
        report = io_paths.metadata_plan(PATHS, OUTPUT)
        report.update(n=args.limit, models=MODELS, seeds=SEEDS, explainers=EXPLAINERS,
                      steps=32, epochs=50, workers=args.workers, fold='screen', causal_claim=False)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    return execute(OUTPUT, args.limit, args.workers)


if __name__ == '__main__':
    raise SystemExit(main())
