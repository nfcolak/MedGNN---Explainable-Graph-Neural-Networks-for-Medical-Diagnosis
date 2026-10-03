"""Fingerprint the incumbent clinical_graph_v2 arms so later edits can be checked.

Records, for the frozen sample10k/max6/top10 protocol with every new option OFF:
  * sha256 of the train-fitted preprocessing state (must equal the frozen run's
    preprocessing.json content),
  * one sha256 over every encoded tensor of every train and validation graph,
  * sha256 of each method's untrained state_dict at seed 1234.

Usage:
  python3 -m comparison.standardized.gchm_v2_protocol.identity_snapshot --out FILE
  python3 -m comparison.standardized.gchm_v2_protocol.identity_snapshot --compare FILE
The compare mode recomputes everything under the current source tree and fails
closed on any difference. It never trains and never loads the test fold.

Compare mode also proves the edge-view premise on the same real records: the
bidirectional view keeps every node tensor and the whole forward edge block, appends
exactly one typed mirror per reversible edge with the forward payload, and is the
view in which complaint/vital/measurement nodes reach the index-visit hub at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ....paths import REPO_ROOT

ROOT = REPO_ROOT
ARTIFACT = ROOT / 'comparison/standardized/event_inputs/clinical_graph_v3_membership_max6_20260923'
TARGETS = ROOT / ('comparison/standardized/event_inputs/'
                  'first_recorded_lab_all_visits_v2_targets_local_v2_max6/targets.csv')
FROZEN_RUN = ROOT / ('comparison/standardized/'
                     'clinical_runs_v3_adapters_sample10k_max6_top10_20260924')
TENSOR_FIELDS = ('x', 'edge_index', 'edge_attr', 'token', 'node_type', 'edge_relation',
                 'edge_triple', 'edge_payload', 'visit_membership_index', 'num_visits',
                 'global_node_mask', 'y')


def _tensor_digest(splits):
    digest = hashlib.sha256()
    for fold in ('train', 'validation'):
        for data in splits[fold]:
            digest.update(data.sample_id.encode())
            for field in TENSOR_FIELDS:
                value = getattr(data, field)
                array = np.ascontiguousarray(value.detach().cpu().numpy())
                digest.update(field.encode())
                digest.update(str(array.dtype).encode())
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def snapshot():
    """Return (record, context); the context lets compare mode reuse the encoded folds."""
    from ....core import train as runner
    from ... import build_method
    from ....core.model import ClinicalGNN
    from ....core.tensorize import (
        PAYLOAD_WIDTH, degree_histogram, preprocessing_state)

    targets = runner.load_targets(TARGETS)
    targets, kept, _ = runner.select_top_labels(targets, 10)
    splits, prep = runner.build_dataset(ARTIFACT, targets, 'all', 10000, 20, 1234)
    state = preprocessing_state(prep)
    frozen_state = json.loads((FROZEN_RUN / 'gchm_pna_seed1234' / 'preprocessing.json').read_text())
    record = {
        'kept_labels': kept,
        'counts': {fold: len(data) for fold, data in splits.items()},
        'preprocessing_equals_frozen_run': state == frozen_state,
        'preprocessing_sha256': hashlib.sha256(
            json.dumps(state, sort_keys=True).encode()).hexdigest(),
        'encoded_tensor_sha256': _tensor_digest(splits),
        'untrained_state_sha256': {},
        'parameter_count': {},
    }
    node_dim = splits['train'][0].x.size(1)
    edge_dim = splits['train'][0].edge_attr.size(1)
    num_triples = len(prep['triples']) + 1
    hist = degree_histogram(splits['train'])
    record['degree_histogram_sha256'] = hashlib.sha256(hist.numpy().tobytes()).hexdigest()
    for method in ('gchm', 'protgnn', 'graphcare', 'gsat'):
        argv = ['--artifact', str(ARTIFACT), '--targets', str(TARGETS), '--output', '/dev/null',
                '--top-k-labels', '10', '--train-limit', '10000', '--epochs', '40']
        argv += (['--method', 'clinical_gnn', '--conv', 'gchm'] if method == 'gchm'
                 else ['--method', method])
        args = runner.normalize_method_args(runner.parser().parse_args(argv))
        torch.manual_seed(1234)
        if method == 'gchm':
            model = ClinicalGNN(num_classes=10, num_tokens=len(prep['vocabulary']),
                                node_dim=node_dim, edge_dim=edge_dim, hidden=args.hidden,
                                layers=args.layers, dropout=args.dropout,
                                token_dim=args.token_dim, conv='gchm',
                                num_triples=num_triples, payload_dim=PAYLOAD_WIDTH,
                                heads=args.heads, degree_histogram=hist,
                                modulation=args.modulation)
            count = model.parameter_count()
        else:
            model = build_method(method, num_tokens=len(prep['vocabulary']), node_dim=node_dim,
                                 edge_dim=edge_dim, num_classes=10, hidden=args.hidden,
                                 layers=args.layers, dropout=args.dropout,
                                 token_dim=args.token_dim, num_triples=num_triples, args=args)
            count = int(model.run_config()['architecture']['parameter_count'])
        record['untrained_state_sha256'][method] = _state_digest(model)
        record['parameter_count'][method] = count
    return record, (runner, targets, splits, prep)


def edge_view_invariants(runner, targets, forward_splits, forward_prep):
    """Premise and construction checks of the bidirectional edge view (no training)."""
    from .... import NODE_KINDS
    from ....core.tensorize import (
        ALL_RELATIONS, REVERSIBLE_RELATIONS, preprocessing_state, relation_vocabulary)

    splits, prep = runner.build_dataset(ARTIFACT, targets, 'all', 10000, 20, 1234,
                                        edge_direction='bidirectional')
    forward_width, width = len(ALL_RELATIONS), len(relation_vocabulary('bidirectional'))
    fitted = len(prep['triples'])
    reverse_id = torch.full((forward_width,), -1, dtype=torch.long)
    for position, relation in enumerate(REVERSIBLE_RELATIONS):
        reverse_id[ALL_RELATIONS.index(relation)] = forward_width + position
    reversible = (reverse_id >= 0).nonzero().flatten()
    hub = NODE_KINDS.index('visit')
    evidence = torch.tensor([NODE_KINDS.index(k) for k in ('complaint', 'vital', 'measurement')])
    failures = Counter()
    reaches_hub = Counter()
    graphs = graphs_with_evidence = reverse_edges = 0
    for fold in ('train', 'validation'):
        if len(splits[fold]) != len(forward_splits[fold]):
            failures['fold_size'] += 1
            continue
        for f, b in zip(forward_splits[fold], splits[fold]):
            graphs += 1
            if f.sample_id != b.sample_id:
                failures['sample_order'] += 1
                continue
            for field in ('x', 'token', 'node_type', 'visit_membership_index', 'num_visits',
                          'global_node_mask', 'y'):
                if not torch.equal(getattr(f, field), getattr(b, field)):
                    failures['node_' + field] += 1
            n = f.edge_index.size(1)
            if not (torch.equal(b.edge_index[:, :n], f.edge_index)
                    and torch.equal(b.edge_relation[:n], f.edge_relation)
                    and torch.equal(b.edge_triple[:n], f.edge_triple)
                    and torch.equal(b.edge_attr[:n, :forward_width], f.edge_attr[:, :forward_width])
                    and not b.edge_attr[:n, forward_width:width].any()
                    and torch.equal(b.edge_attr[:n, width:], f.edge_attr[:, forward_width:])):
                failures['forward_block'] += 1
            mask = torch.isin(f.edge_relation, reversible)
            reverse_edges += int(mask.sum())
            triples = f.edge_triple[mask]
            if not (b.edge_index.size(1) == n + int(mask.sum())
                    and torch.equal(b.edge_index[:, n:], f.edge_index[:, mask].flip(0))
                    and torch.equal(b.edge_relation[n:], reverse_id[f.edge_relation[mask]])
                    and torch.equal(b.edge_triple[n:],
                                    torch.where(triples == 0, triples, triples + fitted))
                    and torch.equal(b.edge_attr[n:, width:], f.edge_attr[mask][:, forward_width:])):
                failures['reverse_block'] += 1
            if torch.isin(f.node_type, evidence).any():
                graphs_with_evidence += 1
            for view, data in (('forward', f), ('bidirectional', b)):
                source, target = data.edge_index
                into_hub = data.node_type[target] == hub
                if torch.isin(data.node_type[source[into_hub]], evidence).any():
                    reaches_hub[view] += 1
    if preprocessing_state(prep) != preprocessing_state(forward_prep):
        failures['preprocessing_state'] += 1
    report = {'graphs': graphs, 'reverse_edges': reverse_edges,
              'graphs_with_evidence_nodes': graphs_with_evidence,
              'graphs_where_evidence_reaches_hub_in_one_hop': dict(reaches_hub),
              'failures': dict(failures)}
    passed = (not failures and reaches_hub['forward'] == 0 and reaches_hub['bidirectional'] > 0)
    return report, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out')
    group.add_argument('--compare')
    args = parser.parse_args()
    record, context = snapshot()
    if args.out:
        path = Path(args.out)
        if path.exists():
            raise FileExistsError('Snapshot exists; choose a fresh path')
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        print(json.dumps(record, indent=2, sort_keys=True))
        return
    reference = json.loads(Path(args.compare).read_text())
    differences = sorted(key for key in reference if reference[key] != record.get(key))
    if differences:
        print(json.dumps({'identical': False, 'differences': differences,
                          'current': record}, indent=2, sort_keys=True))
        raise SystemExit('Incumbent identity changed: ' + ', '.join(differences))
    invariants, passed = edge_view_invariants(*context)
    print(json.dumps({'identical': True, 'current': record,
                      'edge_view_invariants': invariants, 'edge_view_passed': passed},
                     indent=2, sort_keys=True))
    if not passed:
        raise SystemExit('Edge-view invariants failed: ' + json.dumps(invariants))


if __name__ == '__main__':
    main()
