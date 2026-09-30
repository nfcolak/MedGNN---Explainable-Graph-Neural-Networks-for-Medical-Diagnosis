"""Arm definitions, analytic parameter accounting and binding guards for the CEI-GNN v3
extension arms E2w, E2d, E6a and E6b (extensions spec §2.3, §5.2, §5.3, §6, §9 X14).

Nothing here trains, preprocesses, scores a fold or opens a data file; the guards work on
binding documents already loaded by the caller and on synthetic adapters.
"""
from __future__ import annotations

import re
from argparse import Namespace

import torch

from ..methods.cei_gnn_v2 import KIND_PAIR_COUNT
from ..tensorize import (ALL_RELATIONS, EDGE_DIRECTIONS, REVERSE_RELATIONS, REVERSIBLE_RELATIONS,
                         relation_vocabulary)

__all__ = ['CONTROL_ARM', 'EXTENSION_ARMS', 'ARM_DEFINITIONS', 'ARM_FIELDS',
           'V2_REFERENCE_DIMENSIONS', 'V2_REFERENCE_PARAMETER_COUNT',
           'ANALYTIC_V2_SCHEMA_DELTAS', 'COMORBID_RANK', 'analytic_parameter_count',
           'analytic_arm_delta', 'arm_of_run_config', 'inventory_diff',
           'assert_size_arm_support', 'edge_view_record', 'assert_edge_mask_length',
           'reverse_edge_attribution']

CONTROL_ARM = 'C'
EXTENSION_ARMS = ('E2w', 'E2d', 'E6a', 'E6b')
CONTROL_HIDDEN = 128
E2W_HIDDEN = 256
COMORBID_RANK = 16   # E6b bottleneck (spec §5.3)
# `run_config()` fields that define an arm relative to C (spec §6 arm table; §2.2 flag).
ARM_FIELDS = ('hidden', 'encoder_depth', 'edge_direction', 'comorbid_block',
              'common_init_identical_to_c')
ARM_DEFINITIONS = {
    'C': {'hidden': CONTROL_HIDDEN, 'encoder_depth': 1, 'edge_direction': 'forward',
          'comorbid_block': 0, 'common_init_identical_to_c': True},
    'E2w': {'hidden': E2W_HIDDEN, 'encoder_depth': 1, 'edge_direction': 'forward',
            'comorbid_block': 0, 'common_init_identical_to_c': False},
    'E2d': {'hidden': CONTROL_HIDDEN, 'encoder_depth': 2, 'edge_direction': 'forward',
            'comorbid_block': 0, 'common_init_identical_to_c': True},
    'E6a': {'hidden': CONTROL_HIDDEN, 'encoder_depth': 1, 'edge_direction': 'bidirectional',
            'comorbid_block': 0, 'common_init_identical_to_c': True},
    'E6b': {'hidden': CONTROL_HIDDEN, 'encoder_depth': 1, 'edge_direction': 'forward',
            'comorbid_block': 1, 'common_init_identical_to_c': True},
}

# Dimensions of the recorded v2 full-run binding (spec §1): the analytic count below
# reproduces its 92,300 parameters exactly.
V2_REFERENCE_DIMENSIONS = {'num_tokens': 391, 'node_dim': 63, 'edge_dim': 22, 'num_classes': 10,
                           'num_triples': 16, 'num_relations': 15, 'token_dim': 32,
                           'pair_rank': 16, 'num_node_types': 8, 'hidden': 128}
V2_REFERENCE_PARAMETER_COUNT = 92_300
# Spec §2.3 / §5.2 / §5.3 v2-schema deltas relative to arm C (before the v3 blocks).
ANALYTIC_V2_SCHEMA_DELTAS = {'E2w': 177_792, 'E2d': 16_768, 'E6a': 4_224, 'E6b': 2_228}

_DIMENSION_KEYS = ('num_tokens', 'node_dim', 'edge_dim', 'num_classes', 'num_triples',
                   'num_relations', 'token_dim', 'pair_rank', 'num_node_types', 'hidden')


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return value


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer, got {value!r}')
    return value


def analytic_parameter_count(dimensions, *, K=None, universe_size=0, encoder_depth=1,
                             comorbid_block=0) -> int:
    """Closed-form parameter count of `EvidenceNetworkV3` (v2 schema + v3 blocks).

    v2 schema: `cei_gnn_v2.py` lines 101–119. v3 blocks: bias-free PLE projection
    `(K+1) × hidden` (v3 §12.3) and two absence tables `universe × classes`; `K=None`
    omits both (pure v2 count). E2d blocks add `w² + w + 2w` each (spec §2.2); the E6b
    block adds `w·16 + 16·C + C + C` (spec §5.3).
    """
    missing = [key for key in _DIMENSION_KEYS if key not in dimensions]
    if missing:
        raise ValueError(f'dimensions missing {missing}')
    d = {key: _positive_int(dimensions[key], key) for key in _DIMENSION_KEYS}
    w, C, rank = d['hidden'], d['num_classes'], d['pair_rank']
    _positive_int(encoder_depth, 'encoder_depth')
    if comorbid_block not in (0, 1):
        raise ValueError(f'comorbid_block must be 0 or 1, got {comorbid_block!r}')
    total = (d['num_tokens'] * d['token_dim']                      # token_embedding
             + d['num_node_types'] * w                             # node_type_embedding
             + (d['node_dim'] + d['token_dim'] + w) * w + w         # node_encoder
             + 2 * w                                               # node_norm
             + w * 2 * C + 2 * C                                   # node_head
             + d['num_relations'] * w                              # relation_embedding
             + d['num_triples'] * w                                # triple_embedding
             + d['edge_dim'] * w                                   # edge_feature_projection
             + 2 * (w * w + w + w * C + C)                         # edge_source, edge_target
             + 2 * (w * C + C)                                     # edge_context_vote/gate
             + w * rank + rank * C + C                             # pair_projection, pair_vote
             + KIND_PAIR_COUNT * C + C)                            # pair_gate, bias
    if K is not None:
        _positive_int(K, 'K')
        _nonnegative_int(universe_size, 'universe_size')
        total += (K + 1) * w + 2 * universe_size * C
    total += (encoder_depth - 1) * (w * w + w + 2 * w)
    if comorbid_block:
        total += w * COMORBID_RANK + COMORBID_RANK * C + C + C
    return int(total)


def _arm_dimensions(arm, dimensions):
    """(dimensions, encoder_depth, comorbid_block) an arm realises at the control dimensions."""
    if arm == 'E2w':
        return {**dimensions, 'hidden': E2W_HIDDEN}, 1, 0
    if arm == 'E2d':
        return dict(dimensions), 2, 0
    if arm == 'E6a':   # tensorize.py lines 76–93, 359: +R relations, +fitted triples, +R columns
        reverse = len(REVERSE_RELATIONS)
        fitted = _positive_int(dimensions['num_triples'], 'num_triples') - 1
        return {**dimensions, 'num_relations': dimensions['num_relations'] + reverse,
                'num_triples': 2 * fitted + 1, 'edge_dim': dimensions['edge_dim'] + reverse}, 1, 0
    if arm == 'E6b':
        return dict(dimensions), 1, 1
    if arm == CONTROL_ARM:
        return dict(dimensions), 1, 0
    raise ValueError(f'unknown extension arm {arm!r}; expected one of {(CONTROL_ARM,) + EXTENSION_ARMS}')


def analytic_arm_delta(arm, dimensions, *, K=None, universe_size=0) -> int:
    """Parameter delta of an extension arm relative to C at the same control dimensions."""
    arm_dims, depth, comorbid = _arm_dimensions(arm, dimensions)
    control = analytic_parameter_count(dimensions, K=K, universe_size=universe_size)
    return analytic_parameter_count(arm_dims, K=K, universe_size=universe_size,
                                    encoder_depth=depth, comorbid_block=comorbid) - control


def arm_of_run_config(config) -> str:
    """The arm label ('C', 'E2w', 'E2d', 'E6a', 'E6b') a `run_config()` document realises.

    Every extension arm is v3 arm C plus exactly one difference of the §6 table; any other
    combination of the arm fields is refused rather than mapped to the nearest arm.
    """
    if not isinstance(config, dict):
        raise ValueError('run_config must be a dict')
    if config.get('arm') != CONTROL_ARM:
        raise ValueError(f'extension arms are built on v3 arm C; run_config arm is {config.get("arm")!r}')
    if config.get('pair_mode') != 'additive':
        raise ValueError(f'extension arms use pair_mode additive; got {config.get("pair_mode")!r}')
    realised = {field: config.get(field) for field in ARM_FIELDS}
    for name, definition in ARM_DEFINITIONS.items():
        if realised == definition:
            return name
    raise ValueError(f'run_config fields {realised} match no arm of the §6 arm table')


def inventory_diff(control_inventory, arm_inventory) -> dict:
    """Named-tensor diff of an extension inventory against C's (spec §1 / E17).

    Entries are `{name: {'shape': [...], 'active': bool}}` as `run_config()` records them.
    Keys: `added`, `removed`, `widened` (same name, different shape), `unchanged` (same shape
    and activity) and `activity_changed` (same shape, different active flag); every list is
    sorted.
    """
    control, arm = dict(control_inventory), dict(arm_inventory)
    added = sorted(set(arm) - set(control))
    removed = sorted(set(control) - set(arm))
    widened, unchanged, activity = [], [], []
    for name in sorted(set(control) & set(arm)):
        if list(control[name]['shape']) != list(arm[name]['shape']):
            widened.append(name)
        elif bool(control[name]['active']) != bool(arm[name]['active']):
            activity.append(name)
        else:
            unchanged.append(name)
    return {'added': added, 'removed': removed, 'widened': widened, 'unchanged': unchanged,
            'activity_changed': activity}


_LAYERS_REFUSAL = re.compile(r'\blayers\b')


def assert_size_arm_support(v3_state=None, *, layers=2) -> dict:
    """Which registered CEI adapters accept `layers=2`: v3 must, v2 must still refuse (§2.6.3).

    Both adapters are probed with tiny dimensions. The v2 probe needs no state; the v3
    probe constructs the full network when `v3_state={'path': ..., 'k': ...}` is given,
    otherwise it stops at the state requirement, which lies after the depth check, so a
    refusal naming `layers` is the only way the v3 adapter can report `False`.
    """
    from ..methods import METHOD_REGISTRY

    dims = dict(num_tokens=2, node_dim=4, edge_dim=2, num_classes=2, hidden=8, dropout=0.0,
                token_dim=2, num_triples=2)
    support = {}
    for method, options in (('cei_gnn_v3', {'arm': CONTROL_ARM}),
                            ('cei_gnn_v2', {'pair_mode': 'additive'})):
        if method not in METHOD_REGISTRY:
            raise ValueError(f'{method} is not registered')
        if method == 'cei_gnn_v3' and v3_state is not None:
            options = {**options, 'v3_state': str(v3_state['path']), 'k': int(v3_state['k'])}
        args = Namespace(method_options=options, seed=0)
        try:
            adapter = METHOD_REGISTRY[method](layers=layers, args=args, **dims)
        except ValueError as error:
            if _LAYERS_REFUSAL.search(str(error)):
                support[method] = False
                continue
            if method == 'cei_gnn_v3' and 'v3_state' in str(error) and v3_state is None:
                support[method] = True
                continue
            raise
        depth = getattr(adapter, 'encoder_depth', getattr(adapter, 'layers_count', None))
        support[method] = depth == layers
    if support.get('cei_gnn_v3') is not True:
        raise ValueError(f'cei_gnn_v3 must accept layers={layers} (E2d maps --layers to encoder_depth)')
    if support.get('cei_gnn_v2') is not False:
        raise ValueError('cei_gnn_v2 must still refuse layers != 1 (v2 files are byte-identical)')
    return support


# ------------------------------------------------------------ E6a bidirectional view

def _relation_ids(edge_relation):
    if not torch.is_tensor(edge_relation) or edge_relation.ndim != 1:
        raise ValueError('edge_relation must be a 1-D integer tensor')
    if edge_relation.numel() and edge_relation.dtype not in (torch.int8, torch.int16, torch.int32,
                                                              torch.int64, torch.uint8):
        raise ValueError('edge_relation must use an integer dtype')
    return edge_relation.long().tolist()


def edge_view_record(edge_relation, edge_direction) -> dict:
    """F/R accounting of one tensorised edge list under the arm's edge view (spec §5.2).

    `tensorize.py` lines 348–358 append the reverse block strictly after the forward block,
    so the relation ids must be `[< forward ids …] + [>= len(ALL_RELATIONS) …]` with no
    interleaving; `reverse_of_forward` lists, in order, the forward position each reverse
    edge mirrors (the reversible forward edges in their forward order). The forward view
    must carry no reverse id at all.
    """
    if edge_direction not in EDGE_DIRECTIONS:
        raise ValueError(f'edge_direction must be one of {EDGE_DIRECTIONS}, got {edge_direction!r}')
    ids = _relation_ids(edge_relation)
    names = relation_vocabulary(edge_direction)
    forward_count = len(ALL_RELATIONS)
    if any(r < 0 for r in ids) or any(r >= len(relation_vocabulary('bidirectional')) for r in ids):
        raise ValueError('edge_relation id outside the bidirectional relation vocabulary')
    is_reverse = [r >= forward_count for r in ids]
    reverse_edges = sum(is_reverse)
    if edge_direction == 'forward' and reverse_edges:
        raise ValueError(f'forward view carries {reverse_edges} reverse-relation edge(s) '
                         f'(ids >= {forward_count})')
    forward_edges = len(ids) - reverse_edges
    if any(is_reverse[:forward_edges]) or not all(is_reverse[forward_edges:]):
        raise ValueError('reverse block must be appended strictly after the forward block '
                         '(tensorize.py line 350); reverse ids are interleaved with forward ids')
    reversible = [i for i in range(forward_edges) if names[ids[i]] in REVERSIBLE_RELATIONS]
    if edge_direction == 'bidirectional':
        expected = ['rev:' + names[ids[i]] for i in reversible]
        actual = [names[r] for r in ids[forward_edges:]]
        if actual != expected:
            raise ValueError(f'reverse block {actual} is not the reversible forward edges in '
                             f'forward order {expected}')
    return {'edge_direction': edge_direction, 'forward_edges': forward_edges,
            'reverse_edges': reverse_edges, 'mask_length': forward_edges + reverse_edges,
            'reverse_of_forward': reversible if edge_direction == 'bidirectional' else []}


def assert_edge_mask_length(mask, record) -> None:
    """Refuse an edge mask whose length is not the arm's own F + R (v3 §4.5 on E6a).

    A forward-length mask under the bidirectional view is the specific error §5.2 names;
    values must be finite probabilities in [0, 1] (the network's own rule).
    """
    if not isinstance(record, dict) or 'mask_length' not in record:
        raise ValueError('record must be an edge_view_record with mask_length')
    expected = int(record['mask_length'])
    if not torch.is_tensor(mask) or mask.ndim != 1:
        raise ValueError('edge mask must be a 1-D tensor')
    if mask.numel() != expected:
        forward = int(record.get('forward_edges', expected))
        hint = (' (a forward-length mask; the bidirectional list is forward + reverse)'
                if record.get('edge_direction') == 'bidirectional' and mask.numel() == forward else '')
        raise ValueError(f'edge mask length {mask.numel()} differs from the arm edge list length '
                         f'{expected} = {forward} forward + {expected - forward} reverse{hint}')
    if not torch.isfinite(mask).all() or bool((mask < 0).any()) or bool((mask > 1).any()):
        raise ValueError('edge mask values must be finite and in [0, 1]')
    return None


def reverse_edge_attribution(edge_contributions, edge_relation, edge_direction) -> dict:
    """Split per-edge attributions into the forward and reverse blocks; never merged.

    A reverse edge is its own item (relation `rev:*`); the forward + reverse sum is returned
    only under `derived_forward_plus_reverse` as `{label, values, forward_positions}` and is
    `None` for the forward view (spec §5.2).
    """
    record = edge_view_record(edge_relation, edge_direction)
    if not torch.is_tensor(edge_contributions) or edge_contributions.ndim != 2:
        raise ValueError('edge_contributions must be float[edges, classes]')
    if edge_contributions.size(0) != record['mask_length']:
        raise ValueError(f'edge_contributions has {edge_contributions.size(0)} rows, the edge list '
                         f'has {record["mask_length"]} (= {record["forward_edges"]} forward + '
                         f'{record["reverse_edges"]} reverse)')
    forward_edges = record['forward_edges']
    names = relation_vocabulary(edge_direction)
    ids = _relation_ids(edge_relation)
    forward = edge_contributions[:forward_edges]
    reverse = edge_contributions[forward_edges:]
    derived = None
    if edge_direction == 'bidirectional':
        positions = record['reverse_of_forward']
        derived = {'label': 'derived: forward + reverse (not a model attribution)',
                   'values': forward[positions] + reverse,
                   'forward_positions': list(positions)}
    return {'forward': forward, 'reverse': reverse,
            'reverse_relations': [names[r] for r in ids[forward_edges:]],
            'reverse_of_forward': list(record['reverse_of_forward']),
            'derived_forward_plus_reverse': derived}


# ---------------------------------------------------------- extension binding guards

# §6 common settings of every CEI stage, as `binding.json` records them (U5 policy check).
_STAGE_POLICY = {'method': 'cei_gnn_v3', 'train_limit': 10000, 'dev_limit': 5000, 'epochs': 40,
                 'patience': 40, 'sample_seed': 1234, 'selection_fold': 'dev',
                 'final_eval': 'none', 'test_evaluated': False, 'weights': 'sqrt_inverse',
                 'edges': 'all', 'top_k_labels': 10, 'num_classes': 10}
_STAGE_SEEDS = (1234, 2025, 7)
_CONTROL_FIELDS = ('k_selection', 'k_selection_sha256', 'control_binding_sha256')
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def _require(document, key, label):
    if key not in document or document[key] is None:
        raise ValueError(f'{label} missing required field: {key}')
    return document[key]


def _hash_list(value, label):
    if (not isinstance(value, list) or len(value) != len(_STAGE_SEEDS)
            or not all(isinstance(h, str) and _SHA256.match(h) for h in value)):
        raise ValueError(f'{label} must be a list of {len(_STAGE_SEEDS)} hex SHA-256 values')
    return list(value)


def _validated_controls(control_bindings) -> dict:
    """The frozen C reference (v3 §12.17, §12.21): record, its canonical hash, C's hashes."""
    from ..cei_v3_study import k_selection_sha256

    if not isinstance(control_bindings, dict):
        raise ValueError('control_bindings must be a dict')
    for key in _CONTROL_FIELDS:
        _require(control_bindings, key, 'control_bindings')
    record = control_bindings['k_selection']
    if not isinstance(record, dict):
        raise ValueError('control_bindings k_selection must be the frozen k_selection.json content')
    if control_bindings['k_selection_sha256'] != k_selection_sha256(record):
        raise ValueError('control_bindings k_selection_sha256 does not hash the k_selection record')
    if record.get('arm') != CONTROL_ARM:
        raise ValueError(f'k_selection arm {record.get("arm")!r} is not the control arm {CONTROL_ARM!r}')
    if record.get('k_selection_completed_before_screen') is not True:
        raise ValueError('k_selection must record k_selection_completed_before_screen: true')
    k_selected = record.get('k_selected')
    if isinstance(k_selected, bool) or not isinstance(k_selected, int):
        raise ValueError(f'k_selection k_selected {k_selected!r} is not an integer (frozen K)')
    controls = _hash_list(control_bindings['control_binding_sha256'], 'control_bindings control_binding_sha256')
    if controls != _hash_list(record.get('control_binding_sha256'), 'k_selection control_binding_sha256'):
        raise ValueError('control_bindings control_binding_sha256 differs from the k_selection record')
    stages = record.get('stages')
    if not isinstance(stages, list):
        raise ValueError('k_selection stages must be a list')
    winners = [s for s in stages if isinstance(s, dict) and s.get('k') == k_selected]
    if [s.get('seed') for s in winners] != list(_STAGE_SEEDS):
        raise ValueError(f'k_selection stages at the frozen K must be seeds {list(_STAGE_SEEDS)} in order')
    if [s.get('binding_sha256') for s in winners] != controls:
        raise ValueError('control_binding_sha256 differs from the binding hashes of the frozen-K C stages')
    checkpoints = [s.get('checkpoint_sha256') for s in winners]
    if 'control_checkpoint_sha256' in control_bindings:
        if _hash_list(control_bindings['control_checkpoint_sha256'], 'control_checkpoint_sha256') != checkpoints:
            raise ValueError('control_bindings control_checkpoint_sha256 differs from the frozen-K C checkpoints')
    return {'record': record, 'k_selection_sha256': control_bindings['k_selection_sha256'],
            'k_selected': int(k_selected), 'control_binding_sha256': controls,
            'control_checkpoint_sha256': checkpoints,
            'v3_state_sha256': record.get('selected_v3_state_sha256'),
            'knot_table_sha256': record.get('selected_knot_table_sha256'),
            'preprocessing_sha256': control_bindings.get('preprocessing_sha256'),
            'v3_state': control_bindings.get('v3_state'), 'rule': record.get('rule')}


def extension_binding_record(binding, control_bindings) -> dict:
    """The checked fields of one extension-stage binding, or raise (see the assertion)."""
    assert_extension_binding(binding, control_bindings)
    config = binding['method_config']
    return {'arm': binding['extension_arm'], 'seed': int(binding['seed']), 'k': int(config['k']),
            'k_selection_sha256': binding['k_selection_sha256'],
            'control_binding_sha256': list(binding['control_binding_sha256']),
            'control_checkpoint_sha256': list(binding.get('control_checkpoint_sha256', [])),
            'preprocessing_sha256': binding['preprocessing_sha256'],
            'v3_state_sha256': config['v3_state_sha256'], 'final_eval': binding['final_eval'],
            'arm_definition': dict(ARM_DEFINITIONS[binding['extension_arm']]),
            'parameter_count': binding['parameter_count'],
            'common_init_identical_to_c': config['common_init_identical_to_c']}


def assert_extension_binding(binding, control_bindings) -> None:
    """Refuse an extension-stage binding that does not bind C's freeze (spec §1, §6, §9 X14).

    `binding` is the stage's `binding.json` content plus the study-binding fields
    (`extension_arm`, `k_selected`, `k_grid`, `k_selection_rule`, `k_selection_sha256`,
    `control_binding_sha256`, optional `control_checkpoint_sha256`). `control_bindings`
    carries the frozen `k_selection` record, its canonical hash, and C's three binding
    hashes (and optionally the checkpoint hashes, `preprocessing_sha256`, `v3_state_sha256`,
    `v3_state`, `k` of the C stages).

    Checks, in order: the K-freeze hash and frozen K; `control_binding_sha256` (exactly C's
    three, in seed order); `preprocessing_sha256` and the v3 state hash/path;
    `final_eval == 'none'` with no validation/test trace; the §6 protocol settings; the arm
    definition fields (§6 table, `common_init_identical_to_c` of §2.2) and their agreement
    between the runner binding and the adapter's `run_config`. Inputs are never mutated.
    """
    if not isinstance(binding, dict):
        raise ValueError('binding must be a dict')
    controls = _validated_controls(control_bindings)
    config = _require(binding, 'method_config', 'binding')
    if not isinstance(config, dict) or config.get('method') != 'cei_gnn_v3':
        raise ValueError('binding method_config is not a cei_gnn_v3 run_config')

    # --- K-freeze hash and frozen K (v3 §12.17; EXT §1) -----------------------------------
    freeze = _require(binding, 'k_selection_sha256', 'binding')
    if freeze != controls['k_selection_sha256']:
        raise ValueError('binding k_selection_sha256 differs from the frozen K-freeze record hash')
    k_selected = controls['k_selected']
    if binding.get('k_selected') != k_selected:
        raise ValueError(f'binding k_selected {binding.get("k_selected")!r} differs from the frozen K '
                         f'{k_selected}')
    settings = config.get('effective_settings')
    if not isinstance(settings, dict):
        raise ValueError('binding method_config lacks effective_settings')
    architecture = config.get('architecture', {})
    if not (config.get('k') == k_selected == settings.get('k') == architecture.get('k')):
        raise ValueError(f'k drift: binding k {config.get("k")!r} differs from the frozen K {k_selected}')
    if binding.get('k_selection_rule') != controls['rule']:
        raise ValueError('binding k_selection_rule differs from the frozen record rule')
    if list(binding.get('k_grid', [])) != list(controls['record'].get('k_grid', [])):
        raise ValueError('binding k_grid differs from the frozen record grid')

    # --- control_binding_sha256 (v3 §12.21) ----------------------------------------------
    bound = _hash_list(_require(binding, 'control_binding_sha256', 'binding'), 'binding control_binding_sha256')
    if bound != controls['control_binding_sha256']:
        raise ValueError("binding control_binding_sha256 differs from C's three frozen binding hashes "
                         '(seed order 1234, 2025, 7)')
    if 'control_checkpoint_sha256' in binding:
        if _hash_list(binding['control_checkpoint_sha256'], 'binding control_checkpoint_sha256') != controls['control_checkpoint_sha256']:
            raise ValueError("binding control_checkpoint_sha256 differs from C's three frozen checkpoint hashes")

    # --- preprocessing_sha256 and the v3 state (EXT §5.2; v3 §12.18) ---------------------
    prep = _require(binding, 'preprocessing_sha256', 'binding')
    if config.get('preprocessing_sha256') != prep:
        raise ValueError('preprocessing_sha256 of the v3 state differs from the stage binding')
    if controls['preprocessing_sha256'] is not None and prep != controls['preprocessing_sha256']:
        raise ValueError("binding preprocessing_sha256 differs from C's")
    if controls['v3_state_sha256'] is not None and config.get('v3_state_sha256') != controls['v3_state_sha256']:
        raise ValueError('binding v3_state_sha256 differs from the frozen selected_v3_state_sha256')
    if controls['knot_table_sha256'] is not None and config.get('knot_table_sha256') != controls['knot_table_sha256']:
        raise ValueError('binding knot_table_sha256 differs from the frozen selected_knot_table_sha256')
    if controls['v3_state'] is not None:
        if config.get('v3_state_path') != controls['v3_state'] or settings.get('v3_state') != controls['v3_state']:
            raise ValueError("binding v3_state path differs from C's frozen state file")

    # --- final_eval == 'none', no validation/test trace (EXT §6 E4; v3 §12.7) ------------
    if binding.get('final_eval') != 'none':
        raise ValueError(f"binding final_eval {binding.get('final_eval')!r} must be 'none'")
    if 'selected_validation' in binding:
        raise ValueError('binding carries selected_validation: validation was scored')
    if binding.get('test_evaluated') is not False:
        raise ValueError('binding test_evaluated must be False: the test fold is never scored')
    splits = binding.get('split_sample_ids_sha256', {})
    counts = binding.get('counts', {})
    if 'test' in splits or 'test' in counts or 'screen' in splits or 'screen' in counts:
        raise ValueError('binding carries a test/screen split: never present in a stage binding')
    if binding.get('selection_fold') != 'dev':
        raise ValueError(f"binding selection_fold {binding.get('selection_fold')!r} must be 'dev'")

    # --- protocol settings (EXT §6 common settings) --------------------------------------
    for key, value in _STAGE_POLICY.items():
        if binding.get(key) != value:
            raise ValueError(f'study policy mismatch for {key}: expected {value!r}, got {binding.get(key)!r}')
    if binding.get('seed') not in _STAGE_SEEDS:
        raise ValueError(f'binding seed {binding.get("seed")!r} is not one of {list(_STAGE_SEEDS)}')

    # --- arm definition fields (EXT §6 table, §2.2 flag) ---------------------------------
    arm = _require(binding, 'extension_arm', 'binding')
    if arm not in EXTENSION_ARMS:
        raise ValueError(f'binding extension_arm {arm!r} is not one of {list(EXTENSION_ARMS)}')
    if settings.get('arm') != CONTROL_ARM:
        raise ValueError(f'extension arms are built on arm C; effective_settings arm is {settings.get("arm")!r}')
    for key, expected in (('ple_active', True), ('absence_active', True)):
        if config.get(key) is not expected:
            raise ValueError(f'{key} {config.get(key)!r} differs from arm C ({expected})')
    realised = arm_of_run_config(config)   # raises 'arm'/'pair_mode' on any other combination
    if realised != arm:
        raise ValueError(f'binding extension_arm {arm!r} differs from the arm the run_config realises ({realised!r})')
    definition = ARM_DEFINITIONS[arm]
    if config.get('common_init_identical_to_c') is not definition['common_init_identical_to_c']:
        raise ValueError(f'common_init_identical_to_c must be {definition["common_init_identical_to_c"]} for {arm}')
    for key in ('encoder_depth', 'comorbid_block'):
        if settings.get(key) != definition[key]:
            raise ValueError(f'effective_settings {key} {settings.get(key)!r} differs from arm {arm}')
    if binding.get('hidden') != definition['hidden'] or architecture.get('hidden') != definition['hidden']:
        raise ValueError(f'runner hidden {binding.get("hidden")!r} differs from arm {arm} ({definition["hidden"]})')
    if binding.get('layers') != definition['encoder_depth'] or architecture.get('layers') != definition['encoder_depth']:
        raise ValueError(f'runner layers {binding.get("layers")!r} differs from arm {arm} encoder_depth '
                         f'{definition["encoder_depth"]}')
    if binding.get('edge_direction') != definition['edge_direction']:
        raise ValueError(f'runner edge_direction {binding.get("edge_direction")!r} differs from arm {arm} '
                         f'({definition["edge_direction"]})')
    return None
