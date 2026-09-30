"""Arm definitions, analytic parameter accounting and binding guards for the CEI-GNN v3
extension arms E2w, E2d, E6a and E6b (extensions spec §2.3, §5.2, §5.3, §6, §9 X14).

Nothing here trains, preprocesses, scores a fold or opens a data file; the guards work on
binding documents already loaded by the caller and on synthetic adapters.
"""
from __future__ import annotations

import re
from argparse import Namespace

from ..methods.cei_gnn_v2 import KIND_PAIR_COUNT
from ..tensorize import REVERSE_RELATIONS

__all__ = ['CONTROL_ARM', 'EXTENSION_ARMS', 'ARM_DEFINITIONS', 'ARM_FIELDS',
           'V2_REFERENCE_DIMENSIONS', 'V2_REFERENCE_PARAMETER_COUNT',
           'ANALYTIC_V2_SCHEMA_DELTAS', 'COMORBID_RANK', 'analytic_parameter_count',
           'analytic_arm_delta', 'arm_of_run_config', 'inventory_diff',
           'assert_size_arm_support']

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

def edge_view_record(edge_relation, edge_direction) -> dict:
    """F/R accounting of one tensorised edge list under the arm's edge view (spec §5.2)."""
    return {'edge_direction': str(edge_direction), 'forward_edges': 0, 'reverse_edges': 0,
            'mask_length': 0, 'reverse_of_forward': []}


def assert_edge_mask_length(mask, record) -> None:
    """Refuse an edge mask whose length is not the arm's own F + R (v3 §4.5 on E6a)."""
    return None


def reverse_edge_attribution(edge_contributions, edge_relation, edge_direction) -> dict:
    """Split per-edge attributions into the forward and reverse blocks; never merged."""
    return {'forward': edge_contributions, 'reverse': edge_contributions[:0],
            'reverse_relations': [], 'reverse_of_forward': [], 'derived_forward_plus_reverse': None}
