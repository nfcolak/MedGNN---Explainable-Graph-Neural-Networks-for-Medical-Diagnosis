"""Tests for CEI-GNN v3 unit U1: PLE knot table and basis (synthetic arrays only)."""
import numpy as np
import pytest
import torch

from comparison.standardized.clinical_graph_v2 import cei_v3_ple as ple


def _f32(*values):
    return np.asarray(values, dtype=np.float32)


# --- step 1: knot fit -------------------------------------------------------------

def test_fit_uses_quantiles_with_multiplicity_linear_method():
    # Heavily tied item: quantiles over the multiset differ from cut points of the distinct list.
    values = _f32(*([0.0] * 12 + [1.0] * 6 + [2.0, 5.0]))
    table = ple.fit_knots({'lab:x': values}, K=4, min_values=20)
    expected = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0], method='linear')
    # expected float64: [0, 0, 0, 1, 5] -> float32 -> collapse -> [0, 1, 5]
    assert table.method == 'linear'
    assert 'lab:x' in table.items
    assert table.items['lab:x']['knots'] == [0.0, 1.0, 5.0]
    distinct_cut_points = np.quantile(np.unique(values), [0.0, 0.25, 0.5, 0.75, 1.0], method='linear')
    assert table.items['lab:x']['knots'] != sorted(set(np.float32(distinct_cut_points).tolist()))
    assert np.float32(expected[0]) == 0.0 and np.float32(expected[-1]) == 5.0


def test_fit_casts_to_float32_before_collapsing_ties():
    # Two float64 quantiles that coincide once cast to float32 must become one knot.
    base = np.float64(1.0)
    eps = np.float64(1e-9)  # below float32 resolution near 1.0
    values = np.concatenate([np.full(10, base), np.full(10, base + eps)]).astype(np.float64)
    q = np.quantile(values, [0.0, 0.5, 1.0], method='linear')
    assert q[0] != q[-1]  # distinct in float64
    assert np.float32(q[0]) == np.float32(q[-1])  # equal in float32
    table = ple.fit_knots({'lab:tie': values.astype(np.float32)}, K=2, min_values=20)
    assert 'lab:tie' in table.items
    row = table.items['lab:tie']
    assert row['effective_knots'] == 1
    assert row['active'] is False


def test_fit_knots_strictly_increasing_and_effective_count_recorded():
    rng = np.random.default_rng(0)
    values = rng.normal(size=500).astype(np.float32)
    table = ple.fit_knots({'vital:hr': values}, K=8, min_values=20)
    assert 'vital:hr' in table.items
    row = table.items['vital:hr']
    knots = np.asarray(row['knots'], dtype=np.float32)
    assert len(knots) == 9
    assert np.all(np.diff(knots) > 0)
    assert row['effective_knots'] == 9
    assert row['active'] is True
    assert row['count'] == 500
    # positions are j/K, j = 0..K (F16)
    expected = np.quantile(values, np.arange(9) / 8.0, method='linear').astype(np.float32)
    assert knots.tolist() == expected.tolist()


def test_fewer_than_three_distinct_knots_marks_item_inactive():
    values = _f32(*([0.0] * 15 + [1.0] * 15))  # quantiles 0..1 at j/4 -> [0, 0, 0.5, 1, 1] -> 3 knots -> active
    table = ple.fit_knots({'a': values}, K=4, min_values=20)
    assert 'a' in table.items
    assert table.items['a']['active'] is True
    binary = _f32(*([0.0] * 25 + [1.0] * 5))  # [0,0,0,0,1] -> [0, 1] -> 2 knots -> inactive
    table2 = ple.fit_knots({'b': binary}, K=4, min_values=20)
    assert table2.items['b']['active'] is False
    assert table2.items['b']['effective_knots'] == 2
    assert table2.items['b']['knots'] == [0.0, 1.0]


def test_below_min_values_gets_no_row_and_non_finite_values_are_dropped():
    values = np.concatenate([np.arange(19, dtype=np.float32), _f32(np.nan, np.inf)])
    table = ple.fit_knots({'rare': values, 'ok': np.arange(40, dtype=np.float32)}, K=4, min_values=20)
    assert 'rare' not in table.items
    assert table.below_threshold == {'rare': 19}
    assert table.items['ok']['count'] == 40
    assert table.rows() == {'ok': 0}


def test_transform_kind_recorded_and_bounds_at_clip_are_valid():
    # z-scored item with clipped tails at +-10 and a signed-log item
    z = np.concatenate([np.full(5, -10.0), np.linspace(-2, 2, 30), np.full(5, 10.0)]).astype(np.float32)
    sl = np.log1p(np.arange(1, 41, dtype=np.float64)).astype(np.float32)
    table = ple.fit_knots({'z': z, 'sl': sl}, K=4, min_values=20,
                          transform_by_item={'sl': 'signed_log'})
    assert set(table.items) == {'z', 'sl'}
    assert table.items['z']['transform'] == 'zscore'
    assert table.items['sl']['transform'] == 'signed_log'
    assert table.items['z']['knots'][0] == -10.0
    assert table.items['z']['knots'][-1] == 10.0
    assert table.items['z']['active'] is True


def test_tensor_layout_rows_sorted_and_padded():
    rng = np.random.default_rng(1)
    table = ple.fit_knots({
        'b': rng.normal(size=100).astype(np.float32),
        'a': _f32(*([0.0] * 25 + [1.0] * 5)),  # inactive
        'c': _f32(*([0.0] * 15 + [1.0] * 15)),  # 3 knots
    }, K=4, min_values=20)
    assert table.rows() == {'a': 0, 'b': 1, 'c': 2}
    knots, active, transform = table.tensor()
    assert knots.dtype == torch.float32 and knots.shape == (3, 5)
    assert active.dtype == torch.bool and active.tolist() == [False, True, True]
    assert transform == ['zscore', 'zscore', 'zscore']
    assert knots[2, :3].tolist() == [0.0, 0.5, 1.0]
    assert torch.isinf(knots[2, 3:]).all()  # padding beyond the effective knots is +inf
    assert torch.all(knots[1, 1:] > knots[1, :-1])


def test_fit_rejects_bad_k_and_non_float32():
    with pytest.raises(ValueError):
        ple.fit_knots({'a': np.arange(40, dtype=np.float32)}, K=0)
    with pytest.raises(ValueError):
        ple.fit_knots({'a': np.arange(40, dtype=np.float64)}, K=4)


# --- step 2: ple basis ------------------------------------------------------------

def _table_tensors():
    # row 0: active, knots [0, 1, 3] (K=4 -> 5 columns, 2 padded)
    # row 1: inactive (2 knots)
    # row 2: active, full 5 knots
    knots = torch.tensor([
        [0.0, 1.0, 3.0, float('inf'), float('inf')],
        [0.0, 1.0, float('inf'), float('inf'), float('inf')],
        [-2.0, -1.0, 0.0, 1.0, 2.0],
    ], dtype=torch.float32)
    active = torch.tensor([True, False, True])
    return knots, active


def _basis(values, has_value, rows):
    knots, active = _table_tensors()
    return ple.ple_basis(torch.tensor(values, dtype=torch.float32),
                         torch.tensor(has_value, dtype=torch.float32),
                         torch.tensor(rows, dtype=torch.long), knots, active)


def test_basis_at_knots_is_one_hot():
    out = _basis([0.0, 1.0, 3.0, -2.0, 2.0], [1, 1, 1, 1, 1], [0, 0, 0, 2, 2])
    assert out.dtype == torch.float32 and out.shape == (5, 5)
    assert out.tolist() == [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ]


def test_basis_between_knots_interpolates_linearly():
    out = _basis([0.25, 2.0, -0.5], [1, 1, 1], [0, 0, 2])
    assert torch.allclose(out[0], torch.tensor([0.75, 0.25, 0.0, 0.0, 0.0]))
    assert torch.allclose(out[1], torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0]))
    assert torch.allclose(out[2], torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0]))
    assert torch.allclose(out.sum(dim=1), torch.ones(3))


def test_basis_outside_range_clamps_to_endpoint_basis():
    out = _basis([-100.0, 100.0, -7.0, 9.0], [1, 1, 1, 1], [0, 0, 2, 2])
    assert out.tolist() == [
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ]


def test_basis_invalid_value_inactive_row_and_no_table_are_zero():
    # has_value = 0 -> zeros even for an in-range scaled value (scaled_value is 0 by contract)
    out = _basis([0.0, 0.5, 0.5, 0.5], [0, 1, 1, 1], [0, 1, -1, 2])
    assert out[0].tolist() == [0, 0, 0, 0, 0]  # invalid value
    assert out[1].tolist() == [0, 0, 0, 0, 0]  # inactive row
    assert out[2].tolist() == [0, 0, 0, 0, 0]  # row -1: no table
    assert out[3].sum().item() == pytest.approx(1.0)  # valid row still populated
    # valid zero with has_value = 1 is distinguishable from an invalid value
    valid_zero = _basis([0.0], [1], [0])
    assert valid_zero.tolist() == [[1, 0, 0, 0, 0]]


def test_basis_empty_input_and_dtype_contract():
    knots, active = _table_tensors()
    out = ple.ple_basis(torch.zeros(0), torch.zeros(0), torch.zeros(0, dtype=torch.long), knots, active)
    assert out.shape == (0, 5) and out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_basis_rejects_zero_width_interval_and_non_increasing_knots():
    bad = torch.tensor([[0.0, 1.0, 1.0, float('inf'), float('inf')]], dtype=torch.float32)
    active = torch.tensor([True])
    with pytest.raises(ValueError):
        ple.ple_basis(torch.tensor([0.5]), torch.tensor([1.0]), torch.tensor([0]), bad, active)
    decreasing = torch.tensor([[0.0, 2.0, 1.0, float('inf'), float('inf')]], dtype=torch.float32)
    with pytest.raises(ValueError):
        ple.ple_basis(torch.tensor([0.5]), torch.tensor([1.0]), torch.tensor([0]), decreasing, active)


def test_basis_rejects_non_finite_knot_in_active_row_and_bad_row_index():
    knots, active = _table_tensors()
    nan_knots = knots.clone()
    nan_knots[0, 1] = float('nan')
    with pytest.raises(ValueError):
        ple.ple_basis(torch.tensor([0.5]), torch.tensor([1.0]), torch.tensor([0]), nan_knots, active)
    with pytest.raises(ValueError):
        ple.ple_basis(torch.tensor([0.5]), torch.tensor([1.0]), torch.tensor([3]), knots, active)
    with pytest.raises(ValueError):
        ple.ple_basis(torch.tensor([0.5]), torch.tensor([1.0]), torch.tensor([-2]), knots, active)


def test_basis_is_finite_for_non_finite_value_with_has_value_zero():
    out = _basis([float('nan'), float('inf')], [0, 0], [0, 2])
    assert torch.isfinite(out).all()
    assert out.abs().sum().item() == 0.0


# --- step 3: knot state replay ----------------------------------------------------

def _fitted_table():
    rng = np.random.default_rng(7)
    return ple.fit_knots({
        'lab:b': rng.normal(size=200).astype(np.float32),
        'lab:a': _f32(*([0.0] * 25 + [1.0] * 5)),          # inactive
        'vital:c': np.log1p(np.arange(1, 61, dtype=np.float64)).astype(np.float32),
        'rare': np.arange(5, dtype=np.float32),
    }, K=4, min_values=20, transform_by_item={'vital:c': 'signed_log'}, token_min_count=20)


def test_state_binds_method_thresholds_and_k():
    state = _fitted_table().state()
    assert {'method', 'K', 'min_values', 'token_min_count', 'state_version', 'items', 'below_threshold'} <= set(state)
    assert state['method'] == 'linear'
    assert state['K'] == 4
    assert state['min_values'] == 20
    assert state['token_min_count'] == 20
    assert state['state_version'] == 1
    assert state['below_threshold'] == {'rare': 5}
    assert set(state['items']) == {'lab:a', 'lab:b', 'vital:c'}
    assert state['items']['vital:c']['transform'] == 'signed_log'
    assert state['items']['lab:a']['active'] is False
    assert state['items']['lab:a']['effective_knots'] == 2
    import json
    json.dumps(state, allow_nan=False)  # canonical-serialisable, no NaN/inf


def test_sha256_is_stable_and_order_independent():
    table = _fitted_table()
    digest = table.sha256()
    assert isinstance(digest, str) and len(digest) == 64
    assert digest == _fitted_table().sha256()
    reordered = ple.fit_knots({
        'rare': np.arange(5, dtype=np.float32),
        'vital:c': np.log1p(np.arange(1, 61, dtype=np.float64)).astype(np.float32),
        'lab:a': _f32(*([0.0] * 25 + [1.0] * 5)),
        'lab:b': np.random.default_rng(7).normal(size=200).astype(np.float32),
    }, K=4, min_values=20, transform_by_item={'vital:c': 'signed_log'}, token_min_count=20)
    assert reordered.sha256() == digest
    other_k = ple.fit_knots({'lab:b': np.random.default_rng(7).normal(size=200).astype(np.float32)},
                            K=5, min_values=20)
    assert other_k.sha256() != digest


def test_load_round_trip_is_exact():
    import json
    table = _fitted_table()
    state = json.loads(json.dumps(table.state(), sort_keys=True))
    loaded = ple.KnotTable.load(state)
    assert loaded.K == 4
    assert loaded.method == 'linear'
    assert loaded.min_values == 20 and loaded.token_min_count == 20
    assert loaded.items == table.items
    assert loaded.below_threshold == table.below_threshold
    assert loaded.state() == table.state()
    assert loaded.sha256() == table.sha256()
    k1, a1, t1 = table.tensor()
    k2, a2, t2 = loaded.tensor()
    assert torch.equal(k1, k2) and torch.equal(a1, a2) and t1 == t2


def test_load_rejects_unbound_method_version_or_corrupt_rows():
    import copy
    state = _fitted_table().state()
    assert {'method', 'K', 'min_values', 'token_min_count', 'state_version', 'items'} <= set(state)
    for key in ('method', 'K', 'min_values', 'token_min_count', 'state_version'):
        broken = copy.deepcopy(state)
        del broken[key]
        with pytest.raises(ValueError):
            ple.KnotTable.load(broken)
    wrong_method = copy.deepcopy(state)
    wrong_method['method'] = 'nearest'
    with pytest.raises(ValueError):
        ple.KnotTable.load(wrong_method)
    wrong_version = copy.deepcopy(state)
    wrong_version['state_version'] = 99
    with pytest.raises(ValueError):
        ple.KnotTable.load(wrong_version)
    not_increasing = copy.deepcopy(state)
    not_increasing['items']['lab:b']['knots'][1] = not_increasing['items']['lab:b']['knots'][0]
    with pytest.raises(ValueError):
        ple.KnotTable.load(not_increasing)
    too_many = copy.deepcopy(state)
    too_many['items']['lab:b']['knots'].append(1e9)
    with pytest.raises(ValueError):
        ple.KnotTable.load(too_many)
    wrong_count = copy.deepcopy(state)
    wrong_count['items']['lab:b']['effective_knots'] = 2
    with pytest.raises(ValueError):
        ple.KnotTable.load(wrong_count)
    wrong_active = copy.deepcopy(state)
    wrong_active['items']['lab:a']['active'] = True
    with pytest.raises(ValueError):
        ple.KnotTable.load(wrong_active)
    non_finite = copy.deepcopy(state)
    non_finite['items']['lab:b']['knots'][0] = float('nan')
    with pytest.raises(ValueError):
        ple.KnotTable.load(non_finite)


def test_loaded_knots_are_float32_exact():
    state = _fitted_table().state()
    assert 'items' in state and len(state['items']) == 3
    for row in state['items'].values():
        for value in row['knots']:
            assert np.float32(value) == value  # stored as float32-representable floats
