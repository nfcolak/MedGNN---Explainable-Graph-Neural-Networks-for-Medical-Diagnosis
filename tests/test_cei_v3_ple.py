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
