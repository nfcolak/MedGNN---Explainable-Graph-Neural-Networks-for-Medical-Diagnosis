"""Focused guards for the opt-in matched XGBoost runner."""
import numpy as np
import pytest
import xgboost as xgb

from comparison.standardized.matched_gchm_xgb_v1.runner import (
    check_probabilities, first_best, run_cell, selection_metric,
)
from comparison.standardized.performance_review import class_weights
from shared.lib.metrics import multiclass_metrics


def test_selection_matches_shared_metric_and_first_tie():
    y = np.tile(np.arange(30), 2)
    rng = np.random.default_rng(11)
    p = rng.dirichlet(np.ones(30), size=len(y)).astype(np.float32)
    d = xgb.DMatrix(np.zeros((len(y), 2)), label=y)
    assert selection_metric(p, d) == ('macro_f1', multiclass_metrics(y, p.argmax(1), p)['macro_f1'])
    assert first_best([.2, .3, .3, .1]) == 1


@pytest.mark.parametrize('values', [[], [np.nan], [np.inf], [[.1]]])
def test_invalid_history_rejected(values):
    with pytest.raises(ValueError):
        first_best(values)


def test_weight_formula_uses_training_counts():
    y = np.repeat(np.arange(30), np.arange(1, 31))
    expected = np.sqrt(len(y) / (30 * np.bincount(y)))
    np.testing.assert_array_equal(class_weights(y, 30, 'sqrt_inverse'), expected)
    np.testing.assert_array_equal(class_weights(y, 30, 'none'), np.ones(30))


def test_probability_guards():
    p = np.full((2, 30), 1 / 30)
    check_probabilities(p, 2)
    for bad in [p[:, :-1], p * 2, np.full((2, 30), np.nan)]:
        with pytest.raises(ValueError):
            check_probabilities(bad, 2)


def test_scope_rejected_before_reading_data(tmp_path):
    with pytest.raises(ValueError, match='Seed/weight'):
        run_cell(tmp_path / 'out', seed=99)
    with pytest.raises(ValueError, match='bounded'):
        run_cell(tmp_path / 'out', smoke_rounds=3)
    assert not (tmp_path / 'out').exists()


def test_real_xgboost_fixed_budget_selection_and_reload(tmp_path):
    rng = np.random.default_rng(7)
    y = np.tile(np.arange(30), 3)
    d = xgb.DMatrix(rng.normal(size=(len(y), 4)), label=y)
    history = {}
    b = xgb.train({'objective': 'multi:softprob', 'num_class': 30,
                   'tree_method': 'hist', 'nthread': 2, 'max_depth': 2,
                   'disable_default_eval_metric': 1, 'seed': 7},
                  d, num_boost_round=4, evals=[(d, 'validation')],
                  custom_metric=selection_metric, evals_result=history, verbose_eval=False)
    values = history['validation']['macro_f1']
    assert len(values) == b.num_boosted_rounds() == 4
    best = first_best(values)
    selected = b[:best + 1]
    proba = selected.predict(d)
    assert multiclass_metrics(y, proba.argmax(1), proba)['macro_f1'] == values[best]
    selected.save_model(tmp_path / 'model.ubj')
    loaded = xgb.Booster()
    loaded.load_model(tmp_path / 'model.ubj')
    np.testing.assert_array_equal(loaded.predict(d), proba)
