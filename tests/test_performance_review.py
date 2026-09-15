"""Focused leakage/budget regressions for isolated review experiments."""
import numpy as np
import pytest

def test_train_only_weights_ignore_other_folds():
    from comparison.standardized.performance_review import class_weights
    w = class_weights(np.array([0, 0, 0, 1]), 2, 'sqrt_inverse')
    assert np.allclose(w, np.sqrt([4 / 6, 4 / 2]))

def test_feature_views_keep_concepts_and_do_not_pool_hub_twice():
    from comparison.standardized.performance_review import graph_vector
    x = np.zeros((3, 10), dtype=np.float32)
    x[0, 7:] = [2, -1, 4]
    x[1, 2] = 1; x[2, 3] = 1
    assert graph_vector(x, 2, 4, 7, 'concepts').tolist() == [1, 1]
    assert graph_vector(x, 2, 4, 7, 'native').tolist() == [1, 1, 2, -1, 4]

def test_validation_selection_has_no_test_dependency():
    from comparison.standardized.performance_review import best_validation
    rows = [{'epoch': 0, 'macro_f1': .2}, {'epoch': 1, 'macro_f1': .3}, {'epoch': 2, 'macro_f1': .3}]
    assert best_validation(rows)['epoch'] == 1

def test_budget_rejects_extra_trials_or_epochs():
    from comparison.standardized.performance_review import validate_budget
    validate_budget('gsat', 'current', 3, 1234)
    with pytest.raises(ValueError): validate_budget('gsat', 'extra', 3, 1234)
    with pytest.raises(ValueError): validate_budget('gsat', 'current', 4, 1234)
    with pytest.raises(ValueError): validate_budget('gsat', 'current', 3, 1235)
