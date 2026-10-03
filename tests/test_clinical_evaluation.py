"""Bounded synthetic regressions for visit-level clinical evaluation."""
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from core import train


def test_patient_equal_keeps_each_visit_target():
    y = np.array([0, 0, 1, 2])
    pred = np.array([0, 1, 1, 0])
    proba = np.eye(3)[pred]
    subjects = ['returning', 'returning', 'returning', 'single']
    weights = np.array([1 / 3, 1 / 3, 1 / 3, 1])
    result = train.patient_equal_metrics(y, proba, subjects, num_classes=3)
    assert result['accuracy'] == round(accuracy_score(y, pred, sample_weight=weights), 6)
    assert result['balanced_acc'] == round(
        balanced_accuracy_score(y, pred, sample_weight=weights), 6)
    assert result['macro_f1'] == round(f1_score(
        y, pred, labels=np.arange(3), average='macro', sample_weight=weights,
        zero_division=0), 6)
    assert result['patients'] == 2
    assert result['visits'] == 4
    assert result['weight_policy'] == 'inverse_evaluated_visits_per_patient'


def test_macro_f1_uses_declared_classes_even_when_validation_class_absent():
    result = train.metrics(np.array([0, 0]), np.array([[1, 0, 0], [1, 0, 0]]), 3)
    assert result['macro_f1'] == round(1 / 3, 6)
    assert result['top3_acc'] == 1.0


@pytest.mark.parametrize('subjects', [[], ['a']])
def test_patient_equal_rejects_misaligned_arrays(subjects):
    with pytest.raises(ValueError, match='aligned'):
        train.patient_equal_metrics(np.array([0, 1]), np.eye(3)[:2], subjects, 3)
