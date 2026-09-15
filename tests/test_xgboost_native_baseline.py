import numpy as np

from comparison.standardized.xgboost_native_baseline import build_tabular_matrix, split_indices


def test_native_xgboost_tabular_matrix_preserves_counts_and_uses_no_test_rows():
    X, y, folds, feature_names, meta = build_tabular_matrix()
    assert X.shape == (74511, 324)
    assert y.shape == (74511,)
    assert np.bincount(folds, minlength=3).tolist() == [59607, 7448, 7456]
    assert len(feature_names) == 324
    assert meta["contract_sha256"] == "2a2566e662bf87b10d97e58700f868d8d6d6b036582ec046dd5453e6dcd9f7c0"
    train_idx, val_idx = split_indices(folds)
    assert len(train_idx) == 59607
    assert len(val_idx) == 7448
    assert not np.any(folds[train_idx] == 2)
    assert not np.any(folds[val_idx] == 2)


def test_native_xgboost_concept_features_are_binary_presence():
    X, y, folds, feature_names, meta = build_tabular_matrix()
    concept = X[:, :192]
    assert np.array_equal(concept, concept.astype(bool))
    assert feature_names[0].startswith("concept_present[")
    assert feature_names[191].startswith("concept_present[")
    assert feature_names[192] == meta["hub_fields"][0]
