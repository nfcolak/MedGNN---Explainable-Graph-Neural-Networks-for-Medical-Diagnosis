"""
XGBoost baseline on merged_ed_no_icd_sample_20k.csv
Run: python3 xgboost_baseline.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, classification_report
)
import xgboost as xgb

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "merged_ed_no_icd_sample_20k.csv"
)

df = pd.read_csv(CSV_PATH)
df = df.drop(columns=["subject_id"], errors="ignore")

X = df.drop(columns=["disposition"]).values.astype(np.float32)
y = df["disposition"].values.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
print(f"Class distribution (test) — HOME: {(y_test==0).sum()}  ADMITTED: {(y_test==1).sum()}")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("TEST RESULTS")
print("="*50)
print(f"Accuracy    : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 (macro)  : {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1 (ADMITTED): {f1_score(y_test, y_pred, pos_label=1):.4f}")
print(f"ROC-AUC     : {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC      : {average_precision_score(y_test, y_proba):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["HOME", "ADMITTED"]))

# top 20 feature importances
feat_cols = df.drop(columns=["disposition"]).columns.tolist()
importances = pd.Series(model.feature_importances_, index=feat_cols)
print("Top 20 features by importance:")
print(importances.nlargest(20).to_string())
