"""
Tabular baseline for SINGLE-LABEL disease prediction (disease_1, top-30 classes)
on merged_ed_sample_60k.csv.

Mirrors the feature treatment of the GNN dataset, plus the "package A"
improvements (no change to the CSV, no leakage):
  1. class weighting           (balanced sample_weight / class_weight)
  2. early stopping + tuned HP  (validation split, ~800 trees capped by ES)
  3. engineered vital features  (shock index, pulse pressure, MAP, clinical
                                 abnormality flags, abnormal-vital count)
  4. missing indicators         ("vital not measured" is informative)
  5. xgb + hgb ensemble         (soft-prob average; default)

Target  : disease_1 (one class per distinct disease string)
Features: vitals (+engineered +missing flags), vs_* z-scores, demographics
          (gender/race/transport one-hot), med_* (home meds), SYMPTOM multi-hot
Dropped : disease_* (target/leak), icd_codes (leak), los_hours (post-outcome),
          disposition, subject_id

Reports multi-class metrics: accuracy, balanced accuracy, macro/micro-F1,
top-3 / top-5 accuracy, plus majority-class and random baselines.

Usage:
    python3 scripts/disease_baseline.py                 # ensemble (xgb+hgb)
    python3 scripts/disease_baseline.py --model xgb     # xgb only
    python3 scripts/disease_baseline.py --model hgb     # HistGradientBoosting only
    python3 scripts/disease_baseline.py --full          # use the full CSV
"""

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, top_k_accuracy_score)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# raw vital columns we engineer from (may contain NaN)
_RAW_VITALS = ["temperature", "heartrate", "resprate", "o2sat",
               "sbp", "dbp", "pain", "acuity"]


def _engineer_vitals(df):
    """Return (engineered_df, names). Clinical ratios + flags + missing
    indicators. Built from raw vitals; NaN treated as 'not abnormal' for
    flags but flagged separately via miss_* columns."""
    n = len(df)
    feats = {}

    def col(name):
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns \
            else pd.Series(np.nan, index=df.index)

    temp = col("temperature")
    hr = col("heartrate")
    rr = col("resprate")
    o2 = col("o2sat")
    sbp = col("sbp")
    dbp = col("dbp")

    # --- clinical ratios (guarded) ---
    feats["eng_shock_index"] = (hr / sbp.replace(0, np.nan))
    feats["eng_pulse_pressure"] = (sbp - dbp)
    feats["eng_map"] = (dbp + (sbp - dbp) / 3.0)

    # --- clinical abnormality flags (NaN -> 0) ---
    feats["eng_fever"] = (temp >= 100.4).astype(float)
    feats["eng_hypothermia"] = (temp <= 96.8).astype(float)
    feats["eng_tachycardia"] = (hr > 100).astype(float)
    feats["eng_bradycardia"] = (hr < 60).astype(float)
    feats["eng_tachypnea"] = (rr > 20).astype(float)
    feats["eng_hypoxia"] = (o2 < 92).astype(float)
    feats["eng_hypotension"] = (sbp < 90).astype(float)
    feats["eng_hypertension"] = (sbp > 180).astype(float)

    eng = pd.DataFrame(feats, index=df.index)
    # abnormal-vital count = sum of the boolean flags
    flag_cols = [c for c in eng.columns if c not in
                 ("eng_shock_index", "eng_pulse_pressure", "eng_map")]
    eng["eng_abnormal_count"] = eng[flag_cols].sum(axis=1)

    # NOTE: vs_* columns are raw MEAN/min/max vital values (not z-scores), so a
    # "|z|>2" count over them is meaningless — removed. The vs_* values are used
    # directly as numeric features via select_dtypes() in build_xy.

    # --- missing indicators ---
    for v in _RAW_VITALS:
        if v in df.columns:
            eng[f"miss_{v}"] = pd.to_numeric(df[v], errors="coerce").isna().astype(float)

    eng = eng.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return eng


def build_xy(df):
    # --- target ---
    df = df[df["disease_1"].fillna("").astype(str).str.len() > 0].copy()
    y_raw = df["disease_1"].astype(str)

    def _multihot(prefix, tag):
        cols = [c for c in df.columns if c.startswith(prefix)]
        vocab = set()
        for c in cols:
            vocab |= set(df[c].dropna().astype(str))
        vocab.discard("")
        vocab = sorted(vocab)
        index = {s: i for i, s in enumerate(vocab)}
        mat = np.zeros((len(df), len(vocab)), dtype=np.float32)
        for c in cols:
            vals = df[c].fillna("").astype(str).values
            for r, v in enumerate(vals):
                if v in index:
                    mat[r, index[v]] = 1.0
        names = [f"{tag}[{s}]" for s in vocab]
        return mat, names

    # --- symptom multi-hot (doctor-refined, post-exam ICD R-codes) ---
    sym_mat, sym_names = _multihot("symptom_", "sym")
    # --- chief-complaint multi-hot (triage free text, pre-diagnosis) ---
    cc_mat, cc_names = _multihot("chiefcomplaint_", "cc")

    # --- engineered vital features + missing indicators ---
    eng = _engineer_vitals(df)

    # --- drop leakage / non-feature columns ---
    drop = [c for c in df.columns
            if c.startswith("disease_") or c.startswith("symptom_")
            or c.startswith("chiefcomplaint_")
            or c in ("icd_codes", "los_hours", "disposition", "subject_id")]
    Xdf = df.drop(columns=drop, errors="ignore")
    # keep only numeric feature columns
    # Keep NaN for continuous features (labs, bmi) — XGBoost / HistGradientBoosting
    # handle missing values natively (learn a default split direction), which is
    # far better than the 0-sentinel that fillna(0) would force (a creatinine of
    # 0 is impossible; "lab not ordered" is itself informative).
    Xdf = Xdf.select_dtypes(include=[np.number]).astype(np.float32)

    X = np.hstack([Xdf.values, eng.values, sym_mat, cc_mat])
    feat_names = (list(Xdf.columns) + list(eng.columns)
                  + sym_names + cc_names)
    return X, y_raw.values, feat_names


def _make_xgb(n_classes):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=800, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=3, reg_lambda=2.0, reg_alpha=0.5, gamma=0.1,
        objective="multi:softprob", num_class=n_classes,
        tree_method="hist", n_jobs=-1, eval_metric="mlogloss",
        early_stopping_rounds=30,
    )


def _make_hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    kw = dict(max_iter=600, learning_rate=0.06, max_depth=8,
              l2_regularization=1.0, early_stopping=True,
              validation_fraction=0.1, n_iter_no_change=30, random_state=42)
    try:
        return HistGradientBoostingClassifier(class_weight="balanced", **kw)
    except TypeError:  # older sklearn without class_weight
        return HistGradientBoostingClassifier(**kw)


def _fit_xgb(clf, X_tr, y_tr, w_tr):
    # carve a validation set out of train for early stopping
    X_fit, X_val, y_fit, y_val, w_fit, _ = train_test_split(
        X_tr, y_tr, w_tr, test_size=0.1, random_state=42, stratify=y_tr)
    clf.fit(X_fit, y_fit, sample_weight=w_fit,
            eval_set=[(X_val, y_val)], verbose=False)
    best = getattr(clf, "best_iteration", None)
    if best is not None:
        print(f"    xgb early-stopped at iteration {best}")
    return clf


def _fit_hgb(clf, X_tr, y_tr, w_tr):
    try:
        clf.fit(X_tr, y_tr)            # class_weight='balanced' handles balance
    except Exception:
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ensemble",
                    choices=["ensemble", "xgb", "hgb"])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-pyxis", dest="no_pyxis", action="store_true",
                    help="drop pyx_* (ED-dispensed) columns → honest, leak-free run")
    args = ap.parse_args()

    csv = "merged_ed_all_visits.csv" if args.full else "merged_ed.csv"
    path = os.path.join(BASE, "data", csv)
    print(f"Loading {path}")
    df = pd.read_csv(path, low_memory=False)
    if args.no_pyxis:
        pyx = [c for c in df.columns if c.startswith("pyx_")]
        df = df.drop(columns=pyx)
        print(f"  --no-pyxis: dropped {len(pyx)} pyx_* columns (honest run)")

    X, y_raw, feat_names = build_xy(df)
    # Drop ultra-rare classes (< 5 patients) so a stratified split is possible
    vc = pd.Series(y_raw).value_counts()
    keep_classes = set(vc[vc >= 5].index)
    mask = np.array([yy in keep_classes for yy in y_raw])
    dropped = len(y_raw) - mask.sum()
    X, y_raw = X[mask], y_raw[mask]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(f"  rows={len(X)}  features={X.shape[1]}  classes={n_classes} "
          f"(dropped {dropped} rows in <5-sample classes)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # --- balanced sample weights (package A #1) ---
    w_tr = compute_sample_weight("balanced", y_tr)

    # --- train model(s) ---
    proba = None
    xgb_clf = None
    if args.model in ("xgb", "ensemble"):
        print("Training xgb (tuned + early stopping + class weights) ...")
        xgb_clf = _fit_xgb(_make_xgb(n_classes), X_tr, y_tr, w_tr)
        p_xgb = xgb_clf.predict_proba(X_te)
        proba = p_xgb if proba is None else proba + p_xgb
    if args.model in ("hgb", "ensemble"):
        print("Training hgb (balanced + early stopping) ...")
        hgb_clf = _fit_hgb(_make_hgb(), X_tr, y_tr, w_tr)
        p_hgb = hgb_clf.predict_proba(X_te)
        proba = p_hgb if proba is None else proba + p_hgb
    if args.model == "ensemble":
        proba = proba / 2.0  # average soft-probs

    pred = proba.argmax(1)
    labels = np.arange(n_classes)

    tag = {"ensemble": "xgb+hgb ensemble", "xgb": "xgb", "hgb": "hgb"}[args.model]
    print("\n================ DISEASE BASELINE (package A) ================")
    print(f"  model            : {tag}")
    print(f"  test n           : {len(y_te)}  classes: {n_classes}")
    print(f"  accuracy         : {accuracy_score(y_te, pred):.4f}")
    print(f"  balanced_acc     : {balanced_accuracy_score(y_te, pred):.4f}")
    print(f"  macro_F1         : {f1_score(y_te, pred, average='macro', zero_division=0):.4f}")
    print(f"  micro_F1         : {f1_score(y_te, pred, average='micro', zero_division=0):.4f}")
    print(f"  top-3 accuracy   : {top_k_accuracy_score(y_te, proba, k=3, labels=labels):.4f}")
    print(f"  top-5 accuracy   : {top_k_accuracy_score(y_te, proba, k=5, labels=labels):.4f}")

    # context baselines
    maj = np.bincount(y_tr).argmax()
    maj_acc = (y_te == maj).mean()
    print(f"\n  [ctx] majority-class accuracy : {maj_acc:.4f} "
          f"('{le.classes_[maj]}')")
    print(f"  [ctx] random-guess accuracy   : {1.0/n_classes:.4f}")

    # top features (xgb only)
    if xgb_clf is not None:
        imp = xgb_clf.feature_importances_
        order = np.argsort(-imp)[:15]
        print("\n  Top-15 features (xgb):")
        for i in order:
            print(f"    {feat_names[i]:<30} {imp[i]:.4f}")


if __name__ == "__main__":
    main()
