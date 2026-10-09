"""Artifact-only matched validation analysis; never instantiate or run a model.

Run with ``python -m comparison.standardized.matched_gchm_xgb_v1.report``.
Historical sources are checked against their snapshots, NOT changed live code.
Bootstrap conditions on selected checkpoints and observed true-class supports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

from comparison.standardized.native_reference_v1.data import (
    DEFAULT, PINNED_CONTRACT, Reference, digest, sha,
)
from shared.lib.metrics import multiclass_metrics

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
METRICS = ("macro_f1", "balanced_acc", "accuracy", "micro_f1", "top3_acc", "top5_acc")
BOOT_METRICS = ("macro_f1", "balanced_acc", "accuracy")
GCHM_SELECTION = "validation macro_f1; full fixed epoch budget; no early stop"
XGB_SELECTION = "validation macro_f1; first maximum rounded to 6 decimals; full fixed budget"
BOOT_EXPLANATION = (
    "Within each true class, multinomial sampling of observed paired-prediction "
    "categories is distributionally identical to resampling patients with replacement: "
    "each patient's (prediction_A, prediction_B) pair remains together. Fixed class "
    "supports and paired confusion-count sufficient statistics reproduce accuracy, "
    "balanced accuracy, macro F1 and class recall without individual bootstrap indices. "
    "Percentile intervals use unrounded metrics and numpy linear quantiles."
)
LIMITATIONS_TR = [
    "temporal_clean=false: geçmiş tanı alanları mevcut/gelecek tanıları içerebilir; "
    "tüm ziyaret laboratuvarları ve yatış-geneli vital özetlerinin karar anında erişilebilirliği doğrulanmamıştır. "
    "Bunlar klinik erken tahmin değil, sızıntı riski taşıyan doğrulama/rekonstrüksiyon sonuçlarıdır.",
    "Ağırlıklandırma nadir sınıf recall kazanırken precision ve çoğunluk doğruluğunu "
    "düşürebilir; FP/FN ve precision birlikte okunmalı, klinik maliyet analizi yapılmalıdır.",
    "Aynı doğrulama kümesi checkpoint seçimi ve raporlama için kullanıldı. Sabit seçilmiş "
    "checkpoint'lere koşullu bootstrap seçim yanlılığını veya eğitim belirsizliğini kapsamaz.",
    "Bilgi, bölmeler, ağırlık politikası, metrik ve seed eşleşir; hesap bütçesi eşit değildir: "
    "GCHM 30 epoch, XGBoost 1200 tur. Mimari, optimizer ve aranan checkpoint sayısı farklıdır.",
    "Güven aralıkları sınıf bazında noktasaldır; çoklu karşılaştırma düzeltmesi yoktur. "
    "Keşifsel sonuçlar doğrulayıcı üstünlük veya klinik kullanım kanıtı değildir.",
    "test_evaluated=false; test tahmini/metriği üretilmedi. Harici, zaman-güvenli değerlendirme gerekir.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def array_sha(array):
    """Hash shape, dtype and contiguous bytes (not a patient identity substitute)."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(json.dumps([a.dtype.str, list(a.shape)]).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def contained(root, relative):
    """Reject absolute paths, traversal and symlink escapes in artifact maps."""
    root = Path(root).resolve()
    p = root / relative
    require(not Path(relative).is_absolute() and p.resolve().is_relative_to(root),
            f"Unsafe artifact path: {relative}")
    return p


def verify_hashes(root, hashes):
    require(isinstance(hashes, dict) and bool(hashes), "Missing artifact/source hash map")
    for relative, expected in hashes.items():
        p = contained(root, relative)
        require(p.is_file() and sha(p) == expected, f"Hash mismatch: {p}")
    return dict(hashes)


def validate_protocol(protocol):
    require(protocol["version"] == "matched-gchm-xgb-v1", "Wrong protocol version")
    require(protocol["status"] == "frozen_before_new_xgboost_training", "Unfrozen protocol")
    require(protocol["contract_sha256"] == PINNED_CONTRACT, "Wrong native contract")
    require(protocol["test_evaluated"] is False, "Protocol opens test")
    require(protocol["primary_metric"] == "macro_f1", "Wrong selection metric")
    require(protocol["seeds"] == [1234, 1235, 1236], "Exactly three fixed seeds required")
    require(protocol["weight_policies"] == ["none", "sqrt_inverse"], "Wrong weight policies")
    u = protocol["uncertainty"]
    require((u["replicates"], u["seed"], u["confidence"]) == (2000, 20260919, 0.95),
            "Changed frozen bootstrap configuration")
    require(protocol["xgboost"]["num_boost_round"] == 1200, "Changed fixed XGB budget")


def cohort_context(ref):
    """Subset train/validation only. Never request fold 2 or index test labels."""
    tr, va = ref.fold(0), ref.fold(1)
    require(len(tr) > 0 and len(va) > 0, "Empty train/validation cohort")
    labels = ref.contract["labels"]
    y = ref.arrays["y"][va]
    train_y = ref.arrays["y"][tr]
    # Indexed endpoints, rather than computing sizes for the entire dataset.
    ptr = ref.arrays["node_ptr"]
    concepts = ptr[va + 1] - ptr[va] - 1
    require(np.all(concepts >= 0), "Negative concept count")
    require(np.all((y >= 0) & (y < len(labels))), "Invalid validation labels")
    require(np.all((train_y >= 0) & (train_y < len(labels))), "Invalid training labels")
    return {"train": tr, "validation": va, "y": y, "labels": labels,
            "train_support": np.bincount(train_y, minlength=len(labels)), "concepts": concepts,
            "train_ordinals_sha256": digest(tr.tolist()),
            "validation_ordinals_sha256": digest(va.tolist()),
            "labels_sha256": digest(labels)}


def validate_prediction(proba, y, ordinals, context):
    require(np.array_equal(ordinals, context["validation"]), "Validation ordinal/order mismatch")
    require(np.array_equal(y, context["y"]), "Validation label/order mismatch")
    p = np.asarray(proba)
    require(p.shape == (len(y), len(context["labels"])), "Prediction shape mismatch")
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(),
            "Invalid probability values")
    require(np.allclose(p.sum(axis=1), 1., atol=2e-6, rtol=0), "Unnormalized probabilities")
    return p


def check_common_manifest(manifest, method, policy, seed, ref, context):
    require(manifest["status"] == "completed", "Incomplete run")
    require(manifest["method"] == method, "Wrong method")
    require(manifest["test_evaluated"] is False, "Run evaluated test")
    require(manifest["contract_sha256"] == ref.fingerprint, "Run contract mismatch")
    require(manifest["selected_counts"] == [len(context["train"]), len(context["validation"])],
            "Partial or mismatched cohort")
    b = manifest["binding"]
    require(b["contract_sha256"] == ref.fingerprint, "Binding contract mismatch")
    require(b["seed"] == seed and b["weight_policy"] == policy, "Seed/weight mismatch")
    for key in ("train_ordinals_sha256", "validation_ordinals_sha256"):
        require(b[key] == context[key], f"Binding {key} mismatch")
    if "labels_sha256" in b:
        require(b["labels_sha256"] == context["labels_sha256"], "Label hash mismatch")
    if "input_sha256" in b:
        require(b["input_sha256"] == ref.contract["input_sha256"], "Binding input mismatch")
    return b


def first_maximum(history, budget, index_key="epoch", metric_key=None):
    require(len(history) == budget, "Incomplete fixed-budget history")
    require([r[index_key] for r in history] == list(range(budget)), "History order/gaps/duplicates")
    values = np.array([r[metric_key] if metric_key else r["validation"]["macro_f1"]
                       for r in history], dtype=float)
    require(np.isfinite(values).all(), "Nonfinite selection metric")
    return history[int(np.argmax(np.round(values, 6)))]


def load_gchm(run, policy, seed, ref, context):
    """Validate the native replay proof without unpickling or neural inference."""
    import torch

    run = Path(run)
    m = read_json(run / "run_manifest.json")
    b = check_common_manifest(m, "gchm", policy, seed, ref, context)
    require(b["method"] == "gchm" and b["epochs"] == 30 and b["limit"] is None,
            "GCHM is not the full fixed-budget run")
    require(b["loss"] == ("ce" if policy == "none" else "sqrt_inverse"), "Loss mismatch")
    require(b["selection"] == GCHM_SELECTION and m["next_epoch"] == 30,
            "GCHM selection/budget mismatch")
    sources = verify_hashes(run / "source_snapshot", b["source_code"])
    with np.load(run / "cohort.npz", allow_pickle=False) as z:
        require(np.array_equal(z["train_ordinals"], context["train"]), "Training cohort mismatch")
        require(np.array_equal(z["validation_ordinals"], context["validation"]), "Validation cohort mismatch")
    history = read_json(run / "history.json")
    selected = first_maximum(history, 30)
    for row in history:
        require([row["train_count"], row["validation_count"]] == m["selected_counts"],
                "History cohort mismatch")
    ep = selected["epoch"]
    proof = read_json(run / "replay.json")
    require(proof == m["replay"], "Replay file/manifest mismatch")
    require(proof["exact_logits"] is True and proof["test_evaluated"] is False,
            "Missing exact validation replay")
    require(proof["selected_epoch"] == ep and proof["validation_count"] == len(context["y"]),
            "Replay selection/cohort mismatch")
    require(m["checkpoint_path"] == "best.pt" and m["metrics_path"] == "history.json",
            "Unexpected native artifact path")
    require(sha(run / "best.pt") == proof["checkpoint_sha256"], "Checkpoint SHA mismatch")
    prediction_file = f"validation_{ep:03d}.npz"
    with np.load(run / prediction_file, allow_pickle=False) as z:
        logits = z["logits"]
        require(logits.dtype == np.float32 and np.isfinite(logits).all(), "Invalid native logits")
        p = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        validate_prediction(p, z["y"], z["ordinals"], context)
    metrics = multiclass_metrics(context["y"], p.argmax(1), p)
    require(metrics == selected["validation"], "Recomputed selected GCHM metrics mismatch")
    names = ["run_manifest.json", "history.json", "cohort.npz", "best.pt", "replay.json", prediction_file]
    return p, {"run": str(run), "selected_epoch": ep, "selected_epoch_number": ep + 1,
               "selection": b["selection"], "metrics": metrics,
               "artifact_sha256": {name: sha(run / name) for name in names},
               "source_snapshot_sha256": sources, "binding": b,
               "probability_sha256": array_sha(p), "replay": proof,
               "provenance_note": "Stored original exact-logit replay verified by checkpoint hash; no new inference."}


def load_xgboost(run, policy, seed, ref, context, protocol_sha256, budget=1200, expected_params=None):
    from comparison.standardized.performance_review import class_weights

    run = Path(run)
    m = read_json(run / "run_manifest.json")
    b = check_common_manifest(m, "xgboost", policy, seed, ref, context)
    require(m["weight_policy"] == policy and m["seed"] == seed, "XGB seed/weight mismatch")
    require(m["protocol_sha256"] == b["protocol_sha256"] == protocol_sha256, "XGB protocol mismatch")
    require(m["input_sha256"] == b["input_sha256"] == ref.contract["input_sha256"], "XGB input mismatch")
    require(b["labels_sha256"] == context["labels_sha256"], "XGB ordered-label mismatch")
    require(m["selection"] == b["selection"] == XGB_SELECTION, "XGB selection mismatch")
    require(b["rounds"] == budget and b["limit"] is None and m["scope"] == "full_cohort",
            "XGB fixed-budget/full-cohort mismatch")
    if expected_params is not None:
        require(b["params"] == {**expected_params, "seed": seed, "disable_default_eval_metric": 1},
                "XGB parameter mismatch")
    weights = class_weights(ref.arrays["y"][context["train"]], len(context["labels"]), policy).astype(np.float32)
    require(np.array_equal(np.asarray(b["class_weights"], dtype=np.float32), weights), "XGB training weight mismatch")
    hashes = verify_hashes(run, m["artifact_files"])
    required = {"validation.npz", "final_validation.npz", "model.ubj", "final_model.ubj",
                "replay.json", "history.json", "cohort.npz", "labels.json", "protocol.json"}
    require(required <= set(hashes), "XGB artifact binding incomplete")
    verify_hashes(run / "source_snapshot", b["source_code"])
    require(sha(run / "protocol.json") == protocol_sha256, "Copied protocol mismatch")
    require(read_json(run / "labels.json") == context["labels"], "Saved ordered-label mismatch")
    with np.load(run / "cohort.npz", allow_pickle=False) as z:
        require(np.array_equal(z["train_ordinals"], context["train"]), "XGB training cohort mismatch")
        require(np.array_equal(z["validation_ordinals"], context["validation"]), "XGB validation cohort mismatch")
    history = read_json(run / "history.json")
    selected = first_maximum(history, budget, "iteration", "validation_macro_f1")
    require(m["selected_iteration"] == selected["iteration"], "XGB first-max selection mismatch")
    proof = read_json(run / "replay.json")
    require(proof["exact_proba"] is True and proof["test_evaluated"] is False, "XGB replay failed")
    require(proof["selected_iteration"] == selected["iteration"] and
            proof["validation_count"] == len(context["y"]) and proof["max_abs_diff"] == 0.,
            "XGB replay selection/cohort mismatch")
    require(proof["checkpoint_sha256"] == sha(run / "model.ubj"), "XGB replay checkpoint mismatch")
    if "replay" in m:
        require(proof == m["replay"], "XGB replay file/manifest mismatch")
    with np.load(run / "validation.npz", allow_pickle=False) as z:
        p = validate_prediction(z["proba"], z["y"], z["ordinals"], context)
    metrics = multiclass_metrics(context["y"], p.argmax(1), p)
    require(metrics == m["metrics"] and metrics["macro_f1"] == selected["validation_macro_f1"],
            "XGB selected metrics mismatch")
    with np.load(run / "final_validation.npz", allow_pickle=False) as z:
        final_p = validate_prediction(z["proba"], z["y"], z["ordinals"], context)
    final_metrics = multiclass_metrics(context["y"], final_p.argmax(1), final_p)
    require(final_metrics == m["final_metrics"] and final_metrics["macro_f1"] == history[-1]["validation_macro_f1"],
            "XGB final-budget metrics mismatch")
    return p, {"run": str(run), "selected_iteration": m["selected_iteration"],
               "selection": m["selection"], "metrics": metrics, "binding": b,
               "artifact_sha256": {**hashes, "run_manifest.json": sha(run / "run_manifest.json")},
               "probability_sha256": array_sha(p), "replay": proof}


def class_statistics(y, pred, labels):
    k = len(labels)
    matrix = np.bincount(k * np.asarray(y) + pred, minlength=k*k).reshape(k, k)
    support, predicted, tp = matrix.sum(1), matrix.sum(0), np.diag(matrix)
    precision = np.divide(tp, predicted, out=np.zeros(k, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(k, dtype=float), where=support > 0)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(k, dtype=float), where=support+predicted > 0)
    return [{"class_index": i, "label": labels[i], "support": int(support[i]),
             "predicted": int(predicted[i]), "TP": int(tp[i]), "FP": int(predicted[i]-tp[i]),
             "FN": int(support[i]-tp[i]), "precision": float(precision[i]),
             "recall": float(recall[i]) if support[i] else None, "f1": float(f1[i])}
            for i in range(k)]


def group_definitions(train_support, y, concepts, labels):
    k = len(labels)
    counts = np.bincount(y, minlength=k)
    rare = np.argsort(train_support, kind="stable")[:k // 4].tolist()
    classes = {
        "train_rare_quartile": rare,
        "train_nonrare": [i for i in range(k) if i not in rare],
        "exploratory_validation_le100": np.flatnonzero(counts <= 100).tolist(),
        "exploratory_validation_gt400": np.flatnonzero(counts > 400).tolist(),
    }
    masks = {"0-1 concepts": concepts <= 1, "2-4 concepts": (concepts >= 2) & (concepts <= 4),
             "5-9 concepts": (concepts >= 5) & (concepts <= 9), "10+ concepts": concepts >= 10}
    definitions = {name: {"class_indices": ids, "labels": [labels[i] for i in ids],
                          "exploratory": name.startswith("exploratory")}
                   for name, ids in classes.items()}
    return classes, masks, definitions


def metrics_for_subset(y, p):
    if not len(y):
        return None
    # Absent true classes in small groups are expected; retain shared sklearn semantics.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return multiclass_metrics(y, p.argmax(1), p)


def score_prediction(y, p, labels, class_groups, concept_masks):
    per_class = class_statistics(y, p.argmax(1), labels)
    groups = {}
    for name, ids in class_groups.items():
        rows = [per_class[i] for i in ids]
        means = {}
        for key in ("precision", "recall", "f1"):
            values = [r[key] for r in rows if r[key] is not None]
            means[f"mean_{key}"] = float(np.mean(values)) if values else None
        groups[name] = {"class_count": len(ids), "support": sum(r["support"] for r in rows),
                        "TP": sum(r["TP"] for r in rows), "FP": sum(r["FP"] for r in rows),
                        "FN": sum(r["FN"] for r in rows), **means}
    return {"metrics": metrics_for_subset(y, p), "per_class": per_class, "class_groups": groups,
            "concept_groups": {name: {"n": int(mask.sum()), "metrics": metrics_for_subset(y[mask], p[mask])}
                               for name, mask in concept_masks.items()}}


def mean_sample_sd(rows):
    """Recursive numeric summary; SD is across seeds, never patient uncertainty."""
    first = rows[0]
    if isinstance(first, dict):
        return {key: mean_sample_sd([row[key] for row in rows]) for key in first}
    if isinstance(first, list):
        return [mean_sample_sd([row[i] for row in rows]) for i in range(len(first))]
    if first is None or isinstance(first, str):
        require(all(row == first for row in rows), "Inconsistent summary metadata")
        return first
    values = np.asarray(rows, dtype=float)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}


def _count_metrics(tp, predicted, support):
    """Unrounded shared metric definitions, vectorized over bootstrap replicates."""
    denom = predicted + support
    f1 = np.divide(2*tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    present = denom > 0  # sklearn macro F1 uses the union of observed and predicted labels.
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)
    metrics = {"macro_f1": f1.sum(-1) / present.sum(-1),
               "balanced_acc": recall.sum(-1) / np.count_nonzero(support),
               "accuracy": tp.sum(-1) / support.sum()}
    return metrics, recall


def paired_stratified_bootstrap(y, pred_a, pred_b, n_classes, *, replicates=2000,
                                seed=20260919, confidence=0.95):
    """A minus B; paired patient bootstrap implemented with sufficient counts."""
    y, a, b = (np.asarray(v, dtype=np.int64) for v in (y, pred_a, pred_b))
    require(y.ndim == 1 and y.shape == a.shape == b.shape and len(y) > 0, "Invalid bootstrap cohort")
    require(all(np.all((v >= 0) & (v < n_classes)) for v in (y, a, b)), "Invalid bootstrap labels")
    require(replicates >= 2 and 0 < confidence < 1, "Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    support = np.bincount(y, minlength=n_classes)
    ta = np.zeros((replicates, n_classes), dtype=np.int64)
    tb = np.zeros_like(ta)
    pa, pb = np.zeros_like(ta), np.zeros_like(ta)
    for c, n in enumerate(support):
        if not n:
            continue
        pairs, counts = np.unique(a[y == c] * n_classes + b[y == c], return_counts=True)
        draws = rng.multinomial(int(n), counts / n, size=replicates)
        aa, bb = pairs // n_classes, pairs % n_classes
        ta[:, c] = draws[:, aa == c].sum(1)
        tb[:, c] = draws[:, bb == c].sum(1)
        for j in np.unique(aa):
            pa[:, j] += draws[:, aa == j].sum(1)
        for j in np.unique(bb):
            pb[:, j] += draws[:, bb == j].sum(1)
    ma, ra = _count_metrics(ta, pa, support)
    mb, rb = _count_metrics(tb, pb, support)
    def observed(pred):
        tp = np.bincount(y[y == pred], minlength=n_classes)
        return _count_metrics(tp, np.bincount(pred, minlength=n_classes), support)
    oa, ora = observed(a)
    ob, orb = observed(b)
    alpha = (1-confidence) / 2
    def interval(values, point):
        lo, hi = np.quantile(values, [alpha, 1-alpha])
        return {"delta": float(point), "ci_low": float(lo), "ci_high": float(hi)}
    return {"replicates": replicates, "seed": seed, "confidence": confidence,
            "direction": "A minus B", "conditional": True, "multiplicity_corrected": False,
            "metrics": {key: interval(ma[key]-mb[key], oa[key]-ob[key]) for key in BOOT_METRICS},
            "per_class_recall": [{"class_index": c, "support": int(n),
                                  **(interval(ra[:, c]-rb[:, c], ora[c]-orb[c]) if n else
                                     {"delta": None, "ci_low": None, "ci_high": None})}
                                 for c, n in enumerate(support)]}


def comparison_result(a_id, b_id, rows, probabilities, y, settings):
    a, b = rows[a_id], rows[b_id]
    class_delta = []
    for ca, cb in zip(a["per_class"], b["per_class"]):
        class_delta.append({"class_index": ca["class_index"], "label": ca["label"], "support": ca["support"],
                            **{key: ca[key]-cb[key] if ca[key] is not None and cb[key] is not None else None
                               for key in ("precision", "recall", "f1", "TP", "FP", "FN")}})
    group_delta = {}
    for name in a["class_groups"]:
        ga, gb = a["class_groups"][name], b["class_groups"][name]
        group_delta[name] = {key: ga[key]-gb[key] if ga[key] is not None else None
                             for key in ("mean_precision", "mean_recall", "mean_f1", "TP", "FP", "FN")}
    return {"A": a_id, "B": b_id, "direction": "A minus B",
            "metric_delta": {key: a["metrics"][key]-b["metrics"][key] for key in METRICS},
            "per_class_delta": class_delta, "classes_recall_won": sum(r["recall"] is not None and r["recall"] > 0 for r in class_delta),
            "classes_recall_lost": sum(r["recall"] is not None and r["recall"] < 0 for r in class_delta),
            "class_group_delta": group_delta,
            "concept_group_delta": {name: {"n": ga["n"], "metrics": {
                key: ga["metrics"][key]-b["concept_groups"][name]["metrics"][key] for key in METRICS
            } if ga["metrics"] is not None else None} for name, ga in a["concept_groups"].items()},
            "bootstrap": paired_stratified_bootstrap(y, probabilities[a_id].argmax(1), probabilities[b_id].argmax(1),
                                                     probabilities[a_id].shape[1], **settings)}


def analyze_predictions(probabilities, context, protocol, provenance=None):
    """Pure analysis API; probabilities keyed ``model/policy/seed`` (seed as text).

    Production callers must use build_report(), which gates all source artifacts.
    This lower-level function is deliberately usable for small synthetic unit tests.
    """
    seeds, policies = protocol["seeds"], protocol["weight_policies"]
    require(len(seeds) == 3 and len(set(seeds)) == 3, "Exactly three independent seeds required")
    expected = {f"{m}/{p}/{s}" for m in ("gchm", "xgboost") for p in policies for s in seeds}
    require(set(probabilities) == expected, "Missing/extra model-policy-seed cells")
    probs = dict(probabilities)
    labels, y = context["labels"], context["y"]
    classes, masks, definitions = group_definitions(context["train_support"], y, context["concepts"], labels)
    rows, summaries = {}, {}
    for model in ("gchm", "xgboost"):
        for policy in policies:
            prefix = f"{model}/{policy}"
            ids = [f"{prefix}/{s}" for s in seeds]
            for key in ids:
                validate_prediction(probs[key], y, context["validation"], context)
                rows[key] = score_prediction(y, probs[key], labels, classes, masks)
            summaries[prefix] = {"n_seeds": 3, "sd_ddof": 1, "kind": "single_model_seed_mean_sample_sd",
                                 "statistics": mean_sample_sd([rows[key] for key in ids])}
            # Average original probabilities, not logits, votes or per-seed scores.
            ensemble_id = f"{prefix}/ensemble"
            probs[ensemble_id] = np.mean(np.stack([probs[key] for key in ids]), axis=0)
            rows[ensemble_id] = score_prediction(y, probs[ensemble_id], labels, classes, masks)
            rows[ensemble_id]["ensemble_members"] = ids
            rows[ensemble_id]["probability_sha256"] = array_sha(probs[ensemble_id])
    u = protocol["uncertainty"]
    settings = {key: u[key] for key in ("replicates", "seed", "confidence")}
    comparisons = {}
    for policy in policies:
        for unit in [*map(str, seeds), "ensemble"]:
            key = f"gchm_minus_xgboost/{policy}/{unit}"
            comparisons[key] = comparison_result(f"gchm/{policy}/{unit}", f"xgboost/{policy}/{unit}", rows, probs, y, settings)
    for model in ("gchm", "xgboost"):
        for unit in [*map(str, seeds), "ensemble"]:
            key = f"sqrt_minus_none/{model}/{unit}"
            comparisons[key] = comparison_result(f"{model}/sqrt_inverse/{unit}", f"{model}/none/{unit}", rows, probs, y, settings)
    delta_summaries = {}
    for prefix in [*(f"gchm_minus_xgboost/{p}" for p in policies),
                   *(f"sqrt_minus_none/{m}" for m in ("gchm", "xgboost"))]:
        delta_summaries[prefix] = mean_sample_sd([comparisons[f"{prefix}/{s}"]["metric_delta"] for s in seeds])
    return {"schema_version": "matched-gchm-xgb-report-v1", "scope": "exploratory_validation_only_artifact_analysis",
            "test_evaluated": False, "temporal_clean": False, "protocol": protocol,
            "cohort": {"train_count": len(context["train"]), "validation_count": len(y), "labels": labels,
                       "train_support": context["train_support"].tolist(), "validation_support": np.bincount(y, minlength=len(labels)).tolist(),
                       **{k: context[k] for k in ("train_ordinals_sha256", "validation_ordinals_sha256", "labels_sha256")}},
            "group_definitions": definitions,
            "metric_semantics": "Overall/subgroup metrics use shared/lib/metrics.py (6 decimals). Per-class zero precision/F1 denominators map to 0; absent-class recall is null. Class-group macro summaries use full-cohort one-vs-rest counts, so FP outside a rare group remain counted. Subgroup balanced_acc averages observed true classes; macro_f1 uses observed/predicted union. Empty subgroups are retained as null.",
            "bootstrap_method": BOOT_EXPLANATION, "bootstrap_scope": u.get("scope"),
            "bootstrap_rng_policy": "Reset the same protocol seed for each comparison, deterministic canonical order; comparisons are not independent.",
            "rows": rows, "seed_summaries": summaries, "comparisons": comparisons,
            "matched_seed_delta_summaries": delta_summaries, "provenance": provenance or {},
            "limitations_tr": LIMITATIONS_TR}


def build_report(protocol_path=PROTOCOL, *, repo=REPO, artifact=DEFAULT, runs_root=None):
    """Read and validate artifacts, then compute report; no writes or inference."""
    protocol_path, repo = Path(protocol_path), Path(repo)
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    protocol_hash = sha(protocol_path)
    ref = Reference(artifact, expected=protocol["contract_sha256"])
    require(ref.contract["temporal_clean"] is False, "Unexpected native temporal status")
    context = cohort_context(ref)
    require(len(context["labels"]) == 30, "Expected 30 native classes")
    runs_root = Path(runs_root) if runs_root is not None else protocol_path.parent / "runs"
    probabilities, provenance = {}, {}
    for policy in protocol["weight_policies"]:
        for seed in protocol["seeds"]:
            key = f"gchm/{policy}/{seed}"
            path = contained(repo, protocol["gchm_reuse"][policy][str(seed)])
            probabilities[key], provenance[key] = load_gchm(path, policy, seed, ref, context)
            key = f"xgboost/{policy}/{seed}"
            probabilities[key], provenance[key] = load_xgboost(runs_root / policy / f"seed{seed}", policy, seed,
                                                              ref, context, protocol_hash,
                                                              expected_params={k: v for k, v in protocol["xgboost"].items()
                                                                               if k != "num_boost_round"})
    report = analyze_predictions(probabilities, context, protocol, provenance)
    report["input_provenance"] = {"contract_sha256": ref.fingerprint, "input_sha256": ref.contract["input_sha256"],
                                  "source_bindings": ref.contract["source_bindings"], "temporal_clean": False,
                                  "limitations": ref.contract["limitations"], "artifact": str(artifact)}
    report["analysis_provenance"] = {"protocol_sha256": protocol_hash, "report_source_sha256": sha(__file__),
                                     "metrics_source_sha256": sha(REPO / "shared/lib/metrics.py")}
    return report


def render_markdown(report):
    lines = ["# Eşleştirilmiş GCHM–XGBoost: doğrulama raporu", "",
             f"**Yalnız doğrulama: n={report['cohort']['validation_count']}; test_evaluated=false; temporal_clean=false.**",
             "GCHM, GNN ailesidir; XGBoost harici tabular referanstır. Aşağıdaki eşleşmeler aile içi sıralama değildir.",
             "Her checkpoint ilk maksimum (6 ondalık) macro-F1 ile seçildi; ensemble üç seed'in eşit olasılık ortalamasıdır.", "",
             "## Metrikler", "",
             "| Model / ağırlık / seed | Macro F1 | Dengeli doğruluk | Doğruluk | Micro F1 | Top-3 | Top-5 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, row in report["rows"].items():
        lines.append("| " + key + " | " + " | ".join(f"{row['metrics'][m]:.4f}" for m in METRICS) + " |")
    for key, row in report["seed_summaries"].items():
        stats = row["statistics"]["metrics"]
        lines.append("| " + key + " seed ort. ± örnek SS | " + " | ".join(
            f"{stats[m]['mean']:.4f} ± {stats[m]['sample_sd']:.4f}" for m in METRICS) + " |")
    lines += ["", "## Ensemble farkları ve koşullu %95 güven aralıkları", "",
              "A−B yönü satır adındadır. Tek-seed eşleşmeleri, sqrt−none farkları, sınıf recall aralıkları ve tüm ayrıntılar JSON'dadır.",
              "| Karşılaştırma | Δ Macro F1 [GA] | Δ Dengeli doğruluk [GA] | Δ Doğruluk [GA] | Recall kazanılan sınıf |",
              "|---|---:|---:|---:|---:|"]
    for key, row in report["comparisons"].items():
        if key.endswith("/ensemble"):
            vals = [row["bootstrap"]["metrics"][m] for m in BOOT_METRICS]
            lines.append("| " + key + " | " + " | ".join(
                f"{v['delta']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]" for v in vals) +
                f" | {row['classes_recall_won']}/{len(report['cohort']['labels'])} |")
    lines += ["", "## Nadir sınıflar ve precision–recall değiş tokuşu", "",
              "Nadir sınıflar yalnız eğitim desteğinin alt çeyreğiyle (sabit sınıf sıralı bağ çözümü) tanımlandı. "
              "Eski doğrulama ≤100 / >400 grupları yalnız keşifsel karşılaştırmadır. FP tüm doğrulama kümesinden sayılır.",
              "| Ensemble | Eğitim-nadir precision | Eğitim-nadir recall | Eğitim-nadir F1 | Eğitim-nadir FP | Nadir-dışı recall | Eski ≤100 recall | Eski >400 recall |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return "—" if v is None else f"{v:.4f}"
    for key, row in report["rows"].items():
        if key.endswith("/ensemble"):
            groups = row["class_groups"]
            rare = groups["train_rare_quartile"]
            lines.append(f"| {key} | {fmt(rare['mean_precision'])} | {fmt(rare['mean_recall'])} | {fmt(rare['mean_f1'])} | {rare['FP']} | " +
                         " | ".join(fmt(groups[g]["mean_recall"]) for g in ("train_nonrare", "exploratory_validation_le100", "exploratory_validation_gt400")) + " |")
    lines += ["", "## Kavram sayısına göre ensemble", "",
              "| Ensemble | Grup | n | Macro F1 | Dengeli doğruluk | Doğruluk |",
              "|---|---|---:|---:|---:|---:|"]
    for key, row in report["rows"].items():
        if key.endswith("/ensemble"):
            for name, group in row["concept_groups"].items():
                metrics = group["metrics"] or {}
                lines.append(f"| {key} | {name} | {group['n']} | " + " | ".join(fmt(metrics.get(m)) for m in BOOT_METRICS) + " |")
    lines += ["", "## Yöntem ve sınırlar", "",
              "- 2000 tekrar; seed=20260919. Gerçek sınıfa göre tabakalı eşleştirilmiş hasta bootstrap'ı: "
              "her sınıfta ortak (A tahmini, B tahmini) hücreleri multinomial örneklenir. Aynı hastayı iki model için birlikte "
              "yeniden örneklemeyle dağılımsal olarak eşdeğerdir; sınıf destekleri sabittir. Aralıklar yuvarlanmamış metriklerden hesaplanır.",
              "- Seed örnek SS (ddof=1) eğitim rastgeleliğinin betimlemesidir; ensemble veya bootstrap aralığı değildir."]
    lines.extend("- " + text for text in report["limitations_tr"])
    lines += ["", "JSON: tüm tek-seed/ensemble metrikleri; her sınıf precision, recall, F1, TP, FP, FN ve destek; "
              "gruplar; eşleşmiş farklar ve koşullu aralıklar; girdi/kaynak/checkpoint/tahmin SHA-256 kayıtları.", ""]
    return "\n".join(lines)


def write_report(report, output_dir, *, overwrite=False):
    """Only report.json/report.md are written; existing targets require opt-in."""
    output_dir = Path(output_dir)
    targets = [output_dir / "report.json", output_dir / "report.md"]
    for path in targets:
        if path.is_symlink():
            raise ValueError(f"Refusing symlink output: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Report exists; explicit --overwrite required: {path}")
    payloads = [json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", render_markdown(report)]
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an unnoticed concurrent overwrite in the default mode.
    for path, payload in zip(targets, payloads):
        with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
            stream.write(payload)
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--artifact", type=Path, default=DEFAULT)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=HERE / "reports")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace only the new report.json/report.md")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    require(output.is_relative_to(HERE) and not output.is_relative_to(HERE / "runs"),
            "Report output must stay in this study, outside runs; historical artifacts are read-only")
    for name in ("report.json", "report.md"):
        if (output / name).exists() and not args.overwrite:
            raise FileExistsError(output / name)
    report = build_report(args.protocol, artifact=args.artifact, runs_root=args.runs_root)
    paths = write_report(report, output, overwrite=args.overwrite)
    print(json.dumps({"reports": list(map(str, paths)), "test_evaluated": False, "temporal_clean": False}))
    return report


if __name__ == "__main__":
    main()
