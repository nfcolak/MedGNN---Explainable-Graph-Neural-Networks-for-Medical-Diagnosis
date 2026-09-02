"""Assemble the aligned ProtGNN vs GraphCare comparison into comparison/comparison.md.

Both methods were trained + evaluated on the IDENTICAL canonical split
(comparison/canonical_split.json): same patients, same test set, same 30 classes,
same metrics (shared/lib/metrics.py). So differences come from the METHOD.

Run:  PYTHONPATH=. python3 comparison/compare.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "comparison"
METRICS = ["accuracy", "balanced_acc", "macro_f1", "micro_f1", "top3_acc", "top5_acc"]
HIGHER_BETTER = True


def _get(m, k):
    if k in m and m[k] is not None:
        return float(m[k])
    if k == "accuracy" and "micro_f1" in m:   # micro-F1 == accuracy for single-label
        return float(m["micro_f1"])
    return None


def _load_graphcare():
    return json.load(open(HERE / "graphcare" / "metrics.json"))["test"]


def _load_protgnn():
    # canonical rerun first (it also stores the checkpoint + predictions),
    # then the copied canonical report, else the latest ProtGNN run
    canon = HERE / "protgnn_canonical" / "metrics.json"
    if canon.exists():
        return json.load(open(canon))["test"]
    p = HERE / "protgnn" / "test_metrics.json"
    if p.exists():
        m = json.load(open(p))
    else:
        latest = (REPO / "protgnn_analysis" / "outputs" / "results" / "latest_run.txt")
        d = Path(latest.read_text().strip())
        m = json.load(open(d / "test_metrics.json"))
    return m.get("test", m)   # tolerate {"test": {...}} or flat


def _load_optional(path, key=None):
    """Return a metrics dict, or None when that run has not been produced yet."""
    p = HERE / path
    if not p.exists():
        return None
    m = json.load(open(p))
    return m.get(key, m) if key else m


def main():
    g = _load_graphcare()
    p = _load_protgnn()
    tab = _load_optional("tabular/metrics.json")
    gcn = _load_optional("plain_gcn/metrics.json", "test")
    maj = tab.get("majority") if tab else None
    xgb = tab.get("xgboost") if tab else None

    # column label -> metrics dict (None columns are dropped)
    cols = [("Majority", maj), ("XGBoost", xgb), ("Plain-GCN", gcn),
            ("GraphCare", g), ("ProtGNN", p)]
    cols = [(name, m) for name, m in cols if m is not None]

    def fmt(v):
        return f"{v:.4f}" if v is not None else "-"

    rows = []
    for k in METRICS:
        vals = [_get(m, k) for _, m in cols]
        if all(v is None for v in vals):
            continue
        best = max((v for v in vals if v is not None), default=None)
        win = ", ".join(name for (name, _), v in zip(cols, vals) if v == best)
        rows.append((k, vals, win))

    header = "| Metric | " + " | ".join(name for name, _ in cols) + " | Best |"
    md = ["# Aligned comparison on the canonical split", "",
          "Every method is trained + evaluated on the **identical** canonical split",
          "(`comparison/canonical_split.json`): same patients, same test set, same 30",
          "classes, same metrics. Differences come from the **method**, not the data.", "",
          header, "|" + "---|" * (len(cols) + 2)]
    for k, vals, win in rows:
        md.append(f"| {k} | " + " | ".join(fmt(v) for v in vals) + f" | {win} |")
    md += ["",
           "- **macro_f1 / balanced_acc**: per-class balance (rare-class sensitivity).",
           "- **accuracy / micro_f1 / top-k**: overall correctness + ranking.",
           "",
           "Majority + XGBoost come from `comparison/tabular_baseline_canonical.py`,",
           "the prototype-free ablation from `comparison/plain_gcn_canonical.py`.",
           "ProtGNN and GraphCare use the same ontology+PMI KG signal, so that pair",
           "isolates the modelling approach (prototype case-based reasoning vs KG",
           "bi-attention GNN)."]
    (HERE / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("\n  wrote", HERE / "comparison.md")


if __name__ == "__main__":
    main()
