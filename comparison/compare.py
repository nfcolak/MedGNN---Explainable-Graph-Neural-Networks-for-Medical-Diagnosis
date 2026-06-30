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
    # copied canonical report first, else the latest ProtGNN run
    p = HERE / "protgnn" / "test_metrics.json"
    if p.exists():
        m = json.load(open(p))
    else:
        latest = (REPO / "protgnn_analysis" / "outputs" / "results" / "latest_run.txt")
        d = Path(latest.read_text().strip())
        m = json.load(open(d / "test_metrics.json"))
    return m.get("test", m)   # tolerate {"test": {...}} or flat


def main():
    g = _load_graphcare()
    p = _load_protgnn()

    rows = []
    for k in METRICS:
        pv, gv = _get(p, k), _get(g, k)
        if pv is None and gv is None:
            continue
        win = ""
        if pv is not None and gv is not None:
            win = "ProtGNN" if pv > gv else ("GraphCare" if gv > pv else "tie")
        rows.append((k, pv, gv, win))

    def fmt(v):
        return f"{v:.4f}" if v is not None else "-"

    md = ["# Aligned comparison: ProtGNN vs GraphCare", "",
          "Both trained + evaluated on the **identical** canonical split",
          "(`comparison/canonical_split.json`): same patients, same test set, same 30",
          "classes, same metrics. Differences come from the **method**, not the data.", "",
          "| Metric | ProtGNN (A) | GraphCare (B) | Winner |",
          "|---|---|---|---|"]
    for k, pv, gv, win in rows:
        md.append(f"| {k} | {fmt(pv)} | {fmt(gv)} | {win} |")
    md += ["",
           "- **macro_f1 / balanced_acc**: per-class balance (rare-class sensitivity).",
           "- **accuracy / micro_f1 / top-k**: overall correctness + ranking.",
           "",
           "Both use the same ontology+PMI KG signal, so this isolates the modelling",
           "approach (prototype case-based reasoning vs KG bi-attention GNN)."]
    (HERE / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("\n  wrote", HERE / "comparison.md")


if __name__ == "__main__":
    main()
