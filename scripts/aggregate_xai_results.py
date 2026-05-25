"""
aggregate_xai_results.py
Reads per-graph JSON files from outputs/results/explanations/ and computes
mean XAI metrics per explainer. Run from project root after each experiment.

Usage:
    python scripts/aggregate_xai_results.py --run_dir outputs/results
    python scripts/aggregate_xai_results.py --run_dir outputs/runs/shapeggen_house_ba_...
"""
import os, sys, json, argparse, csv
from collections import defaultdict

EXPLAINERS = ["GradExplainer", "IntegratedGradExplainer", "GNNExplainer"]

def aggregate_run(run_dir: str) -> dict | None:
    exp_dir = os.path.join(run_dir, "explanations")
    if not os.path.isdir(exp_dir):
        return None

    # Read dataset name from model_config if available
    cfg_path = os.path.join(run_dir, "model_config.json")
    dataset = "unknown"
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        dataset = cfg.get("dataset", "unknown")

    per_explainer = defaultdict(lambda: {
        "f1": [], "fid_plus": [], "fid_minus": [],
        "sparsity": [], "auroc": [],
        "f1_c1": [], "fid_minus_c1": [],
        "n_total": 0, "n_class1": 0, "n_errors": 0,
    })

    graph_files = sorted(
        f for f in os.listdir(exp_dir)
        if f.startswith("graph_") and f.endswith(".json")
    )
    if not graph_files:
        return None

    for fname in graph_files:
        with open(os.path.join(exp_dir, fname)) as f:
            rec = json.load(f)

        true_label = rec.get("true_label", 0)
        xai = rec.get("xai_metrics", {})
        if not xai:
            continue

        for exp_name in EXPLAINERS:
            m = xai.get(exp_name)
            if not m:
                per_explainer[exp_name]["n_errors"] += 1
                continue

            acc = m.get("accuracy", {})
            f1  = acc.get("f1", 0.0)
            fid_plus  = m.get("fidelity_plus", 0.0)
            fid_minus = m.get("fidelity_minus", 0.0)
            spar      = m.get("sparsity", 0.0)
            auroc     = acc.get("auroc", float("nan"))

            d = per_explainer[exp_name]
            d["f1"].append(f1)
            d["fid_plus"].append(fid_plus)
            d["fid_minus"].append(fid_minus)
            d["sparsity"].append(spar)
            d["auroc"].append(auroc)
            d["n_total"] += 1

            if true_label == 1:
                d["f1_c1"].append(f1)
                d["fid_minus_c1"].append(fid_minus)
                d["n_class1"] += 1

    if not any(per_explainer[e]["n_total"] > 0 for e in EXPLAINERS):
        return None

    import statistics

    def mean(lst):
        valid = [x for x in lst if x == x]  # exclude nan
        return round(statistics.mean(valid), 4) if valid else float("nan")

    results = {"dataset": dataset, "run_dir": run_dir, "explainers": {}}
    for exp_name in EXPLAINERS:
        d = per_explainer[exp_name]
        results["explainers"][exp_name] = {
            "n_total":      d["n_total"],
            "n_class1":     d["n_class1"],
            "n_errors":     d["n_errors"],
            "F1_all":       mean(d["f1"]),
            "F1_class1":    mean(d["f1_c1"]),
            "Fid+":         mean(d["fid_plus"]),
            "Fid-_all":     mean(d["fid_minus"]),
            "Fid-_class1":  mean(d["fid_minus_c1"]),
            "Sparsity":     mean(d["sparsity"]),
            "AUROC":        mean(d["auroc"]),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="outputs/results",
                        help="Path to a single run dir, or outputs/runs to scan all")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: <run_dir>/xai_summary.csv)")
    args = parser.parse_args()

    run_dirs = []
    runs_root = args.run_dir

    # If pointed at outputs/runs, scan subdirs
    if os.path.basename(runs_root) == "runs" and os.path.isdir(runs_root):
        for name in sorted(os.listdir(runs_root)):
            d = os.path.join(runs_root, name)
            if os.path.isdir(d):
                run_dirs.append(d)
    else:
        run_dirs.append(runs_root)

    all_rows = []
    for rd in run_dirs:
        result = aggregate_run(rd)
        if result is None:
            print(f"  [SKIP] {rd} — no xai_metrics found in explanation JSONs")
            continue
        for exp_name, metrics in result["explainers"].items():
            row = {"dataset": result["dataset"], "explainer": exp_name, **metrics}
            all_rows.append(row)
            print(
                f"  {result['dataset']:<35} {exp_name:<28} "
                f"F1={metrics['F1_class1']:.3f}  "
                f"Fid+={metrics['Fid+']:.3f}  "
                f"Fid-={metrics['Fid-_class1']:.3f}  "
                f"Spar={metrics['Sparsity']:.3f}"
            )

    if not all_rows:
        print("No results found.")
        return

    out_path = args.out or os.path.join(runs_root, "xai_summary.csv")
    fieldnames = ["dataset","explainer","n_total","n_class1","n_errors",
                  "F1_all","F1_class1","Fid+","Fid-_all","Fid-_class1",
                  "Sparsity","AUROC"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Saved → {out_path}")

if __name__ == "__main__":
    main()