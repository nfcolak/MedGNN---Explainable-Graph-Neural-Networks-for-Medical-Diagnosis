# XGBoost control

Tabular control: XGBoost over the token/value summary of the same clinical graph
artifact, under the shared comparison contract (same targets, folds, class set). It is
a control, not a graph method; the native 30-class baseline is retired.

## Code (in this folder)

| File | Role |
|---|---|
| `tabular_control.py` | XGBoost control runner |

Old path `clinical_graph_v2/controls/tabular_control.py` is a compatibility shim.

## Run

```bash
python3 -m comparison.standardized.clinical_graph_v2.methods.xgboost.tabular_control \
  --artifact <artifact> --targets <targets.csv> --out <new dir>
```

Needs approval (it trains) and a new, empty output directory. Example arguments:
[`../../README.md`](../../README.md), "tablo kontrolü".

## Reports (`docs/`)

- `docs/gchm-xgboost-matched-results.md` (matched GCHM vs XGBoost, 6 + 6 runs)

## Results

Result directories are not moved. Names only:

- `comparison/standardized/clinical_runs_v3_adapters_sample10k_max6_top10_20260924/xgboost_seed1234` (shared run)
- `comparison/standardized/clinical_runs_v3_full_top10_20260924/xgb_control_seed1234` (shared run)
- `comparison/standardized/matched_gchm_xgb_v1/` (shared run with GCHM)
