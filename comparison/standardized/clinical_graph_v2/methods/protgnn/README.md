# ProtGNN

Prototype-based self-explainable GNN (class prototypes, cluster/separation losses,
warm-up, train-loader projection), adapted to the multi-visit clinical graph.
Upstream checkpoints are not compatible. Native budget: hidden 128, 3 layers,
dropout 0.51, Adam, 300 epochs.

## Code (in this folder)

| File | Role |
|---|---|
| `adapter.py` | ProtGNN network, auxiliary objective, optimizer groups, `--protgnn-*` options |

Old path `clinical_graph_v2/methods/protgnn.py` is a compatibility shim.

## Run

Shared runner: `core/train.py` with `--method protgnn`. Read-only plan first; real
training needs explicit approval and a new, empty output directory.

```bash
python3 -m comparison.standardized.clinical_graph_v2.core.train --method protgnn \
  --artifact <artifact> --targets <targets.csv> --output <new dir> --seed 1234
```

Full command lines: [`../../README.md`](../../README.md), "Adım 4" and adapter contract.

## Reports

`docs/` in this folder holds ProtGNN-only reports (currently none). Cross-method
reports stay in the repo `docs/` (e.g. `docs/graphxai-500-results.md`, legacy 30-class).

## Results

Result directories are not moved. Names only:

- `comparison/standardized/clinical_runs_v3_adapters_sample10k_max6_top10_20260924/protgnn_seed1234` (shared run: all adapters + XGBoost + GCHM-PNA)
- `comparison/standardized/clinical_runs_v3_full_top10_20260924/protgnn_seed1234` (shared run)
- `comparison/standardized/clinical_runs_v3_challenger_sample10k_max6_top10_20260927/protgnn_seed{1234,2025,7}` (shared run with the deleted ProtoNode challenger; ProtGNN is the reference)
- `comparison/standardized/clinical_runs_cei_pilot_20260928/protgnn_control` (CEI pilot control)
- `/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/cei-v3-20261001/protgnn` (CEI v3 comparison)
