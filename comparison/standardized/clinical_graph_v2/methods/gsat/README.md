# GSAT

Graph Stochastic Attention: shared GIN predictor, node-level stochastic attention,
one edge-mask intervention and a Bernoulli KL information bottleneck, adapted to the
multi-visit clinical graph. Native budget: hidden 128, 3 layers, dropout 0.3, Adam,
100 epochs.

## Code (in this folder)

| File | Role |
|---|---|
| `adapter.py` | GSAT network, information-bottleneck curriculum, `--gsat-*` options |

Old path `clinical_graph_v2/methods/gsat.py` is a compatibility shim.

## Run

Shared runner: `core/train.py` with `--method gsat`. Training needs explicit approval
and a new, empty output directory.

```bash
python3 -m comparison.standardized.clinical_graph_v2.core.train --method gsat \
  --artifact <artifact> --targets <targets.csv> --output <new dir> --seed 1234
```

Full command lines: [`../../README.md`](../../README.md).

## Reports

`docs/` in this folder holds GSAT-only reports (currently none). Cross-method
reports stay in the repo `docs/`.

## Results

Result directories are not moved. Names only:

- `comparison/standardized/clinical_runs_v3_adapters_sample10k_max6_top10_20260924/gsat_seed1234` (shared run)
- `comparison/standardized/clinical_runs_v3_full_top10_20260924/gsat_seed1234` (shared run)
