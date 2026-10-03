# GCHM-PNA

Hub-gated principal-neighbourhood-aggregation GNN (`--conv gchm_v2`, `gchm_v3`):
hub-state and relation-type multiplicative gate, compact PNA, hub readout; v3 adds a
wide token path, label-wise readout, jumping knowledge and edge dropout. Experimental;
ADR-007/008 are proposed, not accepted.

## Code (in this folder)

| File | Role |
|---|---|
| `gchm_v2.py` | GCHM-PNA v2 layers and encoder |
| `gchm_v3.py` | v3 readout, imports v2 |
| `protocol/protocol.py` | equal-budget protocol (dry-run by default; `--execute` writes) |
| `protocol/identity_snapshot.py` | protocol identity snapshot helper |

The tracked JSON `comparison/standardized/gchm_v2_protocol/state/incumbent_identity_before.json` stays at its old location (data, not code).

Old paths `core/gchm_v2.py`, `core/gchm_v3.py` and `comparison/standardized/gchm_v2_protocol/`
keep compatibility shims.

## Run

```bash
# plan + ETA only (read-only)
python3 -m comparison.standardized.clinical_graph_v2.methods.gchm_pna.protocol.protocol
python3 -m comparison.standardized.clinical_graph_v2.methods.gchm_pna.protocol.protocol --status
# training-free mechanism checks
python3 -m comparison.standardized.clinical_graph_v2.core.mechanism_check
# training (needs approval): shared runner, --conv gchm_v2 | gchm_v3
python3 -m comparison.standardized.clinical_graph_v2.core.train --conv gchm_v2 --edge-direction bidirectional ...
```

`--stage pilot --execute` and `--execute` of the protocol write results; run only after approval.
Details: [`../../README.md`](../../README.md) ("GCHM-PNA v2 ve eşit bütçeli protokol", "v3").

## Reports (`docs/`)

- `docs/superpowers/specs/2026-09-24-gchm-pna-v2-design.md` (v2 design)
- `docs/gchm-concept-dropout.md`, `docs/gchm-concept-dropout-pilot-results.md` (rejected challenger)
- `docs/pna-interaction.md`, `docs/pna-performance-results.md`, `docs/pna-quality-and-feature-improvement-analysis.md`, `docs/pna-quality-evidence/`

Matched GCHM/XGBoost result: `../xgboost/docs/gchm-xgboost-matched-results.md`.

## Results

Result directories are not moved. Names only:

- `comparison/standardized/clinical_runs_v3_adapters_sample10k_max6_top10_20260924/gchm_pna_seed1234` (shared run)
- `comparison/standardized/clinical_runs_v3_full_top10_20260924/gchm_pna_seed1234` (shared run)
- `comparison/standardized/matched_gchm_xgb_v1/` (shared run with XGBoost; reports tracked)
- Historical 30-class PNA runs (`comparison/standardized/pna_experiments/`, `native_runs/`) are not comparable with the active task.
