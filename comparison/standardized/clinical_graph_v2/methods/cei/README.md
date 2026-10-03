# CEI-GNN (clinical evidence-interaction GNN)

Independent evidence-interaction method written for this benchmark, GraphXAI-compatible.
v1 (`cei_gnn`), v2 (`cei_gnn_v2`, within-visit pairs) and v3 (`cei_gnn_v3`, value encoding,
absence/PLE/preservation extensions). Study results are aggregate only, with provenance.

## Code (in this folder)

| File | Role |
|---|---|
| `cei_gnn.py`, `cei_gnn_v2.py`, `cei_gnn_v3.py` | networks |
| `plugin_cei_gnn.py`, `plugin_cei_gnn_v2.py`, `plugin_cei_gnn_v3.py` | `REGISTER` plugins (auto-discovered) |
| `studies/cei_pilot.py`, `cei_graphxai.py`, `cei_v2_study.py` | v1/v2 pilot, GraphXAI, v2 study |
| `studies/cei_v3_{study,screen,validation,graphxai,run,paths,ple,preserve,absence,vs_protgnn,vs_xgboost}.py` | v3 studies and comparisons |
| `studies/cei_v3_ext/` | extensions: arm_guards, comorbid_block, io, launch, offsets, run, scoring, study, validation_scoring |

Old paths (`methods/cei_gnn*.py`, `methods/plugin_cei_gnn*.py`, `studies/cei/`, root `cei_*.py`,
`cei_v3_ext/`) keep compatibility shims.

## Run

Studies are modules under `studies/`; plan/dry-run modes first, `--execute` only after approval.

```bash
# module paths: ...clinical_graph_v2.methods.cei.studies.<module> (e.g. cei_v3_study); read its docstring, do not call --help
python3 -m comparison.standardized.clinical_graph_v2.core.train --method cei_gnn_v3 ...
```

Exact command lines and approvals: `docs/cei-v3-delivery-2026-10-01.md`.

## Reports (`docs/`)

- `docs/cei-v3-delivery-2026-10-01.md` (v3 delivery), `docs/cei-v3-evidence-2026-10-01.json`, `docs/cei-v3-smoke-2026-10-01.json`
- `docs/cei-gnn-v2-pair-study-result-2026-09-29.md`
- `docs/superpowers/specs/` and `docs/superpowers/plans/` (evidence-interaction, v2 pairs, v3 design/extensions/value encoding)
- `docs/presentations/cei-overview/` (3-slide deck and build script)

## Results

Result directories are not moved. Names only:

- `comparison/standardized/clinical_runs_cei_pilot_20260928/` (CEI pilot: candidate, product-off, smoke, dev GraphXAI, ProtGNN control; shared run)
- `/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/cei-v3-20261001/` (core, graphxai, validation, protgnn; shared run with ProtGNN)
- `/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/cei-v3-smoke-20261001/`
- `/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/cei-speed-20261002/`
- `/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/cei-v3-20261001-receipt.json`, `delivery-evidence-20261001/`
