# Repository navigation

Start with the [clinical v2/v3 runbook](comparison/standardized/clinical_graph_v2/README.md)
for the active max6 / train-derived Top-10 task. The
[cleanup record](docs/max6-top10-cleanup.md) explains what was retired and what
is held as evidence. The old 30-class reproduction is a different scientific
task; its inputs and scores are not interchangeable with the active one.

## Active runtime

| Location | Role / maintenance boundary |
|---|---|
| `comparison/standardized/clinical_graph_v2/` | Multi-visit build, all method adapters and plugins (ProtGNN, GSAT, GraphCare, GCHM-PNA, CEI-GNN v1-v3, GMT, GPS, label-attention, token-fusion, virtual-node), training, audits, tabular control. |
| `comparison/canonical_split.json` | Fixed class ordering and subject folds. Never regenerate. Top-10 selection depends on this original class order. |
| `comparison/standardized/icd_mapping.py` | ICD mapping used by the label chain. |
| `comparison/standardized/event_graph_v1/` | Kept files only: `__init__`, `schema`, `first_lab`, `ingest`, `graph`, `knowledge_seed.csv`. |
| `comparison/standardized/enriched_input_v1/spec.py` | Kept input spec (plus the package marker it needs). |
| `comparison/standardized/event_graph_gchm_xgb_v1/` | Kept label contract: `__init__`, `labels`, `local_labels_v2`. |
| `comparison/standardized/gchm_v2_protocol/` | Optional, frozen protocol. Does not enable ADR-008. |
| `comparison/standardized/xgboost_native_baseline.py` | XGBoost control entry referenced by the runbook. |
| `shared/lib/`, `shared/data_prep/` | Shared contracts, split, metrics, data prep. `merge_ed.py` is hash-pinned: keep byte-identical, do not run or fix it. |
| `visualizer/` | Optional React/TypeScript/Vite viewer for already-exported graph JSON. No supported export command is current. |
| `external/` | Third-party code. GraphXAI is used by the CEI explanation path; GraphCare is retained legacy. Keep upstream layout. |
| `docs/` | Runbooks, evidence and reports. Many describe retired or historical tooling. |
| `docs-vault/` | Retained local navigation; project decisions live in ProjectOS. Do not relocate. |
| `requirements.txt`, `requirements-lock.txt`, `environment.yml` | Main-Python dependencies. |
| `.venv-graphcare/` | Existing isolated legacy GraphCare runtime. Not used by the clinical adapter; left in place. |
| `.claude/`, `.git/` | Local agent state and version control; not scientific source. |

Package names and active paths are unchanged; there is no `src/` layout.

## Data and evidence held in place

These are not code and are not retired by the source cleanup. Some may be
untracked or gitignored locally, so a clean checkout may not contain them.

- `data/`: raw/merged data and graph caches. Never relocate.
- `comparison/standardized/event_inputs/`: max6 membership and event inputs
  (`clinical_graph_v3_membership_max6_20260923`,
  `first_recorded_lab_all_visits_v2_max6`,
  `first_recorded_lab_all_visits_v2_targets_local_v2_max6`) and the historical
  `first_recorded_lab_all_visits_v2_targets_v1/binding_manifest.json`.
  The all-visits and `membership_full_20260923` directories are held as lineage;
  there is no archive or delete decision.
- `comparison/standardized/native_inputs/protgsat_snapshot_v1/contract.json`:
  required label contract.
- Current max6/Top-10, challenger and CEI result directories with
  `source_snapshot`, binding, result and manifest files.
- `comparison/standardized/native_runs/` and other historical scientific
  outputs. Preserved, not verified reproducible.
- `performance_diagnosis/` and `zero_concept_verification/`: contain runnable
  verification sources; held without an explicit scope decision.
- Legacy worktrees with unsaved changes or separate results.

Old root folders (for example the former method directories) may still exist
because protected outputs live inside them. Do not assume whole directories are
gone; check the file manifest in the cleanup record.

## Retired legacy code (frozen)

The old 30-class native/star/cooccur model code, its dependent tests and the
legacy ProtGNN graph exporter were retired as one unit. Source is recoverable
from the Git refs and verified local archives in
[docs/max6-top10-cleanup.md](docs/max6-top10-cleanup.md). Retired entrypoints
are not documented as commands. Docs under `docs/` and READMEs inside held
directories that mention them are historical.

## Rules for work in this repository

- Real preprocessing, cache creation and training need explicit approval; output
  directories must be new. Never evaluate the held-out fold.
- Tests run only in an explicitly authorized testing phase. The approved source
  and test archival does not authorize running tests.
- `shared/data_prep/merge_ed.py` executes work at import time and has a known
  stale path; it is hash-bound to recorded provenance, so leave it as is. Any fix
  is a separate versioned job.
- Do not use `--help` on project modules as a harmless probe (import side effects).
- Use module form (`python3 -m package.module`) from the repository root.
- Do not mass-move scientific code, symlinks, vendored dependencies or historical
  outputs to reduce root entries.
- Open the viewer from `visualizer/` with `npm run dev`; it only displays graph
  JSON that already exists.
