# Repository navigation

Start with the [clinical v2/v3 runbook](comparison/standardized/clinical_graph_v2/README.md)
for the active max6 / train-derived Top-10 task. The
[cleanup record](docs/max6-top10-cleanup.md) explains what was retired and what
is held as evidence. The old 30-class reproduction is a different scientific
task; its inputs and scores are not interchangeable with the active one.

## Active runtime

| Location | Role / maintenance boundary |
|---|---|
| `comparison/standardized/clinical_graph_v2/` | Multi-visit build, all method adapters and plugins, training, audits, tabular control, CEI studies. Subpackages: `core/` (contracts, schema, graph, build, store, tensorize, diagnosis, stratify, rewiring, relation_information, repair_metadata, model, gchm_v2, gchm_v3, train, aggregate, audit, mechanism_check), `controls/` (`tabular_control.py`), `methods/` (ProtGNN, GSAT, GraphCare, GCHM-PNA, CEI-GNN, GMT, GPS, label-attention, token-fusion, virtual-node), `studies/cei/` (CEI v1-v3 studies, `cei_v3_ext/`), `paths.py` (`PACKAGE_ROOT`, `REPO_ROOT`). Every old module path keeps a compatibility shim; see [docs/structure-cleanup-2026-10-03.md](docs/structure-cleanup-2026-10-03.md). To add a method, study or control, follow "Adding things" in its README. |
| `comparison/canonical_split.json` | Fixed class ordering and subject folds. Never regenerate. Top-10 selection depends on this original class order. |
| `comparison/standardized/icd_mapping.py` | ICD mapping used by the label chain. |
| `comparison/standardized/event_graph_v1/` | Kept files only: `__init__`, `schema`, `first_lab`, `ingest`, `graph`, `knowledge_seed.csv`. |
| `comparison/standardized/enriched_input_v1/spec.py` | Kept input spec (plus the package marker it needs). |
| `comparison/standardized/event_graph_gchm_xgb_v1/` | Kept label contract: `__init__`, `labels`, `local_labels_v2`. |
| `comparison/standardized/gchm_v2_protocol/` | Optional, frozen protocol. Does not enable ADR-008. |
| `comparison/standardized/clinical_graph_v2/controls/tabular_control.py` | Current XGBoost control. The native 30-class baseline is retired. |
| `comparison/standardized/build_explanation_cohort.py` | Frozen synthetic-fixture helper retained for shared cohort assertions; not a current command or permission to access heldout data. |
| `shared/lib/`, `shared/data_prep/` | Shared contracts, split, metrics, data prep. `merge_ed.py` is hash-pinned: keep byte-identical, do not run or fix it. |
| `external/` | Third-party code. GraphXAI is used by the CEI explanation path; GraphCare is retained legacy. Keep upstream layout. |
| `docs/` | Runbooks, evidence and reports. Many describe retired or historical tooling. |
| `docs-vault/` | Retained local navigation; project decisions live in ProjectOS. Do not relocate. |
| `requirements.txt`, `requirements-lock.txt`, `environment.yml` | Main-Python dependencies. |
| `.venv-graphcare/` | Existing isolated legacy GraphCare runtime. Not used by the clinical adapter; left in place. |
| `.claude/`, `.git/` | Local agent state and version control; not scientific source. |

Top-level package names are unchanged; there is no `src/` layout. Inside `clinical_graph_v2/` the modules moved into subpackages on 2026-10-03 (shims at the old paths).

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
- `performance_diagnosis/` and `zero_concept_verification/`: HOLD historical
  verification sources/results, dependent on archived legacy runtime. They stay
  untouched and are not active suite entrypoints; execution is not authorized.
- Legacy worktrees with unsaved changes or separate results.

Old root folders (for example the former method directories) may still exist
because protected outputs live inside them. Do not assume whole directories are
gone; check the file manifest in the cleanup record.

## Tracked historical artifacts (frozen, not runtime)

Tracked in Git, but they describe retired or historical work. Do not run, move or edit them.

| Path | What it is |
|---|---|
| `comparison/standardized/benchmark_config.json` | Config of the retired 30-class benchmark; names retired `protgnn_analysis.train`, `gsat_analysis.train`, `graphcare_analysis.run`. No Python consumer. |
| `comparison/standardized/benchmark_results_summary.tsv` | Results summary of the 30-class benchmark. Different task; not comparable. |
| `comparison/standardized/matched_gchm_xgb_v1/` | Four tracked report files of the matched GCHM/XGBoost study (runs/smoke are gitignored). |
| `comparison/standardized/representative_eventgchm_v1/` | README only; runner retired. |
| `comparison/protgnn_explained_subjects.json` | Subject list from an old ProtGNN explanation run; unreferenced. Patient-derived; do not publish. |
| `protgnn_analysis/outputs/**`, `graphcare_analysis/outputs/**` | 236 tracked per-patient explanation files (~70 MB). Patient-derived; publication needs a privacy decision. |
| `comparison/standardized/explanation_subjects.json` | Fixture used by `build_explanation_cohort.py` and retained tests. |

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
- The viewer (`visualizer/`) was retired on 2026-10-03; recover it with `git checkout c54be79a -- visualizer`.
