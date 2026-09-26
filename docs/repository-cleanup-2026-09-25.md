# Repository cleanup — 2026-09-25

## Publication scope

This change publishes the authorized cleanup and navigation updates only. Other
pending clinical model, adapter, protocol and test-source changes in the author's
working tree are deliberately excluded. Navigation describes the intended current
clinical workflow and the preserved native reference; this cleanup is not a
release or runtime verification of those pending implementations.

No new medical records, input artifacts, checkpoints, generated run folders,
private ProjectOS notes or machine-local restoration inventories are included.
Model weights remain excluded. Existing scientific reports under `docs/` retain
the historical findings; removing an experiment directory is not a claim that its
method failed or that its scores are comparable with the current task.

## Retired historical artifacts and runs

The following 15 local directories were moved reversibly out of the working tree.
Only previously tracked files within them appear as deletions in Git; ignored
artifacts were never added for publication.

Paths below are relative to `comparison/standardized/`:

- `event_inputs/clinical_graph_v3_membership_max6_20260923_interrupted_readback/`
- `event_inputs/clinical_graph_v3_membership_smoke_1000_20260923/`
- `event_inputs/clinical_graph_v3_membership_smoke_1000_20260923_v2/`
- `event_inputs/first_recorded_lab_all_visits_v1/`
- `clinical_runs_v3_adapters_invalid_full30_20260923/`
- `clinical_runs_v3_adapters_smoke_20260923/`
- `event_inputs/clinical_graph_v3_full/`
- `event_inputs/clinical_graph_v3_max6/`
- `clinical_runs_v3/`
- `clinical_runs_v3_gchm/`
- `clinical_runs_v3_hgt_pilot/`
- `clinical_runs_v3_loro/`
- `clinical_runs_v3_mv10/`
- `clinical_runs_v3_rewire/`
- `clinical_runs_v3_top10/`

These represent failed or interrupted production, bounded smoke attempts,
wrong-task adapter runs, and superseded pre-repair graph experiments. They must
not be treated as one blanket failed benchmark.

## Archived independent experiments

The following nine local targets were moved together with their available
outputs to an external archive, preserving original relative paths:

- `method_design_probe/`
- `new_input_diagnosis_v1/`
- `patient_graph_v1/`
- `gchm_plq_v1/`
- `gchm_dropout_pilot_v1/`
- `native_runs/gchm_plq_v1/`
- `native_runs/gchm_dropout_pilot_v1/`
- `native_runs/_pilot/pnp/`
- `native_runs/finish_runs.sh`

The archive is not a standalone installation. Reproducing a historical run still
requires its bound input, source version and environment. Reports such as
`new-input-diagnosis.md`, `gchm-concept-dropout-pilot-results.md`, and
`gchm-xgboost-matched-results.md` remain in `docs/`. The archived batch helper
contains destructive retry commands; restoration is not authorization to run it.

## Intentionally retained

- Native reference entrypoints, model packages and shared helpers.
- Legacy ProtGNN CLIs: native source-fingerprint collection covers whole method
  packages; some also have retained test references.
- EventGCHM source chain and current producer/label helpers: shared dependencies
  and an unresolved local migration prevent piecemeal retirement.
- Synthetic performance-diagnosis verification sources.
- Nonredundant agent rollback copies, original inputs, canonical split,
  environments, vendored code and the viewer.
- Current membership-max6 inputs and corrected sample10k/full-Top10 outputs
  remain local and are not published by this cleanup.

## Navigation and recovery

`STRUCTURE.md` distinguishes the current clinical task, native 30-class reference,
and historical star/cooccur tools. `docs-vault/Overview.md` points to the local
ProjectOS project without publishing a machine-specific home path. The old
2026-09-20 cleanup plan is explicitly historical and must not be executed.

Detailed per-file recovery manifests are retained locally as
`docs/cleanup-retired-experiments-20260925-112447.json` and
`docs/cleanup-sections-3-4-20260925-121538.json`, with copies alongside the archived
files. They are intentionally ignored because they contain local paths and
working-tree inventories. Restore only into unoccupied paths and verify recorded
hashes; never overwrite newer work. Git history retains previously committed
source versions, but is not a backup of ignored data or model artifacts.

## Verification boundary

Local relocation checks verified destination hashes and source absence. Static
source analysis found no remaining incoming imports to the selected removed
modules; protected native code was unchanged by cleanup. Generated first-party
caches were removed from the working tree separately.

Tests, training, preprocessing and held-out evaluation were not run. Static
checks and a successful Git push must not be described as runtime correctness or
benchmark replay. No history rewrite or permanent archive deletion is part of
this change.
