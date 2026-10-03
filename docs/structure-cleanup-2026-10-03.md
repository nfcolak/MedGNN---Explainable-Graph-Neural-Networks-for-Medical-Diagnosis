# Structure cleanup 2026-10-03

Follow-up to `docs/max6-top10-cleanup.md`. Base: `cleanup/max6-top10-20261002-final` (`c54be79a`). Local only, no push.

## What changed

1. The 2026-10-02 byte-freeze on `comparison/standardized/clinical_graph_v2/**` is lifted (that package only).
   The package was split into subpackages. File names did not change; files moved with `git mv`.

| Old (under `comparison/standardized/clinical_graph_v2/`) | New |
|---|---|
| `contracts, schema, graph, build, store, tensorize, diagnosis, stratify, rewiring, relation_information, repair_metadata, model, gchm_v2, gchm_v3, train, aggregate, audit, mechanism_check` | `core/<same>.py` |
| `tabular_control.py` | `controls/tabular_control.py` |
| `methods/*` | unchanged location (imports updated) |
| `cei_graphxai, cei_pilot, cei_v2_study, cei_v3_absence, cei_v3_graphxai, cei_v3_paths, cei_v3_ple, cei_v3_preserve, cei_v3_run, cei_v3_screen, cei_v3_study, cei_v3_validation, cei_v3_vs_protgnn, cei_v3_vs_xgboost` | `studies/cei/<same>.py` |
| `cei_v3_ext/` (all modules) | `studies/cei/cei_v3_ext/<same>.py` |
| new | `paths.py` (`PACKAGE_ROOT`, `REPO_ROOT`), `core/`, `controls/`, `studies/`, `studies/cei/` package markers |

2. Every old module path keeps a compatibility shim, so old imports and old
   `python3 -m comparison.standardized.clinical_graph_v2.<module>` commands still work.
   New code and docs use the new paths; moved code never imports through a shim.
3. Source binding (`recursive_source_hashes`) now covers the whole package via `PACKAGE_ROOT`; repo-relative
   paths use `REPO_ROOT`. Guards keep their meaning; none were weakened.
4. The viewer (`visualizer/`) was retired on 2026-10-03 (`git rm -r`; tracked files only). Recover it from
   commit `c54be79a`: `git checkout c54be79a -- visualizer`.
5. Documentation fixes from the 2026-10-03 audit: installed status and live receipt paths in the cleanup
   record, frozen notices on three kept READMEs, tracked `.gitignore` for `/data/` and `/.worktrees/`,
   `psutil` declared, STRUCTURE.md table of tracked historical artifacts.

## Consequences for results

Old result bindings record the old source hashes of the package. They no longer match the new source
hashes. Historical runs stay historical (not reproducible by the new tree, not rewritten). New runs
need new output directories; do not add bypasses for the old bindings.

## What stays frozen

- `shared/**` (including `shared/data_prep/merge_ed.py`, hash-pinned), `comparison/canonical_split.json`.
- `event_graph_v1/`, `event_graph_gchm_xgb_v1/{labels,local_labels_v2}.py`, `enriched_input_v1/spec.py`, `icd_mapping.py`.
- `data/`, `external/`, `docs-vault/`, `.venv-graphcare/`, all result/output/`source_snapshot` directories.
- Held-out fold stays closed; no training, preprocessing or cache builds without approval; tests only in an opened testing phase.

## Method folders (2026-10-03, second pass)

Every main method now has its own folder `clinical_graph_v2/methods/<method>/` with its code, studies,
reports (`docs/`) and a `README.md` saying where its results are: `protgnn`, `cei`, `gsat`, `graphcare`,
`gchm_pna`, `xgboost`. Shared code stays in `core/`, `methods/base.py`, `methods/__init__.py`, `paths.py`.
`methods/__init__.py` discovers `plugin_*.py` only inside method subpackages. Every old module path keeps a
compatibility shim (old package locations keep a plain `__init__.py` plus one shim per submodule).

| Old | New |
|---|---|
| `methods/protgnn.py` | `methods/protgnn/adapter.py` |
| `methods/gsat.py` | `methods/gsat/adapter.py` |
| `methods/graphcare.py` | `methods/graphcare/adapter.py` |
| `core/gchm_v2.py`, `core/gchm_v3.py` | `methods/gchm_pna/gchm_v2.py`, `gchm_v3.py` |
| `comparison/standardized/gchm_v2_protocol/*.py` | `methods/gchm_pna/protocol/<same>.py` |
| `controls/tabular_control.py` | `methods/xgboost/tabular_control.py` |
| `methods/cei_gnn.py`, `cei_gnn_v2.py`, `cei_gnn_v3.py` | `methods/cei/<same>.py` |
| `methods/plugin_cei_gnn.py`, `plugin_cei_gnn_v2.py`, `plugin_cei_gnn_v3.py` | `methods/cei/<same>.py` |
| `studies/cei/*.py` (14 modules), `studies/cei/cei_v3_ext/` | `methods/cei/studies/<same>.py`, `methods/cei/studies/cei_v3_ext/` |

Deleted challengers (code and tests, no shims): GMT, GPS, label-attention, token-fusion, virtual-node,
ProtoNode (`methods/protonode.py`, `plugin_gmt.py`, `plugin_gps.py`, `plugin_labelattn.py`,
`plugin_tokenfusion.py`, `plugin_vnode.py`; ProtoNode also left the core registry). Their result
directories stay untouched. Recover the code from commit `aad56003`.

Doc moves (`git mv`; paths relative to `clinical_graph_v2/methods/`):

| Old | New |
|---|---|
| `docs/cei-*` (v3 delivery, v3 evidence/smoke JSON, v2 pair study) | `cei/docs/` |
| `docs/presentations/cei-overview/` | `cei/docs/presentations/cei-overview/` |
| `docs/superpowers/specs/*cei*`, `*evidence-interaction*`; `docs/superpowers/plans/*` (CEI plans) | `cei/docs/superpowers/{specs,plans}/` |
| `docs/gchm-concept-dropout*.md`, `docs/pna-*.md`, `docs/pna-quality-evidence/` | `gchm_pna/docs/` |
| `docs/superpowers/specs/2026-09-24-gchm-pna-v2-design.md` | `gchm_pna/docs/superpowers/specs/` |
| `docs/gchm-xgboost-matched-results.md` | `xgboost/docs/` |

Cross-method, cleanup, data/input and legacy docs stay in `docs/` (for example `graphxai-500-results.md`,
`max6-top10-*`, `native-identical-input-v1.md`, `model-performance-diagnosis.md`).

## Root cleanup (2026-10-03, third pass)

Decision (user, 2026-10-03): retired legacy experiments, unused old inputs, the old GraphCare runtime,
old agent run dirs and the viewer leftovers are archived to
`/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/repo-archive-20261003/`
(hash-verified tars; `manifest.jsonl`) and deleted from the repo. Base `refactor/method-folders-20261003`
(123a1b8b). No code reads the removed paths (only data literals in `tests/test_benchmark_runner.py` and frozen
`shared/**` docstrings mention them).

Archived and deleted, tracked (`git rm`):

- `protgnn_analysis/`, `graphcare_analysis/`
- `comparison/standardized/matched_gchm_xgb_v1/`, `comparison/standardized/representative_eventgchm_v1/`
- `comparison/standardized/benchmark_config.json`, `comparison/standardized/benchmark_results_summary.tsv`
- `comparison/protgnn_explained_subjects.json`

Archived and deleted, untracked/ignored local copies:

- `comparison/standardized/{native_runs,native_evidence,native_inputs/patient_knn_v1,results,enriched_inputs,enriched_runs,event_training,performance_diagnosis,zero_concept_verification,production_cache_20260913T154134,training.log,graphxai_current_run.txt}`
- `comparison/standardized/event_inputs/{clinical_graph_v3_membership_full_20260923,first_recorded_lab_all_visits_v2_features_v1,first_recorded_lab_all_visits_v2_targets_v1,first_recorded_lab_all_visits_v2_targets_local_v2,v3_build.log}`
- `.worktrees/_runs`, `.hermes`, `visualizer/`

Deleted without archive: `.venv-graphcare/`, `external/GraphCare`, `.superpowers/`.

Kept: `data/`, `external/GraphXAI-main`, `docs-vault/`, `event_inputs` max6 dirs and
`first_recorded_lab_all_visits_v2`, `native_inputs/protgsat_snapshot_v1`, `clinical_runs_*`,
`.worktrees/cei1001-smoke`, `.worktrees/cei-v3-delivery`.

`.gitignore` lost the rules that only served deleted paths; `/data/`, `/.worktrees/`, `.hermes/` and rules for
existing paths (`clinical_runs*`, `event_inputs`, `native_inputs`, `external/` GraphXAI extras) stay.
Docs updated: README, STRUCTURE, `comparison/standardized/README.md`, `docs/max6-top10-cleanup.md`, method READMEs.
