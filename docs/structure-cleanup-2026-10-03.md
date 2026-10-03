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
