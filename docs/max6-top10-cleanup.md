# Max6 / Top-10 cleanup (user-approved)

Status: user-approved cleanup, not an ADR. ADR-007 and ADR-008 remain proposed
and are not enabled. This document is the implementation record; a ProjectOS
follow-up for commit-sync is listed at the end.

## Decision

After a two-round main-agent / Opus discussion the user approved building the
clean structure ("Bu yapiyi kur"). The approval covers a coordinated retirement
of the legacy 30-class source, its dependent tests and the legacy exporter, plus
clean-worktree housekeeping.

It does not authorize: running tests, training, heldout access, changing
scientific logic, deleting data or results, relocating `data/`, `external/` or
`docs-vault/`, changing `.venv-graphcare/`, or publishing patient material.

## References

| Item | Value |
|---|---|
| Original checkout commit | `f94e4076` |
| Delivery base (`feature/cei-v3-delivery`) | `6395f33f` |
| Local archival receipt | to be filled by the integrator |
| Retirement manifest | to be filled by the integrator (`retirement_manifest.json`) |

Exact removal counts are not stated here; use the manifest.

## Kept as active

All current methods, plugins and CEI under `comparison/standardized/clinical_graph_v2/`;
`comparison/canonical_split.json`; `icd_mapping.py`; the kept
`event_graph_v1`, `enriched_input_v1/spec.py` and `event_graph_gchm_xgb_v1`
label files; `gchm_v2_protocol/` (frozen, optional); `shared/`. Full list:
[STRUCTURE.md](../STRUCTURE.md).

Top-10 selection starts from the original fixed class order and target binding,
so those records are kept even though their names say "30-class" or "native".
No package was renamed and there is no `src/` refactor.

## merge_ed byte freeze

`shared/data_prep/merge_ed.py` is compared by SHA-256 against a historical
record in `local_labels_v2.py`. The current file hash was confirmed to match the
max6 sidecar record. Its known stale-path issue is deliberately not fixed here;
any correction would be a separate versioned production job. The same
byte-preservation applies to `clinical_graph_v2/**` and `canonical_split.json`.

## Retired unit

Retired together in one scoped change: legacy 30-class method code, the tests
that import it, and `visualizer/scripts/export_protgnn_graphs.py`. The code
worker's manifest is authoritative for exact paths. Docs and READMEs under held
directories that mention retired commands are historical and were not rewritten.

Old root folders may remain because protected outputs live inside them. Do not
claim whole directories are gone.

## Evidence hold

Held in place, not moved, deleted or archived: all `data/`; the max6 inputs and
their sidecars; all-visits targets and event index as source lineage;
`membership_full_20260923` (not a max6 input, no decision yet);
`native_runs/native_evidence` and other historical scientific outputs;
`performance_diagnosis/` and `zero_concept_verification/`;
`external/GraphCare`, `.venv-graphcare/` (the clinical GraphCare adapter does
not depend on them; removal or relocation needs a separate named decision);
`external/GraphXAI-main` (used by CEI explanations); worktrees with unsaved
changes.

## Provenance gap

The author of the filtering step that produced the max6 sidecar files could not
be determined, and byte-exact reproduction is not demonstrated. These files must
be preserved exactly. The all-visits and max6 `frozen_category_map.json` files
were observed to be hardlinks of one inode, so they are not independent backups.

## Preservation and recovery expectations

- Original source is recoverable from `f94e4076` and `6395f33f`, the commits
  recorded by earlier runs and their `source_snapshot` copies.
- Files outside Git (vendor code, data, uncommitted changes) are covered only by
  the integrator's byte/hash manifest and archive, not by a tag.
- The delivery branch already deleted some tracked files, so one tag is not
  enough to recover old result sources.
- A run may be called re-verifiable only after its required source hash set is
  checked against a ref, snapshot or archive, with the sampling scope written
  down. Matching source hashes alone do not guarantee identical training output;
  environment, data and command are also needed.
- Everything else is "preserved, reproduction not verified".
- Scope is local. Nothing is pushed: Git history contains patient-derived
  payloads and publication is a separate decision.

No tests, training, build or project-module imports were run for this
documentation change; checks were static (link/path and `git diff --check`).

## ProjectOS follow-up

After integration, update the ProjectOS MedGNN note and the commit-sync mapping
(`ProjectOS/scripts/commit_sync.py` atomic notes) so retired paths and the new
frozen-legacy policy are reflected, and update the local `.claude` context
registry (`SUBSYSTEMS`) for the paths retired by the manifest. The `.claude`
tree is local-only and is not tracked here.
