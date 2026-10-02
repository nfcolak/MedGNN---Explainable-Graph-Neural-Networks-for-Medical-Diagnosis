# Max6 / Top-10 cleanup (user-approved)

This is the 2026-10-02 cleanup implementation record, not a new scientific ADR.
ADR-007 and ADR-008 remain proposed and are not enabled. Integration is a local
installation candidate; original-checkout installation is still pending the
parent. Durable preservation passed its separate parent-verified gates; this
candidate is not installable until the protected AGENTS.md write gate succeeds.

## Decision and scope

The user approved the coordinated legacy source/test/exporter retirement with
"Bu yapiyi kur" and separately authorized the main agent only for controlled
final transfer and approved clean-worktree removal. Source/document authoring
and integration remain delegated.

Approval does not authorize project imports, tests or pytest collection, real
preprocessing, training, heldout access, scientific edits, deletion of data or
results, relocation of `data/`, `external/` or `docs-vault/`, changes to
`.venv-graphcare/`, or publishing patient material.

Byte-freeze and no-push instructions here apply FOR THIS cleanup operation.
They do not prohibit future separately authorized scientific development.
Permanent gates still forbid heldout access, unapproved real preprocessing or
training, overwriting occupied outputs, and canonical-split regeneration.
Publication needs a separate privacy decision; historical Git contains
patient-derived payloads and is not safe to push merely because current docs
are clean.

## Exact references and receipts

| Item | Value |
|---|---|
| Original checkout | `f94e4076b0f1425eaae7c418a15eb5876a19ab4c` (`f94e4076`) |
| Delivery/base | `6395f33ffc272cf19f1f9939b12c7f8d50e245b7` (`6395f33f`) |
| Code source | `625e9b557d1d049dfa6d418e1cd761a320a48a3e` |
| Docs source | `af9fae912b52ff847cbc7980dc6de96b61ab87b0` |
| Bounded tracked index | [max6-top10-retirement.json](max6-top10-retirement.json) |
| Durable common preservation root | `/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/max6-cleanup-20261002` |
| Immutable code manifest | `/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/max6-cleanup-20261002/med-clean-code/retirement_manifest.json` |
| Verified code archive | `/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/max6-cleanup-20261002/med-clean-code/retired-source-and-mixed-tests.tar` |
| Code static receipt | `/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/max6-cleanup-20261002/med-clean-code/static-validation.json` |
| Final integration manifest | `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/.worktrees/_runs/med-clean-finish/final-retirement-manifest.json` |
| Integration static receipt | `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/.worktrees/_runs/med-clean-finish/integrated-static-validation.json` |

The code archive SHA256 is
`077a24c674c7524317ca2886defc63d4b0c6d72f2e7e83ca248e9e9866da22b4`.
Its 128 exact originals cover 97 retired sources, 24 retired test files and the
original versions of 7 adjusted mixed test files. The final manifest additionally
archives `tests/test_performance_evidence.py`, for 25 retired tests in total.
That test exclusively drove the already-absent frozen
`performance_review_20260913/evidence.py`; it was archived without execution at
`/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/.worktrees/_runs/med-clean-finish/archived-originals/tests/test_performance_evidence.py`.
Its SHA256 is `57bbfc60297b176ba48e9b0f8c7447a8ba3ff4332b2088f9f666030ecac73146`
and Git blob SHA is `7da72be23642c46a0c02dbaf7dd493bfe4aaf5c8`.
The code worker's archived manifest is immutable, not rewritten by integration.

Final installation and original-checkout byte verification will be recorded in
`/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/max6-cleanup-20261002/installation-receipt.json`
AFTER parent preservation/collision gates and controlled transfer. Neither the
existence of a candidate commit nor these source receipts verifies that final
installation. Durable preservation is complete and parent-verified at
`/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/max6-cleanup-20261002/med-clean-preserve/`:
`deployment-preflight.json`, `receipt-check.json` and
`preservation-policy.summary.json`. These immutable receipts retain old run-dir
paths; resolve that prefix using common-root `preservation-transfer-receipt.json`.
All transferred hashes matched except launcher-finalized `state.json`, whose
recorded runtime-metadata exception is explicit; bundle verification passed.
The receipts cover 52,644 protected root files, not a historical rerun claim.
The exact evidence ignore-policy/source receipt is
`/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/max6-cleanup-20261002/med-clean-evidence/protection-manifest.json`
(source `c27eea268451808e30945895603568e8c3f15af8`). After checkout the parent must
restore all 175 HELD paths byte-identically to their original locations, ignored
and local. The large viewer graph manifest and old clinical-explanation output
summaries are untracked/ignored, never purged. Frontend source is unchanged.

## Kept runtime and frozen exceptions

All current methods, plugins and CEI stay under
`comparison/standardized/clinical_graph_v2/`; current XGBoost guidance points
only to `clinical_graph_v2/tabular_control.py`. The native baseline is retired.
The fixed `comparison/canonical_split.json`, `icd_mapping.py`, kept
`event_graph_v1` schema/first-lab/ingest/graph/knowledge seed, input spec,
`event_graph_gchm_xgb_v1` labels/local-labels and `shared/` stay in place.
`gchm_v2_protocol/` remains optional/frozen, not authorization to enable ADR-008.
See [STRUCTURE.md](../STRUCTURE.md) and the exact retained-exception index.

`comparison/standardized/build_explanation_cohort.py` remains byte-identical as
a frozen synthetic-fixture helper used by shared cohort-count/roundtrip/manifest
assertions. It is NOT a current command and does not authorize heldout access.
Top-10 selection still starts from the original fixed class order and target
binding, even when their records carry "30-class" or "native" names. No active
package or path was renamed and no compatibility stubs were added.

## Task-local source freeze

The code worker records 97 protected source/contract hashes. FOR THIS cleanup,
all protected sources, `clinical_graph_v2/**`, `shared/**` and the canonical split
are byte-frozen. `shared/data_prep/merge_ed.py` is historically SHA256-pinned by
`local_labels_v2.py`; it performs work at import time and has a known stale-path
issue. It is not imported, run or fixed here. Any correction needs a separately
authorized, versioned production job that preserves historical provenance.

The worker also records 43 untouched retained test hashes. One of those is the
explicitly archived evidence-driver test above; 42 remain untouched in the final
suite, alongside 7 mixed files whose surviving shared/clinical assertions are
preserved exactly as verified by the code worker. No tests are written or run.

## Historical instructions and evidence HOLD

Legacy 30-class native/star/cooccur model sources, their dependent tests and
`visualizer/scripts/export_protgnn_graphs.py` retire together. Root legacy and
standardized READMEs carry frozen notices; historical commands beneath those
notices are archival examples, not active instructions. Historical scientific
result reports and scientific payloads are not rewritten.

Former root package directories may remain as evidence-only shells because
protected outputs live inside them. Do not claim whole directories are gone.
The optional visualizer displays existing JSON only: its legacy exporter is
retired and its frontend was not rebuilt for CEI.

Held in place and untouched: all `data/`; max6 inputs and sidecars; all-visits
labels and event index as lineage; `membership_full_20260923` (not a max6 input);
required native label contracts and historical binding manifests; current and
historical results with every `source_snapshot`/binding/result/manifest;
`external/GraphCare`, `.venv-graphcare/`, `external/GraphXAI-main`, `docs-vault/`
and worktrees with unsaved changes or separate outputs.

HOLD `performance_diagnosis/` and `zero_concept_verification/` include historical
verification sources/results that depend on archived legacy runtime. They are
not active suite entrypoints and are not executed, edited or removed here.

## Provenance and static limits

The author of the max6 sidecar-filtering step is not established and byte-exact
reproduction is not demonstrated. The all-visits and max6 category-map records
were observed to share a hardlink inode; they are not independent backups.
Keep those bytes exactly.

Original/delivery Git references and saved source snapshots preserve source
history, but are not by themselves complete backups of ignored data, vendor
code or uncommitted changes. Delivery already removed some historical source
bindings. A run is re-verifiable only after its complete required source hash
set matches an explicit ref/snapshot/archive, with verification scope stated.
Matching source alone does not guarantee identical training output: environment,
data and commands also matter. Otherwise say "preserved, reproduction not
verified".

Integration uses only stdlib AST/symtable/path/hash checks and `git diff --check`.
Inert audit-command strings and provenance docstrings are distinguished from
active imports; HOLD historical checks are outside active targets. No project
module imports, CLI `--help`, tests/collection, builds/exports, installs, real
data preparation, training or heldout evaluation are performed. Static receipts
do not establish runtime/library availability.

## Local-only context and remaining transfer gates

Prepared copies of the original `.claude` registry and launch configuration are
in the integrator run directory's `local-overlay/`, with original/prepared
SHA256, relative path and mode in `local-overlay-manifest.json`. The parent must
fail if originals changed, then transfer only those exact authored bytes after
preservation gates and read back the exact destinations. Only Buffett backend
and frontend launch entries are removed; the visualizer entry remains. Nothing
under `.claude` is tracked or recreated in the target worktree.

The user explicitly authorized the AGENTS.md current-XGBoost and task-scoping
correction ("Evet, AGENTS.md’de bu kapsamlı düzeltmeye izin veriyorum"). The
normal protected-write gate nevertheless timed out; no retry or bypass was
attempted, and AGENTS.md remains unreconciled. The exact proposed correction is
saved under this run as `proposed-agents-reconciliation.patch`, with the gate
failure in `instruction-write-blocker.json`. This candidate is BLOCKED for
installation even if its other static checks pass.
The corrected registry checker resolves ACTIVE tracked targets only in the
candidate. Its only original-root exceptions are the two Gitignored historical
preservation manifests (`docs/cleanup-sections-3-4-20260925-121538.json` and
`docs/cleanup-retired-experiments-20260925-112447.json`, hash-pinned against the
preflight protected-root manifest) and preserved `.claude/context/` symlinks.
No placeholder files are created and no stale original code is a fallback.
ProjectOS/commit-sync follow-up belongs to the parent after installation; this
unit does not write either vault or the original checkout.
