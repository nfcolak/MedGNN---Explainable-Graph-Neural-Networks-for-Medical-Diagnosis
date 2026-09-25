# Repository navigation

Start with the [clinical v2/v3 runbook](comparison/standardized/clinical_graph_v2/README.md)
for the current max6 / train-derived Top-10 task. The
[native identical-input reference](docs/native-identical-input-v1.md) is a
preserved, separate 30-class reproduction. Their inputs and scores are not
interchangeable. The [older star/cooccur runbook](comparison/standardized/README.md)
and [usability report](docs/usability-verification.md) describe historical tooling,
not the current clinical entrypoint. Shared scientific packages stay in place.

## What lives where

| Location | Role / maintenance boundary |
|---|---|
| `comparison/standardized/clinical_graph_v2/` | Current multi-visit build, method adapters, training, audits and tabular control; max6 / train-derived Top-10 contract. |
| `comparison/standardized/train_identical.py`, `native_reference_v1/` | Preserved native identical-input reproduction, separate 30-class task. |
| `comparison/standardized/` | Also contains retained historical star/cooccur tooling and experimental protocols; not one interchangeable pipeline. |
| `comparison/canonical_split.json` | Fixed class ordering and subject folds. Do not regenerate for a resumed benchmark. |
| `protgnn_analysis/` | ProtGNN model, patient graph loader, training and explanation code. `scripts/` also contains legacy exploratory tools. |
| `gsat_analysis/` | GSAT on the shared PyG patient graphs, trainer and explainers. |
| `pna_analysis/` | Opt-in interaction-PNA and plain/wider controls; bounded common-input train/eval/replay. Separate from the three-method matrix; [commands and architecture](docs/pna-interaction.md). |
| `graphcare_analysis/` | Split-aware KG builder, patient adapter, training wrapper and explanations around upstream BAT-GNN. |
| `shared/lib/` | Benchmark/explanation contracts, canonical graph identities, provenance, metrics, split and configuration helpers. |
| `shared/data_prep/` | Raw CSV merging, medication/chief-complaint normalization, optional lab extraction. Legacy batch scripts; see cautions below. |
| `baselines/` | Exploratory tabular baseline; its default split is not the standardized benchmark. |
| `comparison/build_split.py` | Canonical split construction source; retained for provenance. Existing `canonical_split.json` is the active input, not regenerated during cleanup. |
| `data/` | Raw/merged CSVs, external data symlinks and graph caches. Not source code; never relocate as a cleanup side effect. |
| `external/` | Third-party GraphXAI and GraphCare source. Keep upstream layout and separate dependency requirements. |
| `visualizer/` | Independent React/TypeScript/Vite viewer; `scripts/export_protgnn_graphs.py` exports patient graphs. Export rewrites viewer data. |
| `tests/` | Maintained Python tests; only in an explicitly authorized testing phase. Use the explicit `tests` path to avoid vendored collection. |
| `docs/` | Operating runbooks, current scientific evidence, verification and benchmark design specification. |
| `docs-vault/` | Retained local navigation; canonical project decisions and work logs live in ProjectOS. Do not relocate this directory. |
| `README.md`, `STRUCTURE.md` | Landing page and this navigation map. Nonruntime thesis/templates/proposals and obsolete TODO notes are no longer in the working tree. |
| `requirements.txt`, `requirements-lock.txt`, `environment.yml` | Flexible and pinned main-Python dependencies. Not a replacement for the GraphCare environment. |
| `.venv-graphcare/` | Existing isolated GraphCare runtime; do not merge into the main environment. |
| `.claude/`, `.superpowers/`, `.pytest_cache/`, `.git/` | Agent, test and version-control state; not scientific source. |

The substantive [working-tree cleanup report](docs/cleanup-working-tree.md)
and [per-file restoration manifest](docs/cleanup-working-tree.json) record
retired templates, literature/figures, completed agent task packets, obsolete
two-method experiments, and broken unused entrypoints. They were moved outside
the repository to a unique macOS Trash directory, not permanently deleted.
`data.zip` was retired only after all 31 non-metadata payload files matched the
extracted data by SHA-256; all extracted data stays in place. Earlier cleanup
reports are historical evidence, not the current inventory.

On 2026-09-25, the user authorized retirement of 15 old clinical artifact/run
directories (failed/interrupted/smoke attempts and pre-repair v3 graphs/results).
They were moved reversibly to macOS Trash, not permanently deleted. The
[public cleanup record](docs/repository-cleanup-2026-09-25.md)
describes the scope; machine-local restoration manifests retain exact paths and
hashes and are not published. Historical report references to those directories
now require restoration; current membership-max6 inputs, sample10k/full-Top10
results, source indexes, target sidecars and model code remain in place.
The experiment/failure retrospective is indexed in the ProjectOS MedGNN note.

## Current and preserved entrypoints

The current clinical build/train commands and output contracts are in the
[clinical runbook](comparison/standardized/clinical_graph_v2/README.md).
Real preprocessing, cache creation and training require explicit approval;
output directories must be new and unoccupied. Never evaluate the held-out fold.

The native reference can be planned without training:

```bash
python3 -m comparison.standardized.train_identical --output comparison/standardized/native_runs/<new_dir> --dry-run
```

## Historical star/cooccur orchestration (retained)

Run from the repository root:

```bash
# Read-only: show the complete preprocessing/train/explain/summary plan.
python3 -m comparison.standardized.run_all --dry-run

# Read-only: validate inputs and show the training matrix.
python3 -m comparison.standardized.run_benchmark --dry-run

# Read-only: show one three-method explanation set, without checkpoints.
python3 -m comparison.standardized.run_explanations --topology star --seed 1234 --dry-run

# Read-only: preview cache construction; no caches are created by default.
python3 -m comparison.standardized.build_caches --dry-run

# Only after the user explicitly opens the testing phase.
python3 -m pytest tests -q
```

`run_all --execute` is **expensive**: it generates missing standardized caches,
audits record parity, trains the 18 matrix cells, executes six three-method
explanation sets and writes summaries. Do not run it merely to verify setup.
Real preprocessing/training requires separate approval.

`--resume` revalidates completed scientific artifacts and skips completed
commands. `--retry-failed` permits retrying a failed command. An occupied,
incomplete result is not overwritten: the additional `--archive-incomplete`
flag preserves that individual run/set under `comparison/standardized/attempts/`
before starting it again. Ensure no worker is still active. **This is run-level
restart from scratch, not optimizer/epoch-level resume.**

## Output ownership

- Current clinical inputs/results: the hash-bound paths in the clinical runbook;
  preserve `clinical_graph_v3_membership_max6_20260923`, its source indexes and
  target sidecars, and the corrected sample10k/full-Top10 result directories.
- Native reference: `comparison/standardized/native_runs/<new-run>/`.

The following paths belong to retained historical tooling, not the current
max6/Top-10 task:

- Historical matrix: `comparison/standardized/results/<method>/<topology>/seed_<seed>/`.
- Fixed-cohort explanations: `comparison/standardized/explanations/<topology>/seed_<seed>/<method>/`.
- Pipeline state/events: `comparison/standardized/checkpoint/` (outside result cells).
- Preserved retry attempts: `comparison/standardized/attempts/` (excluded from summaries).
- Isolated PNA smokes: `comparison/standardized/pna_experiments/<new-run>/` (no overwrite; not included in the legacy benchmark summaries).
- Legacy method runs: each method's own `outputs/`, not a root `outputs/` directory.
- Shared graphs: `data/graphs/<topology>/{protgnn,graphcare}/`; GSAT uses ProtGNN's PyG cache.

Historical matrix topologies are `star` and `cooccur`. Additional legacy or
method-specific graph variants are defined in `shared/lib/graph_structures.py`;
they must not silently enter the primary aggregate. The contracts are enforced
by code and tests; full-production graph parity still requires cache generation
and the real audit, and has not been established by this usability pass.

## Legacy boundaries and cautions

- Independent design/input probes, PNP entrypoints and its pilot outputs, and
  PLQ/dropout analysis plus their run outputs were moved to a durable external
  archive on 2026-09-25. See the
  [public cleanup record](docs/repository-cleanup-2026-09-25.md); exact hashes and
  recovery destinations remain in the machine-local sections 3–4 manifest.
  Historical commands naming those paths require restoration first; the archive
  is not a standalone installation. Scientific reports remain in `docs/`.
- Legacy ProtGNN CLI files are retained: native `source_bindings()` fingerprints
  the whole method package, and some CLIs have maintained test references.
  A filename that looks unused is not sufficient reason to remove it.
- The old EventGCHM chain remains intact because shared dependencies and an
  unresolved migration need a separate closure decision. Agent `*-before`
  snapshots differ from live files and remain rollback evidence.
- `performance_diagnosis/` contains synthetic mechanism verification sources;
  those are retained rather than changing test-only code outside a testing phase.
- The 2026-09-20 cleanup plan is a historical record, not executable guidance.

- `shared/data_prep/merge_ed.py` executes work at import time and both it and
  `extract_ed_labs.py` still derive `shared/data` rather than the root `data/`.
  Do not run them with `--help` as a harmless probe. Their raw-data CLI and path
  migration require a separate fixture-backed change; the existing files were
  not moved or their processing semantics altered here.
- Several historical scripts use implicit datasets, topology prompts and
  timestamped outputs. A passing `--help` checks imports/parser wiring, not a
  complete training, attribution, export or clinical-report run.
- Use module form (`python3 -m package.module`) from the root. For direct-file
  legacy commands use `PYTHONPATH=.:external/GraphXAI-main`; GraphCare uses
  `PYTHONPATH=.:external/GraphCare .venv-graphcare/bin/python3`.
- Open the viewer from `visualizer/` with `npm run dev`; exporting real graphs is
  a separate write operation, not part of a viewer build or setup check.

Future organization should consolidate documentation links and testable CLI
wrappers before considering a `src/` layout. Do not mass-move scientific code,
symlinks, vendored dependencies or historical outputs just to reduce root entries.
