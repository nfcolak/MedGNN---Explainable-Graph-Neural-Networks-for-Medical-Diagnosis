# Repository navigation

Scientific code stays in its existing packages: moving it would break imports,
checkpoint paths, notebooks and cached graph recipes. Start with the
[standardized runbook](comparison/standardized/README.md) and the
[verification report](docs/usability-verification.md), not the older experiment
scripts.

## What lives where

| Location | Role / maintenance boundary |
|---|---|
| `comparison/standardized/` | Current three-method benchmark: config, cache builder/audit, per-cell runner, orchestration, fixed-cohort explanations, summary. |
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
| `tests/` | Maintained Python tests; invoke `python3 -m pytest tests -q` explicitly to avoid vendored test collections. |
| `docs/` | Operating runbooks, current scientific evidence, verification and benchmark design specification. |
| `docs-vault/` | Project-owned Obsidian knowledge vault; keep inside this repository. |
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

## One pipeline, explicit execution

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

# Maintained tests (temporary fixture preprocessing and model forward/backward).
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

- Current benchmark: `comparison/standardized/results/<method>/<topology>/seed_<seed>/`.
- Fixed-cohort explanations: `comparison/standardized/explanations/<topology>/seed_<seed>/<method>/`.
- Pipeline state/events: `comparison/standardized/checkpoint/` (outside result cells).
- Preserved retry attempts: `comparison/standardized/attempts/` (excluded from summaries).
- Isolated PNA smokes: `comparison/standardized/pna_experiments/<new-run>/` (no overwrite; not included in the legacy benchmark summaries).
- Legacy method runs: each method's own `outputs/`, not a root `outputs/` directory.
- Shared graphs: `data/graphs/<topology>/{protgnn,graphcare}/`; GSAT uses ProtGNN's PyG cache.

Primary standardized topologies are `star` and `cooccur`. Additional legacy or
method-specific graph variants are defined in `shared/lib/graph_structures.py`;
they must not silently enter the primary aggregate. The contracts are enforced
by code and tests; full-production graph parity still requires cache generation
and the real audit, and has not been established by this usability pass.

## Legacy boundaries and cautions

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
