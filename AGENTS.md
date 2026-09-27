# MedGNN

Self-explainable GNN diagnosis benchmark on MIMIC-IV ED patient graphs: ProtGNN, GSAT, GraphCare, GCHM-PNA and an XGBoost control, compared under fixed data contracts.

## Tech stack

| Item | Value |
|---|---|
| Language | Python (system `python3`; repo recommends 3.10/3.11) |
| ML | PyTorch 2.2–2.8, PyTorch Geometric 2.5–2.6, XGBoost 2.x, scikit-learn |
| GraphCare runtime | separate `.venv-graphcare/` — never merge into the main env |
| Viewer | `visualizer/` (React/TypeScript/Vite) |

## Non-negotiable rules

| Rule | Why / source |
|---|---|
| NEVER load or evaluate the held-out test fold; select checkpoints on validation (or ADR-008 dev split) macro-F1 only. | `.claude/context/adr-002.md` |
| NEVER write/run tests unless the user explicitly opens a "testing phase". Parse-only checks are fine. | `.claude/context/adr-005.md` |
| NEVER start real preprocessing, cache builds or training without explicit user approval. Use `--dry-run` / plan modes. | `STRUCTURE.md` |
| NEVER overwrite a filled output/artifact directory; write to a new path. Runners refuse by design. | `comparison/standardized/clinical_graph_v2/README.md` |
| NEVER upgrade historical graphs by editing metadata; rebuild into a new path. | `.claude/context/adr-004.md` |
| NEVER regenerate `comparison/canonical_split.json`. | `STRUCTURE.md` |
| Compared methods MUST share artifact, class set, train/validation sample-ID hashes, preprocessing contract and seed. | `.claude/context/adr-006.md` |
| Do not relocate `data/`, `external/` or `docs-vault/` as a cleanup side effect. | `STRUCTURE.md` |
| Report results with task identity (cohort, class set, sample size, seed). Single-seed gaps are not method superiority. | `.claude/context/adr-006.md` |

## Current comparison contract (max6 / Top-10)

- Patients with more than 6 total visits are excluded.
- Classes: the 10 most frequent diagnoses by TRAIN-fold frequency; other labels are dropped, never merged into `other`.
- Sample comparison: the same 10,000 train IDs (seed `1234`) and the full 4,254-row validation fold.
- Results from this task are NOT comparable with the older 30-class native benchmark (331-slot star input).
- Full details: `.claude/context/adr-006.md`. Proposed, not accepted: `adr-007.md` (bidirectional edges, GCHM-PNA v2) and `adr-008.md` (dev-selected equal-budget protocol).

## Where things live

| Area | Path | Context doc |
|---|---|---|
| Multi-visit clinical graph v2/v3 + method adapters | `comparison/standardized/clinical_graph_v2/` (`methods/`) | `clinical-graph-v3.md`, `adr-003.md`, `adr-004.md` |
| GCHM / GCHM-PNA v2 | `gchm_analysis/`, `clinical_graph_v2/gchm_v2.py`, `comparison/standardized/gchm_v2_protocol/` | `gchm.md` |
| ProtGNN | `protgnn_analysis/`, `clinical_graph_v2/methods/protgnn.py` | `protgnn.md` |
| GSAT | `gsat_analysis/`, `clinical_graph_v2/methods/gsat.py` | `gsat.md` |
| GraphCare | `graphcare_analysis/`, `clinical_graph_v2/methods/graphcare.py` | `graphcare.md` |
| XGBoost control | `comparison/standardized/xgboost_native_baseline.py`, `clinical_graph_v2/tabular_control.py` | `xgboost.md` |
| Native identical-input benchmark (30-class) | `comparison/standardized/train_identical.py` | `native-benchmark.md`, `adr-001.md` |
| Interaction PNA (opt-in) | `pna_analysis/` | `pna.md` |
| Shared contracts, split, data prep | `shared/lib/`, `shared/data_prep/`, `comparison/canonical_split.json` | `shared.md` |
| Tests | `tests/` | `tests.md` |
| Vendored third-party code | `external/` (keep upstream layout) | `external.md` |

Context docs are in `.claude/context/`. They are read-only links to the ProjectOS vault. Read the relevant one before changing a subsystem.

## Commands (run from repo root)

```bash
# Read-only plans / wiring
python3 -m comparison.standardized.train_identical --output comparison/standardized/native_runs/<new_dir> --dry-run
python3 -m comparison.standardized.clinical_graph_v2.mechanism_check        # training-free mechanism checks
python3 -m comparison.standardized.gchm_v2_protocol.protocol               # plan + ETA only
python3 -m comparison.standardized.gchm_v2_protocol.protocol --status

# Only inside an explicitly opened testing phase
python3 -m pytest tests -q        # explicit path avoids vendored test collections
```

Training commands (`clinical_graph_v2.train`, `tabular_control`, `--execute` modes) are listed in `comparison/standardized/clinical_graph_v2/README.md`. Run them only after approval.

## Conventions

- New run outputs go to a new, dated directory under `comparison/standardized/` (existing examples: `clinical_runs_v3_*_YYYYMMDD`).
- Every run writes `binding.json` / `result.json` with hashes. Never claim equal compute from equal epochs or similar parameter counts.
- Method-native CLI options are namespaced (`--protgnn-*`, `--gsat-*`, `--graphcare-*`). `--conv` is valid only with `--method clinical_gnn`.
- Use `.venv-graphcare/bin/python` for GraphCare. `train_identical` selects it automatically; for other entry points, check the runner.

## Project memory

- Decisions, work logs and current status: ProjectOS vault `10-Projects/medgnn/MedGNN.md`. It is the source of truth; this file only distills the currently accepted rules.
- When a durable decision changes, update the ADR in ProjectOS first, then the matching line here.
- This file is the only agent instruction file, and every agent follows all of it. Do not create other instruction files or agent-specific sections.

## Context tooling

- **Before relying on a context doc,** run `python3 .claude/scripts/context-drift-check.py`.
  - `HIGH`: update the doc, or tell the user, before relying on it.
  - `MEDIUM`: mention it to the user.
  - Dismiss with `python3 .claude/scripts/context-drift-check.py --dismiss`.
- **Finding code for a task:** use the "Where things live" table and the doc in `.claude/context/` before a broad search.
  - The subsystem→file registry is `SUBSYSTEMS` in `.claude/mcp/context_retrieval_mcp/server.py`.
  - The same registry is exposed as an optional MCP server (`context-retrieval`) with tools `find_relevant_context`, `get_files_for_subsystem`, `search_context_documents` and `list_subsystems`.
- **Doc/agent templates** (`.claude/agents/*/AGENT.md`):
  - `context-factory` defines how to write a new context doc. Put the canonical note in ProjectOS and symlink it into `.claude/context/`.
  - Use `agent-factory` only after a domain shows repeated mistakes.
- **After structural changes:**
  - Update the ProjectOS note or ADR.
  - If files moved, update `SUBSYSTEMS`.
  - If an accepted rule changed, update this file.
