# MedGNN

Self-explainable GNN diagnosis benchmark on MIMIC-IV ED patient graphs: ProtGNN, GSAT, GraphCare, GCHM-PNA, CEI-GNN and an XGBoost control, compared under fixed data contracts.

## Tech stack

| Item | Value |
|---|---|
| Language | Python (system `python3`; repo recommends 3.10/3.11) |
| ML | PyTorch 2.2–2.8, PyTorch Geometric 2.5–2.6, XGBoost 2.x, scikit-learn |
| Legacy GraphCare runtime | `.venv-graphcare/` is retained legacy; never merge into the main env or change it. The current clinical GraphCare adapter (`clinical_graph_v2/methods/graphcare.py`) is native code and does not use it |

## Non-negotiable rules

| Rule | Why / source |
|---|---|
| NEVER load or evaluate the held-out test fold; select checkpoints on validation macro-F1 only (or on the ADR-008 dev split only when the user explicitly selects that proposed protocol; or on the patient-disjoint TRAIN dev split only for the CEI-GNN studies the user approved under accepted ADR-010 / ADR-013 / ADR-014, whose ADR-012 plan they supersede; never the held-out fold). | `.claude/context/adr-002.md` |
| NEVER write/run tests unless the user explicitly opens a "testing phase". Parse-only checks are fine. The user-approved source/test archival does NOT authorize running tests. | `.claude/context/adr-005.md` |
| NEVER start real preprocessing, cache builds or training without explicit user approval. Use `--dry-run` / plan modes. | `STRUCTURE.md` |
| NEVER overwrite a filled output/artifact directory; write to a new path. Runners refuse by design. | `comparison/standardized/clinical_graph_v2/README.md` |
| NEVER upgrade historical graphs by editing metadata; rebuild into a new path. | `.claude/context/adr-004.md` |
| NEVER regenerate `comparison/canonical_split.json`. | `STRUCTURE.md` |
| Compared methods MUST share artifact, class set, train/validation sample-ID hashes, preprocessing contract and seed. | `.claude/context/adr-006.md` |
| Do not relocate `data/`, `external/` or `docs-vault/` as a cleanup side effect. | `STRUCTURE.md` |
| Preserve the permanent historical SHA256 guard for `shared/data_prep/merge_ed.py`; never silently rewrite its pinned bytes or provenance. Do not import, run or fix it during cleanup; any correction requires a separately authorized, versioned production job. | `docs/max6-top10-cleanup.md` |
| `shared/**`, `comparison/canonical_split.json`, `shared/data_prep/merge_ed.py`, retained rebuild sources and `icd_mapping.py` stay byte-identical. `clinical_graph_v2/**` may change (restructured 2026-10-03); keep a compatibility shim at every old module path. Work locally; no push without approval. | `docs/max6-top10-cleanup.md`, `docs/structure-cleanup-2026-10-03.md` |
| Do not delete held data/results/source snapshots. Publication requires a separate privacy decision: historical Git contains patient-derived payloads. | `docs/max6-top10-cleanup.md` |
| Report results with task identity (cohort, class set, sample size, seed). Single-seed gaps are not method superiority. | `.claude/context/adr-006.md` |

## Current comparison contract (max6 / Top-10)

- Patients with more than 6 total visits are excluded.
- Classes: the 10 most frequent diagnoses by TRAIN-fold frequency; other labels are dropped, never merged into `other`.
- Sample comparison: the same 10,000 train IDs (seed `1234`) and the full 4,254-row validation fold.
- Results from this task are NOT comparable with the retired 30-class native benchmark (331-slot star input). Its code is frozen legacy, retired with its tests and exporter; see `docs/max6-top10-cleanup.md`. Top-10 selection still needs the original class order in `comparison/canonical_split.json`.
- Full details: `.claude/context/adr-006.md`. Proposed, not accepted: `adr-007.md` (bidirectional edges, GCHM-PNA v2) and `adr-008.md` (dev-selected equal-budget protocol; not enabled).

## Where things live

| Area | Path | Context doc |
|---|---|---|
| Multi-visit clinical graph v2/v3 core (build, train, audit, ...) | `comparison/standardized/clinical_graph_v2/core/` (old paths are shims) | `clinical-graph-v3.md`, `adr-003.md`, `adr-004.md` |
| Method adapters and plugins | `clinical_graph_v2/methods/` | `clinical-graph-v3.md`, `adr-003.md` |
| Controls | `clinical_graph_v2/controls/` | `xgboost.md` |
| GCHM / GCHM-PNA v2 | `clinical_graph_v2/core/gchm_v2.py`, `comparison/standardized/gchm_v2_protocol/` | `gchm.md` |
| ProtGNN | `clinical_graph_v2/methods/protgnn.py` | `protgnn.md` |
| GSAT | `clinical_graph_v2/methods/gsat.py` | `gsat.md` |
| GraphCare | `clinical_graph_v2/methods/graphcare.py` (native adapter) | `graphcare.md` |
| CEI-GNN | `clinical_graph_v2/methods/cei_gnn*.py`, `clinical_graph_v2/studies/cei/` (studies, `cei_v3_ext/`) | `docs/cei-v3-delivery-2026-10-01.md` |
| XGBoost control | `clinical_graph_v2/controls/tabular_control.py` | `xgboost.md` |
| Kept label/input chain | `comparison/standardized/{icd_mapping.py,event_graph_v1/,enriched_input_v1/spec.py,event_graph_gchm_xgb_v1/{labels,local_labels_v2}.py}` | `docs/max6-top10-cleanup.md` |
| Shared contracts, split, data prep | `shared/lib/`, `shared/data_prep/`, `comparison/canonical_split.json` | `shared.md` |
| Retired legacy (frozen) | former 30-class method dirs, native/star/cooccur runners, dependent tests, legacy exporter | Recoverable from Git refs/archives listed in `docs/max6-top10-cleanup.md`; some old folders remain for protected outputs. Never run for new work (test-fold access). |
| Tests | `tests/` (only what the retirement manifest left) | `tests.md` |
| Vendored third-party code | `external/` (keep upstream layout; GraphXAI used by CEI, GraphCare legacy) | `external.md` |

Context docs are in `.claude/context/`. They are read-only links to the ProjectOS vault. Read the relevant one before changing a subsystem.

## Commands (run from repo root)

```bash
# Read-only plans / wiring
python3 -m comparison.standardized.clinical_graph_v2.core.mechanism_check        # training-free mechanism checks
python3 -m comparison.standardized.gchm_v2_protocol.protocol               # plan + ETA only
python3 -m comparison.standardized.gchm_v2_protocol.protocol --status

# Only inside an explicitly opened testing phase
python3 -m pytest tests -q        # explicit path avoids vendored test collections
```

Training commands (`clinical_graph_v2.core.train`, `controls.tabular_control`, `--execute` modes) are listed in `comparison/standardized/clinical_graph_v2/README.md`. Run them only after approval.

## Conventions

- New run outputs go to a new, dated directory under `comparison/standardized/` (existing examples: `clinical_runs_v3_*_YYYYMMDD`).
- Every run writes `binding.json` / `result.json` with hashes. Never claim equal compute from equal epochs or similar parameter counts.
- Method-native CLI options are namespaced (`--protgnn-*`, `--gsat-*`, `--graphcare-*`). `--conv` is valid only with `--method clinical_gnn`.
- The clinical GraphCare adapter runs in the main environment; `.venv-graphcare/` is for retained legacy code only.
- Adding a method, study or control: follow "Adding things" in `comparison/standardized/clinical_graph_v2/README.md`. New code goes into `methods/`, `studies/<name>/` or `controls/`; never into the old flat shim files.

## Project memory

- Decisions, work logs and current status: ProjectOS vault `10-Projects/medgnn/MedGNN.md`. It is the source of truth; this file only distills the currently accepted rules.
- When a durable decision changes, update the ADR in ProjectOS first, then the matching line here.
- This file is the only agent instruction file, and every agent follows all of it. Do not create other instruction files or agent-specific sections.

## Context tooling

- The `.claude/` tree (context docs, scripts, MCP) is local-only (gitignored symlinks into ProjectOS) and absent in clones and agent worktrees.
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
