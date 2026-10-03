# MedGNN

Self-explainable GNN diagnosis benchmark on MIMIC-IV ED patient graphs: ProtGNN, GSAT, GraphCare, GCHM-PNA, CEI-GNN and an XGBoost control, compared under fixed data contracts.

## Tech stack

| Item | Value |
|---|---|
| Language | Python (system `python3`; repo recommends 3.10/3.11) |
| ML | PyTorch 2.2–2.8, PyTorch Geometric 2.5–2.6, XGBoost 2.x, scikit-learn |

## Non-negotiable rules

| Rule | Why / source |
|---|---|
| NEVER load or evaluate the held-out test fold; select checkpoints on validation macro-F1 only (or on the ADR-008 dev split only when the user explicitly selects that proposed protocol; or on the patient-disjoint TRAIN dev split only for the CEI-GNN studies the user approved under accepted ADR-010 / ADR-013 / ADR-014, whose ADR-012 plan they supersede; never the held-out fold). | `.claude/context/adr-002.md` |
| NEVER write/run tests unless the user explicitly opens a "testing phase". Parse-only checks are fine. The user-approved source/test archival does NOT authorize running tests. | `.claude/context/adr-005.md` |
| NEVER start real preprocessing, cache builds or training without explicit user approval. Use `--dry-run` / plan modes. | `ProjectOS medgnn/Reference/STRUCTURE.md` |
| NEVER overwrite a filled output/artifact directory; write to a new path. Runners refuse by design. | `ProjectOS medgnn/Reference/clinical-graph-v2-runbook.md` |
| NEVER upgrade historical graphs by editing metadata; rebuild into a new path. | `.claude/context/adr-004.md` |
| NEVER regenerate `comparison/canonical_split.json`. | `ProjectOS medgnn/Reference/STRUCTURE.md` |
| Compared methods MUST share artifact, class set, train/validation sample-ID hashes, preprocessing contract and seed. | `.claude/context/adr-006.md` |
| Do not relocate `data/` or `external/` as a cleanup side effect. | `ProjectOS medgnn/Reference/STRUCTURE.md` |
| Preserve the permanent historical SHA256 guard for `shared/data_prep/merge_ed.py`; never silently rewrite its pinned bytes or provenance. Do not import, run or fix it during cleanup; any correction requires a separately authorized, versioned production job. | `ProjectOS medgnn/Reports/max6-top10-cleanup.md` |
| `shared/**`, `comparison/canonical_split.json`, `shared/data_prep/merge_ed.py`, retained rebuild sources and `icd_mapping.py` stay byte-identical. Code moved to top-level method folders on 2026-10-03 without compatibility shims; the only shim left is `comparison/standardized/clinical_graph_v2/repair_metadata.py` (two frozen label files import it). Historical run source bindings no longer match; new runs need new output dirs. Work locally; no push without approval. | `ProjectOS medgnn/Reports/max6-top10-cleanup.md`, `ProjectOS medgnn/Reports/structure-cleanup-2026-10-03.md` |
| Do not delete held data/results/source snapshots. Publication requires a separate privacy decision: historical Git contains patient-derived payloads. | `ProjectOS medgnn/Reports/max6-top10-cleanup.md` |
| Never keep notes in this repo (no `docs/`, READMEs, specs, reports, `STRUCTURE.md`); only `AGENTS.md` and `README.md`. Notes go to ProjectOS `10-Projects/medgnn/` (history, reports, reference) and Jev-Mem `Notes/MedGNN/` (current rules). | User rule 2026-10-03 |
| Report results with task identity (cohort, class set, sample size, seed). Single-seed gaps are not method superiority. | `.claude/context/adr-006.md` |

## Current comparison contract (max6 / Top-10)

- Patients with more than 6 total visits are excluded.
- Classes: the 10 most frequent diagnoses by TRAIN-fold frequency; other labels are dropped, never merged into `other`.
- Sample comparison: the same 10,000 train IDs (seed `1234`) and the full 4,254-row validation fold.
- Results from this task are NOT comparable with the retired 30-class native benchmark (331-slot star input). Its code is frozen legacy, retired with its tests and exporter; see `ProjectOS medgnn/Reports/max6-top10-cleanup.md`. Top-10 selection still needs the original class order in `comparison/canonical_split.json`.
- Full details: `.claude/context/adr-006.md`. Proposed, not accepted: `adr-007.md` (bidirectional edges, GCHM-PNA v2) and `adr-008.md` (dev-selected equal-budget protocol; not enabled).

## Where things live

| Area | Path | Context doc |
|---|---|---|
| Multi-visit clinical graph v2/v3 core (build, train, audit, schema constants, paths) | `core/` | `clinical-graph-v3.md`, `adr-003.md`, `adr-004.md` |
| Shared method code (registry, base) | `core/registry.py`, `core/method_base.py` (plugins `plugin_*.py` are discovered only inside the method folders) | `clinical-graph-v3.md`, `adr-003.md` |
| Source binding | `core/paths.py` (`REPO_ROOT`, `CODE_ROOTS`: method folders plus every `comparison/<dir>/` with `__init__.py`, never `comparison/standardized/`), `core/contracts.py` (`code_source_hashes()`) | `adr-006.md` |
| ProtGNN | `protgnn/` (`adapter.py`) | `ProjectOS medgnn/Reference/method-protgnn.md` |
| CEI-GNN | `cei/` (`cei_gnn*.py`, `plugin_cei_gnn*.py`, `studies/`, `studies/cei_v3_ext/`) | `ProjectOS medgnn/Reports/cei/cei-v3-delivery-2026-10-01.md` |
| GSAT | `gsat/` (`adapter.py`) | `gsat.md` |
| GraphCare | `graphcare/` (`adapter.py` native) | `graphcare.md` |
| GCHM / GCHM-PNA v2 | `gchm_pna/` (`gchm_v2.py`, `gchm_v3.py`, `protocol/`; state data stays in `comparison/standardized/gchm_v2_protocol/state/`) | `gchm.md` |
| XGBoost control | `xgboost_control/` (`tabular_control.py`; never name it `xgboost`, it would shadow the library) | `xgboost.md` |
| Comparisons (single folder) | `comparison/all/` (all methods together), `comparison/<a>_vs_<b>/` per pair (now `cei_vs_protgnn/`, `cei_vs_xgboost/`); `comparison/` and `comparison/standardized/` stay namespace packages (no `__init__.py`) | `adr-006.md` |
| Inputs and run results (not moved) | `comparison/standardized/{event_inputs,native_inputs,clinical_runs_*}/`, `explanation_subjects.json` | `ProjectOS medgnn/Reference/clinical-graph-v2-runbook.md` |
| Kept label/input chain | `comparison/standardized/{icd_mapping.py,event_graph_v1/,enriched_input_v1/spec.py,event_graph_gchm_xgb_v1/{labels,local_labels_v2}.py}` | `ProjectOS medgnn/Reports/max6-top10-cleanup.md` |
| Shared contracts, split, data prep | `shared/lib/`, `shared/data_prep/`, `comparison/canonical_split.json` | `shared.md` |
| Retired legacy (frozen) | former 30-class method dirs, native/star/cooccur runners, dependent tests, legacy exporter, `protgnn_analysis/`, `graphcare_analysis/`, old runs and viewer | Code and outputs are archived in `/Users/necatifurkancolak/AI-Workplace/Artifacts/MedGNN/repo-archive-20261003/` and in Git history; see `ProjectOS medgnn/Reports/max6-top10-cleanup.md`. Never run for new work (test-fold access). |
| Tests | `tests/` (only what the retirement manifest left) | `tests.md` |
| Vendored third-party code | `external/` (keep upstream layout; GraphXAI only, used by CEI) | `external.md` |

Context docs are in `.claude/context/`. They are read-only links to the ProjectOS vault. Read the relevant one before changing a subsystem.

## Commands (run from repo root)

```bash
# Read-only plans / wiring
python3 -m core.mechanism_check                 # training-free mechanism checks
python3 -m gchm_pna.protocol.protocol           # plan + ETA only
python3 -m gchm_pna.protocol.protocol --status

# Only inside an explicitly opened testing phase
python3 -m pytest tests -q        # explicit path avoids vendored test collections
```

Training commands (`core.train`, `xgboost_control.tabular_control`, `--execute` modes) are listed in `ProjectOS medgnn/Reference/clinical-graph-v2-runbook.md`. Run them only after approval.

## Conventions

- New run outputs go to a new, dated directory under `comparison/standardized/` (existing examples: `clinical_runs_v3_*_YYYYMMDD`).
- Every run writes `binding.json` / `result.json` with hashes. Never claim equal compute from equal epochs or similar parameter counts.
- Method-native CLI options are namespaced (`--protgnn-*`, `--gsat-*`, `--graphcare-*`). `--conv` is valid only with `--method clinical_gnn`.
- Adding a method, study or comparison: a new method is a new top-level folder `<name>/` with `__init__.py`, `adapter.py` or `plugin_<name>.py` and `studies/`; add it to `METHOD_FOLDERS` in `core/registry.py` (plugin discovery) and to `_METHOD_ROOTS` in `core/paths.py` (source binding). A study lives under the owning method's `studies/` with a new dated output directory; a control is its own method folder. A cross-method comparison goes to `comparison/all/` or a new `comparison/<a>_vs_<b>/` with an `__init__.py` (bound automatically). Shared code goes into `core/`. No README or docs in these folders.

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
