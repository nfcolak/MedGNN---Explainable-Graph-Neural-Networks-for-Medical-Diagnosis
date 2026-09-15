# Working-tree cleanup

## Outcome

Moved **207 files / 2,916,015,958 bytes** out of the repository across **31 selected paths**, preserving relative paths in:

`/Users/necatifurkancolak/.Trash/MedGNN-cleanup-20260914T152126+0200-ba819220`

This is substantive source/document/experiment retirement, not cache-only cleanup. No source code was rewritten. `README.md` and `STRUCTURE.md` were minimally updated. The Git index/history, dirty active code, scientific data and current outputs were not reset or committed.

- [Per-file source/destination, size, reason and SHA-256 manifest](cleanup-working-tree.json)
- [Final verification and exact command logs](cleanup-working-tree-verification.json)
- Full pre-cleanup snapshot: `/Users/necatifurkancolak/.Trash/MedGNN-cleanup-20260914T152126+0200-ba819220/_audit/before.json`

## Removed from the working tree

- `Master Project_Template./`, obsolete LOS-era `TODO.txt`, literature PDFs/paper spreadsheets, unused figures and the obsolete two-method architecture document.
- Earlier canonical two-method/ablation scripts and their result folders under `comparison/`; it now contains only `build_split.py`, `canonical_split.json` and `standardized/`.
- Broken `protgnn_analysis/scripts/train_gnns.sh` (called a nonexistent trainer) and unused `explainability/graphxai_integration.py` (incorrect root and random-weight fallback).
- Old prototype/LOS HPO sweeps and projection log; completed benchmark agent task packets and implementation plans. The benchmark design specification remains.
- Stray text document in GraphCare outputs.
- `data.zip`: all **31 non-metadata payload members** were SHA-256-identical to the extracted files. The extracted files were rehashed after cleanup and are unchanged. Only the redundant compressed copy was moved.

`thesis/`, `thesis_proposals/`, the template ZIP and unrelated job spreadsheet were **already absent on entry**; their tracked deletions were present in the initial Git status. This pass does not claim those earlier moves as its own. Earlier cleanup reports remain as historical audit records, not current inventory.

## KEEP closure

- Standardized orchestration → benchmark/cache/cohort/explanation/summarizer CLIs → all three method packages and shared contracts/graph builders.
- Active ProtGNN helper dependencies (`archive_results`, `summarize_all_test`), checkpoint/report tools and compatibility scripts explicitly exercised by maintained tests. `baselines/disease_baseline.py` remains because the maintained confusion-analysis CLI imports it.
- Canonical split and its construction source; all extracted raw/merged data and graph caches; model checkpoints and default legacy result/report paths still consumed by retained tools.
- **All** `comparison/standardized/` scientific evidence, including current 13–14 September common-input, data-quality, performance/zero-concept audits, source snapshots, attempts and **comparison/standardized/graphxai_500_20260914T144701** (3594 files hash-verified unchanged).
- GraphXAI/GraphCare vendor trees, existing GraphCare environment, viewer dependencies and existing browser graph data. No dependency/environment/raw-data recursive crawl was used.
- Maintained tests, dependency specifications, operating runbooks, current scientific reports and the project-owned `docs-vault/` knowledge base.

## Verification

- Every relocated file is absent at source and SHA-256-identical at its Trash destination; zero relocation mismatches.
- **4854 retained files** verified unchanged, excluding the two intentional navigation edits. All 31 extracted archive payload files verified unchanged separately.
- Checked symlinks: 3; broken: 0. Environment executables and dependencies remain operational.
- Main maintained suite plus current data-quality tests: **405 passed, 2 skipped**.
- GraphCare isolated environment tests: **8 passed**; separate real-model smoke: `OK: logits (2, 30)`, backward ran.
- `run_all --help/--dry-run`, benchmark dry-run (**18 unique cells**), explanation/cache dry-runs and all three trainer help checks: exit 0.
- Viewer `npm run build`: TypeScript + Vite production build passed. Existing large-chunk warning remains.
- `git diff --check`: exit 0. No retraining, production preprocessing or explanation execution.
- Initial combined main-environment collection incorrectly included GraphCare-specific tests: 8 failed because main Python lacks its isolated dependencies. Rerunning those tests in `.venv-graphcare` passed; no model changes were made to hide the failure.

Harmless existing warnings: main Python LibreSSL/urllib3 and Matplotlib deprecations. Dry-runs do not certify an executable cohort/checkpoint combination: the existing default cohort-size caveat in the benchmark runbook still applies.

## Restoration

Move an exact `destination` from the JSON manifest back to its `source` only if that source is absent. Folder moves may restore all contained files at once. Do not empty Trash until satisfied. All retired scientific/source/document material is reversible; only 1 untracked Finder metadata files were permanently removed.

## Final root

```text
.claude
.git
.gitignore
.hermes
.superpowers
.venv-graphcare
README.md
STRUCTURE.md
baselines
comparison
data
docs
docs-vault
environment.yml
external
graphcare_analysis
gsat_analysis
protgnn_analysis
requirements-lock.txt
requirements.txt
shared
tests
visualizer
```
