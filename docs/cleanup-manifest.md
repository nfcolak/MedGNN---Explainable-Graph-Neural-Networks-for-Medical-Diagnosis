# Cleanup manifest

Executed 20260913T132941+0200 in `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN`. User-authorized clutter cleanup only; no scientific code or tests changed.

## Reversible moves to macOS Trash

Two ignored, untracked files (313,551 bytes) moved, not permanently deleted. Git history contained no entries for either candidate. Reference search across first-party text, including thesis and project vault documentation, found only `.gitignore`, `STRUCTURE.md` and the historical `docs/usability-verification.md` inventory. Data, external dependencies, environments and output trees were excluded from traversal.

- Original: `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/Master Project_Template..zip`
  - Trash / restoration source: `/Users/necatifurkancolak/.Trash/MedGNN-20260913T132941+0200-a6bb7473-Master Project_Template..zip`
  - Reason: Redundant archive: all 25 non-metadata payload files SHA-256-identical to the retained extracted template; no runtime references found.
  - Size: 307,404 bytes; SHA-256: `aa4519a624a20f23dc8f2acc84a17a64f75495d72cb187879f04b66ce28b5af4`.
- Original: `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/job_postings_2026-07-11.xlsx`
  - Trash / restoration source: `/Users/necatifurkancolak/.Trash/MedGNN-20260913T132941+0200-bb49bd0d-job_postings_2026-07-11.xlsx`
  - Reason: Unrelated recruiting spreadsheet (postings sheet, company/job links; openpyxl metadata dated 2026-07-11); ignored and untracked, no runtime references found.
  - Size: 6,147 bytes; SHA-256: `7b52424ef1f992300cb9d5bc2dd284161237095ef7f3b147c7c7118e2597326d`.

Restore by moving each exact Trash path back to its original path, only if that original path does not already exist. Do not empty Trash until satisfied with the cleanup.

## Regenerable items permanently removed

No tracked files were removed. These contain no user documents. Paths below are absolute; directory rows include all their contents.

| Path | Reason | Files | Bytes |
|---|---|---:|---:|
| `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/.pytest_cache` | Regenerable pytest state | 5 | 45359 |
| `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/.DS_Store` | Finder metadata | 1 | 18436 |
| `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/Master Project_Template./.DS_Store` | Finder metadata | 1 | 6148 |
| `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/comparison/.DS_Store` | Finder metadata | 1 | 6148 |
| `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/visualizer/.DS_Store` | Finder metadata | 1 | 10244 |

Total: 5 removed paths, 9 files, 86,335 bytes. No eligible `__pycache__` directories were present. Regenerable metadata has no restoration copy; no cache regeneration was run.

## Retained deliberately

- `Master Project_Template./`: contains a project-specific thesis title and substantive source; archive duplication does not establish duplication against the active thesis. Retained all 25 non-metadata files; removed only its `.DS_Store`.
- `data.zip` (2,867,475,165 bytes), all `data/` content and symlinks: archive duplication not proven; untouched.
- `thesis/`, `thesis_proposals/`, `docs-vault/`: protected, no cleanup inside.
- `.venv-graphcare/`, all other detected environments, `external/`, `node_modules/`, `.git/`: excluded from cleanup traversal. Symlinks never followed.
- Results, checkpoints, outputs and retry attempts: excluded. Legacy scripts, `TODO.txt`, `.claude/`, `.superpowers/` and all scientific code/tests retained; no unsupported assumption of obsolescence.

## Verification

- Original moved paths absent; exact Trash destinations present; SHA-256 matched before/after each move.
- All removed paths confirmed absent immediately after removal.
- 1,029 pre-existing first-party/non-cache and tracked files verified byte-identical after cleanup, including existing uncommitted code/tests. Git status identical before/after physical cleanup.
- `git diff --check`: exit 0, no output after physical cleanup.
- Documentation-only follow-up: this manifest added; `STRUCTURE.md` updated to reflect completed cleanup instead of the previous retention statement. Historical usability report left unchanged.
- No tests or training run: no code changes; running pytest would recreate deliberately removed state. No commits, pushes, installs, data processing or cache generation.
