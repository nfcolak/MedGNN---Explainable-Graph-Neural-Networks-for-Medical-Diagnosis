"""Stable roots for the repository (source binding and repo-relative paths)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOTS = ('core', 'protgnn', 'cei', 'gsat', 'graphcare', 'gchm_pna', 'xgboost_control', 'comparisons')
