"""Stable roots for the repository (source binding and repo-relative paths)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_METHOD_ROOTS = ('core', 'protgnn', 'cei', 'gsat', 'graphcare', 'gchm_pna')
_COMPARISON_DIR = REPO_ROOT / 'comparison'


def _comparison_roots():
    if not _COMPARISON_DIR.is_dir():
        return ()
    return tuple(f'comparison/{path.name}' for path in sorted(_COMPARISON_DIR.iterdir())
                 if path.is_dir() and path.name != 'standardized' and (path / '__init__.py').is_file())


CODE_ROOTS = _METHOD_ROOTS + _comparison_roots()
