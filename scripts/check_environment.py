"""Check whether the local environment can run the project pipeline."""

from __future__ import annotations

import importlib
import sys


REQUIRED_IMPORTS = [
    ("torch", "PyTorch"),
    ("torch_geometric", "PyTorch Geometric"),
    ("numpy", "NumPy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("scipy", "SciPy"),
    ("networkx", "NetworkX"),
    ("matplotlib", "Matplotlib"),
    ("tqdm", "tqdm"),
    ("optuna", "Optuna"),
    ("xgboost", "XGBoost"),
    ("rdkit", "RDKit"),
]

OPTIONAL_IMPORTS = [
    ("openpyxl", "openpyxl, for native .xlsx writing"),
]

PROJECT_IMPORTS = [
    ("configs.config", "project config"),
    ("prot_gnn.models", "ProtGNN models"),
    ("prot_gnn.load_dataset", "dataset loaders"),
    ("graphxai.explainers.grad", "GraphXAI GradExplainer"),
    ("graphxai.explainers.integrated_grad", "GraphXAI IntegratedGradExplainer"),
    ("graphxai.explainers.gnn_explainer", "GraphXAI GNNExplainer"),
]


def _version(module):
    return getattr(module, "__version__", "unknown")


def _check(imports, *, optional=False):
    ok = True
    for module_name, label in imports:
        try:
            module = importlib.import_module(module_name)
            print(f"[OK] {label}: {_version(module)}")
        except Exception as exc:
            ok = False
            level = "WARN" if optional else "FAIL"
            print(f"[{level}] {label}: {exc}")
    return ok or optional


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print("\nCore dependencies")
    deps_ok = _check(REQUIRED_IMPORTS)

    print("\nOptional dependencies")
    _check(OPTIONAL_IMPORTS, optional=True)

    print("\nProject imports")
    project_ok = _check(PROJECT_IMPORTS)

    if deps_ok and project_ok:
        print("\nEnvironment check passed.")
        return 0

    print("\nEnvironment check failed.")
    print("Install dependencies with environment.yml or requirements-lock.txt,")
    print("and run with PYTHONPATH=src:external/GraphXAI-main:.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
