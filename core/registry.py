"""Clinical graph method-adapter registry.

Tasks adding a method should register its adapter here; shared loading and
validation remain in the clinical_graph_v2 runner.
"""
from __future__ import annotations

import importlib
import pkgutil

from .method_base import ClinicalMethodAdapter, MethodOutput
from graphcare.adapter import GraphCareAdapter
from gsat.adapter import GSATAdapter
from protgnn.adapter import ProtGNNAdapter

METHOD_REGISTRY: dict[str, type[ClinicalMethodAdapter]] = {
    "graphcare": GraphCareAdapter,
    "gsat": GSATAdapter,
    "protgnn": ProtGNNAdapter,
}
CORE_METHODS = frozenset(METHOD_REGISTRY)


def register_plugins(modules, base_registry) -> dict:
    """Merge each module's `REGISTER = {name: AdapterClass}` into a copy of the registry.

    A plugin may not replace a registered method, so core arms stay byte-identical.
    """
    registry = dict(base_registry)
    for module in modules:
        register = getattr(module, "REGISTER", None)
        if not isinstance(register, dict):
            raise ValueError(f"plugin {module.__name__} must define REGISTER = {{name: class}}")
        for name, adapter_type in register.items():
            if name in registry:
                raise ValueError(f"method {name!r} is already registered")
            if not (isinstance(adapter_type, type)
                    and issubclass(adapter_type, ClinicalMethodAdapter)):
                raise ValueError(f"plugin method {name!r} is not a ClinicalMethodAdapter")
            registry[name] = adapter_type
    return registry


METHOD_FOLDERS = ("cei", "gchm_pna", "graphcare", "gsat", "protgnn")


def _discover_plugins():
    """Discover `plugin_*.py` modules inside the method folders only."""
    modules = []
    for method in METHOD_FOLDERS:
        package = importlib.import_module(method)
        for plugin in sorted(pkgutil.iter_modules(package.__path__), key=lambda info: info.name):
            if plugin.name.startswith("plugin_"):
                modules.append(importlib.import_module(f"{method}.{plugin.name}"))
    return register_plugins(modules, base_registry={})


class _PluginSet:
    """All discovered plugins as one module-like REGISTER, checked against the core."""
    __name__ = "discovered_plugins"

    def __init__(self, register):
        self.REGISTER = register


METHOD_REGISTRY = register_plugins([_PluginSet(_discover_plugins())], base_registry=METHOD_REGISTRY)


def build_method(name, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args) -> ClinicalMethodAdapter:
    """Construct one registered clinical adapter using the shared dimensions."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("method name must be a nonempty string")
    key = name.strip().lower()
    adapter_type = METHOD_REGISTRY.get(key)
    if adapter_type is None:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"unknown clinical method {name!r}; registered methods: {available}")
    return adapter_type(
        num_tokens=num_tokens,
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_classes=num_classes,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        token_dim=token_dim,
        num_triples=num_triples,
        args=args,
    )


__all__ = [
    "ClinicalMethodAdapter",
    "MethodOutput",
    "METHOD_REGISTRY",
    "GraphCareAdapter",
    "GSATAdapter",
    "ProtGNNAdapter",
    "build_method",
]
