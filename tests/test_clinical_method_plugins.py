"""Plugin interface for new clinical methods (one file per method, no runner edits).

A plugin module `methods/plugin_<name>.py` defines `REGISTER = {name: AdapterClass}`.
Each adapter class declares `runner_defaults` (the common runner profile) and reads
its own settings from `--method-option key=value` via `method_option`. The runner
consults the adapter for gradient clipping and the early-stopping gate.
Synthetic in-memory tests only; no clinical artifacts are read or trained.
"""
import types
from argparse import Namespace

import pytest
import torch
import torch.nn as nn

from core import registry as methods, train
from core import method_base as base


CORE = {"graphcare", "gsat", "protgnn"}


def require(module, name):
    value = getattr(module, name, None)
    if value is None:
        pytest.fail(f"plugin interface missing: {module.__name__}.{name}")
    return value


class FakeAdapter(base.ClinicalMethodAdapter):
    runner_defaults = dict(hidden=16, layers=2, dropout=0.2, lr=2e-3, weight_decay=0.0,
                           batch_size=8, epochs=5, patience=3, min_delta=0.0)
    grad_clip_value = 0.5

    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3))

    def forward(self, batch, *, epoch):
        raise NotImplementedError

    def on_epoch_start(self, epoch, train_loader):
        pass

    def optimizer_groups(self, args):
        return [{"params": list(self.parameters())}]

    def run_config(self):
        return {"method": "fake_plugin", "architecture": {"parameter_count": 3}}

    def early_stopping_start(self):
        return 4


def plugin_module(**register):
    module = types.ModuleType("fake_plugin_module")
    module.REGISTER = register
    return module


def test_plugins_are_merged_into_the_registry_without_touching_core_methods():
    register_plugins = require(methods, "register_plugins")
    registry = register_plugins([plugin_module(fake_plugin=FakeAdapter)], base_registry={})
    assert registry == {"fake_plugin": FakeAdapter}
    assert CORE <= set(methods.METHOD_REGISTRY)
    with pytest.raises(ValueError, match="already registered"):
        register_plugins([plugin_module(protgnn=FakeAdapter)],
                         base_registry=dict(methods.METHOD_REGISTRY))
    with pytest.raises(ValueError, match="REGISTER"):
        register_plugins([types.ModuleType("no_register")], base_registry={})


def test_method_option_casts_validates_and_rejects_unknown_keys():
    method_option = require(base, "method_option")
    reject_unknown = require(base, "reject_unknown_options")
    args = Namespace(method_options={"temperature": "0.5", "mode": "sum"})
    assert method_option(args, "temperature", 1.0, float, minimum=0.0) == 0.5
    assert method_option(args, "missing", 7, int) == 7
    assert method_option(Namespace(), "missing", 7, int) == 7
    assert method_option(args, "mode", "mean", str, choices=("mean", "sum")) == "sum"
    with pytest.raises(ValueError, match="temperature"):
        method_option(args, "temperature", 1.0, float, maximum=0.1)
    with pytest.raises(ValueError, match="mode"):
        method_option(args, "mode", "mean", str, choices=("mean", "max"))
    reject_unknown(args, ("temperature", "mode"))
    with pytest.raises(ValueError, match="unknown method option"):
        reject_unknown(args, ("temperature",))


def base_argv(tmp_path, *extra):
    return ["--artifact", str(tmp_path / "a"), "--targets", str(tmp_path / "t"),
            "--output", str(tmp_path / "o"), *extra]


def test_runner_accepts_a_registered_plugin_with_its_defaults_and_options(monkeypatch, tmp_path):
    monkeypatch.setitem(methods.METHOD_REGISTRY, "fake_plugin", FakeAdapter)
    parser = train.parser()
    try:
        parsed = parser.parse_args(base_argv(
            tmp_path, "--method", "fake_plugin", "--method-option", "alpha=0.3",
            "--method-option", "mode=sum"))
    except SystemExit:
        pytest.fail("runner does not accept plugin methods / --method-option")
    args = train.normalize_method_args(parsed, parser)
    assert args.hidden == 16 and args.epochs == 5 and args.lr == 2e-3
    assert args.method_options == {"alpha": "0.3", "mode": "sum"}
    assert "option:alpha" in args._method_overrides and "option:mode" in args._method_overrides
    assert train.method_defaults("fake_plugin") == FakeAdapter.runner_defaults
    assert not (tmp_path / "o").exists()


def test_method_option_is_rejected_for_core_methods_and_malformed(monkeypatch, tmp_path):
    monkeypatch.setitem(methods.METHOD_REGISTRY, "fake_plugin", FakeAdapter)
    parser = train.parser()
    with pytest.raises(SystemExit):
        train.normalize_method_args(parser.parse_args(base_argv(
            tmp_path, "--method", "protgnn", "--method-option", "alpha=1")), parser)
    with pytest.raises(SystemExit):
        train.normalize_method_args(parser.parse_args(base_argv(
            tmp_path, "--method", "fake_plugin", "--method-option", "alpha")), parser)


def test_plugin_controls_gradient_clipping_and_early_stopping_gate(monkeypatch):
    monkeypatch.setitem(methods.METHOD_REGISTRY, "fake_plugin", FakeAdapter)
    model = FakeAdapter()
    model.weight.grad = torch.full((3,), 10.0)
    train.clip_gradients("fake_plugin", model)
    assert model.weight.grad.abs().max().item() == pytest.approx(0.5)
    assert train.early_stopping_start_epoch("fake_plugin", model) == 4


def test_plugin_without_runner_defaults_is_rejected(monkeypatch, tmp_path):
    class NoDefaults(FakeAdapter):
        runner_defaults = None

    monkeypatch.setitem(methods.METHOD_REGISTRY, "no_defaults", NoDefaults)
    parser = train.parser()
    with pytest.raises((SystemExit, ValueError)):
        train.normalize_method_args(parser.parse_args(base_argv(
            tmp_path, "--method", "no_defaults")), parser)
