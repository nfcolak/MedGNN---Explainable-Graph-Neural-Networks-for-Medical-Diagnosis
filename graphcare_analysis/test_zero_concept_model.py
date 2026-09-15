"""Real upstream BAT-GNN integration; run in .venv-graphcare, no optimizer steps."""
import pytest
import torch
from graphcare_analysis.adapter import _subgraph, _collate
from graphcare_analysis.run import build_graphcare_model, _forward
from graphcare_analysis.explainability.explain_graphcare import (
    GraphCareGraphXAIWrapper, explain_graphcare_record,
)


def make_batch(codes):
    records = []
    for index, ids in enumerate(codes):
        record = _subgraph(ids, {i: [] for i in range(4)}, 4, structure="star")
        record["y"] = index % 3
        records.append(record)
    return _collate(records)


def make_model():
    torch.manual_seed(1234)
    return build_graphcare_model({"num_nodes": 4, "num_rels": 3}, 3, "cpu")


@pytest.mark.parametrize("codes", [[[]], [[], []], [[], [0, 1]], [[0], []]])
def test_real_model_hub_only_and_mixed_forward_backward(codes):
    model = make_model()
    batch = make_batch(codes)
    logits = _forward(model, batch)
    assert logits.shape == (len(codes), 3)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits, batch["y"])
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_raw_ehr_pool_is_zero_and_mixed_eval_preserves_nonempty():
    model = make_model().eval()
    observed = []
    handle = model.lin.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
    mixed = _forward(model, make_batch([[], [0, 1]]))
    handle.remove()
    raw_ehr = observed[-1]
    assert torch.equal(raw_ehr[0], torch.zeros_like(raw_ehr[0]))
    assert torch.equal(raw_ehr[1], (model.node_emb.weight[0] + model.node_emb.weight[1]).detach().view(1, -1) / 2)
    single = _forward(model, make_batch([[0, 1]]))
    assert torch.allclose(mixed[1], single[0], atol=1e-7)


def test_real_hub_only_explanation_and_fidelity():
    model = make_model().eval()
    batch = make_batch([[]])
    wrapper = GraphCareGraphXAIWrapper(model)
    x = wrapper.set_context(batch)
    assert wrapper.verify() == (True, 0.0)
    x.requires_grad_()
    logits = wrapper(x, batch["edge_index"], batch["batch"])
    logits.sum().backward()
    assert torch.isfinite(x.grad).all()
    batch.update(subject_id="synthetic-hub", topology="star", seed=1234)
    result = explain_graphcare_record(model, batch)
    import json
    json.dumps(result, allow_nan=False)
    from shared.lib.fidelity import fidelity_plus, fidelity_minus, sparsity
    args = (wrapper, x.detach(), batch["edge_index"], [0.0], 0, batch["batch"])
    assert fidelity_minus(*args)["prob"] == 0.0
    assert torch.isfinite(torch.tensor(fidelity_plus(*args)["prob"]))
    assert sparsity([0.0]) == 0.0


@pytest.mark.parametrize("training", [False, True])
def test_nonempty_logits_and_gradients_identical_to_upstream(training):
    model = make_model()
    from graphcare_.model import GraphCare as Upstream
    from graphcare_analysis.config import cfg
    upstream = Upstream(4, 3, 1, cfg.emb_dim, cfg.emb_dim, 3,
                        layers=cfg.num_layers, dropout=cfg.dropout,
                        patient_mode="joint", use_alpha=True, use_beta=True, gnn="BAT")
    upstream.load_state_dict(model.state_dict(), strict=True)
    model.train(training)
    upstream.train(training)
    batch = make_batch([[0, 1], [2]])
    torch.manual_seed(17)
    expected = _forward(upstream, batch)
    expected.sum().backward()
    torch.manual_seed(17)
    actual = _forward(model, batch)
    actual.sum().backward()
    assert torch.equal(actual, expected)
    for (name, param), (other_name, other) in zip(model.named_parameters(), upstream.named_parameters()):
        assert name == other_name
        if param.grad is None:
            assert other.grad is None
        else:
            assert torch.equal(param.grad, other.grad), name
