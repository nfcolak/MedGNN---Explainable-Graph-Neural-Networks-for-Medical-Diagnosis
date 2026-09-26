"""Mechanism checks that pass or fail WITHOUT training.

A benchmark score never proves the intended mechanism ran. These checks do, and they
cost seconds. Run before any training claim.
"""
import sys

import torch

from .model import ClinicalGNN, EdgeConditionedLayer


def check_message_not_separable(dim=16, edge_dim=8, seed=0):
    """m(x_i, x_j, e) must not equal the sum of its one-at-a-time parts.

    If the residual sits at float32 roundoff (~1e-7) the layer is additive and can
    never multiply a neighbour's content by the receiver's state -- meaning
    `baseline_of`'s delta could not condition on the receiving measurement.
    """
    torch.manual_seed(seed)
    layer = EdgeConditionedLayer(dim, edge_dim, dropout=0.0)
    xi, xj = torch.randn(4, dim), torch.randn(4, dim)
    e = torch.randn(4, edge_dim)
    zi, zj, ze = torch.zeros_like(xi), torch.zeros_like(xj), torch.zeros_like(e)
    with torch.no_grad():
        full = layer.message(xj, xi, e)
        base = layer.message(zj, zi, ze)
        only_j = layer.message(xj, zi, ze) - base
        only_i = layer.message(zj, xi, ze) - base
        only_e = layer.message(zj, zi, e) - base
        residual = (full - (base + only_i + only_j + only_e)).abs().max().item()
    return residual, residual > 1e-4


def check_receiver_changes_message(dim=16, edge_dim=8, seed=1):
    """Fixing sender and edge, varying only the receiver, must change the message."""
    torch.manual_seed(seed)
    layer = EdgeConditionedLayer(dim, edge_dim, dropout=0.0)
    xj, e = torch.randn(1, dim), torch.randn(1, edge_dim)
    with torch.no_grad():
        a = layer.message(xj, torch.randn(1, dim), e)
        b = layer.message(xj, torch.randn(1, dim), e)
    delta = (a - b).abs().max().item()
    return delta, delta > 1e-4


def check_edge_payload_ablation(dim=16, edge_dim=8, seed=2):
    """The payload ablation must change the output but NOT the parameter count."""
    torch.manual_seed(seed)
    free = EdgeConditionedLayer(dim, edge_dim, dropout=0.0, use_edge_payload=True)
    torch.manual_seed(seed)
    ablated = EdgeConditionedLayer(dim, edge_dim, dropout=0.0, use_edge_payload=False)
    x = torch.randn(5, dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    e = torch.randn(4, edge_dim)
    with torch.no_grad():
        a = free(x, edge_index, e)
        b = ablated(x, edge_index, e)
    same_params = sum(p.numel() for p in free.parameters()) == \
        sum(p.numel() for p in ablated.parameters())
    delta = (a - b).abs().max().item()
    return delta, (delta > 1e-4 and same_params)


def check_degenerate_graph_finite(seed=3):
    """An edgeless single-node graph must produce finite logits, not NaN."""
    torch.manual_seed(seed)
    model = ClinicalGNN(num_tokens=10, node_dim=14, edge_dim=22, hidden=32, layers=2)
    from torch_geometric.data import Batch, Data
    d = Data(x=torch.randn(1, 14), edge_index=torch.zeros((2, 0), dtype=torch.long),
             edge_attr=torch.zeros((0, 22)))
    d.token = torch.zeros(1, dtype=torch.long)
    out = model(Batch.from_data_list([d]))
    return float(out.abs().max()), bool(torch.isfinite(out).all())


def check_batching_does_not_mix(seed=4):
    """Two graphs batched together must give the same logits as run separately."""
    torch.manual_seed(seed)
    model = ClinicalGNN(num_tokens=10, node_dim=14, edge_dim=22, hidden=32, layers=2)
    model.eval()
    from torch_geometric.data import Batch, Data

    def make(n, m):
        d = Data(x=torch.randn(n, 14),
                 edge_index=torch.randint(0, n, (2, m)),
                 edge_attr=torch.randn(m, 22))
        d.token = torch.randint(0, 10, (n,))
        return d

    a, b = make(6, 9), make(4, 5)
    with torch.no_grad():
        separate = torch.cat([model(Batch.from_data_list([a])),
                              model(Batch.from_data_list([b]))])
        together = model(Batch.from_data_list([a, b]))
    delta = (separate - together).abs().max().item()
    return delta, delta < 1e-4


def check_gradients_reach_every_block(seed=5):
    """Every parameter block must receive a gradient, or a module is dead code."""
    torch.manual_seed(seed)
    model = ClinicalGNN(num_tokens=10, node_dim=14, edge_dim=22, hidden=32, layers=2)
    from torch_geometric.data import Batch, Data
    d = Data(x=torch.randn(6, 14), edge_index=torch.randint(0, 6, (2, 9)),
             edge_attr=torch.randn(9, 22))
    d.token = torch.randint(1, 10, (6,))
    out = model(Batch.from_data_list([d]))
    out.sum().backward()
    missing = [n for n, p in model.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
    return missing, not missing


CHECKS = [
    ('message_not_additively_separable', check_message_not_separable),
    ('receiver_state_changes_message', check_receiver_changes_message),
    ('edge_payload_ablation_equal_params', check_edge_payload_ablation),
    ('degenerate_graph_finite', check_degenerate_graph_finite),
    ('batching_does_not_mix_graphs', check_batching_does_not_mix),
    ('gradients_reach_every_block', check_gradients_reach_every_block),
]


def main():
    failures = 0
    for name, fn in CHECKS:
        value, ok = fn()
        print(f'{"PASS" if ok else "FAIL"}  {name:38} {value}')
        failures += not ok
    print(f'\n{len(CHECKS) - failures}/{len(CHECKS)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
