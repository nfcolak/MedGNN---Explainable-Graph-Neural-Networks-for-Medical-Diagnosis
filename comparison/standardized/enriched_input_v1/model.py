"""Opt-in numerical patient hub; original PNA source/checkpoint bindings unchanged."""
import torch
from torch import nn

from pna_analysis.data import concept_graph
from pna_analysis.model import PNAPredictor, GraphXAIWrapper


class HubEncoder(nn.Module):
    """Constant concept/hub identity projection plus separate numeric projection."""
    def __init__(self, identity, numeric_dim):
        super().__init__()
        self.identity = identity
        self.numeric = nn.Linear(numeric_dim, identity.out_features, bias=False)

    def forward(self, x):
        cut = self.identity.in_features
        return self.identity(x[:, :cut]) + self.numeric(x[:, cut:])


class EnrichedPNA(PNAPredictor):
    """Same PNA messages/pair content/readout; numerical state enters before MP.

    Zero numeric width is exactly the original implementation/state schema. With
    positive width, zero numeric values add zero and preserve base-path logits.
    """
    def __init__(self, input_dim, classes, degree_histogram, numeric_dim=0, **kwargs):
        super().__init__(input_dim, classes, degree_histogram, **kwargs)
        if numeric_dim < 0:
            raise ValueError('Negative numeric dimension')
        self.numeric_dim = numeric_dim
        if numeric_dim:
            self.encoder = HubEncoder(self.encoder, numeric_dim)


def numeric_graph(codes, label, names, numeric):
    """Shared PyG payload: zero numeric leaf slots, payload only on patient hub."""
    graph = concept_graph(codes, label, names)
    vector = torch.as_tensor(numeric, dtype=torch.float32)
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError('Finite one-dimensional hub payload required')
    payload = graph.x.new_zeros((graph.num_nodes, vector.numel()))
    payload[graph.node_type == 2] = vector
    graph.x = torch.cat([graph.x, payload], dim=1)
    return graph


def build_comparator(method, concepts, numeric_dim, device='cpu'):
    """Actual common-input ProtGNN/GSAT models consume the same appended hub x.

    This only changes this model instance's input width. Original CLI/cache/run
    defaults are untouched. No GraphCare categorical-ID expansion is pretended
    to represent a continuous measurement; its consumer is explicitly pending.
    """
    if method not in {'protgnn', 'gsat'}:
        raise NotImplementedError('GraphCare categorical embedding needs a numeric encoder; not integrated')
    from comparison.standardized.common_input_improvement import build_model
    model, _, _, device = build_model(method, concepts + numeric_dim, device=device)
    return model, device


def explain_numeric(model, graph, feature_names, steps=16, epochs=10):
    """Actual GraphXAI algorithms, retaining signed per-channel Grad/IG results.

    The identity aggregation callback retains GraphXAI's full feature matrix;
    this is not a custom replacement algorithm. IG uses vendor zero baseline.
    """
    from shared.lib.graphxai_standardized import explain_algorithms
    from graphxai.explainers import GradExplainer, IntegratedGradExplainer
    model.eval()
    wrapper = GraphXAIWrapper(model, graph.node_type)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=graph.x.device)
    algorithms = explain_algorithms(wrapper, graph.x, graph.edge_index, batch=batch, steps=steps, epochs=epochs)
    target = wrapper(graph.x, graph.edge_index, batch=batch).argmax(-1).detach()
    criterion = lambda logits, labels: logits.gather(1, labels.view(-1, 1)).sum()
    retain_features = lambda matrix, dim: matrix
    values = {}
    for name in ['GradExplainer', 'IntegratedGradExplainer']:
        wrapper.zero_grad(set_to_none=True)
        x = graph.x.detach().clone()
        if name == 'GradExplainer':
            explanation = GradExplainer(wrapper, criterion).get_explanation_graph(
                x, graph.edge_index, target, aggregate_node_imp=retain_features, forward_kwargs={'batch': batch})
        else:
            explanation = IntegratedGradExplainer(wrapper, criterion).get_explanation_graph(
                graph.edge_index, x, target, node_agg=retain_features, steps=steps, forward_kwargs={'batch': batch})
        scores = explanation.node_imp[graph.node_type == 2, -len(feature_names):].detach().flatten()
        if scores.numel() != len(feature_names) or not torch.isfinite(scores).all():
            raise ValueError('Invalid GraphXAI numeric feature attribution')
        values[name] = scores.tolist()
    wrapper.zero_grad(set_to_none=True)
    return {'algorithms': algorithms, 'numeric_attribution': values, 'feature_names': feature_names,
            'scope': 'synthetic/selected-record wiring; not a clinical explanation validation',
            'edge_mask_interpretation': 'numeric hub residual remains frozen under edge-only interventions',
            'ig_baseline': 'vendor all-zero features; fixed original node types and topology'}
