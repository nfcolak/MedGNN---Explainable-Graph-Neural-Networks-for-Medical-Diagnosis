import torch
from torch_geometric.nn import GCNConv, global_mean_pool

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GCNConv(2, 2)
    def forward(self, x, edge_index, batch=None):
        return global_mean_pool(self.conv(x, edge_index), batch)

def test_real_graphxai_algorithms_and_edge_gradient():
    from shared.lib.graphxai_standardized import explain_algorithms
    torch.manual_seed(1234)
    model = Tiny().eval()
    x = torch.tensor([[1., 2.], [3., 1.], [2., 4.]])
    edges = torch.tensor([[0,1,1,2],[1,0,2,1]])
    result = explain_algorithms(model,x,edges,batch=torch.zeros(3,dtype=torch.long),steps=4,epochs=3)
    assert set(result) == {'GradExplainer','IntegratedGradExplainer','GNNExplainer'}
    for value in result.values():
        assert value['status'] == 'success'
        assert len(value['node_explanation']['node_importance']) == 3
    assert result['GNNExplainer']['provenance']['edge_gradient_verified']
    assert not model.conv.explain

def test_zero_predictive_edge_gradient_is_valid_not_disconnected():
    from shared.lib.graphxai_standardized import explain_algorithms
    model = Tiny().eval()
    with torch.no_grad():
        model.conv.lin.weight.zero_()
    x = torch.ones(3, 2)
    edges = torch.tensor([[0,1,1,2],[1,0,2,1]])
    result = explain_algorithms(model, x, edges, batch=torch.zeros(3,dtype=torch.long), steps=4, epochs=3)
    provenance = result['GNNExplainer']['provenance']
    assert provenance['edge_gradient_verified']
    assert provenance['edge_gradient_l1'] == 0
    assert provenance['zero_predictive_edge_gradient'] is True


def test_real_graphxai_edgeless_graph_has_finite_optimizer_gradients(monkeypatch):
    from shared.lib.graphxai_standardized import explain_algorithms
    original_backward = torch.Tensor.backward
    def checked_backward(loss, *args, **kwargs):
        assert torch.isfinite(loss).all(), 'GNNExplainer objective must remain finite'
        return original_backward(loss, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'backward', checked_backward)
    original_step = torch.optim.Adam.step
    observed = []
    def checked_step(optimizer, *args, **kwargs):
        observed.extend(p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None)
        assert all(torch.isfinite(g).all() for g in observed)
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, 'step', checked_step)
    model = Tiny().eval()
    result = explain_algorithms(model, torch.ones(1,2), torch.empty((2,0),dtype=torch.long), batch=torch.zeros(1,dtype=torch.long), steps=4, epochs=3)
    p = result['GNNExplainer']['provenance']
    assert p['edge_gradient_verified'] is False
    assert p['edge_gradient_status'] == 'not_applicable_edgeless'
    assert p['empty_edge_entropy'] == 'zero'
    assert observed
    for value in result.values():
        assert len(value['node_explanation']['node_importance']) == 1
    from shared.lib.explanation_contract import validate_explanation_record, EXPLANATION_SCHEMA
    explanation = result['GradExplainer']['node_explanation']
    record=dict(schema=EXPLANATION_SCHEMA,schema_version=2,method='protgnn',subject_id='synthetic',topology='star',seed=1234,true_class_id=0,prediction_class_id=explanation['target']['class_id'],attribution_method='deterministic_input_x_gradient',node_explanation=explanation,graphxai=result)
    validate_explanation_record(record,method='protgnn',topology='star',seed=1234)


def test_checkpoint_config_reconstruction_is_cpu_and_strict():
    from protgnn_analysis.explainability.explain_standardized import checkpoint_model_args
    from protgnn_analysis.config import model_args
    c={'model':'gcn','latent_dim':[7,7],'mlp_hidden':[4],'readout':'sum','dropout':0.2,'adj_normalize':False,'emb_normalize':True,'enable_prototypes':False,'num_prototypes_per_class':2}
    args=checkpoint_model_args(c)
    assert args.device == 'cpu'
    assert args.latent_dim == [7,7]
    assert not args.adj_normlize and args.emb_normlize
    assert args is not model_args

def test_version_two_requires_all_real_algorithms():
    from shared.lib.explanation_contract import validate_explanation_record, EXPLANATION_SCHEMA, build_node_explanation
    import pytest
    model=Tiny().eval()
    x=torch.ones(3,2)
    edges=torch.tensor([[0,1,1,2],[1,0,2,1]])
    explanation=build_node_explanation(model,x,edges,batch=torch.zeros(3,dtype=torch.long))
    record=dict(schema=EXPLANATION_SCHEMA,schema_version=2,method='protgnn',subject_id='synthetic',topology='star',seed=1234,true_class_id=0,prediction_class_id=explanation['target']['class_id'],attribution_method='deterministic_input_x_gradient',node_explanation=explanation,graphxai={})
    with pytest.raises(ValueError,match='three GraphXAI'):
        validate_explanation_record(record,method='protgnn',topology='star',seed=1234)
