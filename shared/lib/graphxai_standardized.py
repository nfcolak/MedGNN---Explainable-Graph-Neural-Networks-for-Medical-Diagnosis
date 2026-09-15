"""Real vendored GraphXAI graph algorithms, with explicit PyG compatibility fixes.

Upstream GNNExplainer's optimizer is retained. Its base prediction helper disables
all gradients and its legacy double-underscore masks are ignored by modern PyG;
this adapter repairs those two API defects, not the optimization algorithm.
"""
from pathlib import Path
import hashlib
import os
import sys
import torch
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.explain.algorithm.utils import set_masks, clear_masks
from shared.lib.explanation_contract import build_node_explanation

VENDOR = Path(__file__).resolve().parents[2] / 'external/GraphXAI-main'
sys.path.insert(0, str(VENDOR))
from graphxai.explainers import GradExplainer, IntegratedGradExplainer, GNNExplainer

ALGORITHMS = ('GradExplainer', 'IntegratedGradExplainer', 'GNNExplainer')

class CompatibleGNNExplainer(GNNExplainer):
    def _predict(self, x, edge_index, return_type='label', forward_kwargs=None):
        out = self.model(x, edge_index, **(forward_kwargs or {}))
        if return_type == 'label':
            return out.argmax(-1)
        if return_type == 'prob':
            return out.softmax(-1)
        if return_type == 'log_prob':
            return out.log_softmax(-1)
        raise ValueError('Unknown prediction return type')

    def _set_masks(self, x, edge_index, **kwargs):
        super()._set_masks(x, edge_index, **kwargs)
        set_masks(self.model, self.edge_mask, edge_index, apply_sigmoid=True)

    def _clear_masks(self):
        clear_masks(self.model)
        super()._clear_masks()

def explain_algorithms(wrapper, x, edge_index, *, batch, steps=32, epochs=50):
    if steps < 1 or epochs < 1:
        raise ValueError('positive explanation budgets required')
    wrapper.eval()
    with torch.no_grad():
        label = wrapper(x, edge_index, batch=batch).argmax(-1)
    # Attribute the predicted logit, rather than an unrelated training label.
    criterion = lambda logits, target: logits.gather(1, target.view(-1,1)).sum()
    aggregate = lambda values, dim: values.abs().sum(dim=dim)
    result = {}
    sources = ['grad.py','integrated_grad.py','gnn_explainer.py','_base.py']
    hashes = {name:hashlib.sha256((VENDOR/'graphxai/explainers'/name).read_bytes()).hexdigest() for name in sources}
    for name in ALGORITHMS:
        wrapper.zero_grad(set_to_none=True)
        features = x.detach().clone()
        provenance = {'implementation':'vendored GraphXAI', 'source_sha256':hashes,
                      'torch':torch.__version__, 'pyg':torch_geometric.__version__,
                      'target':'predicted_logit' if name != 'GNNExplainer' else 'predicted_class_log_probability',
                      'steps':steps if name == 'IntegratedGradExplainer' else None,
                      'epochs':epochs if name == 'GNNExplainer' else None}
        if name == 'GradExplainer':
            exp = GradExplainer(wrapper, criterion).get_explanation_graph(features,edge_index,label,
                aggregate_node_imp=aggregate,forward_kwargs={'batch':batch})
        elif name == 'IntegratedGradExplainer':
            exp = IntegratedGradExplainer(wrapper,criterion).get_explanation_graph(edge_index,features,label,
                steps=steps,node_agg=aggregate,forward_kwargs={'batch':batch})
        else:
            ex = CompatibleGNNExplainer(wrapper)
            old = os.environ.get('GNNEXP_EPOCHS')
            try:
                ex._set_masks(features,edge_index,explain_feature=True,device=x.device)
                logits = wrapper(features,edge_index,batch=batch)
                gradient = torch.autograd.grad(logits[0,label.item()],ex.edge_mask,allow_unused=True)[0]
                edgeless = edge_index.size(1) == 0
                if not edgeless and (gradient is None or not torch.isfinite(gradient).all()):
                    raise RuntimeError('GNNExplainer edge mask gradient disconnected or nonfinite; unsupported graph/model')
                # Connected zero derivatives are legitimate flat regions. An empty
                # graph has no edge derivative to verify; never label that verified.
                provenance['edge_count'] = int(edge_index.size(1))
                provenance['edge_gradient_verified'] = not edgeless
                provenance['edge_gradient_status'] = 'not_applicable_edgeless' if edgeless else 'connected_finite'
                provenance['edge_gradient_l1'] = float(gradient.abs().sum()) if gradient is not None else 0.0
                provenance['zero_predictive_edge_gradient'] = provenance['edge_gradient_l1'] == 0.0
                provenance['empty_edge_entropy'] = 'zero'
                ex._clear_masks()
                os.environ['GNNEXP_EPOCHS'] = str(epochs)
                exp = ex.get_explanation_graph(features,edge_index,forward_kwargs={'batch':batch})
            finally:
                ex._clear_masks()
                if old is None:
                    os.environ.pop('GNNEXP_EPOCHS',None)
                else:
                    os.environ['GNNEXP_EPOCHS'] = old
            # Continuous incident-edge mass avoids upstream's binary node mask ties.
            importance = torch.zeros(x.size(0),device=x.device)
            importance.index_add_(0,edge_index[0],exp.edge_imp.detach())
            importance.index_add_(0,edge_index[1],exp.edge_imp.detach())
            exp.node_imp = importance
            provenance['node_reduction'] = 'sum_incident_continuous_edge_mask'
        result[name] = {'status':'success','provenance':provenance,
                       'node_explanation':build_node_explanation(wrapper,x,edge_index,batch=batch,node_importance=exp.node_imp)}
    wrapper.zero_grad(set_to_none=True)
    return result
