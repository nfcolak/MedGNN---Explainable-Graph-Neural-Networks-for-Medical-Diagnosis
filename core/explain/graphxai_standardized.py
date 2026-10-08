"""Real vendored GraphXAI graph algorithms, with explicit PyG compatibility fixes.

Upstream GNNExplainer's optimizer is retained. Its base prediction helper disables
all gradients and its legacy double-underscore masks are ignored by modern PyG;
this adapter repairs those two API defects, not the optimization algorithm.

GNNExplainer normally drives its learned edge mask into the model by patching
every `torch_geometric.nn.MessagePassing` submodule (`set_masks`/`clear_masks`,
below). Models that aggregate by hand instead of subclassing `MessagePassing`
(e.g. the clinical_graph_v2 adapters) have nothing for that patch to attach to,
so `set_masks` silently finds zero targets and the optimizer climbs a mask that
never touches a single message. `CompatibleGNNExplainer._predict` below adds a
second, explicit delivery path for exactly that case: if the wrapped model's
own `forward` declares an `edge_mask` parameter, the live mask (same Parameter,
every optimisation step, sigmoid-activated like the MessagePassing path) is
passed as that keyword argument. Models that don't declare `edge_mask` -- every
existing wrapper on the star/cooccur/ontology/full graph included -- are
completely unaffected: the signature check is `False` for them, so the call
shape is byte-identical to before. Both delivery paths can be active at once
(harmless) for a model that is both `MessagePassing`-based and separately
declares `edge_mask`; no current wrapper does both.
"""
from pathlib import Path
import hashlib
import inspect
import os
import sys
import torch
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.explain.algorithm.utils import set_masks, clear_masks
from core.explain.explanation_contract import build_node_explanation

VENDOR = Path(__file__).resolve().parents[2] / 'external/GraphXAI-main'
sys.path.insert(0, str(VENDOR))
from graphxai.explainers import GradExplainer, IntegratedGradExplainer, GNNExplainer

ALGORITHMS = ('GradExplainer', 'IntegratedGradExplainer', 'GNNExplainer')

def _model_declares_edge_mask(model):
    try:
        return 'edge_mask' in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False

def aggregate_reverse_edge_pairs(edge_imp, reversible_edge_pairs):
    """Collapse each (forward_index, reverse_index) pair to their mean, in
    place of two independently-learned mask weights for what is really one
    clinical fact (GraphXAI improvement plan item #5: "do not count reverse
    edges as separate evidence"). A reverse edge (edge_direction=
    'bidirectional') duplicates a single fact as two directed edges; left
    alone, that fact gets two chances to push its incident nodes' importance
    up under the incident-edge node reduction below, inflating anything
    touching a reversible relation relative to a forward-only one. `None` or
    `[]` is a no-op, returning `edge_imp` unchanged (the entire function is
    skipped by forward-only callers and every non-clinical_graph_v2 caller)."""
    if not reversible_edge_pairs:
        return edge_imp
    edge_imp = edge_imp.clone()
    for forward_i, reverse_i in reversible_edge_pairs:
        combined = 0.5 * (edge_imp[forward_i] + edge_imp[reverse_i])
        edge_imp[forward_i] = combined
        edge_imp[reverse_i] = combined
    return edge_imp

class CompatibleGNNExplainer(GNNExplainer):
    def _predict(self, x, edge_index, return_type='label', forward_kwargs=None):
        kwargs = dict(forward_kwargs or {})
        mask = getattr(self, 'edge_mask', None)
        if mask is not None and _model_declares_edge_mask(self.model):
            kwargs['edge_mask'] = mask.sigmoid()
        out = self.model(x, edge_index, **kwargs)
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

def explain_algorithms(wrapper, x, edge_index, *, batch, steps=32, epochs=50,
                       reversible_edge_pairs=None, node_reduction='sum', target_class=None):
    if steps < 1 or epochs < 1:
        raise ValueError('positive explanation budgets required')
    # 'sum' is the historical reduction every existing caller (e.g. the CEI GraphXAI
    # studies) was run with; 'mean' divides by incident-edge count to remove the
    # degree bias and is opt-in so existing results stay reproducible.
    if node_reduction not in ('sum', 'mean'):
        raise ValueError("node_reduction must be 'sum' or 'mean'")
    wrapper.eval()
    with torch.no_grad():
        label = wrapper(x, edge_index, batch=batch).argmax(-1)
    # target_class=None explains the model's own prediction (every existing caller).
    # An explicit class explains that class's logit instead -- e.g. the TRUE class of a
    # misclassified patient. GNNExplainer cannot be redirected: the vendored optimiser
    # hard-codes the model's predicted class as its target.
    predicted_label = label
    if target_class is not None:
        label = torch.tensor([int(target_class)], device=label.device)
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
                      'target':('predicted_logit' if target_class is None else f'class_{int(target_class)}_logit')
                               if name != 'GNNExplainer' else 'predicted_class_log_probability',
                      'steps':steps if name == 'IntegratedGradExplainer' else None,
                      'epochs':epochs if name == 'GNNExplainer' else None}
        # Isolated per algorithm: one algorithm's failure (e.g. GNNExplainer's
        # disconnected-gradient refusal above) must not discard the other two
        # algorithms' already-computed, perfectly valid results along with it.
        # Previously the whole function raised and returned nothing at all.
        if name == 'GNNExplainer' and target_class is not None and int(label.item()) != int(predicted_label.item()):
            result[name] = {'status': 'unsupported', 'provenance': provenance,
                            'unsupported_reason': 'GNNExplainer always explains the model\'s own predicted '
                                                  'class; it cannot be directed at another class'}
            continue
        try:
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
                    # This connectivity pre-check calls `wrapper` directly, not
                    # through `_predict` (unlike the optimisation loop inside
                    # get_explanation_graph below) -- so it needs the same explicit
                    # edge_mask handoff repeated here, or it would (correctly, for
                    # the wrong reason) report "disconnected" for any model that
                    # only supports the mask via this keyword, before ever reaching
                    # the loop that actually uses it.
                    precheck_kwargs = {'batch': batch}
                    if _model_declares_edge_mask(wrapper):
                        precheck_kwargs['edge_mask'] = ex.edge_mask.sigmoid()
                    logits = wrapper(features,edge_index,**precheck_kwargs)
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
                edge_imp = aggregate_reverse_edge_pairs(exp.edge_imp.detach(), reversible_edge_pairs)
                provenance['reverse_pairs_aggregated'] = len(reversible_edge_pairs or [])
                # Continuous incident-edge mass avoids upstream's binary node mask ties.
                # 'mean' (opt-in): summing raw mask mass over incident edges makes node
                # importance scale with node degree almost independently of which
                # edges the mask actually learned to weight highly -- a high-degree
                # hub accumulates the largest total even when no single incident edge
                # is individually important. Dividing by incident-edge count isolates
                # the mask's actual signal from this degree confound.
                importance = torch.zeros(x.size(0),device=x.device)
                degree = torch.zeros(x.size(0),device=x.device)
                importance.index_add_(0,edge_index[0],edge_imp)
                importance.index_add_(0,edge_index[1],edge_imp)
                degree.index_add_(0,edge_index[0],torch.ones_like(edge_imp))
                degree.index_add_(0,edge_index[1],torch.ones_like(edge_imp))
                exp.node_imp = importance / degree.clamp_min(1.0) if node_reduction == 'mean' else importance
                provenance['node_reduction'] = f'{node_reduction}_incident_continuous_edge_mask'
        except Exception as exc:
            result[name] = {'status':'failed','error':str(exc),'error_type':type(exc).__name__,
                           'provenance':provenance}
            continue
        result[name] = {'status':'success','provenance':provenance,
                       'node_explanation':build_node_explanation(wrapper,x,edge_index,batch=batch,node_importance=exp.node_imp)}
    wrapper.zero_grad(set_to_none=True)
    return result
