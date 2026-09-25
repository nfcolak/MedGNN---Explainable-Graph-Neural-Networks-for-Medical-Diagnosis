"""Model registry on the one native 331-dimensional input, no legacy fallback."""
import copy
import sys
import torch
from torch import nn
from torch_geometric.data import Data

METHODS = ('protgnn','gsat','graphcare','pna','pna_interaction','gchm','gchm_additive','gchm_pairs',
           'gchm_concept_dropout','gchm_plq','gchm_tabgnn','dtv_gnn')
# Fitted-constant recipe for gchm_plq. Changing any value changes the model input
# encoding, so the run binding records the fitted edge hash, not just this dict.
PLQ = {'bins': 8, 'min_unique': 3, 'missing_value': 0.0, 'fit_fold': 0,
       'recipe': 'train-fold quantiles of non-missing values; piecewise-linear code + missing flag'}
# DTV-GNN hyperparameters, prespecified from the depth sweep (order two is all
# this dataset supports) rather than tuned on validation.
DTV = {'rank': 24, 'dropout': 0.1, 'interaction_scale': 1.0}


def _fitted_sha(edges, continuous, binary):
    """Hash the fitted encoder constants so a run binding pins them exactly.

    A recipe dict alone is not enough: the same recipe on a different fold or a
    different artifact yields different edges, and that silently changes the model
    input. Hashing the realized arrays makes any such drift fail the resume check.
    """
    import hashlib
    import numpy as np
    h = hashlib.sha256()
    for a in (np.ascontiguousarray(edges), np.ascontiguousarray(continuous),
              np.ascontiguousarray(binary)):
        h.update(str([a.dtype.str, a.shape]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def canonical_input(batch):
    if batch.x.ndim != 2 or batch.x.shape[1] != 331 or not torch.isfinite(batch.x).all():
        raise ValueError('Expected finite native 331-slot input')
    if batch.node_ids.shape != (batch.x.shape[0],) or batch.node_type.shape != batch.node_ids.shape:
        raise ValueError('Missing native categorical metadata')
    if not torch.equal(batch.batch[batch.edge_index[0]],batch.batch[batch.edge_index[1]]):
        raise ValueError('Cross-patient edges forbidden')
    return batch.x, batch.edge_index, batch.y


def graphcare_arguments(batch):
    canonical_input(batch)
    count = int(batch.batch.max()) + 1
    membership = batch.x.new_zeros((count,193))
    clinical = batch.node_ids != 192
    membership[batch.batch[clinical],batch.node_ids[clinical]] = 1.
    return dict(node_ids=batch.node_ids,rel_ids=torch.full((batch.edge_index.shape[1],),2,device=batch.x.device,dtype=torch.long),
                edge_index=batch.edge_index,batch=batch.batch,visit_node=membership[:,None,:],ehr_nodes=membership,
                numeric=batch.x[:,199:])


def build_model(method, ref, device='cpu'):
    if method not in METHODS: raise ValueError(method)
    if method == 'protgnn':
        from protgnn_analysis.models import GnnNets
        from protgnn_analysis.config import model_args, train_args
        config = copy.deepcopy(model_args); config.device=str(device);config.enable_prot=True
        model=GnnNets(331,30,config);model.to_device()
        return model,train_args.learning_rate,train_args.weight_decay
    if method == 'gsat':
        from gsat_analysis.train import build_model as build
        from gsat_analysis.config import cfg
        return build(331,30,device),cfg.lr,cfg.weight_decay
    if method == 'graphcare':
        from graphcare_analysis.config import cfg
        sys.path.insert(0,str(cfg.UPSTREAM_DIR))
        from .graphcare import NumericGraphCare
        model=NumericGraphCare(num_nodes=193,num_rels=3,max_visit=1,embedding_dim=cfg.emb_dim,
            hidden_dim=cfg.emb_dim,out_channels=30,layers=cfg.num_layers,dropout=cfg.dropout,
            patient_mode='joint',use_alpha=True,use_beta=True,gnn='BAT').to(device)
        return model,cfg.lr,cfg.weight_decay
    if method.startswith('gchm') or method == 'dtv_gnn':
        from gchm_analysis.model import GCHM, PiecewiseLinearQuantileHub, fit_hub_quantiles
        modulation = 'additive' if method == 'gchm_additive' else 'multiplicative'
        encoder = None
        if method in ('gchm_plq', 'gchm_tabgnn', 'dtv_gnn'):
            # Fitted on fold 0 only; validation/test rows never enter the quantiles.
            hub = ref.arrays['hub']
            edges, continuous, binary, degenerate = fit_hub_quantiles(
                hub, ref.arrays['folds'] == PLQ['fit_fold'], bins=PLQ['bins'],
                missing_value=PLQ['missing_value'], min_unique=PLQ['min_unique'])
            encoder = PiecewiseLinearQuantileHub(edges, continuous, binary, hub.shape[1])
            encoder.fit_report = {**PLQ, 'continuous': int(len(continuous)),
                                  'binary': int(len(binary)), 'degenerate_edges': degenerate,
                                  'encoded_width': int(encoder.out_dim),
                                  'edges_sha256': _fitted_sha(edges, continuous, binary)}
        if method == 'dtv_gnn':
            from gchm_analysis.dtv import DTVGNN
            model = DTVGNN(classes=30, hub_encoder=encoder,
                           degree_histogram=ref.degree_histogram(), **DTV).to(device)
            return model, 1e-3, 1e-5
        model = GCHM(classes=30, degree_histogram=ref.degree_histogram(),
                     modulation=modulation,
                     concept_pairs=method == 'gchm_pairs',
                     concept_dropout=0.15 if method == 'gchm_concept_dropout' else 0.0,
                     hub_encoder=encoder,
                     hub_skip=method == 'gchm_tabgnn',
                     ).to(device)
        return model, 1e-3, 1e-5
    from pna_analysis.model import PNAPredictor
    return PNAPredictor(331,30,ref.degree_histogram(),interactions=method=='pna_interaction').to(device),1e-3,0.


def predict(model, method, batch, epoch, training):
    canonical_input(batch)
    if method == 'graphcare': return model(**graphcare_arguments(batch)),0.
    if method.startswith('gchm') or method == 'dtv_gnn': return model(batch),0.
    if method == 'gsat':
        out=model(batch,epoch=epoch,training=training)
        return out['logits'],out['info_loss'] if training else 0.
    if method.startswith('pna'): return model(batch),0.
    logits,_,_,_,dist=model(batch)
    if not training: return logits,0.
    identity=model.model.prototype_class_identity.to(batch.x.device)
    correct=identity[:,batch.y.view(-1)].T.bool(); n=identity.shape[0]//30
    cluster=dist[correct].reshape(-1,n).min(1)[0].mean()
    separation=torch.clamp(1.-dist[~correct].reshape(-1,29*n).min(1)[0],min=0).mean()
    l1=(model.model.last_layer.weight*(1-identity.T)).norm(p=1)
    return logits,.1*cluster+.1*separation+5e-4*l1


class GraphXAIWrapper(nn.Module):
    """Continuous native feature wrapper; categorical membership is fixed metadata.

    Numeric hub channels stay on the computational path during Grad/IG/feature
    perturbation. Edge-only interventions do not erase the observed hub values.
    GraphCare categorical ids are not falsely described as differentiable inputs.
    """
    def __init__(self,model,method,graph,epoch=0):
        super().__init__();self.model=model;self.method=method;self.epoch=epoch
        self.register_buffer('node_ids',graph.node_ids.detach().clone())
        self.register_buffer('node_type',graph.node_type.detach().clone())
    def forward(self,x,edge_index,batch=None):
        if batch is None: batch=torch.zeros(x.shape[0],dtype=torch.long,device=x.device)
        g=Data(x=x,edge_index=edge_index,batch=batch,node_ids=self.node_ids,node_type=self.node_type,
               y=torch.zeros(int(batch.max())+1,dtype=torch.long,device=x.device))
        return predict(self.model,self.method,g,self.epoch,False)[0]
