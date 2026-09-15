"""Read-only all-graph adapter audit plus preservation evidence (no training)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch
from .data import REPO, DEFAULT, Reference, sha, save
from .models import METHODS, canonical_input, graphcare_arguments


def audit(method, output):
    ref=Reference();h=hashlib.sha256();count=0
    for i in range(len(ref)):
        g=ref[i];b=Batch.from_data_list([g])
        x,edges,y=canonical_input(b)
        if method=='graphcare':
            args=graphcare_arguments(b)
            # Invert the actual categorical+continuous consumer, not a parallel builder.
            decoded=torch.tensor(ref.arrays['templates'])[args['node_ids']].clone()
            decoded[:,199:]=args['numeric']
            if not torch.equal(decoded,x):raise ValueError('GraphCare categorical/numeric information mismatch')
            if int(args['ehr_nodes'].sum())!=g.num_nodes-1:raise ValueError('GraphCare membership mismatch')
            if not torch.equal(args['visit_node'][:,0],args['ehr_nodes']):raise ValueError('GraphCare visit mismatch')
            if not (args['rel_ids']==2).all():raise ValueError('Relation mismatch')
            x=decoded;edges=args['edge_index']
        for t in [x,edges,y]:h.update(t.numpy().tobytes())
        count+=1
    if h.hexdigest()!=ref.contract['native_tensor_sha256']:raise ValueError('Native baseline tensor parity failed')
    result={'method':method,'graphs':count,'counts':ref.contract['counts'],'native_tensor_sha256':h.hexdigest(),
            'contract_sha256':ref.fingerprint,'native_payload_bitwise_equal':True,'before_learned_transformations':True}
    save(output,result);print(json.dumps(result));return result


def preservation(output, before=None):
    if before is None:
        paths=set()
        for folder in ['data','comparison','protgnn_analysis/outputs','gsat_analysis/outputs','graphcare_analysis/outputs','pna_analysis/outputs']:
            for p in (REPO/folder).rglob('*'):
                if p.is_file() and not any(x in p.parts for x in ['native_runs','native_inputs','native_reference_v1','native_evidence','__pycache__']):paths.add(p)
        for folder in ['protgnn_analysis','gsat_analysis','graphcare_analysis','pna_analysis','shared']:
            paths.update((REPO/folder).rglob('*.py'))
        result={str(p.relative_to(REPO)):sha(p) for p in sorted(paths)}
    else:
        old=json.loads(Path(before).read_text());result={p:sha(REPO/p) for p in old}
        if result!=old:raise ValueError('Preserved legacy file changed')
    save(output,result);print(json.dumps({'preserved_files':len(result),'matched_before':before is not None}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preserve',action='store_true');p.add_argument('--before',type=Path)
    a=p.parse_args()
    if a.preserve:preservation(a.output,a.before)
    elif a.method:audit(a.method,a.output)
    else:p.error('method or preserve required')
