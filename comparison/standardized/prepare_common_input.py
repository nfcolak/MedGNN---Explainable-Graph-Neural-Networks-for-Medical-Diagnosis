"""One-shot real source audit and isolated common-input materialization."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch
from comparison.standardized.common_input_improvement import ROOT, fit_snapshot, pyg_graph, graphcare_graph
from comparison.standardized.performance_review_20260913.evidence import save
from graphcare_analysis.build_kg import canonical_row_indices

ROOT.mkdir(exist_ok=False)
old = Path('comparison/standardized/performance_review_20260913')
paths = set(json.loads((old/'preservation_before.json').read_text()))
paths.update(str(p) for p in old.rglob('*') if p.is_file())
paths.update(p for p in subprocess.check_output(['git', 'diff', '--name-only'], text=True).splitlines() if Path(p).is_file())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
preservation = {p:{'size':Path(p).stat().st_size,'sha256':sha(p)} for p in sorted(paths)}
save(ROOT/'preservation_before.json', preservation)
(ROOT/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
source=Path('data/merged_ed.csv'); split_path=Path('comparison/canonical_split.json')
df=pd.read_csv(source,low_memory=False); split=json.loads(split_path.read_text())
keep, fit, keys = canonical_row_indices(df, split)
names, presence = fit_snapshot(df, fit)
assert len(names)==192 and len(keep)==74511 and len(fit)==59607
folds=np.array([split['fold'][keys[i]] for i in keep],dtype=np.int8)
y=df.disease_1.map({c:i for i,c in enumerate(split['classes'])}).to_numpy()[keep]
assert not pd.isna(y).any(); y=y.astype(np.int64); presence=presence[keep]
assert np.bincount(folds).tolist()==[59607,7448,7456]
# Independent parity against existing PyG cached-concept evidence, not its native hub.
prior=np.load(old/'audit/features_no_identifiers.npz')
meta=json.loads(next(Path('data/graphs/star/protgnn').glob('*recipe55*/metadata.json')).read_text())
prior_names=['med:'+c[4:] for c in meta['med_vocab']]+['cc:'+c for c in meta['cc_vocab']]
reorder=[prior_names.index(n) for n in names]
assert np.array_equal(y,prior['y']) and np.array_equal(folds,prior['folds'])
assert np.array_equal(presence,prior['x'][:,reorder])
# Actual all-row feature/edge equality at effective encoder inputs, including hub.
h=hashlib.sha256(); torch.set_num_threads(4)
for i,row in enumerate(presence):
    codes=np.flatnonzero(row); p=pyg_graph(codes,int(y[i]),192); g=graphcare_graph(codes,int(y[i]),192)
    expected=torch.nn.functional.one_hot(g['node_ids'],193).float()
    assert torch.equal(p.x,expected) and torch.equal(p.edge_index,g['edge_index'])
    assert p.x[-1,-1]==1 and p.x[-1,:192].sum()==0
    assert g['ehr_nodes'][-1]==0 and g['visit_node'][-1]==1
    h.update(p.x.numpy().tobytes()); h.update(p.edge_index.numpy().tobytes())
np.savez_compressed(ROOT/'inputs_no_identifiers.npz',presence=presence,y=y,folds=folds)
# Availability evidence: inspect raw schema and aggregate medication charttime vs intake.
raw=Path('data/Original CSVs')
headers={n:next(csv.reader((raw/n).open())) for n in ('medrecon.csv','triage.csv','edstays.csv')}
med=pd.read_csv(raw/'medrecon.csv',usecols=['stay_id','charttime'])
stays=pd.read_csv(raw/'edstays.csv',usecols=['stay_id','intime'])
joined=med.merge(stays,on='stay_id',how='left',validate='many_to_one')
delta=(pd.to_datetime(joined.charttime,errors='coerce')-pd.to_datetime(joined.intime,errors='coerce')).dt.total_seconds()
timing={'raw_medication_rows':len(med),'valid_intake_charttime_pairs':int(delta.notna().sum()),
        'charttime_after_intake':int((delta>0).sum()),'charttime_at_or_before_intake':int((delta<=0).sum()),
        'median_seconds_after_intake':float(delta.median()),'scope':'all raw medication rows, not canonical selected visits'}
contract={'name':'medication-history-available source-snapshot diagnostic',
 'temporal_clean':False,'early_triage_eligible':False,'raw_to_model_train_only':False,
 'blockers':['Merged source lacks stay_id/intime/availability cutoff; selected stay cannot be uniquely recovered by simple subject join.',
             'Raw medrecon has charttime but merge_ed aggregates every medication row in the stay without cutoff.',
             'Raw triage has no charttime; prediction/diagnosis availability timestamp not established.',
             'Whole-source top300 medication selection and row filter, complaint prevalence and selected cohort already baked into snapshot.'],
 'source_schema':headers,'timing_audit':timing,'source_sha256':sha(source),'split_sha256':sha(split_path),
 'fit_scope':'canonical train rows of already-filtered source snapshot only','fit_rows':len(fit),
 'fit_identity_sha256':hashlib.sha256('\n'.join(sorted(keys[i] for i in fit)).encode()).hexdigest(),
 'names':names,'classes':split['classes'],'concept_channels':192,'constant_hub_channels':1,
 'effective_feature_dim':193,'patient_specific_hub_channels':0,'excluded_native_hub_channels':132,
 'counts':np.bincount(folds).tolist(),'hub_only_by_fold':[int(((presence.sum(1)==0)&(folds==f)).sum()) for f in range(3)],
 'audited_graphs':len(y),'feature_edge_parity_sha256':h.hexdigest(),
 'encoder_input_contract':'PyG float one-hot(node_id,193) exactly equals GraphCare categorical node_id expanded to same one-hot. Learned embeddings, pooling, visit/EHR membership treatment remain method-specific; not hidden embedding equality.',
 'hub':'last node identity 192; one constant one-hot; no demographic/numeric/label payload',
 'input_sha256':sha(ROOT/'inputs_no_identifiers.npz'),'test_evaluated':False}
save(ROOT/'input_contract_and_audit.json',contract)
save(ROOT/'plan.json',{'seed':1234,'topology':'star','epoch_cap':30,'batch_size':128,'updates_per_epoch':int(np.ceil(59607/128)),
 'cells':[['protgnn','current'],['protgnn','sqrt_inverse'],['graphcare','current'],['graphcare','sqrt_inverse'],['gsat','current']],
 'selection':'validation macro_f1 only; no early stop; best and epoch30 both reported',
 'scope':contract['name'],'temporal_clean':False,'test_evaluated':False,
 'protgnn':{'warm_epochs':10,'projection_epoch_zero_based':20,'nearest_graphs_per_prototype':10,'projection':'original MCTS; train candidates only'},
 'batch_change':'GraphCare native32 ->128 for both conditions; ProtGNN/GSAT128 unchanged; equal updates not equal wall time/capacity'})
print(json.dumps({k:contract[k] for k in ('counts','concept_channels','hub_only_by_fold','audited_graphs','timing_audit','temporal_clean')},indent=2))
