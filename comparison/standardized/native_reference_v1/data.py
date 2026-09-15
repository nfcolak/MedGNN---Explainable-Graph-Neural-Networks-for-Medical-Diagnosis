"""Lossless native 331-slot graphs; no method-specific preprocessing or fallback."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[3]
DEFAULT = REPO / 'comparison/standardized/native_inputs/protgsat_snapshot_v1'
VERSION = 'native-protgsat-331-star-v1'
PINNED_CONTRACT = '2a2566e662bf87b10d97e58700f868d8d6d6b036582ec046dd5453e6dcd9f7c0'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def check_fields(names):
    for name in names:
        if name.startswith(('disease_', 'symptom_')) or name in {'subject_id','stay_id','hadm_id','icd_codes','disposition','los_hours'}:
            raise ValueError('Direct target/identifier feature forbidden: ' + name)


def transform_native(df, names, fit, expected=None):
    check_fields(names)
    if set(names) - set(df): raise ValueError('Missing native fields')
    stats = {c: {'mean': float(df.iloc[fit][c].mean()), 'std': float(df.iloc[fit][c].std())} for c in names}
    if expected is not None and stats != expected: raise ValueError('Native train-only scaler mismatch')
    result = df[names].copy()
    for c in names:
        s = stats[c]
        result[c] = ((result[c] - s['mean']) / s['std'] if s['std'] > 0 else 0.)
    return result.fillna(0.).to_numpy(dtype=np.float32), stats


class Reference:
    def __init__(self, root=DEFAULT, expected=None, verify_sources=True):
        self.root = Path(root)
        self.contract = json.loads((self.root / 'contract.json').read_text())
        c = self.contract
        if c['version'] != VERSION: raise ValueError('Wrong input version')
        binding = c.copy(); recorded = binding.pop('contract_sha256')
        if digest(binding) != recorded: raise ValueError('Contract fingerprint mismatch')
        if recorded != PINNED_CONTRACT: raise ValueError('Authoritative native contract fingerprint mismatch')
        if expected is not None and expected != recorded: raise ValueError('Expected input fingerprint mismatch')
        if sha(self.root / 'inputs.npz') != c['input_sha256']: raise ValueError('Input fingerprint mismatch')
        if verify_sources:
            for p, h in c['source_bindings'].items():
                if sha(REPO / p) != h: raise ValueError('Source/split/cache fingerprint mismatch: ' + p)
        self.arrays = dict(np.load(self.root / 'inputs.npz', allow_pickle=False))
        a = self.arrays
        if set(a) != {'templates','node_ids','node_ptr','edges','edge_ptr','hub','y','folds'}: raise ValueError('Missing/extra artifact fields')
        if a['hub'].shape != (74511,132) or a['templates'].shape != (193,331): raise ValueError('Native feature width mismatch')
        if np.bincount(a['folds'], minlength=3).tolist() != [59607,7448,7456]: raise ValueError('Split mismatch')
        if not np.isfinite(a['hub']).all(): raise ValueError('Nonfinite native payload')
        self.fingerprint = recorded

    def __len__(self): return len(self.arrays['y'])

    def __getitem__(self, i):
        a = self.arrays
        lo, hi = a['node_ptr'][i:i+2]; el, eh = a['edge_ptr'][i:i+2]
        ids = a['node_ids'][lo:hi].astype(np.int64)
        x = a['templates'][ids].copy()
        x[0,199:] = a['hub'][i]
        types = np.where(ids == 192, 2, np.where(ids < 104, 0, 1))
        return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(a['edges'][:,el:eh].copy()),
                    y=torch.tensor([int(a['y'][i])]), node_type=torch.from_numpy(types),
                    node_ids=torch.from_numpy(ids), graph_ordinal=torch.tensor([i]))

    def fold(self, fold, limit=None):
        rows = np.flatnonzero(self.arrays['folds'] == fold)
        return rows if limit is None else rows[:limit]

    def degree_histogram(self):
        counts = np.diff(self.arrays['node_ptr'])[self.arrays['folds'] == 0] - 1
        hist = np.bincount(counts, minlength=2)
        hist[1] += counts.sum()
        return torch.tensor(hist, dtype=torch.long)


def build(output=DEFAULT):
    import pandas as pd
    from protgnn_analysis.load_dataset import get_dataset, standardized_cache_paths, _canonical_row_indices
    from gsat_analysis.train import get_dataset as gsat_get_dataset
    from shared.lib.benchmark_contract import load_canonical_split
    if gsat_get_dataset is not get_dataset: raise ValueError('Native GSAT loader diverged')
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    split_path = REPO / 'comparison/canonical_split.json'
    split = load_canonical_split(split_path)
    cp, mp = standardized_cache_paths(REPO/'data', split_path, 'star')
    if not cp.is_file(): raise FileNotFoundError('Reference cache missing; never rebuild silently')
    native = get_dataset(str(REPO/'data'), 'mimic_intra_patient_disease', graph_structure='star', canonical_split=split_path)
    meta = json.loads(mp.read_text())
    names = list(native.feature_cols)
    if len(native) != 74511 or len(names) != 331: raise ValueError('Native reference dimensions changed')
    if len(meta['med_vocab']) != 104 or len(meta['cc_vocab']) != 88: raise ValueError('Reference vocabulary changed')
    check_fields(meta['demo_vocab'] + meta['patient_num_vocab'] + meta['med_vocab'])
    if meta['icd_vocab'] or meta['symptom_vocab'] or meta['vital_vocab']: raise ValueError('Unexpected native modality')
    df = pd.read_csv(REPO/'data/merged_ed.csv')
    keep, fit, _ = _canonical_row_indices(df, split)
    df = df.iloc[keep].reset_index(drop=True)
    numeric, stats = transform_native(df, meta['patient_num_vocab'], fit, meta['patient_num_stats'])
    hub = np.concatenate([df[meta['demo_vocab']].to_numpy(dtype=np.float32), numeric], axis=1)
    templates = np.zeros((193,331), dtype=np.float32)
    templates[192,0] = 1
    for j in range(192):
        templates[j, 1 if j < 104 else 3] = 1
        templates[j,4+j] = 1
        templates[j,196] = 1
    node_ids=[]; node_ptr=[0]; edges=[]; edge_ptr=[0]; ys=[]; folds=[]
    tensor_hash = hashlib.sha256(); empty=0
    for i in range(len(native)):
        g = native[i]
        if int(g.subject_id) != int(df.iloc[i]['subject_id']): raise ValueError('Native row identity mismatch')
        ids = np.r_[192, g.x[1:,4:196].argmax(1).numpy()].astype(np.int64)
        x = templates[ids].copy(); x[0,199:] = hub[i]
        if not np.array_equal(x, g.x.numpy()): raise ValueError('Native tensor mismatch at ordinal ' + str(i))
        n = len(ids); expected_edges = {(0,j) for j in range(1,n)} | {(j,0) for j in range(1,n)}
        if set(map(tuple,g.edge_index.T.tolist())) != expected_edges or g.edge_index.shape[1] != 2*(n-1): raise ValueError('Non-star native topology')
        label = int(g.y); sid = str(int(g.subject_id))
        if split['classes'][label] != df.iloc[i]['disease_1']: raise ValueError('Label ordering mismatch')
        e = g.edge_index.numpy()
        for t in [x, e, np.array([label],dtype=np.int64)]: tensor_hash.update(t.tobytes())
        node_ids.extend(ids.tolist()); node_ptr.append(len(node_ids)); edges.append(e); edge_ptr.append(edge_ptr[-1]+e.shape[1]); ys.append(label); folds.append(split['fold'][sid]); empty += n==1
    if np.bincount(folds).tolist() != [59607,7448,7456]: raise ValueError('Fold counts differ')
    bindings = {str(p.relative_to(REPO)): sha(p) for p in [cp,mp,split_path,REPO/'data/merged_ed.csv']}
    output.mkdir(parents=True)
    np.savez_compressed(output/'inputs.npz',templates=templates,node_ids=np.array(node_ids,dtype=np.int64),
        node_ptr=np.array(node_ptr,dtype=np.int64),edges=np.concatenate(edges,axis=1),edge_ptr=np.array(edge_ptr,dtype=np.int64),
        hub=hub,y=np.array(ys,dtype=np.int64),folds=np.array(folds,dtype=np.int64))
    c={'version':VERSION,'input_sha256':sha(output/'inputs.npz'),'source_bindings':bindings,
       'split_sha256':sha(split_path),'labels':split['classes'],'feature_names':names,
       'hub_fields':meta['demo_vocab']+meta['patient_num_vocab'],'patient_num_stats':stats,
       'native_metadata':meta,'native_tensor_sha256':tensor_hash.hexdigest(),
       'native_equal_graphs':len(native),'counts':[59607,7448,7456],'hub_only_graphs':int(empty),
       'temporal_clean':False,'scope':'Native source-snapshot reproduction, not clinical early prediction',
       'missingness':'Exactly native: pnum NaN -> z=0; no new missingness channels; demo unchanged',
       'limitations':{'hx':'Original upstream history, may include current/future visit diagnoses; NOT cleaned',
       'n_ed_visits':'Whole-subject visit count may include future visits',
       'labs':'Whole-visit summaries/abnormal flags; result availability and units not revalidated',
       'vs':'Stay-wide min/max/std, not initial vitals', 'age_bmi':'Native source anchors/documentation time not validated',
       'medications_complaints':'Source filtering/selected-stay lineage/availability not recovered by train-only fitting'}}
    c['contract_sha256']=digest(c); save(output/'contract.json',c)
    ref=Reference(output)
    replay=hashlib.sha256()
    for i in range(len(ref)):
        g=ref[i]
        for t in [g.x.numpy(),g.edge_index.numpy(),g.y.numpy()]: replay.update(t.tobytes())
    if replay.hexdigest()!=tensor_hash.hexdigest(): raise ValueError('Artifact readback differs')
    save(output/'native_audit.json',{'graphs':len(ref),'native_tensor_sha256':replay.hexdigest(),'scaler_exact':True,'artifact_readback_exact':True,'counts':c['counts'],'hub_only_graphs':int(empty)})
    return c


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--build',action='store_true');p.add_argument('--output',type=Path,default=DEFAULT)
    a=p.parse_args()
    if not a.build: p.error('Explicit --build required')
    c=build(a.output);print(json.dumps({'artifact':str(a.output),'graphs':c['native_equal_graphs'],'contract_sha256':c['contract_sha256']}))
