"""Exact subject/stay binding; preserve literal ICD codes and frozen class order.

Reconstruct the historical category-title ontology on its HASH-PINNED original
source, before selecting event visits. Never refit category names or rank classes
on the new event cohort. The inherited ontology is frequency-derived and is not
claimed to be train-fitted. No clinical mapping is invented here.
"""
from pathlib import Path
import json
import pandas as pd
from .labels import DEFAULT_GRAPH_ROOT, DEFAULT_RAW_ROOT, DEFAULT_LABELS, REPO, sha256, _load_labels, _load_disease_merges, _json_save


def literal_ids(frame):
    for key in ('subject_id', 'stay_id'):
        if not frame[key].str.fullmatch(r'[1-9][0-9]*').all():
            raise ValueError('Invalid literal identifier: ' + key)
    if frame.groupby('stay_id').subject_id.nunique().gt(1).any():
        raise ValueError('Stay belongs to conflicting subjects')


def build(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    historical_path = DEFAULT_GRAPH_ROOT.parent / 'first_recorded_lab_all_visits_v2_targets_v1/binding_manifest.json'
    historical = json.loads(historical_path.read_text())
    source_hashes = {
        'raw_diagnosis_sha256': sha256(DEFAULT_RAW_ROOT / 'diagnosis.csv'),
        'icd_mapping_sha256': sha256(DEFAULT_RAW_ROOT / 'icd9_to_icd10_mapping.csv'),
        'merge_policy_sha256': sha256(REPO / 'shared/data_prep/merge_ed.py'),
    }
    if any(historical[k] != v for k, v in source_hashes.items()):
        raise ValueError('Original ontology source changed; refuse new-outcome reconstruction')
    labels = _load_labels(DEFAULT_LABELS)
    gm = json.loads((DEFAULT_GRAPH_ROOT / 'manifest.json').read_text())
    if gm['status'] != 'completed' or gm['reference_class_order_only'] != labels or historical['labels'] != labels:
        raise ValueError('Frozen graph/class order mismatch')
    d = pd.read_csv(DEFAULT_RAW_ROOT / 'diagnosis.csv', dtype=str, keep_default_na=False)
    literal_ids(d)
    if not d.icd_version.isin(['9', '10']).all() or not d.seq_num.str.fullmatch(r'[1-9][0-9]*').all():
        raise ValueError('Invalid diagnosis version/priority')
    # Code case/whitespace canonicalization follows merge_ed.py; IDs are never normalized.
    d['icd_code'] = d.icd_code.str.strip().str.upper()
    d['icd_title'] = d.icd_title.str.strip()
    d['seq_num'] = d.seq_num.astype('int64')
    titles = d[d.icd_version == '10'].drop_duplicates('icd_code').set_index('icd_code').icd_title.to_dict()
    m = pd.read_csv(DEFAULT_RAW_ROOT / 'icd9_to_icd10_mapping.csv', dtype=str, keep_default_na=False)
    m['approximate'] = pd.to_numeric(m.approximate, errors='raise')
    # Keep the historical sorting/tie policy, but NEVER infer ICD columns as numeric.
    m = m[m.no_map == '0'].sort_values('approximate').drop_duplicates('icd9_code')
    crosswalk = dict(zip(m.icd9_code.str.upper(), m.icd10_code.str.upper()))
    d['icd10_code'] = [c if v == '10' else crosswalk.get(c, '') for c, v in zip(d.icd_code, d.icd_version)]
    d['title'] = d.icd10_code.map(lambda c: titles.get(c, '').split(',')[0].strip())
    d['cat3'] = d.icd10_code.str[:3]
    usable = ~d.cat3.str[:1].isin(['V', 'W', 'X', 'Y', 'Z', 'R', '']) & ~d.title.isin(['', 'nan'])
    # This is the inherited merge_ed ontology reconstruction, not a new event-cohort vote.
    cat_title = d[usable].groupby('cat3').title.agg(lambda s: s.value_counts().index[0]).to_dict()
    merges = _load_disease_merges()
    category_map = {k: merges.get(v, v) for k, v in cat_title.items() if merges.get(v, v) in labels}
    d['label'] = d.cat3.map(category_map).fillna('')
    selected = d[usable & d.label.isin(labels)].sort_values(['subject_id', 'stay_id', 'seq_num'], kind='stable')
    groups = selected.groupby(['subject_id', 'stay_id']).label.agg(lambda s: list(dict.fromkeys(s))).reset_index()
    groups['n_selected_diseases'] = groups.label.map(len)
    groups['label'] = groups.label.map(lambda x: x[0] if len(x) == 1 else '')
    groups['target'] = groups.label.map({label: i for i, label in enumerate(labels)}).fillna(-1).astype('int64')
    cohort = pd.read_csv(DEFAULT_GRAPH_ROOT / 'cohort.csv', dtype=str, keep_default_na=False)
    literal_ids(cohort)
    if cohort.sample_id.duplicated().any() or cohort.stay_id.duplicated().any():
        raise ValueError('Duplicate cohort identity')
    if cohort.groupby('subject_id').split.nunique().gt(1).any():
        raise ValueError('Patient crosses canonical folds')
    # Cross-check ALL diagnosis pairs, not only labelled rows, before the two-key join.
    pairs = d[['subject_id', 'stay_id']].drop_duplicates()
    check = cohort[['subject_id', 'stay_id']].merge(pairs, on='stay_id', how='inner', suffixes=('_graph', '_diagnosis'))
    if check.subject_id_graph.ne(check.subject_id_diagnosis).any():
        raise ValueError('Graph/diagnosis subject mismatch')
    bound = cohort.merge(groups, on=['subject_id', 'stay_id'], how='left', validate='one_to_one')
    bound['target'] = bound.target.fillna(-1).astype('int64')
    bound['label'] = bound.label.fillna('')
    bound['n_selected_diseases'] = bound.n_selected_diseases.fillna(0).astype('int64')
    bound['label_status'] = bound.n_selected_diseases.map(lambda n: 'labelled' if n == 1 else ('multiple_frozen_classes' if n > 1 else 'no_frozen_class'))
    matched = bound[bound.target >= 0]
    output.mkdir(parents=True)
    bound.to_csv(output / 'targets.csv', index=False)
    (output / 'targets.csv').chmod(0o600)
    _json_save(output / 'frozen_category_map.json', {'category_to_label': category_map, 'category_to_historical_title': cat_title,
        'source_hashes': source_hashes, 'labels': labels, 'policy': 'Reconstructed inherited merge_ed.py ontology on the identical hash-pinned historical raw source; no event-cohort vote or top-N reselection.'})
    manifest = {'schema_version': 'event_graph_target_binding_v1', 'status': 'completed',
        'graph_artifact': str(DEFAULT_GRAPH_ROOT), 'graph_manifest_sha256': sha256(DEFAULT_GRAPH_ROOT / 'manifest.json'),
        'graph_sha256': gm['graphs_sha256'], 'cohort_sha256': gm['cohort_sha256'],
        'labels': labels, 'labels_path': str(DEFAULT_LABELS), 'labels_sha256': sha256(DEFAULT_LABELS), **source_hashes,
        'source_code_sha256': sha256(Path(__file__)), 'historical_binding_sha256': sha256(historical_path),
        'target_policy': 'Exact literal subject_id+stay_id; ordered unique frozen-class diagnoses; exactly one final selected disease as in merge_ed.py single-disease filter. Multilabel/unmatched remain target=-1. ICD strings preserve leading zeros.',
        'counts': {'cohort_samples': len(bound), 'matched_samples': len(matched), 'unmatched_samples': int((bound.target < 0).sum()),
            'status_counts': {str(k): int(v) for k,v in bound.label_status.value_counts().items()},
            'by_split': {split: {'samples': int((matched.split == split).sum()), 'patients': int(matched.loc[matched.split == split, 'subject_id'].nunique())} for split in ['train','validation','test']},
            'class_counts': {str(i): int((matched.target == i).sum()) for i in range(len(labels))}},
        'limitations': ['temporal_clean=false; storetime is an availability proxy.', 'Inherited label ontology was historically frequency-derived, not train-only fitted.', 'Visit-level single-disease target policy, not reproduction of all legacy demographic/complete-case cohort filters.', 'Diagnosis targets never become graph covariates.'],
        'artifact_files': {name: sha256(output / name) for name in ['targets.csv','frozen_category_map.json']}}
    _json_save(output / 'binding_manifest.json', manifest)
    print(json.dumps(manifest['counts']), flush=True)
    return manifest

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
