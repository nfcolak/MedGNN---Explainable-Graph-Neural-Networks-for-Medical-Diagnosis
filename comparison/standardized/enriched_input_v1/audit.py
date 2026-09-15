"""Aggregate-only lineage checks; current diagnosis is NEVER a model feature."""
import numpy as np
import pandas as pd


def guard_history(frame, index, stays, diagnoses, fields):
    """Reconstruct indicators from strictly prior ended visits, not source hx bits.

    End-of-visit is a conservative event-time guard, NOT documentation-time proof.
    Unknown lineage becomes missing. Zero means no supported prior recorded
    category, not absence of disease. Unsafe source bits cannot leak via missingness.
    """
    if index.subject_id.duplicated().any():
        raise ValueError('Index lineage must be unique')
    clean = frame.copy()
    prior = stays.merge(index[['subject_id', 'stay_id', 'intime']], on='subject_id',
                        suffixes=('', '_index'), validate='many_to_one')
    def time(series):
        return pd.to_datetime(series, errors='coerce', format='mixed')
    eligible = ((time(prior.intime) < time(prior.intime_index)) &
                (time(prior.outtime) < time(prior.intime_index)) &
                prior.stay_id.ne(prior.stay_id_index))
    safe = prior.loc[eligible, ['subject_id', 'stay_id']].merge(
        diagnoses[['subject_id', 'stay_id', 'history_field']], on=['subject_id', 'stay_id'])
    supported = set(zip(safe.subject_id, safe.history_field))
    known = frame.subject_id.isin(index.subject_id).to_numpy()
    removed = 0
    per_field = {}
    for name in fields:
        value = pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=float)
        support = np.array([(sid, name) in supported for sid in frame.subject_id])
        invalid = (value == 1) & ~support
        removed += int(invalid.sum())
        per_field[name] = {'source_positive': int((value == 1).sum()),
                           'supported_prior_ended_positive': int(((value == 1) & support & known).sum()),
                           'unsupported_positive_removed': int(invalid.sum()),
                           'reconstructed_positive': int((support & known).sum())}
        value = support.astype(float)
        value[~known] = np.nan
        clean[name] = value
    return clean, {'documentation_time_verified': False,
                   'status': 'historical_source_summary_timing_unverified',
                   'unsupported_positive_values_removed': removed,
                   'unknown_index_rows': int((~known).sum()), 'fields': per_field,
                   'prior_ended_stays_considered': int(eligible.sum())}


def audit_source(frame, raw_root, fields):
    """Recover only unique exact triage(+available lab-summary) fingerprints.

    No target or current ICD enters the matching key. Clinical fingerprints are
    reconstruction evidence, not retained encounter lineage in the original CSV.
    """
    import re
    from pathlib import Path
    from shared.lib.benchmark_contract import file_sha256
    raw_root = Path(raw_root)
    required = ['triage.csv', 'edstays.csv', 'diagnosis.csv', 'icd9_to_icd10_mapping.csv']
    if any(not (raw_root / n).is_file() for n in required):
        clean = frame.copy(); clean[fields] = np.nan
        return clean, {'status': 'raw_lineage_unavailable_history_excluded',
                       'lineage': {'unique_matches': 0, 'ambiguous_or_unmatched': len(frame)},
                       'documentation_time_verified': False, 'raw_hashes': {}}
    raw_hashes = {n: file_sha256(raw_root / n) for n in required}
    triage = pd.read_csv(raw_root / 'triage.csv', low_memory=False)
    stays = pd.read_csv(raw_root / 'edstays.csv', low_memory=False)
    if stays.stay_id.duplicated().any() or triage.stay_id.duplicated().any():
        raise ValueError('Raw stays/triage must have unique stay keys')
    triage['temperature'] = ((triage.temperature - 32.) * 5. / 9.).round(3)
    keys = ['subject_id', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'acuity']
    # All present initial fields must agree exactly after documented source rounding.
    match = frame[keys].merge(triage[keys + ['stay_id']], on=keys, how='left')
    candidate_counts = match.groupby('subject_id').stay_id.count()
    initial_unique = int((candidate_counts == 1).sum())
    lab_path = raw_root / 'ed_labs.csv'
    if lab_path.is_file():
        labs = pd.read_csv(lab_path, low_memory=False)
        from .spec import LAB_ITEMS
        lab_keys = ['lab_' + n for n in LAB_ITEMS if 'lab_' + n in frame and 'lab_' + n in labs]
        if lab_keys:
            match = match.merge(labs[['stay_id'] + lab_keys], on='stay_id', how='left', validate='many_to_one')
            match = match.merge(frame[['subject_id'] + lab_keys], on='subject_id', suffixes=('_raw', '_source'), validate='many_to_one')
            equal = np.ones(len(match), dtype=bool)
            for n in lab_keys:
                left, right = match[n + '_raw'], match[n + '_source']
                equal &= (left.eq(right) | (left.isna() & right.isna())).to_numpy()
            match = match[equal]
        raw_hashes['ed_labs.csv'] = file_sha256(lab_path)
    counts = match.groupby('subject_id').stay_id.count()
    index = match[match.subject_id.isin(counts[counts == 1].index)][['subject_id', 'stay_id']]
    index = index.merge(stays[['subject_id', 'stay_id', 'intime']], on=['subject_id', 'stay_id'], validate='one_to_one')
    # Reconstruct the legacy category-to-title map for validation only (never fit new
    # feature vocabulary from all subjects). Existing history vocabulary is frozen.
    diagnoses = pd.read_csv(raw_root / 'diagnosis.csv', dtype={'icd_code': str}, low_memory=False)
    diagnoses.icd_code = diagnoses.icd_code.str.strip().str.upper()
    title_map = diagnoses[diagnoses.icd_version == 10].drop_duplicates('icd_code').set_index('icd_code').icd_title.to_dict()
    crosswalk = pd.read_csv(raw_root / 'icd9_to_icd10_mapping.csv')
    crosswalk = crosswalk[crosswalk.no_map == 0].sort_values('approximate').drop_duplicates('icd9_code')
    map9 = dict(zip(crosswalk.icd9_code.astype(str).str.upper(), crosswalk.icd10_code.astype(str).str.upper()))
    code10 = diagnoses.icd_code.where(diagnoses.icd_version == 10, diagnoses.icd_code.map(map9))
    diagnoses['cat3'] = code10.fillna('').str[:3]
    diagnoses['title'] = code10.map(title_map).fillna('').astype(str).str.split(',').str[0].str.strip()
    disease = ~diagnoses.cat3.str[:1].isin(['', 'V', 'W', 'X', 'Y', 'Z', 'R']) & diagnoses.title.ne('')
    names = diagnoses[disease].groupby('cat3').title.agg(lambda s: s.value_counts().index[0])
    cat_field = {cat: 'hx_' + re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_') for cat, title in names.items()}
    diagnoses['history_field'] = diagnoses.cat3.map(cat_field)
    clean, evidence = guard_history(frame, index, stays, diagnoses, fields)
    earliest = stays.groupby('subject_id').intime.min()
    first = set(index.loc[index.intime.eq(index.subject_id.map(earliest)), 'subject_id'])
    first_hx = frame.loc[frame.subject_id.isin(first), fields].sum(axis=1)
    evidence.update({'raw_hashes': raw_hashes,
                     'lineage': {'initial_triage_unique_matches': initial_unique,
                                 'unique_matches': len(index),
                                 'ambiguous_or_unmatched': len(frame) - len(index),
                                 'matching_policy': 'exact subject+initial triage fields+all available 26 lab means; unique only; never current diagnosis'},
                     'first_raw_visit_matched': len(first),
                     'first_raw_visit_with_source_history': int((first_hx > 0).sum()),
                     'category_mapping': {n: sorted(c for c, f in cat_field.items() if f == n) for n in fields},
                     'lab_timing': 'whole ED stay charttime-window mean; storetime/valueuom dropped; no early-result claim',
                     'raw_labevents_available': (raw_root / 'labevents.csv').is_file(),
                     'raw_labevents_scanned': False})
    return clean, evidence
