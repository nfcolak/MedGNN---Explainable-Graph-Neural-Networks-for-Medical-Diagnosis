"""Shared ICD crosswalk, preserving the frozen local_labels_v2 target policy.

Do not replace the historical pandas quicksort tie order with first-file-row or
stable-sort order: either can change existing targets. Class selection/order is
owned by the frozen target ontology, never by this helper. Pin pandas/numpy and
mapping-file hashes when reproducing the historical tie policy.
"""
import pandas as pd

ICD_MAPPING_POLICY = 'local_labels_v2_gem_approximate_quicksort_first_v1'


def load_icd_map(path):
    """Return (crosswalk, approximate_codes) for the SAME selected GEM rows."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'icd9_code', 'icd10_code', 'approximate', 'no_map'}
    if not required.issubset(frame.columns):
        raise ValueError('ICD mapping is missing required columns')
    frame['approximate'] = pd.to_numeric(frame.approximate, errors='raise')
    selected = (frame[frame.no_map == '0'].sort_values('approximate', kind='quicksort')
                .drop_duplicates('icd9_code'))
    # Uppercasing, without new stripping or numeric coercion, matches the target
    # producer exactly; diagnosis input strings are normalized separately below.
    mapping, flags = {}, {}
    for row in selected.itertuples(index=False):
        code, target = row.icd9_code.upper(), row.icd10_code.upper()
        if not code or not target:
            raise ValueError('Empty code in selected ICD mapping')
        mapping[code] = target
        flags[code] = row.approximate == 1
    if not mapping:
        raise ValueError('Empty ICD-9 to ICD-10 mapping')
    return mapping, {code for code, flag in flags.items() if flag}


def normalize_icd10(version, code, mapping):
    """Map a diagnosis to ICD-10; return None for empty/unmapped ICD-9 codes."""
    code = code.strip().upper()
    if version not in ('9', '10'):
        raise ValueError('Unsupported ICD version: ' + version)
    if not code:
        return None
    return code if version == '10' else mapping.get(code)
