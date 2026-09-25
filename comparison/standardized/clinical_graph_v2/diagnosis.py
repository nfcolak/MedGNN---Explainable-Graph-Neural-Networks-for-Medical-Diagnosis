"""Prior-visit diagnosis history for the v3 clinical graph.

The single hard rule: the index encounter's own diagnoses are the LABEL. They may
never enter a graph. This module therefore never accepts the index `stay_id` in a
diagnosis query, and `build.py` re-derives the same guarantee after writing by
counting how many emitted diagnosis nodes trace back to the index stay. A nonzero
count fails the production.

Timing caveat, stated because it cannot be resolved from the source: MIMIC's
`diagnosis.csv` carries no timestamp at all -- only `stay_id` and `seq_num`. So
"before the cutoff" is established structurally (the diagnosis belongs to an
encounter that had already FINISHED before the index encounter began), never by a
recorded diagnosis time. A diagnosis is billing-coded after its encounter closes,
which is why a completed prior encounter is the correct unit.

ICD-9 and ICD-10 are both present (49%/51%) and 38.6% of patients have their index
and prior encounters coded in different versions. Comparing raw codes across that
boundary silently misses recurrences, so codes are normalized to ICD-10 through the
repository's own GEM table before comparison. Measured effect: recurrence detection
rises from 43.9% to 48.7% of eligible patients; GEM covers 99.2% of ICD-9 codes seen.
Approximate mappings are flagged on the node, never silently treated as exact.
"""
from collections import defaultdict
import csv

from .schema import identifier, rows


def load_icd_map(path):
    """Load the repository's ICD-9 -> ICD-10 GEM table.

    `no_map=1` rows are dropped: those ICD-9 codes have no ICD-10 equivalent and
    inventing one would be a clinical judgement. `approximate=1` is retained as a
    flag so downstream consumers can see the mapping is not exact.
    """
    mapping, approximate = {}, set()
    for row in rows(path, {'icd9_code', 'icd10_code', 'approximate', 'no_map'}):
        code = row['icd9_code'].strip()
        target = row['icd10_code'].strip()
        if not code or not target or row['no_map'] == '1':
            continue
        if code in mapping and mapping[code] != target:
            # Keep the first mapping deterministically rather than picking arbitrarily.
            continue
        mapping[code] = target
        if row['approximate'] == '1':
            approximate.add(code)
    if not mapping:
        raise ValueError('Empty ICD-9 to ICD-10 mapping')
    return mapping, approximate


def normalize(version, code, mapping, approximate):
    """Return (token, is_approximate). ICD-10 passes through; ICD-9 goes via GEM.

    An unmappable ICD-9 code keeps its own namespace instead of being forced into
    the ICD-10 space, so it can still match other ICD-9 occurrences of itself.
    """
    code = code.strip()
    if not code:
        return None, False
    if version == '10':
        return 'dx:icd10:' + code, False
    if version == '9':
        target = mapping.get(code)
        if target is None:
            return 'dx:icd9:' + code, False
        return 'dx:icd10:' + target, code in approximate
    raise ValueError('Unsupported ICD version: ' + version)


class DiagnosisIndex:
    """Diagnoses of the cohort's encounters, keyed by stay.

    Holding index-encounter diagnoses in memory is deliberate: `build.py` needs them
    to PROVE none of them reached a graph. Access is through `prior_history`, which
    refuses the index stay outright, so the guarded and unguarded paths cannot be
    confused at a call site.
    """

    def __init__(self, path, stays, mapping, approximate):
        self.by_stay = defaultdict(list)
        self.counts = defaultdict(int)
        self.mapping = mapping
        self.approximate = approximate
        for row in rows(path, {'subject_id', 'stay_id', 'seq_num', 'icd_code',
                               'icd_version', 'icd_title'}):
            stay = row['stay_id']
            if stay not in stays:
                continue
            self.counts['rows'] += 1
            token, approx = normalize(row['icd_version'], row['icd_code'],
                                      mapping, approximate)
            if token is None:
                self.counts['empty_code'] += 1
                continue
            if token.startswith('dx:icd9:'):
                self.counts['unmapped_icd9'] += 1
            if approx:
                self.counts['approximate_mapping'] += 1
            self.by_stay[stay].append({
                'stay': stay,
                'token': token,
                'seq_num': int(row['seq_num']) if row['seq_num'].strip().isdigit() else None,
                'source_version': row['icd_version'],
                'source_code': row['icd_code'].strip(),
                'title': row['icd_title'].strip(),
                'approximate_mapping': approx,
            })

    def index_tokens(self, stay):
        """Diagnoses of the index encounter -- the LABEL. Never emitted as nodes.

        Exposed only so the producer can verify that none of them appear in the graph.
        """
        return {d['token'] for d in self.by_stay.get(stay, ())}

    def prior_history(self, index_stay, prior_stays):
        """Deduplicated diagnosis history from COMPLETED prior encounters only.

        Refuses the index stay even if a caller passes it in the prior list, so the
        guarantee does not depend on the caller getting the list right.
        """
        if index_stay in prior_stays:
            raise ValueError('Index encounter is not prior history; its diagnoses are the label')
        history = {}
        for stay in prior_stays:
            for record in self.by_stay.get(stay, ()):
                if record['stay'] == index_stay:
                    raise ValueError('Index-encounter diagnosis reached the history path')
                existing = history.get(record['token'])
                # Keep every occurrence so recurrence count and first/last are real.
                if existing is None:
                    history[record['token']] = {**record, 'occurrences': [stay],
                                                'titles': {record['title']},
                                                'approximate_mapping': record['approximate_mapping']}
                else:
                    existing['occurrences'].append(stay)
                    existing['titles'].add(record['title'])
                    existing['approximate_mapping'] |= record['approximate_mapping']
        return [history[token] for token in sorted(history)]
