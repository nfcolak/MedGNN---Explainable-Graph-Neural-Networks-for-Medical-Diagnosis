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
recorded diagnosis time. Availability at encounter completion is an unverified
assumption: this source does not prove when a billing diagnosis became visible.

ICD-9 and ICD-10 can occur in different encounters of the same patient. Comparing
raw codes across that boundary misses recurrences, so both history and targets
use the shared, historical target producer's GEM row-selection policy.
Approximate mappings are flagged on the node, never silently treated as exact.
"""
from collections import defaultdict
from comparison.standardized.icd_mapping import load_icd_map, normalize_icd10

from .schema import rows


def normalize(version, code, mapping, approximate):
    """Return (token, is_approximate). ICD-10 passes through; ICD-9 goes via GEM.

    An unmappable ICD-9 code keeps its own namespace instead of being forced into
    the ICD-10 space, so it can still match other ICD-9 occurrences of itself.
    """
    code = code.strip().upper()
    target = normalize_icd10(version, code, mapping)
    if not code:
        return None, False
    if version == '10':
        return 'dx:icd10:' + code, False
    if version == '9':
        if target is None:
            return 'dx:icd9:' + code, False
        return 'dx:icd10:' + target, code in approximate


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
        prior_stays = list(dict.fromkeys(prior_stays))
        if index_stay in prior_stays:
            raise ValueError('Index encounter is not prior history; its diagnoses are the label')
        history = {}
        for stay in prior_stays:
            for record in self.by_stay.get(stay, ()):
                if record['stay'] == index_stay:
                    raise ValueError('Index-encounter diagnosis reached the history path')
                existing = history.get(record['token'])
                # Recurrence is a count of completed stays, not diagnosis rows.
                # Multiple source codes can also map to the same token in a stay.
                if existing is None:
                    history[record['token']] = {**record, 'occurrences': {stay},
                                                'titles': {record['title']},
                                                'approximate_mapping': record['approximate_mapping']}
                else:
                    existing['occurrences'].add(stay)
                    existing['titles'].add(record['title'])
                    existing['approximate_mapping'] |= record['approximate_mapping']
        return [{**history[token], 'occurrences': sorted(history[token]['occurrences'])}
                for token in sorted(history)]
