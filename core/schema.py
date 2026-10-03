"""Closed input contracts for the v2 clinical decision-point graph.

Every rule here is a refusal, not a repair. Identifiers stay metadata, timestamps
stay in source form, and no clinical mapping is invented: the complaint vocabulary
is a frequency allowlist fitted on the TRAIN fold only, never a curated symptom
ontology and never derived from the label list.
"""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

from comparison.standardized.event_graph_v1.schema import (  # reuse audited primitives
    LAB_IDS, identifier, numeric, rows, sha256, timestamp,
)

KNOWLEDGE_RELATIONS = {'is_a', 'member_of', 'measures', 'assesses', 'may_affect', 'interacts_with'}

# Triage fields recorded at arrival. Temperature is Fahrenheit in the raw ED table
# and is converted exactly once, here, with the source unit recorded alongside.
TRIAGE_VITALS = {'temperature': 'Celsius', 'heartrate': 'beats/min',
                 'resprate': 'breaths/min', 'o2sat': 'percent',
                 'sbp': 'mmHg', 'dbp': 'mmHg'}
# Plausibility gates transcribed from the audited legacy spec. Out-of-range values
# are dropped with a counter, never clipped into range.
VITAL_RANGES = {'temperature': (25., 45.), 'heartrate': (5., 300.), 'resprate': (2., 120.),
                'o2sat': (50., 100.), 'sbp': (40., 300.), 'dbp': (0., 200.)}

# Arrival-time encounter fields. `disposition` is deliberately absent: it is the
# outcome of the visit and is not knowable at the decision point.
ARRIVAL_FIELDS = ('gender', 'race', 'arrival_transport')
FORBIDDEN_EDSTAYS_FIELDS = {'disposition', 'hadm_id'}

_WHITESPACE = re.compile(r'\s+')


def normalize_complaint(value):
    """Casefold and collapse whitespace. No synonym mapping, no clinical grouping.

    Returning the surface form keeps the vocabulary auditable: every token in the
    artifact can be grepped back to raw triage rows. Any mapping of `abd pain` onto
    `abdominal pain` would be an unreviewed clinical judgement, so it is not made.
    """
    if not isinstance(value, str):
        return None
    token = _WHITESPACE.sub(' ', value.strip().casefold())
    return token or None


def split_complaints(value):
    """Split the raw comma-separated triage free-text field into surface tokens."""
    if not isinstance(value, str):
        return []
    seen, result = set(), []
    for part in value.split(','):
        token = normalize_complaint(part)
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def fit_complaint_vocabulary(counts, min_count):
    """Frequency allowlist fitted on training-fold visits only.

    Rare surface forms are dropped and counted, never bucketed into an `other`
    node: an `other` node would make graphs comparable on a quantity that has no
    consistent meaning across patients.
    """
    if min_count < 1:
        raise ValueError('Minimum complaint count must be positive')
    kept = sorted(token for token, count in counts.items() if count >= min_count)
    if not kept:
        raise ValueError('Empty complaint vocabulary; lower --complaint-min-count')
    return kept


def read_cohort(path):
    """Read the inherited v1 cohort unchanged.

    Sample identity, visit lineage, cutoff and fold all come from the existing
    artifact. This producer is forbidden from re-deriving a cutoff: doing so would
    silently move the prediction moment and break comparability with v1.
    """
    required = {'sample_id', 'subject_id', 'stay_id', 'cutoff', 'split'}
    samples, ids, subject_fold, encounters, stay_subjects = [], set(), {}, set(), {}
    for row in rows(path, required):
        if not row['sample_id'] or row['sample_id'] in ids:
            raise ValueError('Empty or duplicate sample_id')
        ids.add(row['sample_id'])
        for key in ('subject_id', 'stay_id'):
            identifier(row[key])
        if stay_subjects.setdefault(row['stay_id'], row['subject_id']) != row['subject_id']:
            raise ValueError('One stay cannot belong to multiple subjects')
        if row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('split must be train, validation or test')
        if subject_fold.setdefault(row['subject_id'], row['split']) != row['split']:
            raise ValueError('One subject cannot cross folds')
        cutoff = timestamp(row['cutoff'])
        key = (row['stay_id'], cutoff)
        if key in encounters:
            raise ValueError('Duplicate stay/cutoff sample')
        encounters.add(key)
        samples.append({k: row[k] for k in required} | {'cutoff': cutoff})
    if not samples:
        raise ValueError('Empty cohort')
    return samples


def read_knowledge(path):
    """Versioned, provenance-bearing relation allowlist. Source-checked, not clinically reviewed."""
    result, seen = [], set()
    required = {'source_token', 'relation', 'target_token', 'source_uri',
                'source_version', 'reviewed_by'}
    for row in rows(path, required):
        if any(not row[key].strip() for key in required):
            raise ValueError('Every knowledge relation requires complete provenance')
        if row['relation'] not in KNOWLEDGE_RELATIONS:
            raise ValueError('Unsupported medical relation')
        if not row['source_token'].startswith(('lab:', 'vital:')):
            raise ValueError('Knowledge source must be an observed concept namespace')
        if not row['target_token'].startswith('knowledge:'):
            raise ValueError('Knowledge target must use knowledge: namespace')
        if not row['source_uri'].startswith(('https://', 'http://', 'urn:')):
            raise ValueError('Knowledge requires a traceable URI')
        key = tuple(row[k] for k in ('source_token', 'relation', 'target_token'))
        if key in seen:
            raise ValueError('Duplicate knowledge triple')
        seen.add(key)
        result.append({key: row[key] for key in required})
    if not result:
        raise ValueError('Knowledge input is empty')
    return result


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    Path(path).chmod(0o600)


__all__ = ['ARRIVAL_FIELDS', 'FORBIDDEN_EDSTAYS_FIELDS', 'KNOWLEDGE_RELATIONS',
           'LAB_IDS', 'TRIAGE_VITALS', 'VITAL_RANGES', 'fit_complaint_vocabulary',
           'identifier', 'normalize_complaint', 'numeric', 'read_cohort',
           'read_knowledge', 'rows', 'save_json', 'sha256', 'split_complaints',
           'timestamp']
