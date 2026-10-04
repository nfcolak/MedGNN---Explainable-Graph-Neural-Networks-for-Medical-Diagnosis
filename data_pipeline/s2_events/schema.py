"""Closed input contracts. Clinical identifiers are metadata, never features."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

from comparison.standardized.enriched_input_v1.spec import LAB_ITEMS

LAB_IDS = {str(value) for value in LAB_ITEMS.values()}
# Raw ED temperatures are Fahrenheit; conversion occurs exactly once at ingestion.
VITAL_UNITS = {'temperature': 'Celsius', 'heartrate': 'beats/min',
               'resprate': 'breaths/min', 'o2sat': 'percent',
               'sbp': 'mmHg', 'dbp': 'mmHg'}
KNOWLEDGE_RELATIONS = {'is_a', 'member_of', 'measures', 'assesses', 'may_affect', 'interacts_with'}


def identifier(value):
    """Reject malformed numeric identifiers instead of repairing 123.0 to 123."""
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]+', value):
        raise ValueError('Expected an unchanged decimal source identifier')
    return value


def timestamp(value):
    if not isinstance(value, str) or not value:
        raise ValueError('Explicit nonempty source timestamp required')
    result = datetime.fromisoformat(value)
    if result.tzinfo is not None:
        raise ValueError('MIMIC local shifted timestamps must be timezone-naive')
    return result


def numeric(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def rows(path, required):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f'{Path(path).name}: missing columns {sorted(set(required) - set(reader.fieldnames or []))}')
        yield from reader


def read_cohort(path):
    """Retain every supplied visit/cutoff, including repeated subjects.

    Each visit has its own target and prediction cutoff. All visits of a patient
    must share a fold; subject-level deduplication is deliberately forbidden.
    """
    required = {'sample_id', 'subject_id', 'stay_id', 'cutoff', 'split'}
    samples, ids, subjects, encounters = [], set(), {}, set()
    stay_subjects = {}
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
        previous = subjects.setdefault(row['subject_id'], row['split'])
        if previous != row['split']:
            raise ValueError('One subject cannot cross folds')
        cutoff = timestamp(row['cutoff'])
        key = (row['stay_id'], cutoff)
        if key in encounters:
            raise ValueError('Duplicate stay/cutoff sample')
        encounters.add(key)
        target = row.get('target', '')
        if target:
            identifier(target)
        samples.append({k: row[k] for k in required} | {'cutoff': cutoff,
                       'target': int(target) if target else None})
    if not samples:
        raise ValueError('Empty cohort')
    return samples


def read_knowledge(path):
    """Read a caller-reviewed, versioned allowlist, not LLM-generated medical facts.

    Requiring provenance does not independently establish clinical correctness.
    Concepts use lab:<itemid>, vital:<field>, med:ndc:<id> or med:name:<name>.
    External nodes use knowledge:<vocabulary>:<code>. Disease/target namespaces
    are deliberately unsupported in this first input version.
    """
    result, seen = [], set()
    required = {'source_token', 'relation', 'target_token', 'source_uri',
                'source_version', 'reviewed_by'}
    for row in rows(path, required):
        if any(not row[key].strip() for key in required):
            raise ValueError('Every knowledge relation requires complete provenance')
        if row['relation'] not in KNOWLEDGE_RELATIONS:
            raise ValueError('Unsupported medical relation')
        if not row['source_token'].startswith(('lab:', 'vital:', 'med:')):
            raise ValueError('Knowledge source must be an observed concept namespace')
        if not row['target_token'].startswith('knowledge:'):
            raise ValueError('Knowledge target must use knowledge: namespace')
        if not row['source_uri'].startswith(('https://', 'http://', 'urn:')):
            raise ValueError('Knowledge requires a traceable URI')
        key = tuple(row[k] for k in ('source_token', 'relation', 'target_token'))
        if key in seen:
            raise ValueError('Duplicate knowledge triple')
        seen.add(key)
        result.append({key: row[key] for key in required} |
                      ({'evidence': row['evidence']} if row.get('evidence') else {}))
    if not result:
        raise ValueError('Knowledge input is empty; use explicit event-only mode instead')
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    Path(path).chmod(0o600)
