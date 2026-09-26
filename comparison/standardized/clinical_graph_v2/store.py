"""Read-only access to the v1 event index plus a v2-local triage/arrival index.

The 18 GB `labevents.csv` scan is NOT repeated. The v1 artifact already produced a
source-bound SQLite index of every cutoff-eligible lab result, and the producer
verifies its sha256 against the v1 manifest before reading a single row. Only the
cheap tables v1 deliberately excluded -- triage and arrival demographics -- are
ingested here, into a separate v2-owned database that never mutates v1.
"""
from collections import Counter, defaultdict
import sqlite3

from .schema import (ARRIVAL_FIELDS, FORBIDDEN_EDSTAYS_FIELDS, TRIAGE_VITALS,
                     VITAL_RANGES, identifier, normalize_complaint, numeric, rows,
                     split_complaints, timestamp)


class ClinicalStore:
    """Joins the inherited read-only event index with the v2 triage index."""

    def __init__(self, event_index, triage_index):
        self.events = sqlite3.connect('file:' + str(event_index) + '?mode=ro', uri=True)
        self.triage = sqlite3.connect('file:' + str(triage_index) + '?mode=ro', uri=True)
        self.events.row_factory = sqlite3.Row
        self.triage.row_factory = sqlite3.Row
        self._visits = {}
        self._prior = {}

    def close(self):
        self.events.close()
        self.triage.close()

    # ------------------------------------------------------------------ encounters
    def visit(self, stay):
        if stay not in self._visits:
            row = self.events.execute(
                'SELECT stay, subject, start, finish FROM visits WHERE stay=?', (stay,)).fetchone()
            self._visits[stay] = dict(row) if row else None
        return self._visits[stay]

    def prior_visits(self, subject, index_start):
        """Completed encounters strictly before the index encounter began.

        An overlapping or unfinished earlier encounter is not settled history, so
        `finish < index_start` is required as well as `start < index_start`.
        """
        key = (subject, index_start)
        if key not in self._prior:
            self._prior[key] = [r['stay'] for r in self.events.execute(
                'SELECT stay FROM visits WHERE subject=? AND start<? AND finish IS NOT NULL '
                'AND finish<? ORDER BY start, stay', (subject, index_start, index_start))]
        return self._prior[key]

    # ------------------------------------------------------------------ triage tier
    def arrival(self, stay):
        row = self.triage.execute('SELECT * FROM arrival WHERE stay=?', (stay,)).fetchone()
        return dict(row) if row else {}

    def complaints(self, stay):
        return [r['token'] for r in self.triage.execute(
            'SELECT token FROM complaints WHERE stay=? ORDER BY ordinal', (stay,))]

    def triage_vitals(self, stay):
        return [(r['field'], r['value'], r['unit'], r['source_unit'])
                for r in self.triage.execute(
                    'SELECT field, value, unit, source_unit FROM vitals WHERE stay=? ORDER BY field', (stay,))]

    # ---------------------------------------------------------------- measurements
    def measurements(self, stay, cutoff):
        """Every representable result of this encounter available by the cutoff."""
        iso = cutoff.isoformat()
        return [{'id': r['id'], 'token': r['token'], 'value': r['value'], 'unit': r['unit'],
                 'time': r['time'], 'available': r['available'], 'source': r['source'],
                 'timing_basis': r['timing_basis']}
                for r in self.events.execute(
                    'SELECT id, token, value, unit, time, available, source, timing_basis '
                    'FROM events WHERE stay=? AND time<=? AND available<=? ORDER BY time, id',
                    (stay, iso, iso))]

    def analyte_history(self, subject, prior_stays, token, cutoff):
        """Prior-visit results of ONE analyte, oldest first.

        Restricting to a named analyte is what keeps this artifact small: v1 pulled
        every historical event of every kind, which is where its 18 GB went.
        """
        if not prior_stays:
            return []
        iso = cutoff.isoformat()
        placeholders = ','.join('?' * len(prior_stays))
        return [{'id': r['id'], 'token': r['token'], 'value': r['value'], 'unit': r['unit'],
                 'time': r['time'], 'available': r['available'], 'source': r['source'],
                 'timing_basis': r['timing_basis']}
                for r in self.events.execute(
                    'SELECT id, token, value, unit, time, available, source, timing_basis '
                    'FROM events WHERE subject=? AND token=? AND stay IN (%s) '
                    'AND time<=? AND available<=? ORDER BY time, id' % placeholders,
                    (subject, token, *prior_stays, iso, iso))]


def build_triage_index(path, raw_root, stays):
    """Ingest triage + arrival demographics for the cohort stays only.

    Both tables are small (~39 MB and ~40 MB) and are read once. Values that fail a
    plausibility gate are counted and dropped, never clipped. `disposition` is
    refused outright: it is the visit's outcome, not decision-time evidence.
    """
    db = sqlite3.connect(path)
    path.chmod(0o600)
    db.executescript('''
        CREATE TABLE arrival (stay TEXT PRIMARY KEY, subject TEXT NOT NULL,
            gender TEXT, race TEXT, arrival_transport TEXT, age REAL, acuity REAL);
        CREATE TABLE complaints (stay TEXT NOT NULL, ordinal INTEGER NOT NULL,
            token TEXT NOT NULL, PRIMARY KEY (stay, ordinal));
        CREATE TABLE vitals (stay TEXT NOT NULL, field TEXT NOT NULL, value REAL NOT NULL,
            unit TEXT NOT NULL, source_unit TEXT NOT NULL, PRIMARY KEY (stay, field));
    ''')
    counts = Counter()

    ages = {}
    for row in rows(raw_root / 'patients.csv', {'subject_id', 'anchor_age'}):
        value = numeric(row['anchor_age'])
        if value is not None:
            ages[row['subject_id']] = value
    counts['patients:with_anchor_age'] = len(ages)

    arrival_rows = []
    required_arrival = {'subject_id', 'stay_id', *ARRIVAL_FIELDS}
    if FORBIDDEN_EDSTAYS_FIELDS & required_arrival:
        raise ValueError('Outcome field requested from edstays; disposition is not decision-time evidence')
    for row in rows(raw_root / 'edstays.csv', required_arrival):
        if row['stay_id'] not in stays:
            continue
        identifier(row['subject_id'])
        identifier(row['stay_id'])
        arrival_rows.append((row['stay_id'], row['subject_id'],
                             row['gender'].strip() or None, row['race'].strip() or None,
                             row['arrival_transport'].strip() or None,
                             ages.get(row['subject_id']), None))
        counts['arrival:rows'] += 1
    db.executemany('INSERT INTO arrival VALUES (?,?,?,?,?,?,?)', arrival_rows)
    db.commit()

    complaint_rows, vital_rows, acuities = [], [], []
    for row in rows(raw_root / 'triage.csv', {'subject_id', 'stay_id', 'chiefcomplaint',
                                              'acuity', *TRIAGE_VITALS}):
        if row['stay_id'] not in stays:
            continue
        counts['triage:rows'] += 1
        tokens = split_complaints(row['chiefcomplaint'])
        if not tokens:
            counts['triage:no_complaint'] += 1
        for ordinal, token in enumerate(tokens):
            complaint_rows.append((row['stay_id'], ordinal, token))
        acuity = numeric(row['acuity'])
        if acuity is not None:
            acuities.append((acuity, row['stay_id']))
        for field, unit in TRIAGE_VITALS.items():
            value = numeric(row[field])
            if value is None:
                counts['vitals:missing:' + field] += 1
                continue
            source_unit = 'Fahrenheit' if field == 'temperature' else unit
            if field == 'temperature':
                value = (value - 32.) * 5. / 9.
            low, high = VITAL_RANGES[field]
            if not low <= value <= high:
                counts['vitals:out_of_range:' + field] += 1
                continue
            vital_rows.append((row['stay_id'], field, value, unit, source_unit))
    db.executemany('INSERT INTO complaints VALUES (?,?,?)', complaint_rows)
    db.executemany('INSERT INTO vitals VALUES (?,?,?,?,?)', vital_rows)
    db.executemany('UPDATE arrival SET acuity=? WHERE stay=?', acuities)
    db.commit()
    counts['complaints:rows'] = len(complaint_rows)
    counts['vitals:rows'] = len(vital_rows)
    counts['acuity:rows'] = len(acuities)
    db.close()
    return dict(counts)


def complaint_counts(triage_index, train_stays):
    """Count complaint surface forms over TRAIN-fold visits only."""
    db = sqlite3.connect('file:' + str(triage_index) + '?mode=ro', uri=True)
    counts = Counter()
    for stay, token in db.execute('SELECT stay, token FROM complaints'):
        if stay in train_stays:
            counts[token] += 1
    db.close()
    return counts
