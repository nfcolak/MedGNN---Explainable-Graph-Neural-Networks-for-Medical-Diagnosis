"""Chunked raw MIMIC ingestion into an output-local SQLite event index.

No legacy merge imports, aggregate labs, diagnosis rows or untimestamped triage
are consumed. A reconciliation record is not a medication administration.
"""
from collections import Counter
import json
from pathlib import Path
import sqlite3

import pandas as pd

from .schema import LAB_IDS, VITAL_UNITS, identifier, numeric, rows, timestamp


class EventStore:
    def __init__(self, path):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.path.chmod(0o600)
        self.db.executescript('''
            CREATE TABLE visits (stay TEXT PRIMARY KEY, subject TEXT NOT NULL,
                                 start TEXT NOT NULL, finish TEXT);
            CREATE INDEX visit_subject ON visits(subject, start);
            CREATE TABLE events (id TEXT PRIMARY KEY, subject TEXT NOT NULL,
                stay TEXT NOT NULL, time TEXT NOT NULL, available TEXT NOT NULL,
                token TEXT NOT NULL, value REAL, unit TEXT, source TEXT NOT NULL,
                timing_basis TEXT NOT NULL);
            CREATE INDEX event_visit ON events(stay, available, time);
        ''')
        self.counts = Counter()

    def close(self):
        self.db.close()

    def ingest_visits(self, path, samples):
        subjects = {s['subject_id'] for s in samples}
        for row in rows(path, {'subject_id', 'stay_id', 'intime', 'outtime'}):
            if row['subject_id'] not in subjects:
                continue
            identifier(row['stay_id'])
            start = timestamp(row['intime'])
            finish = timestamp(row['outtime']) if row['outtime'] else None
            if finish is not None and finish < start:
                raise ValueError('Encounter ends before it starts')
            self.db.execute('INSERT INTO visits VALUES (?, ?, ?, ?)',
                            (row['stay_id'], row['subject_id'], start.isoformat(),
                             finish.isoformat() if finish else None))
            self.counts['visits'] += 1
        self.db.commit()
        for sample in samples:
            row = self.db.execute('SELECT subject, start, finish FROM visits WHERE stay=?',
                                  (sample['stay_id'],)).fetchone()
            if row is None or row[0] != sample['subject_id']:
                raise ValueError('Cohort stay lineage not found in raw edstays')
            if sample['cutoff'] < timestamp(row[1]) or (row[2] and sample['cutoff'] > timestamp(row[2])):
                raise ValueError('Cutoff must be inside the explicit index encounter')

    def chunks(self, path, columns, subjects, chunk_size):
        scanned = 0
        for frame in pd.read_csv(path, usecols=columns, dtype=str,
                                 keep_default_na=False, chunksize=chunk_size):
            offset = scanned
            scanned += len(frame)
            # Preserve file row ordinal for provenance, not model features.
            frame = frame.copy()
            frame['_source_row'] = range(offset + 2, scanned + 2)
            selected = frame[frame['subject_id'].isin(subjects)]
            yield selected
            self.counts[Path(path).name + ':scanned'] = scanned
            print(json.dumps({'stage': 'ingest', 'source': Path(path).name,
                              'rows_scanned': scanned}), flush=True)

    def insert(self, event):
        self.db.execute('INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', event)
        self.counts['events:' + event[8].split(':')[0]] += 1

    def ingest_labs(self, path, subjects, chunk_size):
        columns = ['labevent_id', 'subject_id', 'itemid', 'charttime', 'storetime',
                   'valuenum', 'valueuom']
        # Subject-local intervals avoid a huge subject x all-visits DataFrame join.
        intervals = {}
        for stay, subject, start, finish in self.db.execute('SELECT * FROM visits'):
            intervals.setdefault(subject, []).append((stay, start, finish))
        for frame in self.chunks(path, columns, subjects, chunk_size):
            frame = frame[frame['itemid'].isin(LAB_IDS)]
            for row in frame.to_dict('records'):
                value = numeric(row['valuenum'])
                if value is None or not row['valueuom'].strip():
                    self.counts['labs:missing_value_or_unit'] += 1
                    continue
                try:
                    time, available = timestamp(row['charttime']), timestamp(row['storetime'])
                except ValueError:
                    self.counts['labs:invalid_or_missing_time'] += 1
                    continue
                if available < time:
                    self.counts['labs:store_before_chart'] += 1
                    continue
                time_s = time.isoformat()
                candidates = [stay for stay, start, end in intervals[row['subject_id']]
                              if start <= time_s and end is not None and time_s <= end]
                if len(candidates) != 1:
                    self.counts['labs:ambiguous_or_unmatched_stay'] += 1
                    continue
                lab_id = identifier(row['labevent_id'])
                self.insert(('lab:' + lab_id, row['subject_id'], candidates[0], time_s,
                             available.isoformat(), 'lab:' + row['itemid'], value,
                             row['valueuom'], 'labevents.csv:' + str(row['_source_row']),
                             'storetime'))
            self.db.commit()

    def ingest_chart_proxy(self, raw_root, subjects, chunk_size):
        """Opt-in retrospective proxy. charttime is NOT verified availability."""
        visits = {stay: (subject, start, finish) for stay, subject, start, finish
                  in self.db.execute('SELECT * FROM visits')}
        for filename, columns in (
            ('vitalsign.csv', ['subject_id', 'stay_id', 'charttime', *VITAL_UNITS]),
            ('medrecon.csv', ['subject_id', 'stay_id', 'charttime', 'name', 'ndc']),
        ):
            for frame in self.chunks(Path(raw_root) / filename, columns, subjects, chunk_size):
                for row in frame.to_dict('records'):
                    visit = visits.get(row['stay_id'])
                    try:
                        time = timestamp(row['charttime']).isoformat()
                    except ValueError:
                        self.counts[filename + ':invalid_time'] += 1
                        continue
                    if visit is None or visit[0] != row['subject_id'] or time < visit[1] or (visit[2] and time > visit[2]):
                        self.counts[filename + ':outside_stay'] += 1
                        continue
                    source = filename + ':' + str(row['_source_row'])
                    if filename == 'vitalsign.csv':
                        for field, unit in VITAL_UNITS.items():
                            value = numeric(row[field])
                            if value is None:
                                continue
                            if field == 'temperature':
                                value = (value - 32.) * 5. / 9.
                            self.insert((source + ':' + field, row['subject_id'], row['stay_id'],
                                         time, time, 'vital:' + field, value, unit, source,
                                         'charttime_proxy'))
                    else:
                        # Keep an invalid NDC out, never silently normalize it.
                        ndc = row['ndc']
                        if ndc and ndc != '0':
                            try:
                                token = 'med:ndc:' + identifier(ndc)
                            except ValueError:
                                self.counts['medrecon:invalid_ndc'] += 1
                                continue
                        elif row['name'].strip():
                            token = 'med:name:' + row['name'].strip().casefold()
                        else:
                            self.counts['medrecon:missing_identity'] += 1
                            continue
                        self.insert((source, row['subject_id'], row['stay_id'], time, time,
                                     token, None, None, source, 'charttime_proxy'))
                self.db.commit()
