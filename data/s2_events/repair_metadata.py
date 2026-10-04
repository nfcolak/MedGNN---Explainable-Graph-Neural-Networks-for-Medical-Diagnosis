"""Repair absent-graph event-index metadata WITHOUT scanning raw labs or generating graphs.

Default is a read-only preview. Apply only with --execute --offline, after stopping
all producers/filter jobs. For a capped derivative, supply --source-artifact (the
uncapped index) and --max-visits-per-subject explicitly; never infer scope from the
corrupt manifest. Existing graph files, SQLite journals, symlinks, and interrupted
repairs fail closed. Original JSON bytes are retained in content-addressed,
read-only backups; per-file atomic replacement breaks metadata hardlinks.

This is NOT a multi-file transaction or raw-source audit. An interrupted apply
leaves a pending marker and immutable backups for manual recovery. Target sidecars
are never rewritten: their graph-manifest hash bindings require explicit migration.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import uuid

from core.schema import identifier, read_cohort, timestamp

REPAIR_VERSION = 'event_index_metadata_repair_v1'
METADATA_NAMES = ('manifest.json', 'prepared_index.json', 'verification.json',
                  'delivery_report.json', 'filter_manifest.json')
FOLDS = {0: 'train', 1: 'validation', 2: 'test'}
PENDING = '.metadata-repair.pending.json'
LOCK = '.artifact-metadata.lock'


def _bytes_json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _signature(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Expected a regular non-symlink file: ' + str(path))
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _hash(path):
    before = _signature(path)
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if _signature(path) != before:
        raise ValueError('Input changed while hashing: ' + str(path))
    return digest.hexdigest()


def _exclusive_write(path, data, mode=0o600):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != data:
        raise ValueError('Write readback mismatch: ' + str(path))


def _sync_directory(root):
    fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def artifact_lock(root):
    """Shared with first_lab resume; stale/active locks are never auto-cleared."""
    root = Path(root).resolve(strict=True)
    path = root / LOCK
    token = uuid.uuid4().hex.encode()
    _exclusive_write(path, token)
    try:
        if (root / PENDING).exists():
            raise ValueError('Interrupted metadata repair; inspect pending marker and backups')
        yield
    finally:
        if not path.is_symlink() and path.read_bytes() == token:
            path.unlink()


def _index(root):
    path = root / 'events.sqlite'
    _signature(path)
    for suffix in ('-wal', '-shm', '-journal'):
        if (root / ('events.sqlite' + suffix)).exists():
            raise ValueError('SQLite sidecar present; require a stopped, checkpointed index')
    # Immutable avoids creating any journal/SHM file during a preview. The offline
    # requirement and before/after identities are essential to this mode.
    db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        visits = {r[0]: tuple(r[1:]) for r in db.execute('SELECT stay,subject,start,finish FROM visits')}
        excluded = {r[0]: tuple(r[1:]) for r in db.execute(
            'SELECT stay,subject,raw_start,raw_finish,reason FROM excluded_visits')}
        first = dict(db.execute('SELECT stay,available FROM first_results'))
        presence = {r[0] for r in db.execute('SELECT stay FROM lab_presence')}
        counts = {table: db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
                  for table in ('visits', 'excluded_visits', 'first_results', 'lab_presence', 'events')}
    finally:
        db.close()
    if any(counts[key] != len(value) for key, value in (
            ('visits', visits), ('excluded_visits', excluded),
            ('first_results', first), ('lab_presence', presence))):
        raise ValueError('Duplicate indexed stay identity')
    if set(visits) & set(excluded) or not set(first) <= set(visits) or not presence <= set(visits):
        raise ValueError('Invalid index stay lineage')
    if not set(first) <= presence:
        raise ValueError('First result without laboratory presence')
    for stay, (subject, start, finish) in visits.items():
        identifier(stay)
        identifier(subject)
        begin = timestamp(start)
        if finish is not None and timestamp(finish) < begin:
            raise ValueError('Invalid indexed visit interval')
        if stay in first and (finish is None or not begin <= timestamp(first[stay]) <= timestamp(finish)):
            raise ValueError('First result outside completed visit interval')
    return {'visits': visits, 'excluded': excluded, 'first': first, 'presence': presence, 'counts': counts}


def _evidence(root, source_artifact, maximum, manifest):
    tracked = {}

    def track(path):
        tracked[path] = _signature(path)
        return path

    canonical_path = track(root / 'canonical_split.json')
    track(root / 'cohort.csv')
    track(root / 'events.sqlite')
    canonical_hash = _hash(canonical_path)
    if manifest.get('source_bindings', {}).get('canonical_split', {}).get('sha256') != canonical_hash:
        raise ValueError('Canonical split differs from historical source binding')
    split = json.loads(canonical_path.read_bytes())
    canonical, labels = split['fold'], split['classes']
    if (not canonical or any(identifier(k) != k or type(v) is not int or v not in FOLDS
                             for k, v in canonical.items())):
        raise ValueError('Invalid canonical subject fold map')
    if (not labels or len(set(labels)) != len(labels)
            or labels != manifest.get('reference_class_order_only')):
        raise ValueError('Frozen class order mismatch')
    current = _index(root)
    reference = current
    reference_hash = None
    if maximum is not None:
        if type(maximum) is not int or maximum < 1 or source_artifact is None:
            raise ValueError('A positive visit cap requires the uncapped --source-artifact')
        source_artifact = Path(source_artifact).resolve(strict=True)
        if source_artifact == root:
            raise ValueError('Filtered source must be a separate uncapped artifact')
        source_split = track(source_artifact / 'canonical_split.json')
        source_index = track(source_artifact / 'events.sqlite')
        if _hash(source_split) != canonical_hash:
            raise ValueError('Source and derivative canonical splits differ')
        reference = _index(source_artifact)
        reference_hash = _hash(source_index)
    elif source_artifact is not None:
        raise ValueError('--source-artifact is only valid with an explicit visit cap')
    raw_counts = Counter(row[0] for row in reference['visits'].values())
    raw_counts.update(row[0] for row in reference['excluded'].values())
    if set(raw_counts) - set(canonical):
        raise ValueError('Index subject outside canonical population')
    if set(canonical) - set(raw_counts):
        raise ValueError('Uncapped reference lacks canonical subject visits; require the full source index')
    kept = {s for s in canonical if maximum is None or raw_counts[s] <= maximum}
    for key in ('visits', 'excluded'):
        expected = {stay: row for stay, row in reference[key].items() if row[0] in kept}
        if current[key] != expected:
            raise ValueError('Index is not the complete requested subject-filtered scope')
    expected_first = {stay: cutoff for stay, cutoff in reference['first'].items()
                      if reference['visits'][stay][0] in kept}
    if current['first'] != expected_first:
        raise ValueError('Filtered first-result index differs from the source subset')
    visit_filter = {'max_visits_per_subject': maximum,
                    'visit_count_basis': 'distinct stay_id values in indexed visits + excluded_visits',
                    'subjects_before': len(canonical), 'subjects_after': len(kept),
                    'subjects_dropped': len(canonical) - len(kept),
                    'raw_visits_dropped': sum(raw_counts[s] for s in set(canonical) - kept)}
    samples = read_cohort(root / 'cohort.csv')
    stay_samples, patient_stays = {}, defaultdict(set)
    sample_counts, subjects_by_fold = Counter(), defaultdict(set)
    for sample in samples:
        subject, stay = sample['subject_id'], sample['stay_id']
        if stay in stay_samples or stay not in current['first']:
            raise ValueError('Cohort must contain exactly one sample per eligible stay')
        if (subject not in kept or subject != current['visits'][stay][0]
                or sample['split'] != FOLDS[canonical[subject]]
                or sample['cutoff'] != timestamp(current['first'][stay])):
            raise ValueError('Cohort cutoff/subject/fold does not match index')
        expected_id = _digest(('event-first-lab-v1:' + subject + ':' + stay).encode())
        if sample['sample_id'] != expected_id:
            raise ValueError('Cohort sample_id differs from first-lab identity policy')
        stay_samples[stay] = sample
        patient_stays[subject].add(stay)
        sample_counts[sample['split']] += 1
        subjects_by_fold[sample['split']].add(subject)
    if set(stay_samples) != set(current['first']):
        raise ValueError('Cohort is missing indexed first-result stays')
    eligibility, by_fold = Counter(), {fold: Counter() for fold in FOLDS.values()}
    for stay, (subject, _, finish) in current['visits'].items():
        reason = ('eligible' if stay in current['first'] else 'missing_visit_end' if finish is None
                  else 'no_lab_charttime_in_window' if stay not in current['presence']
                  else 'no_eligible_recorded_result_in_window')
        eligibility.update([reason])
        by_fold[FOLDS[canonical[subject]]].update([reason, 'raw_visits'])
    invalid_subjects = set()
    for subject, _, _, reason in current['excluded'].values():
        invalid_subjects.add(subject)
        eligibility.update([reason])
        by_fold[FOLDS[canonical[subject]]].update([reason, 'raw_visits'])
    policy = {'sampling_unit': 'visit_cutoff', 'multiple_visits_per_patient': True,
              'subject_deduplication': False, 'split_unit': 'subject', 'samples': len(samples),
              'patients': len(patient_stays), 'index_visits': len(samples),
              'patients_with_multiple_index_visits': sum(len(v) > 1 for v in patient_stays.values()),
              'canonical_subjects': len(kept), 'canonical_subjects_before_filter': len(canonical),
              'canonical_subjects_without_eligible_visit': len(kept - set(patient_stays)),
              'subjects_with_invalid_source_visits': len(invalid_subjects),
              'visit_eligibility': dict(eligibility),
              'visit_eligibility_by_fold': {k: dict(v) for k, v in by_fold.items()},
              'by_fold': {fold: {'samples': sample_counts[fold], 'patients': len(subjects_by_fold[fold]),
                                 'index_visits': sample_counts[fold]} for fold in FOLDS.values()},
              'visit_filter': visit_filter}
    evidence = {'status': 'verified', 'scope': 'cohort_and_sqlite_index_only; not_graphs_or_raw_source',
                'cohort_sha256': _hash(root / 'cohort.csv'),
                'event_index_sha256': _hash(root / 'events.sqlite'),
                'canonical_split_sha256': canonical_hash, 'index_counts': current['counts'],
                'cohort_policy': policy, 'visit_filter': visit_filter,
                'source_artifact': str(source_artifact) if source_artifact else None,
                'source_event_index_sha256': reference_hash,
                'limitations': ['Raw input and historical source-code hashes preserved, not revalidated.',
                                'Index completeness relative to raw labs is not established by this repair.',
                                'No graph bytes were read, generated, modified or certified.']}
    return evidence, tracked


def _plan(root, source_artifact, maximum):
    if (root / PENDING).exists():
        raise ValueError('Interrupted metadata repair; inspect pending marker and backups')
    if (root / 'graphs.jsonl').exists() or (root / 'graphs.jsonl').is_symlink():
        raise ValueError('Refuse metadata-only repair when graphs.jsonl exists; never mutate graphs')
    originals = {}
    signatures = {}
    for name in METADATA_NAMES:
        path = root / name
        if path.exists() or path.is_symlink():
            signatures[path] = _signature(path)
            originals[name] = path.read_bytes()
            if not isinstance(json.loads(originals[name]), dict):
                raise ValueError('Metadata must be a JSON object: ' + name)
    if not {'manifest.json', 'prepared_index.json'} <= originals.keys():
        raise ValueError('Both manifest.json and prepared_index.json are required')
    manifest = json.loads(originals['manifest.json'])
    if manifest.get('schema_version') != 'event_graph_v1':
        raise ValueError('This command repairs event indices, not clinical graph artifacts')
    if manifest.get('status') not in ('completed', 'index_ready'):
        raise ValueError('Repair requires a formerly completed artifact, not an active producer')
    if json.loads(originals['prepared_index.json']).get('status') not in ('prepared', 'index_ready'):
        raise ValueError('Repair requires a completed prepared index')
    evidence, tracked = _evidence(root, source_artifact, maximum, manifest)
    signatures.update(tracked)
    receipt = manifest.get('metadata_repair') or {
        'version': REPAIR_VERSION,
        'prior_metadata_sha256': {name: _digest(data) for name, data in originals.items()},
        'backup_directory': 'metadata_backups',
        'target_binding_status': 'unchanged; old graph_manifest_sha256 bindings require explicit migration',
    }
    if receipt['version'] != REPAIR_VERSION:
        raise ValueError('Unsupported prior metadata repair version')
    common = {'status': 'index_ready', 'graph_status': 'absent', 'temporal_clean': False,
              'early_triage_eligible': False, 'cohort_sha256': evidence['cohort_sha256'],
              'event_index_sha256': evidence['event_index_sha256'],
              'cohort_policy': evidence['cohort_policy'], 'index_counts': evidence['index_counts'],
              'visit_filter': evidence['visit_filter'], 'index_verification': evidence,
              'metadata_repair': receipt, 'graph_regeneration_required': True,
              'resume_prepared_supported': False}
    # Manifest retains historical raw/source-code bindings exactly. Everything
    # overwritten is also recoverable as the original exact JSON backup bytes.
    updated = dict(manifest)
    for key in ('counts', 'coverage', 'graphs_sha256', 'graphs_bytes', 'verification_sha256',
                'ingest_counts', 'error', 'error_type', 'logic_contract_version'):
        updated.pop(key, None)
    updated.update(common)
    updated['limitations'] = evidence['limitations'] + [
        'Targets are not rebound by metadata repair; old manifest hashes remain historical.',
        'storetime is an availability proxy, not proven clinician visibility.']
    updated['source_binding_verification'] = 'historical_hashes_preserved_not_rehashed'
    replacements = {'manifest.json': _bytes_json(updated)}
    for name in originals.keys() - {'manifest.json'}:
        value = dict(common)
        # Preserve any explicit historical source hashes on auxiliary manifests.
        old = json.loads(originals[name])
        for key in ('source_bindings', 'source_code'):
            if key in old:
                value[key] = old[key]
        replacements[name] = _bytes_json(value)
    changed = {name: data for name, data in replacements.items() if data != originals[name]}
    if manifest.get('metadata_repair') and changed:
        raise ValueError('Previously repaired metadata or inputs changed; investigate rather than re-bless')
    return originals, changed, signatures, evidence


def _apply(root, originals, changed, signatures):
    backup = root / 'metadata_backups'
    if backup.exists():
        if backup.is_symlink() or not backup.is_dir():
            raise ValueError('Unsafe metadata backup directory')
    else:
        backup.mkdir(mode=0o700)
    # Complete and verify ALL backups before replacing a single active JSON.
    for name, data in originals.items():
        path = backup / (name + '.' + _digest(data) + '.json')
        if path.exists() or path.is_symlink():
            _signature(path)
            if path.stat().st_nlink != 1 or path.read_bytes() != data or path.stat().st_mode & 0o222:
                raise ValueError('Existing immutable backup conflicts; no overwrite allowed')
        else:
            _exclusive_write(path, data, mode=0o400)
    _sync_directory(backup)
    for path, signature in signatures.items():
        if _signature(path) != signature:
            raise ValueError('Input changed during metadata repair')
    if (root / 'graphs.jsonl').exists():
        raise ValueError('Graph appeared during metadata repair')
    pending = {'version': REPAIR_VERSION, 'old_sha256': {n: _digest(v) for n, v in originals.items()},
               'new_sha256': {n: _digest(v) for n, v in changed.items()}}
    _exclusive_write(root / PENDING, _bytes_json(pending))
    _sync_directory(root)
    # A partial failure intentionally retains pending marker + backups. Consumers
    # must reject the marker. The manifest is replaced last as the commit record.
    for name in sorted(changed, key=lambda n: (n == 'manifest.json', n)):
        temporary = root / ('.' + name + '.' + uuid.uuid4().hex + '.tmp')
        _exclusive_write(temporary, changed[name])
        # Earlier replacements may decrement a shared inode's link count/ctime;
        # bytes + inode/size/mtime still have to equal this exact saved source.
        if (_signature(root / name)[:4] != signatures[root / name][:4]
                or (root / name).read_bytes() != originals[name]):
            raise ValueError('Metadata target changed before replacement')
        os.replace(temporary, root / name)
    _sync_directory(root)
    for name, data in changed.items():
        if (root / name).read_bytes() != data or (root / name).stat().st_nlink != 1:
            raise ValueError('Repaired metadata readback failed')
    (root / PENDING).unlink()
    _sync_directory(root)


def validate_index_ready(root, manifest):
    """Validate the repaired metadata receipt; callers must also hash inputs."""
    root = Path(root)
    if (root / PENDING).exists() or (root / LOCK).exists():
        raise ValueError('Artifact metadata is locked or repair is incomplete')
    if (manifest.get('status') != 'index_ready'
            or manifest.get('metadata_repair', {}).get('version') != REPAIR_VERSION
            or manifest.get('index_verification', {}).get('status') != 'verified'):
        raise ValueError('Inherited index is not verified index_ready metadata')
    _signature(root / 'prepared_index.json')
    prepared = json.loads((root / 'prepared_index.json').read_bytes())
    if prepared.get('status') != 'index_ready':
        raise ValueError('Prepared index status mismatch')
    for key in ('cohort_sha256', 'event_index_sha256', 'cohort_policy', 'index_counts', 'visit_filter'):
        if (key not in manifest or manifest[key] != prepared.get(key)
                or manifest[key] != manifest['index_verification'].get(key)):
            raise ValueError('Repaired index evidence mismatch: ' + key)


def repair(artifact, *, source_artifact=None, max_visits_per_subject=None, execute=False, offline=False):
    """Preview or apply metadata-only repair. Never rewrite targets or graph bytes."""
    root = Path(artifact).resolve(strict=True)
    if execute and not offline:
        raise ValueError('--execute requires --offline acknowledgement: stop all producers first')

    def perform():
        originals, changed, signatures, evidence = _plan(root, source_artifact, max_visits_per_subject)
        for path, signature in signatures.items():
            if _signature(path) != signature:
                raise ValueError('Input changed during metadata inspection')
        if execute and changed:
            _apply(root, originals, changed, signatures)
        return {'status': 'repaired' if execute and changed else 'already_repaired' if not changed else 'dry_run',
                'artifact': str(root), 'changed_files': sorted(changed),
                'evidence': evidence, 'target_sidecars_modified': False,
                'graph_bytes_modified': False,
                'warning': 'Old target graph-manifest bindings require explicit migration; never silently rebind.'}
    if execute:
        lock_roots = {root}
        if source_artifact is not None:
            lock_roots.add(Path(source_artifact).resolve(strict=True))
        with ExitStack() as stack:
            for lock_root in sorted(lock_roots):
                stack.enter_context(artifact_lock(lock_root))
            return perform()
    if ((root / LOCK).exists()
            or source_artifact is not None and (Path(source_artifact) / LOCK).exists()):
        raise ValueError('Active/stale artifact lock; stop producers and inspect before preview')
    return perform()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--source-artifact', type=Path)
    parser.add_argument('--max-visits-per-subject', type=int)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--offline', action='store_true', help='Acknowledge that all artifact writers are stopped')
    args = parser.parse_args()
    result = repair(args.artifact, source_artifact=args.source_artifact,
                    max_visits_per_subject=args.max_visits_per_subject,
                    execute=args.execute, offline=args.offline)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
