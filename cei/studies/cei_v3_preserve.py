"""Copy five historical result roots without changing any original bytes.

All source roots must be explicit. The private preservation_manifest.json records
provenance and per-file SHA256; never commit it or the archived clinical outputs.
--verify reads an archive (including after relocation), never writes or loads graphs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

from ....paths import REPO_ROOT

REPO = REPO_ROOT
NAMES = ('core', 'protgnn', 'validation', 'xgboost', 'graphxai')
MANIFEST = 'preservation_manifest.json'
SCHEMA = 'medgnn.cei_v3_preservation.v1'
DEFAULT_ROOT = REPO / 'comparison/standardized/clinical_runs_preserved_20261001'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'Expected a regular directory, not a symlink: {root}')
    files = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in sorted(dirs + names):
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f'Unexplained symlink: {path}')
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError(f'Not a regular file: {path}')
            files.append({'path': path.relative_to(root).as_posix(),
                          'bytes': path.stat().st_size, 'sha256': digest(path)})
    return sorted(files, key=lambda item: item['path'])


def load_manifest(root):
    root = Path(root)
    document = json.loads((root / MANIFEST).read_text())
    if document.get('schema') != SCHEMA or document.get('status') != 'verified':
        raise ValueError('Incomplete or unsupported preservation manifest')
    roots = document.get('roots', [])
    if [item.get('name') for item in roots] != list(NAMES):
        raise ValueError('Manifest must contain exactly the five named roots')
    for entry in roots:
        if entry.get('relative_root') != entry['name']:
            raise ValueError('Invalid named-relative-root mapping')
        if entry.get('source_before_equals_after') is not True:
            raise ValueError('Missing source-before/source-after proof')
        seen = set()
        for item in entry['files']:
            path = Path(item['path'])
            if path.is_absolute() or '..' in path.parts or item['path'] in seen:
                raise ValueError('Invalid or duplicate inventory relative path')
            seen.add(item['path'])
    count = sum(len(entry['files']) for entry in roots)
    size = sum(item['bytes'] for entry in roots for item in entry['files'])
    if count != document['file_count'] or size != document['total_bytes']:
        raise ValueError('Manifest totals disagree with inventory')
    return document


def verify(root):
    root = Path(root).expanduser().absolute()
    document = load_manifest(root)
    mismatches = []
    if root.is_symlink():
        raise ValueError('Archive root must not be a symlink')
    if {p.name for p in root.iterdir()} != set(NAMES) | {MANIFEST}:
        mismatches.append('archive top-level inventory differs')
    for entry in document['roots']:
        actual = inventory(root / entry['relative_root'])
        if actual != entry['files']:
            mismatches.append(f"{entry['name']}: inventory/hash mismatch")
    return {'status': 'verified' if not mismatches else 'mismatch',
            'file_count': document['file_count'], 'total_bytes': document['total_bytes'],
            'mismatches': mismatches}


def preserve(sources, destination):
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Refusing occupied destination: {destination}')
    ignored = subprocess.run(['git', 'check-ignore', str(destination) + '/'],
                             cwd=REPO, capture_output=True)
    if ignored.returncode != 0:
        raise ValueError('Archive destination must be inside a git-ignored result root')
    try:
        destination.relative_to(REPO)
    except ValueError as error:
        raise ValueError('Archive destination must be inside this repository') from error
    roots = []
    for name in NAMES:
        source = Path(sources[name]).expanduser().absolute()
        if source.is_symlink():
            raise ValueError(f'Unexplained source root symlink: {source}')
        source = source.resolve(strict=True)
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError('Source and destination must not overlap')
        files = inventory(source)
        if not files:
            raise ValueError(f'Refusing empty evidence root: {name}')
        roots.append({'name': name, 'old_root': str(source), 'relative_root': name,
                      'files': files, 'source_before_equals_after': False})
    destination.mkdir(parents=True, exist_ok=False)
    # Failure leaves a visibly incomplete archive; never delete evidence on failure.
    for entry in roots:
        source, target = Path(entry['old_root']), destination / entry['name']
        target.mkdir(exist_ok=False)
        for item in entry['files']:
            out = target / item['path']
            out.parent.mkdir(parents=True, exist_ok=True)
            with (source / item['path']).open('rb') as src, out.open('xb') as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        if inventory(source) != entry['files']:
            raise ValueError(f"Source changed during copy: {entry['name']}")
        if inventory(target) != entry['files']:
            raise ValueError(f"Destination differs after copy: {entry['name']}")
        entry['source_before_equals_after'] = True
    # Recheck all originals at the end, not just immediately after each root's copy.
    for entry in roots:
        if inventory(Path(entry['old_root'])) != entry['files']:
            raise ValueError(f"Source changed before final verification: {entry['name']}")
    document = {'schema': SCHEMA, 'status': 'verified', 'roots': roots,
                'file_count': sum(len(entry['files']) for entry in roots),
                'total_bytes': sum(item['bytes'] for entry in roots for item in entry['files'])}
    with (destination / MANIFEST).open('x') as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write('\n')
    checked = verify(destination)
    if checked['mismatches']:
        raise ValueError(checked)
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path, metavar='ARCHIVE')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_ROOT)
    for name in NAMES:
        parser.add_argument('--' + name + '-root', type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        checked = verify(args.verify)
    else:
        sources = {name: getattr(args, name + '_root') for name in NAMES}
        if any(path is None for path in sources.values()):
            parser.error('Copy requires all five explicit --<name>-root arguments')
        checked = preserve(sources, args.output_root)
    print(json.dumps(checked, indent=2, sort_keys=True))
    return 1 if checked['mismatches'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
