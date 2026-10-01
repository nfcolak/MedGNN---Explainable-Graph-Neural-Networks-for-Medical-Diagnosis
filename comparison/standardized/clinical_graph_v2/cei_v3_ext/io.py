"""Explicit relocation and selected-row I/O; never mutate historical documents."""
from __future__ import annotations

import hashlib
import json
import re
from itertools import zip_longest
from pathlib import Path
from types import FunctionType


def read_json(path):
    with Path(path).open('rb') as stream:
        return json.load(stream)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class PathMap:
    """Longest-prefix mapping applied only at file-open boundaries.

    Manifest: {"mappings": [{"old": "/old/root", "new": "/new/root"}]}.
    Exact string paths remain in all binding/provenance documents.
    """
    def __init__(self, manifest=None):
        self.pairs = []
        self.inventory = {}
        if manifest:
            document = read_json(manifest)
            base = Path(manifest).parent
            if document.get('schema') == 'medgnn.cei_v3_preservation.v1':
                if document.get('status') != 'verified' or Path(manifest).name != 'preservation_manifest.json':
                    raise ValueError('incomplete preservation path-map manifest')
                roots = document.get('roots', [])
                if [row.get('name') for row in roots] != ['core', 'protgnn', 'validation', 'xgboost', 'graphxai']:
                    raise ValueError('preservation manifest has incomplete root inventory')
                rows = []
                total_bytes = total_files = 0
                for root in roots:
                    if root.get('source_before_equals_after') is not True or root.get('relative_root') != root['name']:
                        raise ValueError('preservation manifest lacks verified source proof')
                    rows.append({'old': root['old_root'], 'new': str(base / root['relative_root'])})
                    for item in root['files']:
                        relative = Path(item['path'])
                        key = Path(root['old_root']) / relative
                        if relative.is_absolute() or '..' in relative.parts or key in self.inventory:
                            raise ValueError('invalid preservation inventory path')
                        self.inventory[key] = item
                        total_files += 1
                        total_bytes += item['bytes']
                if total_files != document['file_count'] or total_bytes != document['total_bytes']:
                    raise ValueError('preservation manifest totals mismatch')
            else:
                rows = document.get('mapping', document.get('mappings', document.get('path_map', document)))
            if isinstance(rows, dict):
                rows = [{'old': old, 'new': new} for old, new in rows.items()]
            if not isinstance(rows, list):
                raise ValueError('path-map must contain mappings (old/new prefixes)')
            for row in rows:
                old = row.get('old', row.get('from'))
                new = row.get('new', row.get('to'))
                if not isinstance(old, str) or not isinstance(new, str):
                    raise ValueError('path-map requires string old/new prefixes')
                if not Path(old).is_absolute():
                    raise ValueError('path-map historical prefixes must be absolute')
                destination = Path(new)
                if not destination.is_absolute():
                    destination = base / destination
                self.pairs.append((Path(old), destination))
            if len({str(old) for old, _ in self.pairs}) != len(self.pairs):
                raise ValueError('ambiguous duplicate path-map prefix')
            self.pairs.sort(key=lambda pair: len(str(pair[0])), reverse=True)

    def resolve(self, recorded, expected_sha256=None):
        path = Path(recorded)
        translated = path
        for old, new in self.pairs:
            try:
                translated = new / path.relative_to(old)
                break
            except ValueError:
                pass
        entry = self.inventory.get(path)
        if translated.is_symlink():
            raise ValueError('path-map boundary refuses unexplained symlink')
        if entry is not None:
            if translated.stat().st_size != entry['bytes'] or digest(translated) != entry['sha256']:
                raise ValueError('mapped file differs from immutable preservation inventory')
        if expected_sha256 is not None and digest(translated) != expected_sha256:
            raise ValueError('mapped file differs from historical bound hash')
        return translated


# The producer serializes one sample_id key per graph. Refuse ambiguity rather
# than falling back to decoding an excluded graph. JSON-decoding only the ID
# string is not graph deserialization; membership metadata is read independently.
_SAMPLE_ID = re.compile(r'"sample_id"\s*:\s*("(?:[^"\\]|\\.)*")')


def selected_graphs(graphs_path, membership_path, wanted, limit=None):
    from ..contracts import validate_visit_membership_record

    wanted = set(wanted)
    seen, found = set(), set()
    sentinel = object()
    with Path(graphs_path).open() as graphs, Path(membership_path).open() as members:
        for index, (line, member_line) in enumerate(zip_longest(graphs, members,
                                                               fillvalue=sentinel)):
            if limit is not None and index >= limit:
                break
            if not isinstance(line, str) or not isinstance(member_line, str):
                raise ValueError('graph/membership row counts differ')
            matches = _SAMPLE_ID.findall(line)
            if len(matches) != 1:
                raise ValueError('graph identity envelope is missing or ambiguous')
            sid = json.loads(matches[0])
            record = json.loads(member_line)
            if sid != record.get('sample_id') or sid in seen:
                raise ValueError('graph/membership identity mismatch or duplicate')
            seen.add(sid)
            if sid not in wanted:
                continue  # no graph JSON decoder invoked for any excluded fold
            graph = json.loads(line)
            validate_visit_membership_record(graph, record)
            found.add(sid)
            yield graph, record
    if limit is None and found != wanted:
        raise ValueError('selected rows missing from immutable artifact')


def selected_preprocessing(graphs_path, train_ids, token_min_count=20, *, membership_path):
    """Use the existing fitter's exact code with a scoped reader dependency.

    No module monkeypatch, temporary graph artifact or duplicated fitting policy.
    """
    from .. import tensorize

    function = tensorize.fit_preprocessing
    scope = dict(function.__globals__)
    scope['iter_graphs_with_membership'] = lambda path, members, limit=None: selected_graphs(
        path, members, train_ids, limit)
    selected = FunctionType(function.__code__, scope, function.__name__,
                            function.__defaults__, function.__closure__)
    return selected(graphs_path, train_ids, token_min_count,
                    membership_path=membership_path)


def smoke_fit_inputs(artifact, targets_path, train_limit=256):
    """Same state inputs/transform as U5, using strictly selected TRAIN reads."""
    import numpy as np
    from .. import train, tensorize, cei_v3_study as study
    from ..contracts import VISIT_MEMBERSHIP_FILENAME

    targets, _, _ = train.select_top_labels(train.load_targets(targets_path), 10)
    ids = train.sample_train_ids(targets, train_limit, study.SAMPLE_SEED)
    graphs = Path(artifact) / 'graphs.jsonl'
    members = Path(artifact) / VISIT_MEMBERSHIP_FILENAME
    prep = selected_preprocessing(graphs, ids, membership_path=members)
    state = tensorize.preprocessing_state(prep)
    values, counts = {}, {}
    for graph, _ in selected_graphs(graphs, members, ids):
        present = set()
        for node in graph['nodes']:
            if node['kind'] not in ('measurement', 'vital'):
                continue
            token = tensorize.node_token(node)
            present.add(token)
            value, has_value = prep['scaler'].transform(token, node.get('value'))
            if has_value:
                values.setdefault(token, []).append(value)
        for token in present:
            counts[token] = counts.get(token, 0) + 1
    return study.V3FitInputs(
        values_by_item={token: np.asarray(v, dtype=np.float32) for token, v in values.items()},
        transform_by_item={token: ('zscore' if token in prep['scaler'].stats else 'signed_log')
                           for token in values}, identity_graph_counts=counts,
        vocabulary=dict(prep['vocabulary'].index),
        vocabulary_tokens=tuple(prep['vocabulary'].tokens), token_min_count=20,
        preprocessing_sha256=hashlib.sha256(study._canonical_file_bytes(state)).hexdigest(),
        node_feature_layout=tuple(state['node_feature_layout']), train_count=len(ids))
