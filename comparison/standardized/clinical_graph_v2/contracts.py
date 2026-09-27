"""Fail-closed contracts for corrected clinical graph experiments.

Historical graph/result bytes remain historical. A source edit or repaired source
index manifest does not upgrade the diagnosis history inside an existing graph.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from itertools import islice, zip_longest
from pathlib import Path

from .schema import sha256

LOGIC_CONTRACT_VERSION = 'clinical_graph_logic_v2'
VISIT_MEMBERSHIP_CONTRACT_VERSION = 'clinical_visit_membership_v1'
VISIT_MEMBERSHIP_FILENAME = 'visit_membership.jsonl'
VISIT_MEMBERSHIP_KEYS = frozenset({
    'contract_version', 'sample_id', 'visit_ordinals', 'membership_pairs', 'global_node_mask',
})


def sample_ids_sha256(sample_ids):
    """Bind visit order without printing patient/stay identifiers in run reports."""
    return hashlib.sha256(json.dumps(list(sample_ids), separators=(',', ':'),
                                     ensure_ascii=True).encode('utf-8')).hexdigest()


def recursive_source_hashes(package_root):
    """Hash every Python source under a package using stable relative paths."""
    package_root = Path(package_root).resolve()
    files = sorted(path for path in package_root.rglob('*.py')
                   if '__pycache__' not in path.parts)
    if not files:
        raise ValueError('Source binding is empty')
    return {path.relative_to(package_root).as_posix(): sha256(path) for path in files}


def validate_artifact_manifest(manifest):
    if manifest.get('status') != 'completed':
        raise ValueError('Artifact is not completed')
    if manifest.get('logic_contract_version') != LOGIC_CONTRACT_VERSION:
        raise ValueError('Historical clinical artifact: rebuild graphs into a fresh directory '
                         'with the corrected producer; metadata repair alone is insufficient')
    if manifest.get('temporal_clean') is not False:
        raise ValueError('Untimestamped triage/history cannot be declared temporal_clean; rebuild')
    if manifest.get('visit_membership_contract_version') != VISIT_MEMBERSHIP_CONTRACT_VERSION:
        raise ValueError('Historical clinical artifact lacks source-derived visit membership; '
                         'rebuild into a fresh directory')
    if manifest.get('visit_membership_file') != VISIT_MEMBERSHIP_FILENAME:
        raise ValueError('Artifact visit-membership sidecar filename mismatch; rebuild')
    row_count = manifest.get('visit_membership_rows')
    if type(row_count) is not int or row_count < 0:
        raise ValueError('Artifact visit-membership row count is missing or invalid')
    digest = manifest.get('visit_membership_sha256')
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(character not in '0123456789abcdef' for character in digest)):
        raise ValueError('Artifact visit-membership checksum is missing or invalid')


def validate_visit_membership_record(graph, record, expected_visit_ordinals=None):
    """Validate one sidecar row and return node-index to source visit ordinals."""
    if not isinstance(record, dict) or set(record) != VISIT_MEMBERSHIP_KEYS:
        raise ValueError('Visit-membership row has missing or unexpected fields')
    if record.get('contract_version') != VISIT_MEMBERSHIP_CONTRACT_VERSION:
        raise ValueError('Unsupported visit-membership contract version')
    if not isinstance(graph, dict):
        raise ValueError('Graph row is missing or malformed')
    sample_id = graph.get('sample_id')
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError('Graph sample identity is missing or malformed')
    if record.get('sample_id') != sample_id:
        raise ValueError('Graph and visit-membership sample IDs differ')

    visit_ordinals = record.get('visit_ordinals')
    if (not isinstance(visit_ordinals, list) or not visit_ordinals or
            any(type(value) is not int for value in visit_ordinals)):
        raise ValueError('Visit ordinals must be a nonempty integer list')
    expected = (list(range(len(visit_ordinals))) if expected_visit_ordinals is None
                else list(expected_visit_ordinals))
    if visit_ordinals != expected:
        raise ValueError('Visit ordinals are unordered, missing, or outside source lineage')

    nodes = graph.get('nodes')
    if not isinstance(nodes, list) or any(not isinstance(node, dict) for node in nodes):
        raise ValueError('Graph nodes are missing or malformed')
    if sum(node.get('kind') == 'patient' for node in nodes) != 1:
        raise ValueError('Graph must contain exactly one global patient context node')
    visit_nodes = [node for node in nodes if node.get('kind') == 'visit']
    if len(visit_nodes) != 1 or visit_nodes[0].get('token') != 'visit:index':
        raise ValueError('Graph must contain exactly one index-visit node')
    global_mask = record.get('global_node_mask')
    if (not isinstance(global_mask, list) or len(global_mask) != len(nodes) or
            any(type(value) is not bool for value in global_mask)):
        raise ValueError('Global-node mask does not match graph nodes')
    for node, is_global in zip(nodes, global_mask):
        if is_global != (node.get('kind') == 'patient'):
            raise ValueError('Only patient context nodes may be marked global')

    pairs = record.get('membership_pairs')
    if not isinstance(pairs, list):
        raise ValueError('Visit membership pairs are missing or malformed')
    parsed_pairs = []
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2 or
                any(type(value) is not int for value in pair)):
            raise ValueError('Each visit membership pair must contain two integers')
        ordinal, node_index = pair
        if ordinal not in expected or not 0 <= node_index < len(nodes):
            raise ValueError('Visit membership pair is out of range')
        parsed_pairs.append((ordinal, node_index))
    if parsed_pairs != sorted(set(parsed_pairs)):
        raise ValueError('Visit membership pairs must be unique and ordered')

    memberships = {index: set() for index in range(len(nodes))}
    for ordinal, node_index in parsed_pairs:
        memberships[node_index].add(ordinal)
    for node_index, visits in memberships.items():
        is_global = global_mask[node_index]
        if is_global and visits:
            raise ValueError('Global node also has visit-specific membership')
        if not is_global and not visits:
            raise ValueError('Visit-specific graph node has no membership')
    return memberships


def verify_graph_file(artifact, manifest):
    """Bind the consumed bytes, not only a copied hash in an old manifest."""
    path = artifact / 'graphs.jsonl'
    if manifest.get('graphs_sha256') != sha256(path):
        raise ValueError('Graph file checksum differs from the artifact manifest')


def verify_visit_membership_file(artifact, manifest):
    """Require the manifest-bound sidecar; never synthesize historical membership."""
    if manifest.get('visit_membership_file') != VISIT_MEMBERSHIP_FILENAME:
        raise ValueError('Artifact visit-membership sidecar filename mismatch; rebuild')
    path = artifact / VISIT_MEMBERSHIP_FILENAME
    if not path.is_file() or manifest.get('visit_membership_sha256') != sha256(path):
        raise ValueError('Visit-membership sidecar is missing or differs from its manifest')
    with path.open('rb') as stream:
        rows = sum(1 for _ in stream)
    if rows != manifest.get('visit_membership_rows'):
        raise ValueError('Visit-membership row count differs from the artifact manifest')
    return rows


def iter_graphs_with_membership(graphs_path, membership_path, limit=None) -> Iterator[tuple[dict, dict]]:
    """Stream aligned graph/sidecar rows and fail closed on contract drift."""
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError('limit must be a nonnegative integer or None')

    missing = object()
    seen_sample_ids = set()
    with Path(graphs_path).open() as graph_stream, Path(membership_path).open() as membership_stream:
        rows = zip_longest(graph_stream, membership_stream, fillvalue=missing)
        if limit is not None:
            rows = islice(rows, limit)
        for row_index, (graph_line, membership_line) in enumerate(rows, 1):
            if not isinstance(graph_line, str) or not isinstance(membership_line, str):
                raise ValueError('Graph and visit-membership sample counts differ')
            try:
                graph = json.loads(graph_line)
                record = json.loads(membership_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Malformed graph or visit-membership JSONL row {row_index}') from exc
            validate_visit_membership_record(graph, record)
            sample_id = graph['sample_id']
            if sample_id in seen_sample_ids:
                raise ValueError('Duplicate sample_id in graph/visit-membership streams')
            seen_sample_ids.add(sample_id)
            yield graph, record


def validate_control_configuration(rewire_relations, use_edge_payload):
    if rewire_relations and use_edge_payload:
        raise ValueError('Rewiring requires --no-edge-payload and a matching real-edge '
                         '--no-edge-payload control; old numeric payloads cannot be '
                         'treated as facts about new endpoint pairs')


def verify_target_binding(manifest, targets_path, labels):
    """Labels attach to the exact inherited cohort, independently of graph format."""
    targets_path = Path(targets_path)
    path = targets_path.parent / 'binding_manifest.json'
    binding = json.loads(path.read_text())
    if binding.get('status') != 'completed':
        raise ValueError('Target binding is not completed')
    if binding.get('artifact_files', {}).get(targets_path.name) != sha256(targets_path):
        raise ValueError('Target sidecar checksum mismatch')
    cohort_hash = manifest.get('inherited_cohort_sha256')
    if not cohort_hash or binding.get('cohort_sha256') != cohort_hash:
        raise ValueError('Target cohort lineage differs or is missing; rebuild the sidecar')
    if binding.get('labels') != list(labels):
        raise ValueError('Target label order differs from the canonical class order')
    return sha256(path)
