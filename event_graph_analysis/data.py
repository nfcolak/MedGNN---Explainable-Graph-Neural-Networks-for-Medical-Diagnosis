"""Bound, lazy model access to completed event-graph artifacts.

Opening an artifact indexes JSONL without fitting anything. Adapter fitting reads
only the train fold. Test tensor access requires an explicit allow_test=True;
loading/indexing metadata is not test evaluation. No pickle input is accepted.
"""
import hashlib
import json
from pathlib import Path

from torch.utils.data import Dataset

from .model import EventGraphGNN
from .tensorize import EventGraphTensorizer, SCHEMA_VERSION
from comparison.standardized.event_graph_v1.schema import sha256


class EventGraphArtifact:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'graphs.jsonl'
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('schema_version') != SCHEMA_VERSION or self.manifest.get('status') != 'completed':
            raise ValueError('Only completed event_graph_v1 artifacts may reach a model')
        expected = self.manifest.get('graphs_sha256')
        if not expected or sha256(self.path) != expected:
            raise ValueError('Graph artifact fingerprint mismatch')
        self.fingerprint = expected
        self._offsets = {'train': [], 'validation': [], 'test': []}
        self._line_hashes = {}
        seen = set()
        with self.path.open('rb') as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                graph = json.loads(line)
                fold = graph.get('split')
                sample = graph.get('sample_id')
                if graph.get('schema_version') != SCHEMA_VERSION or fold not in self._offsets:
                    raise ValueError('Mixed schemas or unknown graph fold')
                if not isinstance(sample, str) or not sample or sample in seen:
                    raise ValueError('Empty or duplicate graph sample identity')
                seen.add(sample)
                self._offsets[fold].append(offset)
                self._line_hashes[offset] = hashlib.sha256(line).digest()
        if sum(map(len, self._offsets.values())) != self.manifest.get('counts', {}).get('graphs'):
            raise ValueError('Manifest graph count mismatch')
        coverage = self.manifest.get('coverage', {})
        for fold, offsets in self._offsets.items():
            if len(offsets) != coverage.get(fold, {}).get('graphs', 0):
                raise ValueError('Manifest fold coverage mismatch')
        if sha256(self.path) != self.fingerprint:
            raise ValueError('Graph artifact changed during indexing')

    def _read(self, offset):
        with self.path.open('rb') as stream:
            stream.seek(offset)
            line = stream.readline()
        if hashlib.sha256(line).digest() != self._line_hashes[offset]:
            raise ValueError('Graph record changed after artifact binding')
        return json.loads(line)

    def graphs(self, split, *, allow_test=False):
        """Yield one explicit fold; no implicit inclusion of held-out patients."""
        if split not in self._offsets:
            raise ValueError('Unknown fold')
        if split == 'test' and not allow_test:
            raise ValueError('Held-out graph access requires explicit allow_test=True')
        for offset in self._offsets[split]:
            yield self._read(offset)

    def fit_adapter(self, **kwargs):
        adapter = EventGraphTensorizer(**kwargs).fit(self.graphs('train'))
        return {'artifact_sha256': self.fingerprint, 'labels': self.manifest.get('labels'),
                'fit_split': 'train', 'train_count': len(self._offsets['train']),
                'adapter': adapter.to_dict()}

    def restore_adapter(self, state):
        """Reject a scaler/vocabulary fitted against a different graph artifact."""
        if (state.get('artifact_sha256') != self.fingerprint or state.get('fit_split') != 'train'
                or state.get('train_count') != len(self._offsets['train'])
                or state.get('labels') != self.manifest.get('labels')):
            raise ValueError('Adapter artifact/fold/label binding mismatch')
        return EventGraphTensorizer.from_dict(state['adapter'])

    def dataset(self, split, state, *, allow_test=False):
        if split not in self._offsets:
            raise ValueError('Unknown fold')
        if split == 'test' and not allow_test:
            raise ValueError('Held-out dataset access requires explicit allow_test=True')
        return EventGraphDataset(self, split, self.restore_adapter(state))

    def make_model(self, state, **kwargs):
        """Construct an untrained classifier from the explicit full label order."""
        labels = self.manifest.get('labels')
        if (not isinstance(labels, list) or not labels
                or any(not isinstance(label, str) or not label for label in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError('Classification requires a valid explicit ordered label list')
        adapter = self.restore_adapter(state)
        return EventGraphGNN(adapter.num_tokens, adapter.num_relations, len(labels), **kwargs)


class EventGraphDataset(Dataset):
    """Reopen one bounded record per item; no full in-memory tensor collection."""
    def __init__(self, artifact, split, adapter):
        self.artifact = artifact
        self.offsets = tuple(artifact._offsets[split])
        self.adapter = adapter

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        graph = self.artifact._read(self.offsets[index])
        labels = self.artifact.manifest.get('labels')
        if 'target' in graph and (labels is None or type(graph['target']) is not int
                                  or not 0 <= graph['target'] < len(labels)):
            raise ValueError('Graph target outside bound label order')
        return self.adapter.transform(graph)
