"""Explicit, repo-relative CEI-v3 I/O configuration and immutable path mapping.

--data-root denotes the repository-layout root of private inputs, not a search
location. --results-root denotes the preserved archive with named children.
--path-map accepts its preservation_manifest.json or {"mapping": {old: new}}.
Relative configured paths are rooted at the source repo, not the process CWD;
relative mapping destinations are rooted at the mapping file's directory.
Mapping never edits historical bindings. Every mapped file needs a recorded or
caller-supplied SHA256. Plans inspect metadata only, not graphs or predictions.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import json
from pathlib import Path

from .cei_v3_preserve import MANIFEST, SCHEMA, digest, load_manifest
from core.paths import PACKAGE_ROOT, REPO_ROOT

REPO = REPO_ROOT
DEFAULT_RESULTS = REPO / 'comparison/standardized/clinical_runs_preserved_20261001'
ARTIFACT_REL = 'comparison/standardized/event_inputs/clinical_graph_v3_membership_max6_20260923'
TARGETS_REL = 'comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_local_v2_max6/targets.csv'
PILOT_REL = 'comparison/standardized/clinical_runs_cei_pilot_20260928/protgnn_control'


def absolute(value, base=REPO):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()


class PathMap:
    """Read-only prefix translation; verify bytes at each mapped file boundary."""
    def __init__(self, filename=None):
        self.mapping = {}
        self.inventory = {}
        self.filename = None if filename is None else absolute(filename)
        if self.filename is None:
            return
        document = json.loads(self.filename.read_text())
        if document.get('schema') == SCHEMA:
            if self.filename.name != MANIFEST:
                raise ValueError('Preservation mapping must name its archive manifest')
            document = load_manifest(self.filename.parent)
            for root in document['roots']:
                old = Path(root['old_root'])
                self.mapping[old] = self.filename.parent / root['relative_root']
                for item in root['files']:
                    self.inventory[old / item['path']] = item
        else:
            mapping = document.get('mapping')
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError('Path map requires a preservation manifest or nonempty mapping object')
            for old, new in mapping.items():
                if not Path(old).is_absolute():
                    raise ValueError('Historical mapping keys must be absolute paths')
                self.mapping[Path(old)] = absolute(new, self.filename.parent)

    def resolve(self, value, expected_sha256=None):
        original = Path(value).expanduser()
        translated = original
        matched = False
        for old in sorted(self.mapping, key=lambda path: len(path.parts), reverse=True):
            try:
                suffix = original.relative_to(old)
            except ValueError:
                continue
            translated = self.mapping[old] / suffix
            matched = True
            break
        if not translated.is_absolute():
            translated = REPO / translated
        if translated.is_symlink():
            raise ValueError(f'Unexplained mapped symlink: {translated}')
        translated = translated.resolve(strict=True)
        entry = self.inventory.get(original)
        if matched and not translated.is_file():
            raise ValueError('Mapped I/O boundary must name an inventoried regular file')
        if matched and entry is None and expected_sha256 is None:
            raise ValueError('Mapped file has no SHA256 proof; provide its bound hash')
        if entry is not None:
            if translated.stat().st_size != entry['bytes'] or digest(translated) != entry['sha256']:
                raise ValueError(f'Mapped file differs from preservation inventory: {translated}')
        if expected_sha256 is not None and digest(translated) != expected_sha256:
            raise ValueError(f'Mapped input differs from its historical bound SHA256: {translated}')
        return translated


@dataclass(frozen=True)
class Paths:
    data_root: Path
    results_root: Path
    v3_root: Path
    protgnn_root: Path
    validation_root: Path
    pilot_root: Path
    artifact: Path
    targets: Path
    canonical: Path
    path_map: PathMap

    def cli_args(self):
        result = []
        for key in ('data_root', 'results_root', 'v3_root', 'protgnn_root',
                    'validation_root', 'pilot_root', 'artifact', 'targets', 'canonical'):
            result.extend(['--' + key.replace('_', '-'), str(getattr(self, key))])
        if self.path_map.filename is not None:
            result.extend(['--path-map', str(self.path_map.filename)])
        return result


def add_arguments(parser):
    for key in ('data-root', 'results-root', 'v3-root', 'protgnn-root',
                'validation-root', 'pilot-root', 'artifact', 'targets', 'canonical', 'path-map'):
        parser.add_argument('--' + key, type=Path, default=None,
                            help='Explicit I/O path; relative paths resolve against this source repository')
    parser.add_argument('--plan', action='store_true', help='Read-only metadata plan (default without --execute)')


def resolve(args=None):
    args = args or argparse.Namespace()
    data = absolute(getattr(args, 'data_root', None) or REPO)
    results = absolute(getattr(args, 'results_root', None) or DEFAULT_RESULTS)
    def chosen(key, fallback):
        return absolute(getattr(args, key, None) or fallback)
    mapping = getattr(args, 'path_map', None)
    if mapping is None and (results / MANIFEST).is_file():
        mapping = results / MANIFEST
    return Paths(data, results, chosen('v3_root', results / 'core'),
                 chosen('protgnn_root', results / 'protgnn'),
                 chosen('validation_root', results / 'validation'),
                 chosen('pilot_root', data / PILOT_REL),
                 chosen('artifact', data / ARTIFACT_REL), chosen('targets', data / TARGETS_REL),
                 chosen('canonical', data / 'comparison/canonical_split.json'), PathMap(mapping))


def check_output(paths, root):
    """Never place new output inside an immutable input/archive (or vice versa)."""
    root = absolute(root)
    protected = {paths.results_root, paths.v3_root, paths.protgnn_root,
                 paths.validation_root, paths.pilot_root, paths.artifact}
    protected.update(paths.path_map.mapping)
    protected.update(paths.path_map.mapping.values())
    for value in protected:
        value = absolute(value)
        if root == value or value in root.parents or root in value.parents:
            raise ValueError('Output overlaps an immutable input or preservation root')
    return root


def mapped_model_factory(binding, checkpoint_path, *, paths, factory):
    """Map only the state-file read, retaining original run_config provenance.

    The existing study factory still rebuilds/loads the model strictly. A private
    argument copy carries the translated I/O path. The validated, byte-identical
    in-memory state's provenance path is restored before exact config comparison;
    no tensor, hash, historical binding, or checkpoint is changed.
    """
    config = binding['method_config']
    original = config['v3_state_path']
    if config['effective_settings']['v3_state'] != original:
        raise ValueError('Historical state-path parity differs within method_config')
    mapped = paths.path_map.resolve(original, config['v3_state_sha256'])
    local = copy.deepcopy(binding)
    local['method_config']['effective_settings']['v3_state'] = str(mapped)
    model = factory(local, checkpoint_path)
    if model.state.sha256 != config['v3_state_sha256']:
        raise ValueError('Reconstructed state differs from historical bound bytes')
    model.state = replace(model.state, path=original)
    if model.run_config() != config:
        raise ValueError('Reconstructed method_config differs from immutable binding')
    return model


def metadata_plan(paths, root, *, include_protgnn=True, include_validation=False):
    """Inspect bindings, frozen hashes and completed decision metadata only.

    This is NOT execution preflight: source parity and fold/input checks remain
    mandatory inside --execute, and are not waived by a successful metadata plan.
    """
    root = absolute(root)
    if root.exists():
        raise FileExistsError(f'Plan requires a fresh output root: {root}')
    inputs = [paths.v3_root]
    if include_protgnn:
        inputs.append(paths.protgnn_root)
    if include_validation:
        inputs.append(paths.validation_root)
    decisions = []
    for directory in inputs:
        document = json.loads((directory / 'decision.json').read_text())
        if document.get('status') != 'completed' or document.get('test_evaluated') is not False:
            raise ValueError(f'Incomplete evidence or test-access trace: {directory.name}')
        decisions.append({'name': directory.name, 'completed': True,
                          'historical_validation_evaluated': document['validation_evaluated']})
    frozen = json.loads((paths.v3_root / 'k_selection.json').read_text())
    if frozen.get('k_selected') != 4:
        raise ValueError('Historical frozen K is not 4')
    decision = json.loads((paths.v3_root / 'decision.json').read_text())
    if digest(paths.v3_root / 'k_selection.json') != decision['k_selection_sha256']:
        raise ValueError('Frozen K selection hash differs from historical decision')
    states = set()
    bindings = 0
    source = {path.relative_to(PACKAGE_ROOT).as_posix(): digest(path)
              for path in PACKAGE_ROOT.rglob('*.py') if '__pycache__' not in path.parts}
    source_gaps = set()
    for arm in ('A', 'B', 'C', 'P') if include_protgnn else ('A', 'B', 'C'):
        for seed in (1234, 2025, 7):
            stage = (paths.protgnn_root if arm == 'P' else paths.v3_root) / (
                f'C_K4_seed{seed}' if arm == 'C' else f'{arm}_seed{seed}')
            binding = json.loads((stage / 'binding.json').read_text())
            result = json.loads((stage / 'result.json').read_text())
            if result.get('status') != 'completed' or result.get('binding') != binding:
                raise ValueError(f'Incomplete stage/binding parity: {stage.name}')
            if binding['seed'] != seed or binding['selection_fold'] != 'dev' or binding['final_eval'] != 'none':
                raise ValueError(f'Historical seed/fold contract differs: {stage.name}')
            if binding['test_evaluated'] is not False:
                raise ValueError('Historical stage carries test trace')
            source_gaps.update(key for key, value in binding['source_code'].items()
                               if source.get(key) != value)
            saved = json.loads((stage / 'screen/screen_result.json').read_text())
            if digest(stage / 'binding.json') != saved['binding_sha256']:
                raise ValueError('Screen-bound binding hash differs')
            if digest(stage / 'best.pt') != saved['checkpoint_sha256']:
                raise ValueError('Screen-bound checkpoint hash differs')
            if digest(stage / 'preprocessing.json') != binding['preprocessing_sha256']:
                raise ValueError('Preprocessing hash differs')
            config = binding['method_config']
            if arm != 'P':
                if config['effective_settings']['v3_state'] != config['v3_state_path']:
                    raise ValueError('Bound nested state paths differ')
                states.add(str(paths.path_map.resolve(config['v3_state_path'], config['v3_state_sha256'])))
            bindings += 1
    return {'status': 'plan_only', 'output_root': str(root),
            'read_only_inputs': [str(path) for path in inputs], 'historical_evidence': decisions,
            'verified_binding_count': bindings, 'mapped_state_files_verified': len(states),
            'k_selected': 4, 'graphs_deserialized': 0, 'predictions_loaded': 0,
            'validation_evaluated': False, 'test_evaluated': False,
            'execution_preflight_performed': False,
            'historical_source_parity_verified': not source_gaps,
            'historical_source_mismatch_files': sorted(source_gaps),
            'execution_gap': ('Restore the matching historical scientific source in a separate replay environment before execution; no source/hash checks are waived.'
                              if source_gaps else None)}
