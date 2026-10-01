"""Extension-specific scorer: immutable controls are inputs, not rerun targets."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import cei_v3_study as core
from ..contracts import sample_ids_sha256, VISIT_MEMBERSHIP_FILENAME, recursive_source_hashes
from ..tensorize import encode_graph, load_preprocessing
from .arm_guards import assert_extension_binding
from .io import digest, read_json, selected_graphs
from .study import comorbid_share, comorbid_nonempty


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def mapped_model(binding, checkpoint, path_map):
    """Map only the state open, retaining original config for replay comparisons."""
    current_source = recursive_source_hashes(Path(__file__).parents[1])
    protected = ('tensorize.py', 'contracts.py', 'schema.py', 'cei_v3_absence.py', 'cei_v3_ple.py')
    for name, expected in binding['source_code'].items():
        if name.startswith('methods/') or name in protected:
            if current_source.get(name) != expected:
                raise ValueError(f'historical scientific source mismatch: {name}')
    view = copy.deepcopy(binding)
    config = view['method_config']
    mapped = str(path_map.resolve(config['v3_state_path'], config['v3_state_sha256']))
    config['v3_state_path'] = mapped
    config['effective_settings']['v3_state'] = mapped
    model = core._default_model_factory(view, checkpoint)
    actual = model.run_config()
    for key in ('v3_state_sha256', 'knot_table_sha256', 'universe_sha256',
                'preprocessing_sha256', 'arm', 'k', 'encoder_depth', 'edge_direction',
                'hidden', 'comorbid_block', 'pair_mode', 'common_init_identical_to_c'):
        if actual.get(key) != binding['method_config'].get(key):
            raise ValueError(f'rebuilt model config drift: {key}')
    model.state = replace(model.state, path=binding['method_config']['v3_state_path'])
    if model.run_config() != binding['method_config']:
        raise ValueError('rebuilt model full run_config differs from retained provenance')
    return model


def retained_screen(core_root, frozen, controls, path_map):
    """Verify stored C rows and pairing without ever asking a graph encoder."""
    record = read_json(Path(core_root) / 'screen_record.json')
    record_hash = core.screen_record_sha256(record)
    reference = None
    results, arrays = {}, {}
    for seed, directory in controls['directories'].items():
        binding = read_json(directory / 'binding.json')
        result = read_json(directory / 'screen' / 'screen_result.json')
        for key, expected in (
                ('arm', 'C'), ('seed', seed), ('k', frozen['k_selected']),
                ('checkpoint_sha256', digest(directory / 'best.pt')),
                ('binding_sha256', digest(directory / 'binding.json')),
                ('k_selection_sha256', core.k_selection_sha256(frozen)),
                ('screen_record_sha256', record_hash), ('validation_evaluated', False),
                ('test_evaluated', False)):
            if result.get(key) != expected:
                raise ValueError(f'retained C screen result mismatch: {key}')
        for key, expected in (
                ('artifact_sha256', binding['artifact_graphs_sha256']),
                ('targets_sha256', binding['targets_sha256']),
                ('kept_label_indices', binding['kept_label_indices']),
                ('preprocessing_sha256', binding['preprocessing_sha256'])):
            if record.get(key) != expected:
                raise ValueError(f'retained screen binding mismatch: {key}')
        values = {}
        for name, field in (('logits', 'logits_path'), ('proba', 'proba_path')):
            with np.load(path_map.resolve(result[field]), allow_pickle=False) as saved:
                if set(saved.files) != {name, 'y', 'subjects', 'sample_ids'}:
                    raise ValueError('retained screen NPZ fields differ')
                current = {key: saved[key].copy() for key in saved.files}
            if array_digest(current[name]) != result[name + '_sha256']:
                raise ValueError('retained screen arrays differ from immutable hash')
            if values and any(not np.array_equal(values[key], current[key])
                              for key in ('y', 'subjects', 'sample_ids')):
                raise ValueError('retained logits/proba identities differ')
            values.update(current)
        ids = values['sample_ids'].astype(str).tolist()
        if (len(ids) != record['row_count'] or len(set(ids)) != len(ids)
                or sample_ids_sha256(ids) != record['screen_sample_ids_sha256']
                or values['logits'].dtype != np.float32
                or values['logits'].shape != (len(ids), 10)
                or not np.isfinite(values['logits']).all()
                or not np.isfinite(values['proba']).all()
                or core.weighted_macro_f1(values['y'], values['logits'].argmax(1))
                    != result['macro_f1']):
            raise ValueError('retained screen row/score contract failed')
        import torch
        expected_proba = torch.softmax(torch.from_numpy(values['logits']), 1).numpy()
        if not np.array_equal(expected_proba, values['proba']):
            raise ValueError('retained proba is not softmax of retained logits')
        identity = tuple(values[key] for key in ('y', 'subjects', 'sample_ids'))
        if reference is not None and any(not np.array_equal(a, b)
                                          for a, b in zip(reference, identity)):
            raise ValueError('retained C rows are not paired across seeds')
        reference = identity
        results[seed], arrays[seed] = result, values
    return record, results, arrays


def encode_selected(artifact, preprocessing, ids, targets, *, fold, edge_direction,
                    preprocessing_sha256):
    import torch
    if fold not in ('screen', 'validation'):
        raise ValueError('scorer has no test or dev graph path')
    expected_split = 'train' if fold == 'screen' else 'validation'
    for sid in ids:
        if sid not in targets or targets[sid][1] != expected_split:
            raise ValueError('requested graph belongs to wrong fold')
    if digest(preprocessing) != preprocessing_sha256:
        raise ValueError('preprocessing bytes changed')
    prep = load_preprocessing(read_json(preprocessing))
    rows = {}
    for graph, membership in selected_graphs(
            Path(artifact) / 'graphs.jsonl', Path(artifact) / VISIT_MEMBERSHIP_FILENAME, ids):
        sid = graph['sample_id']
        row = encode_graph(graph, prep, membership, edge_direction=edge_direction)
        row.y = torch.tensor([targets[sid][0]], dtype=torch.long)
        row.subject = targets[sid][2]
        rows[sid] = row
    return tuple(rows[sid] for sid in ids)


def score_extension(directory, output, record, reference, frozen, controls,
                    artifact, targets, path_map, batch_size=128):
    """Guard everything before deserializing any requested screen graph."""
    import torch
    from torch_geometric.loader import DataLoader

    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError('occupied extension screen stage')
    binding = read_json(directory / 'binding.json')
    successor = read_json(directory / 'study_binding.json')
    assert_extension_binding({**binding, **successor}, controls)
    core.check_stage_result(directory, binding, read_json(directory / 'result.json'))
    if successor.get('binding_sha256') != digest(directory / 'binding.json'):
        raise ValueError('extension study binding hash differs')
    control = read_json(controls['directories'][binding['seed']] / 'binding.json')
    for key in ('artifact_graphs_sha256', 'artifact_visit_membership_sha256', 'targets_sha256',
                'kept_label_indices', 'preprocessing_sha256', 'split_sample_ids_sha256',
                'label_order', 'class_weight_values', 'class_counts_train'):
        if binding.get(key) != control.get(key):
            raise ValueError(f'extension/control input parity drift: {key}')
    ids = reference['sample_ids'].astype(str).tolist()
    if sample_ids_sha256(ids) != record['screen_sample_ids_sha256']:
        raise ValueError('screen row draw hash differs')
    model = mapped_model(binding, directory / 'best.pt', path_map)
    rows = encode_selected(artifact, directory / 'preprocessing.json', ids, targets,
                           fold='screen', edge_direction=binding['edge_direction'],
                           preprocessing_sha256=binding['preprocessing_sha256'])
    model.eval()
    logits, ys, subjects, seen_ids = [], [], [], []
    secondary = {'absence_share': [], 'comorbid_share': [], 'comorbid_nonempty': []}
    with torch.no_grad():
        for batch in DataLoader(rows, batch_size=batch_size, shuffle=False):
            parts = model.forward_continuous(model.continuous_inputs(batch),
                                             batch.edge_index, batch, return_parts=True)
            logits.append(parts['logits'].detach().cpu().to(torch.float32))
            ys.append(batch.y.view(-1).cpu())
            subjects.extend(str(s) for s in batch.subject)
            seen_ids.extend(str(s) for s in batch.sample_id)
            kwargs = dict(batch_index=batch.batch, edge_index=batch.edge_index,
                          graph_count=int(batch.num_graphs))
            secondary['absence_share'].append(core.absence_share(parts, **kwargs))
            secondary['comorbid_share'].append(comorbid_share(parts, **kwargs))
            secondary['comorbid_nonempty'].append(comorbid_nonempty(
                parts, batch_index=batch.batch, graph_count=int(batch.num_graphs)))
    raw = torch.cat(logits).numpy()
    y = torch.cat(ys).numpy()
    if (seen_ids != ids or not np.array_equal(y, reference['y'])
            or not np.array_equal(np.asarray(subjects).astype(str), reference['subjects'])
            or raw.shape != (len(ids), 10) or not np.isfinite(raw).all()):
        raise ValueError('extension screen rows differ from the retained control')
    proba = torch.softmax(torch.from_numpy(raw), 1).numpy()
    arrays = dict(y=y, subjects=np.asarray(subjects).astype(str),
                  sample_ids=np.asarray(ids).astype(str))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'logits.npz', logits=raw, **arrays)
    np.savez_compressed(output / 'proba.npz', proba=proba, **arrays)
    shares = {name: np.concatenate(chunks) for name, chunks in secondary.items()}
    np.savez_compressed(output / 'shares.npz', sample_ids=arrays['sample_ids'], **shares)
    result = dict(
        arm=successor['extension_arm'], seed=binding['seed'], k=frozen['k_selected'],
        checkpoint_sha256=digest(directory / 'best.pt'),
        binding_sha256=digest(directory / 'binding.json'),
        study_binding_sha256=digest(directory / 'study_binding.json'),
        k_selection_sha256=core.k_selection_sha256(frozen),
        screen_record_sha256=core.screen_record_sha256(record), row_count=len(ids),
        macro_f1=core.weighted_macro_f1(y, raw.argmax(1)),
        logits_path=str(output / 'logits.npz'), logits_sha256=array_digest(raw),
        proba_path=str(output / 'proba.npz'), proba_sha256=array_digest(proba),
        shares_path=str(output / 'shares.npz'), shares_sha256=digest(output / 'shares.npz'),
        screen_sample_ids_sha256=sample_ids_sha256(ids),
        absence_share_mean=float(shares['absence_share'].mean()),
        training_source_code=binding['source_code'],
        scorer_source_code=recursive_source_hashes(Path(__file__).parents[1]),
        control_binding_sha256=controls['control_binding_sha256'],
        interpretation='exploratory reuse of an already inspected screen; not prospective confirmation',
        validation_evaluated=False, test_evaluated=False)
    core._write_new_json(output / 'screen_result.json', result)
    return result
