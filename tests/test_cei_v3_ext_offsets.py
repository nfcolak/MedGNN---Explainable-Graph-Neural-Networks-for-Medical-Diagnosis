"""Synthetic tests for item 5 — validation-tuned per-class logit offsets (unit X3).

Everything here is synthetic: no data file, run directory, checkpoint or fold is opened.
Spec: EXT design §4 (4.1–4.6, E3/E7/E13) and §9 X3.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from comparison.standardized.clinical_graph_v2 import cei_v3_screen as screen
from comparison.standardized.clinical_graph_v2 import cei_v3_study as study
from comparison.standardized.clinical_graph_v2.cei_v3_ext import offsets
from comparison.standardized.clinical_graph_v2.cei_v3_ext import validation_scoring as vs
from comparison.standardized.clinical_graph_v2.contracts import sample_ids_sha256

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_cei_v3_study as u5  # noqa: E402  (U5 synthetic study fixtures; read-only reuse)


# ------------------------------------------------------------------ fixtures


def _validation_rows(n=12, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        x = torch.tensor(rng.normal(size=(3, 10)), dtype=torch.float32)
        data = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 2]]))
        data.y = torch.tensor([int(rng.integers(0, 10))])
        data.subject = f'pv-{i // 2}'
        data.sample_id = f'va-{i:05d}'
        rows.append(data)
    return rows


def _frozen_c_study(tmp_path, rows):
    """Nine C stages + k_selection.json (K* = 8) whose bindings carry the validation
    fold of `rows` (ordered id hash and count), plus an A stage; nothing is trained."""
    ids = [r.sample_id for r in rows]
    validation_hash = sample_ids_sha256(ids)
    plan_ = study.plan(u5._config(tmp_path))
    overrides = {stage.name: {
        'split_sample_ids_sha256': {'train': 'e' * 64, 'dev': 'f' * 64,
                                    'validation': validation_hash},
        'counts': {'train': 10000, 'dev': 5000, 'validation': len(ids)},
    } for stage in plan_.stages[:9]}
    bindings = u5._grid_bindings(plan_, u5.METRICS_CLEAR, **overrides)
    stage_dirs = {}
    for stage in plan_.stages[:9]:
        stage_dir = Path(stage.output)
        stage_dir.mkdir(parents=True)
        (stage_dir / 'binding.json').write_text(
            json.dumps(bindings[stage.name], indent=2, sort_keys=True) + '\n')
        (stage_dir / 'best.pt').write_bytes(f'checkpoint {stage.name}'.encode())
        stage_dirs[stage.name] = stage_dir
    selection = study.select_k(bindings)
    record = study.write_k_selection(selection, stage_dirs, Path(plan_.k_selection_path))
    assert record['k_selected'] == 8
    for stage in plan_.stages[:9]:
        if stage.k == 8:
            (stage_dirs[stage.name] / 'result.json').write_text(json.dumps(
                u5._result(bindings[stage.name]), indent=2, sort_keys=True) + '\n')
            study.write_study_binding(stage_dirs[stage.name], stage, record)
    frozen = study.plan(u5._config(tmp_path, k_selection=record))
    a_stage = u5._stage(frozen, 'A_seed1234')
    a_dir = Path(a_stage.output)
    a_dir.mkdir(parents=True)
    a_binding = u5._binding(a_stage, root=frozen.output_root,
                            **overrides['C_K8_seed1234'])
    (a_dir / 'binding.json').write_text(json.dumps(a_binding, indent=2, sort_keys=True) + '\n')
    (a_dir / 'result.json').write_text(json.dumps(u5._result(a_binding), indent=2, sort_keys=True) + '\n')
    (a_dir / 'best.pt').write_bytes(b'checkpoint A_seed1234')
    study.write_study_binding(a_dir, a_stage, record)
    c_dirs = {seed: stage_dirs[f'C_K8_seed{seed}'] for seed in study.SEEDS}
    approval = {
        'allow_validation': True,
        'approval_reference': 'G3 item-5 approval (synthetic fixture)',
        'checkpoint_sha256': [hashlib.sha256((d / 'best.pt').read_bytes()).hexdigest()
                              for d in c_dirs.values()],
        'validation_sample_ids_sha256': validation_hash,
        'timestamp': '2026-09-30T00:00:00+02:00',
    }
    return dict(frozen=frozen, record=record, stage_dirs=stage_dirs, c_dirs=c_dirs,
                a_dir=a_dir, approval=approval, ids=ids, validation_hash=validation_hash,
                root=Path(frozen.output_root))


def _encoder(rows, fold='validation', reads=None):
    return study.ScreenEncoder(
        ids=tuple(r.sample_id for r in rows), fold=fold,
        rows=lambda: (reads.append(1) if reads is not None else None) or iter(rows))


def _reference_fit(logits, y, grid=range(-20, 21), sweeps=5):
    """Independent brute-force reference of EXT §4.2 (coordinate ascent, tie rule)."""
    z = np.asarray(logits, dtype=np.float64)
    i_vec = np.zeros(10, dtype=np.int64)
    for _ in range(sweeps):
        for c in range(10):
            best = None
            for g in grid:
                trial = i_vec.copy()
                trial[c] = g
                score = study.weighted_macro_f1(y, (z + trial / 10.0).argmax(1))
                key = (score, -abs(g), -g)
                if best is None or key > best[0]:
                    best = (key, g)
            i_vec[c] = best[1]
    return i_vec


def _shifted_logits(n=400, seed=3, shift=(0.0, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)):
    """Rows whose true-class logit is best but class 1 is biased down and class 9 up."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, n)
    logits = rng.normal(scale=0.4, size=(n, 10))
    logits[np.arange(n), y] += 0.8
    logits += np.asarray(shift)
    return logits.astype(np.float32), y.astype(np.int64)


# ----------------------------------------------------------- step 1: offset fitter


def test_fit_offsets_matches_the_reference_coordinate_ascent_and_is_deterministic():
    logits, y = _shifted_logits()
    delta = offsets.fit_offsets(logits, y)
    assert isinstance(delta, np.ndarray) and delta.shape == (10,)
    assert np.issubdtype(delta.dtype, np.integer)
    assert all(-20 <= int(v) <= 20 for v in delta)
    expected = _reference_fit(logits, y)
    assert delta.tolist() == expected.tolist()
    assert np.array_equal(offsets.fit_offsets(logits, y), delta)   # deterministic
    before = study.weighted_macro_f1(y, logits.argmax(1))
    after = study.weighted_macro_f1(y, offsets.apply_offsets(logits, delta))
    assert after > before
    assert delta[1] > 0 and delta[9] < 0   # the biased classes are corrected in direction


def test_fit_offsets_runs_exactly_five_sweeps_in_label_order_with_unrounded_scores():
    logits, y = _shifted_logits(n=120)
    trace = []
    delta = offsets.fit_offsets(logits, y, trace=trace)
    assert len(trace) == 5 * 10
    assert [entry['sweep'] for entry in trace] == [s for s in range(5) for _ in range(10)]
    assert [entry['label'] for entry in trace] == list(range(10)) * 5
    state = np.zeros(10, dtype=np.int64)
    for entry in trace:
        state[entry['label']] = entry['i']
        expected = study.weighted_macro_f1(y, offsets.apply_offsets(logits, state))
        assert entry['score'] == expected        # exact float64, no rounding (E13)
    assert any(entry['score'] != round(entry['score'], 6) for entry in trace)
    assert trace[-1]['score'] == study.weighted_macro_f1(y, offsets.apply_offsets(logits, delta))
    assert state.tolist() == delta.tolist()
    short = []
    offsets.fit_offsets(logits, y, sweeps=2, trace=short)
    assert len(short) == 20


def test_fit_offsets_compares_scores_at_full_precision(monkeypatch):
    """A gain of 1e-9 per row moved into class 3 must be taken; a 6-decimal comparison
    would see ties everywhere and keep i = 0 (tie rule), which is the rounded behaviour
    (E13). The ascent ends with every row predicted as 3 (i_3 = +20, the others pushed
    down as far as needed)."""
    logits, y = _shifted_logits(n=60)

    def tiny_gain_metric(y_true, pred, weights=None, *, num_classes=10):
        return 0.5 + 1e-9 * float(np.sum(np.asarray(pred) == 3))
    monkeypatch.setattr(offsets, 'weighted_macro_f1', tiny_gain_metric)
    delta = offsets.fit_offsets(logits, y)
    assert delta[3] == 20
    assert (offsets.apply_offsets(logits, delta) == 3).all()
    assert delta.tolist() != [0] * 10


def test_fit_offsets_tie_rule_prefers_smallest_magnitude_then_the_negative_value():
    # Row A (true 0): z0 = 0, z1 = 0.25 -> correct only with i_0 >= 3.
    # Row B (true 1): z0 = 0.25, z1 = 0 -> correct only with i_0 <= -3.
    # Both branches score the same macro-F1 -> tie set {-20..-3} u {3..20} -> -3.
    logits = np.full((2, 10), -10.0, dtype=np.float32)
    logits[0, :2] = (0.0, 0.25)
    logits[1, :2] = (0.25, 0.0)
    y = np.asarray([0, 1])
    delta = offsets.fit_offsets(logits, y)
    assert delta.tolist() == [-3] + [0] * 9
    # Clear margins: every grid value ties at the same score -> i = 0 everywhere,
    # and the five sweeps still run in full.
    clear, y_clear = _shifted_logits(n=50, shift=(0.0,) * 10)
    clear[np.arange(len(y_clear)), y_clear] += 10.0
    trace = []
    assert offsets.fit_offsets(clear, y_clear, trace=trace).tolist() == [0] * 10
    assert len(trace) == 50


def test_fit_offsets_accepts_validation_rows_only_and_raw_float32_logits():
    logits, y = _shifted_logits(n=30)
    with pytest.raises(ValueError, match='validation'):
        offsets.fit_offsets(logits, y, fold='screen')
    with pytest.raises(ValueError, match='validation'):
        offsets.fit_offsets(logits, y, fold='test')
    splits = ['validation'] * 30
    assert offsets.fit_offsets(logits, y, row_splits=splits).shape == (10,)
    for bad in ('train', 'dev', 'screen', 'test'):
        wrong = list(splits)
        wrong[7] = bad
        with pytest.raises(ValueError, match='validation'):
            offsets.fit_offsets(logits, y, row_splits=wrong)
    with pytest.raises(ValueError, match='float32'):
        offsets.fit_offsets(logits.astype(np.float64), y)
    log_softmax = torch.log_softmax(torch.from_numpy(logits), dim=1).numpy()
    with pytest.raises(ValueError, match='softmax'):
        offsets.fit_offsets(log_softmax, y)
    with pytest.raises(ValueError):
        offsets.fit_offsets(logits[:, :9], y)
    with pytest.raises(ValueError):
        offsets.fit_offsets(logits, y[:-1])
    with pytest.raises(ValueError, match='grid'):
        offsets.fit_offsets(logits, y, grid=range(1, 5))   # no start point 0


def test_apply_offsets_is_argmax_of_logits_plus_delta_over_ten():
    logits, y = _shifted_logits(n=40)
    delta = np.asarray([0, 9, 0, 0, 0, 0, 0, 0, 0, -5])
    pred = offsets.apply_offsets(logits, delta)
    expected = (logits.astype(np.float64) + delta / 10.0).argmax(1)
    assert pred.tolist() == expected.tolist()
    assert pred.tolist() != logits.argmax(1).tolist()
    assert offsets.apply_offsets(logits, np.zeros(10, dtype=int)).tolist() == logits.argmax(1).tolist()
    with pytest.raises(ValueError):
        offsets.apply_offsets(logits, np.zeros(9, dtype=int))
    with pytest.raises(ValueError):
        offsets.apply_offsets(logits, np.full(10, 0.5))   # offsets are bound as integers


# --------------------------------------------- step 1: validation scorer refusals


def test_score_validation_writes_raw_logits_with_hashes_and_binds_the_approval(tmp_path):
    rows = _validation_rows()
    fx = _frozen_c_study(tmp_path, rows)
    c_dir = fx['c_dirs'][7]
    stub = u5._StubAdapter()
    reads = []
    out = vs.score_validation(study.Checkpoint(str(c_dir), fx['frozen'].k_selection_path),
                              _encoder(rows, reads=reads), fx['approval'],
                              model_factory=lambda binding, path: stub, batch_size=5)
    out = Path(out)
    assert out == c_dir / 'validation' and reads == [1]
    assert stub.calls == 3 and stub.training is False
    with np.load(out / 'logits.npz', allow_pickle=False) as saved:
        assert set(saved.files) == {'logits', 'y', 'subjects', 'sample_ids'}
        logits = saved['logits']
        assert logits.dtype == np.float32 and logits.shape == (12, 10)
        assert np.array_equal(saved['y'], np.asarray([int(r.y) for r in rows]))
        assert list(saved['subjects'].astype(str)) == [r.subject for r in rows]
        assert list(saved['sample_ids'].astype(str)) == [r.sample_id for r in rows]
    expected = np.concatenate([
        stub.forward_continuous(r.x, r.edge_index, Data(batch=torch.zeros(3, dtype=torch.long),
                                                         num_graphs=1)).numpy() for r in rows])
    assert np.array_equal(logits, expected.astype(np.float32))
    result = json.loads((out / 'validation_result.json').read_text())
    assert result['fold'] == 'validation' and result['arm'] == 'C'
    assert (result['seed'], result['k'], result['row_count']) == (7, 8, 12)
    assert result['logits_sha256'] == hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest()
    assert result['logits_path'] == str(out / 'logits.npz')
    assert result['checkpoint_sha256'] == hashlib.sha256((c_dir / 'best.pt').read_bytes()).hexdigest()
    assert result['binding_sha256'] == hashlib.sha256((c_dir / 'binding.json').read_bytes()).hexdigest()
    assert result['k_selection_sha256'] == study.k_selection_sha256(fx['record'])
    assert result['approval_record_sha256'] == vs.approval_record_sha256(fx['approval'])
    assert result['validation_sample_ids_sha256'] == fx['validation_hash']
    assert result['metric'] == 'weighted_macro_f1_unrounded'
    assert result['tuning_macro_f1'] == study.weighted_macro_f1([int(r.y) for r in rows], logits.argmax(1))
    assert result['validation_evaluated'] is True and result['test_evaluated'] is False
    assert not (c_dir / 'validation.npz').exists() and not (out / 'proba.npz').exists()
    with pytest.raises(FileExistsError):
        vs.score_validation(study.Checkpoint(str(c_dir), fx['frozen'].k_selection_path),
                            _encoder(rows), fx['approval'], model_factory=lambda b, p: stub)


@pytest.mark.parametrize('kind', ['no_record', 'not_allowed', 'string_true', 'no_reference',
                                  'other_checkpoint', 'ids_hash', 'no_timestamp'])
def test_score_validation_refuses_a_missing_or_unbound_approval_record(tmp_path, kind):
    rows = _validation_rows()
    fx = _frozen_c_study(tmp_path, rows)
    approval = json.loads(json.dumps(fx['approval']))
    if kind == 'no_record':
        approval = None
    elif kind == 'not_allowed':
        approval['allow_validation'] = False
    elif kind == 'string_true':
        approval['allow_validation'] = 'true'
    elif kind == 'no_reference':
        approval['approval_reference'] = ''
    elif kind == 'other_checkpoint':
        approval['checkpoint_sha256'] = ['0' * 64] * 3
    elif kind == 'ids_hash':
        approval['validation_sample_ids_sha256'] = '1' * 64
    elif kind == 'no_timestamp':
        del approval['timestamp']
    reads = []
    c_dir = fx['c_dirs'][1234]
    with pytest.raises(ValueError, match='approval|validation'):
        vs.score_validation(study.Checkpoint(str(c_dir), fx['frozen'].k_selection_path),
                            _encoder(rows, reads=reads), approval,
                            model_factory=lambda binding, path: u5._StubAdapter())
    assert reads == []
    assert not (c_dir / 'validation').exists()


@pytest.mark.parametrize('kind', ['arm_a', 'losing_k', 'fold_screen', 'fold_test', 'test_id',
                                  'ids_order', 'row_count', 'no_freeze'])
def test_score_validation_refuses_non_c_checkpoints_and_non_validation_rows(tmp_path, kind):
    rows = _validation_rows()
    fx = _frozen_c_study(tmp_path, rows)
    stage_dir, fold, ids, freeze = fx['c_dirs'][2025], 'validation', None, fx['frozen'].k_selection_path
    targets = {r.sample_id: (int(r.y), 'validation', r.subject) for r in rows}
    if kind == 'arm_a':
        stage_dir = fx['a_dir']
    elif kind == 'losing_k':
        stage_dir = fx['stage_dirs']['C_K4_seed1234']
        (stage_dir / 'result.json').write_text(json.dumps(u5._result(
            json.loads((stage_dir / 'binding.json').read_text())), indent=2, sort_keys=True) + '\n')
    elif kind == 'fold_screen':
        fold = 'screen'
    elif kind == 'fold_test':
        fold = 'test'
    elif kind == 'test_id':
        targets[rows[3].sample_id] = (int(rows[3].y), 'test', rows[3].subject)
    elif kind == 'ids_order':
        ids = tuple(r.sample_id for r in reversed(rows))
    elif kind == 'row_count':
        rows = rows[:-1]
    elif kind == 'no_freeze':
        freeze = str(tmp_path / 'absent.json')
    reads = []
    encoder = study.ScreenEncoder(ids=ids or tuple(r.sample_id for r in rows), fold=fold,
                                  rows=lambda: reads.append(1) or iter(rows))
    with pytest.raises(ValueError):
        vs.score_validation(study.Checkpoint(str(stage_dir), freeze), encoder, fx['approval'],
                            model_factory=lambda binding, path: u5._StubAdapter(),
                            targets=targets)
    assert reads == []
    assert not (stage_dir / 'validation').exists()


def test_validation_encoder_opens_the_fold_through_u4_with_the_approval_record(tmp_path, monkeypatch):
    rows = _validation_rows()
    fx = _frozen_c_study(tmp_path, rows)
    seen = []

    def fake_encode_rows(artifact, prep_state, ids, **kwargs):
        seen.append((str(artifact), prep_state, list(ids), kwargs))
        return iter(rows)
    monkeypatch.setattr(screen, 'encode_rows', fake_encode_rows)
    encoder = vs.validation_encoder(tmp_path / 'artifact', {'preprocessing_version': 'x'},
                                    fx['ids'], approval_record=fx['approval'],
                                    targets={r.sample_id: (0, 'validation', r.subject) for r in rows},
                                    edge_direction='forward', preprocessing_sha256='b' * 64)
    assert encoder.fold == 'validation' and tuple(encoder.ids) == tuple(fx['ids'])
    assert seen == []                       # lazy: nothing is encoded until rows() is called
    assert [d.sample_id for d in encoder.rows()] == fx['ids']
    assert len(seen) == 1
    artifact, prep_state, ids, kwargs = seen[0]
    assert ids == fx['ids'] and prep_state == {'preprocessing_version': 'x'}
    assert kwargs['fold'] == 'validation' and kwargs['allow_validation'] is True
    assert kwargs['approval_record'] == fx['approval']
    assert kwargs['edge_direction'] == 'forward' and kwargs['preprocessing_sha256'] == 'b' * 64
    # The approval gate is checked here too, before U4 is asked for anything.
    with pytest.raises(ValueError, match='approval|validation'):
        vs.validation_encoder(tmp_path / 'artifact', {}, fx['ids'],
                              approval_record={'allow_validation': False}, targets={},
                              edge_direction='forward')
    assert len(seen) == 1


def test_approval_record_sha256_is_canonical_and_matches_the_written_file(tmp_path):
    rows = _validation_rows()
    fx = _frozen_c_study(tmp_path, rows)
    digest = vs.approval_record_sha256(fx['approval'])
    assert len(digest) == 64
    assert digest == vs.approval_record_sha256(dict(reversed(list(fx['approval'].items()))))
    assert digest != vs.approval_record_sha256(dict(fx['approval'], timestamp='other'))
    path = fx['root'] / vs.APPROVAL_FILENAME
    written = vs.write_approval_record(path, fx['approval'])
    assert written == fx['approval']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    loaded, loaded_digest = vs.load_approval_record(path)
    assert loaded == fx['approval'] and loaded_digest == digest
    with pytest.raises(FileExistsError):
        vs.write_approval_record(path, fx['approval'])
    with pytest.raises(ValueError, match='approval'):
        vs.write_approval_record(fx['root'] / 'other.json', dict(fx['approval'], allow_validation=False))


# ------------------------------------------------------ step 2: offset freeze replay


class _PresetAdapter:
    """Adapter stub returning preset logits per sample id (raw float32, shape [G, 10])."""

    def __init__(self, logits_by_id):
        self.logits_by_id = logits_by_id
        self.training = True

    def eval(self):
        self.training = False
        return self

    def continuous_inputs(self, batch):
        return batch.x

    def forward_continuous(self, features, edge_index, metadata, *, return_parts=False):
        ids = metadata.sample_id if isinstance(metadata.sample_id, (list, tuple)) else [metadata.sample_id]
        logits = torch.tensor(np.stack([self.logits_by_id[str(s)] for s in ids]), dtype=torch.float32)
        return {'logits': logits} if return_parts else logits


def _scored_validation(tmp_path, n=60, seed=3):
    """Frozen study whose validation fold has `n` rows with EXT-shaped logits; the C_K8_seed7
    checkpoint is scored on validation through `score_validation` (synthetic adapter)."""
    logits, y = _shifted_logits(n=n, seed=seed)
    rows = _validation_rows(n=n, seed=seed)
    for row, label in zip(rows, y):
        row.y = torch.tensor([int(label)])
    fx = _frozen_c_study(tmp_path, rows)
    fx['rows'], fx['logits'], fx['y'] = rows, logits, y
    fx['adapter'] = _PresetAdapter({r.sample_id: logits[i] for i, r in enumerate(rows)})
    fx['approval_path'] = fx['root'] / vs.APPROVAL_FILENAME
    vs.write_approval_record(fx['approval_path'], fx['approval'])
    c_dir = fx['c_dirs'][7]
    fx['c_dir'] = c_dir
    fx['validation_dir'] = Path(vs.score_validation(
        study.Checkpoint(str(c_dir), fx['frozen'].k_selection_path), _encoder(rows),
        fx['approval'], model_factory=lambda binding, path: fx['adapter'], batch_size=16))
    return fx


def _replay_kwargs(fx):
    return dict(stage_dir=fx['c_dir'], approval_path=fx['approval_path'])


def test_offset_record_binds_every_hash_and_the_fitted_offsets(tmp_path):
    fx = _scored_validation(tmp_path)
    c_dir, out = fx['c_dir'], fx['validation_dir']
    with np.load(out / 'logits.npz', allow_pickle=False) as saved:
        stored_logits, stored_y = saved['logits'], saved['y']
    assert np.array_equal(stored_logits, fx['logits']) and np.array_equal(stored_y, fx['y'])
    expected_delta = offsets.fit_offsets(stored_logits, stored_y)
    assert expected_delta.tolist() != [0] * 10       # the fixture exercises a real fit
    record = offsets.offset_record(out, fx['approval_path'])
    assert record['version'] == offsets.OFFSET_RECORD_VERSION
    assert record['arm'] == 'O' and record['control_arm'] == 'C'
    assert (record['stage'], record['seed'], record['k']) == ('C_K8_seed7', 7, 8)
    assert record['delta_int'] == expected_delta.tolist()
    assert all(type(v) is int for v in record['delta_int'])
    assert record['delta'] == [v / 10 for v in expected_delta.tolist()]
    assert record['grid'] == list(range(-20, 21)) and record['sweeps'] == 5
    assert record['delta_scale'] == 10
    assert record['metric'] == 'weighted_macro_f1_unrounded'
    assert record['optimizer_rule'] == offsets.OPTIMIZER_RULE
    assert record['checkpoint_sha256'] == hashlib.sha256((c_dir / 'best.pt').read_bytes()).hexdigest()
    assert record['binding_sha256'] == hashlib.sha256((c_dir / 'binding.json').read_bytes()).hexdigest()
    assert record['k_selection_sha256'] == study.k_selection_sha256(fx['record'])
    assert record['validation_sample_ids_sha256'] == fx['validation_hash']
    assert record['validation_row_count'] == 60
    assert record['validation_logits_sha256'] == hashlib.sha256(
        np.ascontiguousarray(stored_logits).tobytes()).hexdigest()
    assert record['validation_logits_path'] == str(out / 'logits.npz')
    assert record['approval_record_sha256'] == vs.approval_record_sha256(fx['approval'])
    result = json.loads((out / 'validation_result.json').read_text())
    assert record['validation_result_sha256'] == hashlib.sha256(
        (out / 'validation_result.json').read_bytes()).hexdigest()
    assert record['validation_result_sha256'] != record['validation_logits_sha256']
    assert result['approval_record_sha256'] == record['approval_record_sha256']
    assert record['tuning_macro_f1_before'] == result['tuning_macro_f1']
    assert record['tuning_macro_f1_after'] == study.weighted_macro_f1(
        stored_y, offsets.apply_offsets(stored_logits, expected_delta))
    assert record['tuning_macro_f1_after'] > record['tuning_macro_f1_before']
    assert 'tuning score' in record['tuning_score_caveat']
    assert record['validation_evaluated'] is True and record['test_evaluated'] is False
    assert record['screen_read_before_freeze'] is False
    # Frozen to disk once, canonical bytes, hash of the record equals the file hash.
    path = out / offsets.OFFSET_RECORD_FILENAME
    assert path.is_file() and json.loads(path.read_text()) == record
    assert offsets.offset_record_sha256(record) == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        offsets.offset_record(out, fx['approval_path'])


def test_offset_record_refuses_unbound_inputs_before_fitting(tmp_path):
    fx = _scored_validation(tmp_path)
    out = fx['validation_dir']
    # approval record on disk differs from the one bound at scoring time
    other = dict(fx['approval'], timestamp='2026-10-01T00:00:00+02:00')
    other_path = fx['root'] / 'other_approval.json'
    vs.write_approval_record(other_path, other)
    with pytest.raises(ValueError, match='approval'):
        offsets.offset_record(out, other_path)
    # logits tampered after scoring
    with np.load(out / 'logits.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    tampered = dict(arrays)
    tampered['logits'] = arrays['logits'] * np.float32(1.5)
    (out / 'logits.npz').unlink()
    np.savez_compressed(out / 'logits.npz', **tampered)
    with pytest.raises(ValueError, match='logits'):
        offsets.offset_record(out, fx['approval_path'])
    (out / 'logits.npz').unlink()
    np.savez_compressed(out / 'logits.npz', **arrays)
    # checkpoint bytes changed after scoring
    (fx['c_dir'] / 'best.pt').write_bytes(b'another checkpoint')
    with pytest.raises(ValueError, match='checkpoint'):
        offsets.offset_record(out, fx['approval_path'])
    assert not (out / offsets.OFFSET_RECORD_FILENAME).exists()


def test_offset_replay_recomputes_delta_bit_for_bit_and_detects_every_tamper(tmp_path):
    fx = _scored_validation(tmp_path)
    out = fx['validation_dir']
    record = offsets.offset_record(out, fx['approval_path'])
    assert offsets.assert_offset_replay(record, out, fx['approval_path']) is None
    frozen_path = out / offsets.OFFSET_RECORD_FILENAME
    assert frozen_path.is_file(), 'the offset record must be frozen to disk before any screen read'
    loaded = json.loads(frozen_path.read_text())
    assert offsets.assert_offset_replay(loaded, out, fx['approval_path']) is None
    assert loaded == record   # pure: replay does not modify the record

    def expect(mutation, message):
        tampered = json.loads(json.dumps(record))
        mutation(tampered)
        with pytest.raises(ValueError, match=message):
            offsets.assert_offset_replay(tampered, out, fx['approval_path'])

    expect(lambda r: r.__setitem__('delta_int', [v + 1 for v in r['delta_int']]), 'delta')
    expect(lambda r: r.__setitem__('delta_int', r['delta_int'][:9]), 'delta')
    expect(lambda r: r.__setitem__('delta', [v + 0.1 for v in r['delta']]), 'delta')
    expect(lambda r: r.__setitem__('checkpoint_sha256', '0' * 64), 'checkpoint')
    expect(lambda r: r.__setitem__('validation_sample_ids_sha256', '0' * 64), 'sample')
    expect(lambda r: r.__setitem__('validation_logits_sha256', '0' * 64), 'logits')
    expect(lambda r: r.__setitem__('approval_record_sha256', '0' * 64), 'approval')
    expect(lambda r: r.__setitem__('k_selection_sha256', '0' * 64), 'k_selection')
    expect(lambda r: r.__setitem__('binding_sha256', '0' * 64), 'binding')
    expect(lambda r: r.__setitem__('grid', list(range(-10, 11))), 'grid')
    expect(lambda r: r.__setitem__('sweeps', 4), 'sweep')
    expect(lambda r: r.__setitem__('metric', 'macro_f1_rounded'), 'metric')
    expect(lambda r: r.__setitem__('optimizer_rule', 'other'), 'rule')
    expect(lambda r: r.__setitem__('tuning_macro_f1_after', r['tuning_macro_f1_after'] + 1e-9), 'tuning')
    expect(lambda r: r.__setitem__('version', 'other'), 'version')
    expect(lambda r: r.pop('validation_result_sha256'), 'validation_result')
    # the stored validation logits changed on disk -> the fit is not reproducible
    with np.load(out / 'logits.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    arrays['logits'] = np.ascontiguousarray(arrays['logits'][:, ::-1])
    (out / 'logits.npz').unlink()
    np.savez_compressed(out / 'logits.npz', **arrays)
    with pytest.raises(ValueError, match='logits'):
        offsets.assert_offset_replay(record, out, fx['approval_path'])


def test_offset_records_are_one_per_checkpoint_and_seed_specific(tmp_path):
    fx = _scored_validation(tmp_path)
    record_7 = offsets.offset_record(fx['validation_dir'], fx['approval_path'])
    # a second C checkpoint (seed 1234) with different logits gets its own record and delta
    other_logits = np.ascontiguousarray(fx['logits'][:, ::-1])
    adapter = _PresetAdapter({r.sample_id: other_logits[i] for i, r in enumerate(fx['rows'])})
    c_1234 = fx['c_dirs'][1234]
    out_1234 = Path(vs.score_validation(
        study.Checkpoint(str(c_1234), fx['frozen'].k_selection_path), _encoder(fx['rows']),
        fx['approval'], model_factory=lambda binding, path: adapter))
    record_1234 = offsets.offset_record(out_1234, fx['approval_path'])
    assert record_1234['seed'] == 1234 and record_7['seed'] == 7
    assert record_1234['checkpoint_sha256'] != record_7['checkpoint_sha256']
    assert record_1234['validation_logits_sha256'] != record_7['validation_logits_sha256']
    assert record_1234['delta_int'] != record_7['delta_int']
    assert record_1234['approval_record_sha256'] == record_7['approval_record_sha256']
    assert record_1234['validation_sample_ids_sha256'] == record_7['validation_sample_ids_sha256']
    # replaying one record against the other checkpoint's directory fails
    with pytest.raises(ValueError):
        offsets.assert_offset_replay(record_7, out_1234, fx['approval_path'])
