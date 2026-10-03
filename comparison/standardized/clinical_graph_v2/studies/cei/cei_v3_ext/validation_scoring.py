"""Item 5 validation scoring (EXT spec §4.2, E3/E7; §9 X3).

The first and only validation scoring of the v3 programme: one inference per frozen arm-C
checkpoint (winning K) on the full validation fold, opened only by the item-5 approval
record. Stores raw float32 logits (`logits.npz`: logits, y, subjects, sample_ids) and their
SHA-256 in a new path (`<stage_dir>/validation/`), never overwriting; never re-infers, never
touches the test fold (U4 `encode_rows` drops `split == 'test'` rows before `encode_graph`).

From the first call onward validation is a tuning fold of the programme (EXT §4.4).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .. import cei_v3_screen as screen
from .. import cei_v3_study as study
from ....core.contracts import sample_ids_sha256
from .offsets import METRIC_NAME, NUM_CLASSES, VALIDATION_FOLD, weighted_macro_f1

VALIDATION_DIRNAME = 'validation'
VALIDATION_RESULT_FILENAME = 'validation_result.json'
APPROVAL_FILENAME = 'item5_validation_approval.json'
APPROVAL_KEYS = ('allow_validation', 'approval_reference', 'checkpoint_sha256',
                 'validation_sample_ids_sha256', 'timestamp')


# ------------------------------------------------------------ approval record


def _is_hex_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_approval_record(record) -> None:
    """Refuse anything but a complete item-5 approval record (EXT §4.2).

    Required: `allow_validation: true` (the boolean), the user's G3 approval reference for
    item 5 (nonempty text), the C checkpoint SHA-256 list, the validation ordered
    sample-id hash and a timestamp.
    """
    if not isinstance(record, dict):
        raise ValueError('item-5 approval record is required to open the validation fold '
                         '(EXT §4.2); none was given')
    missing = [key for key in APPROVAL_KEYS if key not in record]
    if missing:
        raise ValueError(f'item-5 approval record lacks bound fields: {missing}')
    if record['allow_validation'] is not True:
        raise ValueError("item-5 approval record must carry allow_validation: true (the "
                         f"boolean); got {record['allow_validation']!r}; validation stays closed")
    reference = record['approval_reference']
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("item-5 approval record lacks the user's G3 approval reference")
    hashes = record['checkpoint_sha256']
    if not isinstance(hashes, list) or not hashes or not all(_is_hex_digest(h) for h in hashes):
        raise ValueError('item-5 approval record must list the C checkpoint SHA-256 digests')
    if not _is_hex_digest(record['validation_sample_ids_sha256']):
        raise ValueError('item-5 approval record must bind the validation ordered sample-id hash')
    timestamp = record['timestamp']
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError('item-5 approval record lacks a timestamp')
    return None


def approval_record_sha256(record) -> str:
    """SHA-256 of the approval record in the canonical study file layout (so the digest of
    a record equals the digest of the bytes `write_approval_record` writes)."""
    return hashlib.sha256(study._canonical_file_bytes(record)).hexdigest()


def write_approval_record(path, record) -> dict:
    """Write `<output_root>/item5_validation_approval.json` once (never overwriting)."""
    validate_approval_record(record)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing occupied approval record {path}')
    study._write_new_json(path, record)
    return record


def load_approval_record(path):
    """Load an approval record; returns `(record, sha256 of the file bytes)`."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'item-5 approval record missing at {path}')
    payload = path.read_bytes()
    try:
        record = json.loads(payload.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f'item-5 approval record is not JSON: {path}') from error
    validate_approval_record(record)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != approval_record_sha256(record):
        raise ValueError('approval record bytes are not the canonical serialisation')
    return record, digest


# ------------------------------------------------------- validation encoder


def _check_validation_ids(ids, targets) -> None:
    if targets is None:
        return
    for sid in ids:
        entry = targets.get(sid)
        if entry is None:
            raise ValueError(f'id is not a member of the targets: the validation fold refuses it')
        if entry[1] != VALIDATION_FOLD:
            raise ValueError(f"id carries split {entry[1]!r}, not 'validation'; only "
                             'validation rows are scored here (test is never loaded)')


def validation_encoder(artifact, prep_state, ids, *, approval_record, targets, edge_direction,
                       preprocessing_sha256: Optional[str] = None) -> study.ScreenEncoder:
    """Lazy validation-fold encoder over U4 `encode_rows(fold='validation', ...)` (E3).

    The approval gate is checked here before U4 is asked for anything; rows are encoded
    only when `.rows()` is called (by `score_validation`, after every refusal check).
    """
    validate_approval_record(approval_record)
    ids = tuple(str(sid) for sid in ids)
    if len(set(ids)) != len(ids):
        raise ValueError('validation ids contain duplicates')
    if isinstance(targets, dict):
        _check_validation_ids(ids, targets)

    def rows():
        return screen.encode_rows(artifact, prep_state, list(ids), fold=VALIDATION_FOLD,
                                  edge_direction=edge_direction, allow_validation=True,
                                  approval_record=approval_record, targets=targets,
                                  preprocessing_sha256=preprocessing_sha256)
    return study.ScreenEncoder(ids=ids, fold=VALIDATION_FOLD, rows=rows)


# --------------------------------------------------------- validation scorer


def score_validation(checkpoint, encoder, approval_record, *, output_dir=None,
                     model_factory=None, batch_size=128, targets=None) -> Path:
    """Score one frozen arm-C checkpoint once on the validation fold (EXT §4.2, E3, E7).

    Refuses, before any validation row is read: a missing or unbound approval record
    (allow_validation, reference, this checkpoint's SHA-256, the validation id hash);
    a missing K-freeze record; a checkpoint that is not a freeze-bound arm-C stage at the
    frozen K; a binding/result with validation or test traces; an encoder fold other than
    `'validation'`; encoder ids whose count or ordered hash differ from the checkpoint
    binding's validation split; any id that `targets` (when given) does not mark
    `'validation'`. Writes raw float32 `logits.npz` (hashed) and `validation_result.json`
    into `<stage_dir>/validation/` (or `output_dir`), never overwriting, and returns that
    directory. No `proba`, no metrics beyond the tuning macro-F1.
    """
    import torch
    from torch_geometric.loader import DataLoader

    validate_approval_record(approval_record)
    approval_hash = approval_record_sha256(approval_record)
    stage_dir = Path(checkpoint.stage_dir)
    freeze_path = Path(checkpoint.k_selection_path)
    if not freeze_path.is_file():
        raise ValueError(f'no K-freeze record at {freeze_path}; validation is never scored '
                         'before K is frozen')
    k_selection = study._load_json(freeze_path, study.K_SELECTION_FILENAME)
    study._validate_k_selection_record(k_selection)
    freeze_hash = study._file_sha256(freeze_path)
    if freeze_hash != study.k_selection_sha256(k_selection):
        raise ValueError('k_selection.json bytes are not the canonical serialisation')
    binding_path, result_path = stage_dir / 'binding.json', stage_dir / 'result.json'
    checkpoint_path = stage_dir / 'best.pt'
    study_path = stage_dir / study.STUDY_BINDING_FILENAME
    for path in (binding_path, result_path, checkpoint_path):
        if not path.is_file():
            raise ValueError(f'checkpoint directory lacks {path.name}: {stage_dir}')
    if not study_path.is_file():
        raise ValueError(f'checkpoint binding lacks k_selection_sha256 (no '
                         f'{study.STUDY_BINDING_FILENAME} in {stage_dir})')
    binding = study._load_json(binding_path, 'binding.json')
    result = study._load_json(result_path, 'result.json')
    study_binding = study._load_json(study_path, study.STUDY_BINDING_FILENAME)
    if study_binding.get('k_selection_sha256') != freeze_hash:
        raise ValueError('checkpoint k_selection_sha256 differs from the K-freeze record')
    k_selected = int(k_selection['k_selected'])
    stage = study._stage_of_binding(stage_dir, binding, study_binding)
    if stage.arm != 'C':
        raise ValueError(f'offsets are fitted on arm C only (D10.3); refusing arm {stage.arm!r}')
    if stage.k != k_selected or study_binding.get('k_selected') != k_selected:
        raise ValueError(f'checkpoint K {stage.k} differs from the frozen K* {k_selected}; '
                         'only the three winning-K C checkpoints are scored on validation')
    study.check_stage_result(stage_dir, binding, result)
    study.validate_v3_binding(binding, stage, k_selection=k_selection, study_binding=study_binding)
    if study._file_sha256(binding_path) not in k_selection.get('control_binding_sha256', []):
        raise ValueError('C checkpoint binding is not one of the frozen control bindings')
    checkpoint_sha = study._file_sha256(checkpoint_path)
    if checkpoint_sha not in approval_record['checkpoint_sha256']:
        raise ValueError('item-5 approval record does not list this checkpoint SHA-256; '
                         'validation stays closed for it')
    if encoder.fold != VALIDATION_FOLD:
        raise ValueError(f"validation scoring encodes fold='validation' only, got "
                         f'{encoder.fold!r}; the screen and test folds are never scored here')
    ids = [str(sid) for sid in encoder.ids]
    if len(set(ids)) != len(ids):
        raise ValueError('validation ids contain duplicates')
    _check_validation_ids(ids, targets)
    expected_count = binding['counts'].get(VALIDATION_FOLD)
    if len(ids) != expected_count:
        raise ValueError(f'{len(ids)} encoder ids differ from the checkpoint binding validation '
                         f'count {expected_count!r}; the full validation fold is scored once')
    ids_hash = sample_ids_sha256(ids)
    if ids_hash != binding['split_sample_ids_sha256'].get(VALIDATION_FOLD):
        raise ValueError('ordered validation sample ids differ from the checkpoint binding')
    if ids_hash != approval_record['validation_sample_ids_sha256']:
        raise ValueError('item-5 approval record binds another validation sample-id hash')
    out = Path(output_dir) if output_dir is not None else stage_dir / VALIDATION_DIRNAME
    if out.exists():
        raise FileExistsError(f'Refusing occupied validation output {out}')

    factory = model_factory or study._default_model_factory
    model = factory(binding, checkpoint_path)
    model.eval()
    logits_chunks, y_chunks, subjects, sample_ids = [], [], [], []
    with torch.no_grad():
        for batch in DataLoader(list(encoder.rows()), batch_size=int(batch_size), shuffle=False):
            features = model.continuous_inputs(batch)
            logits = model.forward_continuous(features, batch.edge_index, batch)
            logits_chunks.append(logits.detach().cpu().to(torch.float32))
            y_chunks.append(batch.y.view(-1).cpu())
            batch_subjects = batch.subject if isinstance(batch.subject, (list, tuple)) else [batch.subject]
            batch_ids = batch.sample_id if isinstance(batch.sample_id, (list, tuple)) else [batch.sample_id]
            subjects.extend(str(s) for s in batch_subjects)
            sample_ids.extend(str(s) for s in batch_ids)
    if sample_ids != ids:
        raise ValueError('encoded validation rows differ from the bound ids (order or content)')
    logits = torch.cat(logits_chunks).numpy().astype(np.float32, copy=False)
    if logits.shape != (len(ids), NUM_CLASSES):
        raise ValueError(f'validation logits have shape {logits.shape}, expected '
                         f'({len(ids)}, {NUM_CLASSES})')
    if not np.isfinite(logits).all():
        raise ValueError('non-finite validation logits')
    y = torch.cat(y_chunks).numpy()
    subjects_array = np.asarray(subjects).astype(str)
    ids_array = np.asarray(sample_ids).astype(str)
    out.mkdir(parents=True, exist_ok=False)
    logits_path = out / 'logits.npz'
    np.savez_compressed(logits_path, logits=logits, y=y, subjects=subjects_array,
                        sample_ids=ids_array)
    document = {
        'fold': VALIDATION_FOLD, 'stage': stage.name, 'arm': stage.arm, 'seed': int(stage.seed),
        'k': int(stage.k), 'row_count': len(ids),
        'checkpoint_sha256': checkpoint_sha,
        'binding_sha256': study._file_sha256(binding_path),
        'k_selection_sha256': freeze_hash,
        'approval_record_sha256': approval_hash,
        'validation_sample_ids_sha256': ids_hash,
        'logits_path': str(logits_path),
        'logits_sha256': hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest(),
        'metric': METRIC_NAME,
        'tuning_macro_f1': float(weighted_macro_f1(y, logits.argmax(1))),
        'tuning_score_caveat': ('validation macro-F1 of the untuned checkpoint; a tuning '
                                'score, not a result (EXT §4.2, §4.4)'),
        'validation_evaluated': True, 'test_evaluated': False,
    }
    study._write_new_json(out / VALIDATION_RESULT_FILENAME, document)
    return out
