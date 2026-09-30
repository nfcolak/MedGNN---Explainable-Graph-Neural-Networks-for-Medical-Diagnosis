"""Item 5 validation scoring (EXT spec §4.2, E3/E7; §9 X3).

The first and only validation scoring of the v3 programme: one inference per frozen arm-C
checkpoint on the full validation fold, opened only by the item-5 approval record. Stores
raw float32 logits (`logits.npz`: logits, y, subjects, sample_ids) and their SHA-256 in a new
path; never re-infers, never touches the test fold.

Stub: behaviour is added under TDD (red: offset fitter).
"""
from __future__ import annotations

from pathlib import Path

VALIDATION_DIRNAME = 'validation'
APPROVAL_FILENAME = 'item5_validation_approval.json'
APPROVAL_KEYS = ('allow_validation', 'approval_reference', 'checkpoint_sha256',
                 'validation_sample_ids_sha256', 'timestamp')


def approval_record_sha256(record) -> str:
    """SHA-256 of the approval record (stub)."""
    return '0' * 64


def validate_approval_record(record) -> None:
    """Refuse anything but a complete item-5 approval record (stub: accepts everything)."""
    return None


def write_approval_record(path, record) -> dict:
    """Write the approval record once (stub: writes nothing)."""
    return {}


def load_approval_record(path):
    """Load the approval record and its file SHA-256 (stub)."""
    return {}, '0' * 64


def validation_encoder(artifact, prep_state, ids, *, approval_record, targets, edge_direction,
                       preprocessing_sha256=None):
    """Lazy validation-fold encoder through U4 `encode_rows` (stub: empty rows)."""
    from ..cei_v3_study import ScreenEncoder
    return ScreenEncoder(ids=tuple(ids), fold='screen', rows=lambda: iter(()))


def score_validation(checkpoint, encoder, approval_record, *, output_dir=None,
                     model_factory=None, batch_size=128, targets=None) -> Path:
    """Score one frozen C checkpoint once on the validation fold (stub: writes nothing)."""
    return Path(checkpoint.stage_dir) / VALIDATION_DIRNAME
