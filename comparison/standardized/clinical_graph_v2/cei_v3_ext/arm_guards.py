"""Arm definitions, analytic parameter accounting and binding guards for the CEI-GNN v3
extension arms E2w, E2d, E6a and E6b (extensions spec §2.3, §5.2, §5.3, §6, §9 X14).

Nothing here trains, preprocesses, scores a fold or opens a data file; the guards work on
binding documents already loaded by the caller.
"""
from __future__ import annotations

# Dimensions of the recorded v2 full-run binding (spec §1): the analytic counts below
# reproduce its 92,300 parameters exactly.
V2_REFERENCE_DIMENSIONS = {'hidden': 0, 'num_relations': 0, 'num_triples': 0}
V2_REFERENCE_PARAMETER_COUNT = 0

# Spec §2.3 / §5.2 / §5.3 v2-schema deltas relative to arm C (before the v3 blocks).
ANALYTIC_V2_SCHEMA_DELTAS = {}

# Arm definitions relative to C (spec §6 table) as `run_config()` fields.
ARM_DEFINITIONS = {}


def analytic_parameter_count(dimensions, *, K=None, universe_size=0, encoder_depth=1,
                             comorbid_block=0) -> int:
    """Closed-form parameter count of `EvidenceNetworkV3` (v2 schema + v3 blocks)."""
    return 0


def analytic_arm_delta(arm, dimensions, *, K=None, universe_size=0) -> int:
    """Parameter delta of an extension arm relative to C at the same control dimensions."""
    return 0


def arm_of_run_config(config) -> str:
    """The arm label ('C', 'E2w', 'E2d', 'E6a', 'E6b') a `run_config()` document realises."""
    return ''


def inventory_diff(control_inventory, arm_inventory) -> dict:
    """Named-tensor diff of an extension inventory against C's (spec §1 / E17)."""
    return {'added': [], 'removed': [], 'widened': [], 'unchanged': []}


def assert_size_arm_support() -> dict:
    """Which registered CEI adapters accept `layers=2` (v3 must, v2 must still refuse)."""
    return {}
