"""Integrated Gradients with a clinical baseline (GraphXAI improvement plan item 7).

The vendored IG (external/GraphXAI-main/.../integrated_grad.py:142) uses an all-zero
baseline, with the authors' own "TODO: baseline all 0s, all 1s, ...?". For these
node features that is not a patient: a zero row has no node kind, no value and no
"was measured" flag, so the integration path runs through states that never occur.

Baselines here (pre-registered; recorded in every record's provenance):

- ``not_recorded`` (primary). Each node keeps its own kind (what the node *is*) and
  loses everything recorded about it: value, time and availability scaled to 0 with
  their presence flags at 0, context numbers at 0, and the patient's categorical
  context at its explicit ``<missing>`` code. "Value 0 with has_value 0" is the data's
  own encoding of a missing value, so this is a state the encoder was built to see.
  IG then answers: "what did *recording* this node contribute?".
- ``real_patient_refs`` (robustness check, expected-gradients style). Per node, the
  baseline row is drawn from real training nodes of the same token and kind; the
  attributions of several draws are averaged. Costs one IG pass per draw.

Completeness: IG attributions must sum to f(x) - f(baseline) (here f is the explained
class's logit). The step count is chosen from that error, not by default.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from core import NODE_KINDS

PATIENT_KIND_ID = NODE_KINDS.index("patient")
DEFAULT_STEP_CANDIDATES = (8, 16, 32, 64, 128, 256)
DEFAULT_TOLERANCE = 0.05
# |f(x) - f(baseline)| below this is "no change to explain": relative error is
# measured against this floor so a near-zero denominator cannot fake a failure.
DELTA_FLOOR = 1e-3


def not_recorded_baseline(x: torch.Tensor, node_type: torch.Tensor, layout: Sequence[str],
                          *, numeric_width: int | None = None, zeroed_tail: int = 0,
                          kept_tail: int = 0) -> torch.Tensor:
    """The primary baseline: every node keeps its kind, loses what was recorded.

    A model whose explained input is a composite row (CEI-GNN: numeric features, then
    token vectors, then node-type vectors) passes the widths of the two tails: the token
    block is zeroed (nothing recorded -> the pad embedding) and the type block is kept
    (what the node IS). Plain models leave both at 0."""
    numeric_width = len(layout) if numeric_width is None else numeric_width
    if len(layout) != numeric_width or numeric_width + zeroed_tail + kept_tail != x.size(1):
        raise ValueError(f"layout has {len(layout)} names but x has {x.size(1)} columns")
    baseline = torch.zeros_like(x)
    if kept_tail:
        baseline[:, x.size(1) - kept_tail:] = x[:, x.size(1) - kept_tail:]
    patient = node_type.long() == PATIENT_KIND_ID
    for column, name in enumerate(layout):
        if name in NODE_KINDS:
            baseline[:, column] = x[:, column]
        elif name.endswith(":<missing>"):
            baseline[patient, column] = 1.0
    return baseline


class ReferenceBank:
    """Real training node rows, keyed by (token, node kind)."""

    def __init__(self, rows: dict[tuple[int, int], torch.Tensor]):
        self.rows = rows

    @classmethod
    def from_graphs(cls, graphs, *, per_key_cap: int = 100) -> "ReferenceBank":
        collected: dict[tuple[int, int], list[torch.Tensor]] = {}
        for graph in graphs:
            for row, token, kind in zip(graph.x, graph.token.tolist(), graph.node_type.tolist()):
                bucket = collected.setdefault((int(token), int(kind)), [])
                if len(bucket) < per_key_cap:
                    bucket.append(row.detach().clone())
        return cls({key: torch.stack(rows) for key, rows in collected.items()})

    def draw(self, token: torch.Tensor, node_type: torch.Tensor, fallback: torch.Tensor,
             rng: np.random.Generator) -> torch.Tensor:
        """One baseline: a real row for each node whose key has one, else `fallback`."""
        baseline = fallback.clone()
        for i, (t, k) in enumerate(zip(token.tolist(), node_type.tolist())):
            rows = self.rows.get((int(t), int(k)))
            if rows is not None:
                baseline[i] = rows[int(rng.integers(rows.size(0)))]
        return baseline


def integrated_gradients(wrapper, x: torch.Tensor, edge_index: torch.Tensor,
                         baseline: torch.Tensor, target_class: int, steps: int):
    """IG of the target logit w.r.t. x along the straight path from `baseline`.

    Trapezoid rule over steps+1 points; gradients are accumulated as a running sum
    (one copy, not steps+1 -- plan item 9). Returns (attribution [N, F], completeness).
    Tokens and edges stay fixed: only the numeric features are interpolated.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    delta = x - baseline
    total = torch.zeros_like(x)
    for i in range(steps + 1):
        point = (baseline + (i / steps) * delta).detach().requires_grad_(True)
        logit = wrapper(point, edge_index)[0, target_class]
        (grad,) = torch.autograd.grad(logit, point)
        total += grad * (0.5 if i in (0, steps) else 1.0)
    attribution = delta * total / steps
    with torch.no_grad():
        f_x = float(wrapper(x, edge_index)[0, target_class])
        f_base = float(wrapper(baseline, edge_index)[0, target_class])
    change = f_x - f_base
    error = abs(float(attribution.sum()) - change)
    return attribution.detach(), {
        "target_logit_change": change,
        "attribution_sum": float(attribution.sum()),
        "completeness_error_abs": error,
        "completeness_error_rel": error / max(abs(change), DELTA_FLOOR),
    }


def node_importance(attribution: torch.Tensor) -> torch.Tensor:
    """Same reduction the vendored explainers use: sum of |attribution| per node."""
    return attribution.abs().sum(dim=1)


def choose_steps(errors_by_steps: dict[int, Sequence[float]],
                 tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """Smallest candidate whose WORST relative error over the pilot graphs is within
    `tolerance`; if none qualifies, the largest candidate, flagged as not met."""
    worst = {int(s): float(max(errs)) for s, errs in sorted(errors_by_steps.items())}
    for steps, error in worst.items():
        if error <= tolerance:
            return {"steps": steps, "completeness_met": True, "tolerance": tolerance,
                    "worst_relative_error_by_steps": worst}
    largest = max(worst)
    return {"steps": largest, "completeness_met": False, "tolerance": tolerance,
            "worst_relative_error_by_steps": worst}


def expected_gradients(wrapper, x, edge_index, token, node_type, bank: ReferenceBank,
                       fallback: torch.Tensor, target_class: int, steps: int, *,
                       draws: int, seed: int):
    """Average IG attributions over `draws` real-patient baselines."""
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(int(seed))
    total, errors = torch.zeros_like(x), []
    for _ in range(draws):
        baseline = bank.draw(token, node_type, fallback, rng)
        attribution, completeness = integrated_gradients(
            wrapper, x, edge_index, baseline, target_class, steps)
        total += attribution
        errors.append(completeness["completeness_error_rel"])
    return total / draws, {"draws": draws, "mean_relative_error": float(np.mean(errors)),
                           "max_relative_error": float(np.max(errors))}
