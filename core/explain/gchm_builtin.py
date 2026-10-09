"""Built-in node-level explanation of GCHM-PNA v3, read off the model's own computation.

GCHM-PNA v3's class logit is the sum of three terms (gchm_pna/gchm_v3.py, `forward`):

    logit_c = head_c(pooled graph vector)                 # graph-level, NOT node-additive
            + [ sum_i alpha_ic * <state_i, W_c> + b_c ]   # label-wise readout
            + sum_i ( W_tok[t_i, c] + v_i * W_val[t_i, c] )   # wide per-token votes

The last two terms are exactly additive over nodes, so each node has a true signed
contribution to every class: label-wise (alpha_ic * <state_i, W_c>) plus wide vote.
The built-in importance of node i for the explained class c is |that net contribution|,
the same convention as CEI's native accounting. Nothing here is a new scoring rule.

What this does NOT cover, and says so: the pooled head (sum/mean/hub pool -> MLP) is a
nonlinear function of the whole graph, so its share of the logit cannot be assigned to
nodes. It is reported as `graph_level_head_logit`, next to the class bias, and is
never pushed onto nodes. If the head dominates a prediction, the node-level explanation
covers little of it: compare `node_additive_part` with `graph_level_head_logit`.

The label-wise attention weights alpha_ic (CAML-style, the usual "what did the class
look at") are recorded in the detail, but the importance uses the full signed
contribution, because attention alone ignores whether the attended state pushes the
class up or down.

The label-wise attention is recomputed here from the model's own public layers (four
lines of v3's `labelwise_logits`). That duplication is guarded: the parts must sum back
to the model's real logit within 1e-4, otherwise this raises instead of reporting.

GCHM-PNA v2 has neither a label-wise readout nor a wide path (readouts: 'hub'/'pool'),
so it has no node-additive built-in; the wrapper raises NotImplementedError for it.
"""
from __future__ import annotations

import math

import torch
from torch_geometric.utils import softmax as segment_softmax

from gchm_pna.gchm_v3 import HAS_VALUE_COLUMN, VALUE_COLUMN

ACCOUNTING_TOLERANCE = 1e-4


def explain(model, data, target_class: int | None = None):
    """(importance [N], detail) for one graph. Raises RuntimeError when this model
    configuration has no node-additive path (readout='pool' and wide=False)."""
    if not (model.readout == "labelwise" or model.wide):
        raise RuntimeError(
            "GCHM-PNA v3 was built with readout='pool' and wide=False: its whole logit "
            "comes from the pooled head, which cannot be assigned to nodes")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            h, batch, hub_mask, size = model.node_states(data)
            if size != 1:
                raise ValueError("the built-in explanation scores one graph at a time")
            logits = model(data)
            predicted = int(logits.argmax(-1))
            target = predicted if target_class is None else int(target_class)

            pooled = torch.cat([h.sum(0, keepdim=True), h.mean(0, keepdim=True),
                                model.hub_state(h, batch, hub_mask, size)], dim=-1)
            head = model.head(model.pool_norm(pooled))[0]                       # graph-level
            node_terms = torch.zeros(h.size(0), model.num_classes)
            labelwise_total = torch.zeros(model.num_classes)
            attention = None
            if model.readout == "labelwise":
                states = model.labelwise_norm(h)
                score = states @ model.class_query.t() / math.sqrt(states.size(-1))
                attention = segment_softmax(score, batch, num_nodes=size, dim=0)   # [N, C]
                labelwise = attention * (states @ model.class_weight.t())          # [N, C]
                node_terms += labelwise
                labelwise_total = labelwise.sum(0) + model.class_bias
            wide_total = torch.zeros(model.num_classes)
            if model.wide:
                value = (data.x[:, VALUE_COLUMN] * data.x[:, HAS_VALUE_COLUMN]).unsqueeze(-1)
                wide = model.wide_token(data.token) + value * model.wide_value(data.token)
                node_terms += wide
                wide_total = wide.sum(0)

            accounted = head + labelwise_total + wide_total
            gap = float((accounted - logits[0]).abs().max())
            if gap > ACCOUNTING_TOLERANCE:
                raise RuntimeError(
                    f"the node-additive decomposition does not reproduce the model's logits "
                    f"(max |difference| = {gap:.3g}); refusing to report an explanation that "
                    "may not describe this model")
            contribution = node_terms[:, target]
            additive = float(contribution.sum() + (model.class_bias[target]
                                                   if model.readout == "labelwise" else 0.0))
            logit = float(logits[0, target])
            detail = {
                "kind": "gchm_v3_node_additive_accounting",
                "predicted_class": predicted,
                "explained_class": target,
                "logit": logit,
                "graph_level_head_logit": float(head[target]),
                "class_bias": float(model.class_bias[target]) if model.readout == "labelwise" else 0.0,
                "labelwise_node_total": float(labelwise_total[target]),
                "wide_node_total": float(wide_total[target]),
                "node_additive_part": additive,
                "accounting_logit_abs_diff": gap,
                "signed_node_contributions": [float(v) for v in contribution],
                "labelwise_attention": None if attention is None
                                       else [float(v) for v in attention[:, target]],
                "active_paths": {"labelwise": model.readout == "labelwise", "wide": bool(model.wide)},
                "note": "the pooled head is graph-level and is not assigned to nodes; compare "
                        "graph_level_head_logit with node_additive_part to see how much of the "
                        "logit this node-level explanation can speak for",
            }
            return contribution.abs().detach(), detail
    finally:
        if was_training:
            model.train()
