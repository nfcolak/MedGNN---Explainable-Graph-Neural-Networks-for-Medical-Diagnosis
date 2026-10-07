"""Explanation runner for clinical_graph_v2 (GraphXAI improvement plan, items
#1 and #4).

Reloads a real trained run (`binding.json` + `preprocessing.json` + `best.pt`,
written by `train.py`), reconstructs the exact model and the exact validation
split, REFUSES to proceed unless a replay of validation reproduces the
recorded prediction hash exactly, then runs the three real GraphXAI explainers
(via core/explain/graphxai_standardized.explain_algorithms, the same entry point
the legacy star/cooccur/ontology/full pipeline uses) plus each model's
built-in explanation where one exists, for every validation-fold subject (or a
capped prefix of them). For every importance source that succeeded, also
computes `fidelity_v2.py`'s full-evidence-removal fidelity+/- and deletion/
insertion AUC curves (item #4) -- a stricter, separate metric from the
`fidelity_plus`/`fidelity_minus` already nested in each `graphxai` entry
(those come from `core/explain/fidelity.py`, v1, x-row-only removal; do not
compare the two).

Cohort contract: `explanation_contract_v2.py` (COHORT_SCHEMA_VERSION = 2), NOT
explanation_contract.py's v1. v1 is locked to the canonical star-graph split,
TEST_FOLD=2 and exactly 500 subjects, which conflicts with ADR-002. v2 explains
the validation fold (or dev split) of ONE finished run, hash-pinned to that
run's binding.json, and is written next to the records as cohort.json. v1 is
untouched and is not imported here.

Method coverage: gsat, protgnn, graphcare (the ClinicalMethodAdapter methods that
accept `external_edge_mask`), GCHMv2/v3, and the black-box baselines (clinical_gnn
with --conv edge_conditioned | hgt | gchm, explained post hoc only). Any other registered method (CEI-GNN
plugins: their forward has no `external_edge_mask`, and CEI has its own GraphXAI
studies under cei/studies/) is refused with a clear message.

Output governance (plan item #10): every run writes to a new, empty, caller-
chosen directory with its own manifest (checkpoint/preprocessing/split hashes,
explainer config); an occupied directory is refused, never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from core import train
from core.contracts import recursive_source_hashes, sample_ids_sha256
from core.explain.explanation_contract_v2 import build_cohort_v2, validate_cohort_v2
from gchm_pna.gchm_v2 import GCHMv2
from gchm_pna.gchm_v3 import GCHMv3
from core.explain.graphxai_wrapper import (
    ClinicalGNNGraphXAIWrapper, ClinicalGraphXAIWrapper, GraphCareGraphXAIWrapper,
)
from core.model import ClinicalGNN
from core.registry import build_method
from core.schema import sha256
from core.tensorize import PAYLOAD_WIDTH, triple_count

GRAPHXAI_METHODS = frozenset({"gsat", "protgnn", "graphcare"})
RUNNER_SCHEMA = "medgnn.clinical_graph_v2_explanation_run"
RUNNER_SCHEMA_VERSION = 1
# Black-box baselines: ClinicalGNN over genuine MessagePassing layers, explained
# post hoc with GNNExplainer through PyG's own mask hook (no model change).
BLACK_BOX_CONVS = ("edge_conditioned", "hgt", "gchm")


def proba_digest(proba) -> str:
    """Identical hashing scheme to train.py's own (private) proba_digest, so a
    replayed prediction array hashes to the exact value binding.json recorded."""
    return hashlib.sha256(np.ascontiguousarray(proba).tobytes()).hexdigest()


@dataclass(frozen=True)
class RunBundle:
    run_dir: Path
    binding: dict[str, Any]
    state_dict: dict[str, torch.Tensor]


def load_run_bundle(run_dir) -> RunBundle:
    """Load and hash-verify binding.json/preprocessing.json/best.pt. Refuses
    to proceed on any mismatch -- this is the gate the plan's 'contract
    conflict' and 'replay check' items both exist to put in front of
    explaining a model that might not be the one actually trained."""
    run_dir = Path(run_dir)
    binding_path, prep_path, ckpt_path = (
        run_dir / "binding.json", run_dir / "preprocessing.json", run_dir / "best.pt"
    )
    for path, label in ((binding_path, "binding.json"), (prep_path, "preprocessing.json"),
                        (ckpt_path, "best.pt")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} missing from run directory: {run_dir}")
    binding = json.loads(binding_path.read_text())
    if sha256(prep_path) != binding.get("preprocessing_sha256"):
        raise ValueError(
            "preprocessing.json hash does not match binding.json's recorded "
            "preprocessing_sha256 -- this run directory is not internally "
            "consistent; refusing to explain a possibly-mismatched model/data pair."
        )
    if binding.get("selected_validation") is None:
        raise ValueError(
            "binding.json has no selected_validation block (this run never "
            "evaluated validation -- e.g. a --selection-fold dev run with "
            "--final-eval none); nothing to replay or explain against."
        )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    return RunBundle(run_dir=run_dir, binding=binding, state_dict=state_dict)


def rebuild_validation_split(binding: dict[str, Any]):
    """Reconstruct train/validation(/dev) splits exactly as train.py built
    them, from binding.json's own recorded (already-resolved) arguments --
    not re-derived from a CLI invocation, so there is nothing left to guess."""
    targets = train.load_targets(binding["targets_path"])
    if binding.get("top_k_labels") is not None:
        targets, _, _ = train.select_top_labels(targets, binding["top_k_labels"])
    splits, _prep = train.build_dataset(
        binding["artifact"], targets, binding["edges"], binding["train_limit"],
        binding["token_min_count"], binding["seed"],
        tuple(binding.get("dropped_relations") or ()),
        tuple(binding.get("rewired_relations") or ()),
        binding.get("min_prior_visits", 0),
        edge_direction=binding["edge_direction"],
        dev_limit=binding.get("dev_limit"),
        sample_seed=binding.get("sample_seed"),
    )
    if "validation" not in splits:
        raise ValueError("rebuilt splits have no validation fold")
    recorded = binding.get("split_sample_ids_sha256", {}).get("validation")
    actual = sample_ids_sha256(d.sample_id for d in splits["validation"])
    if recorded != actual:
        raise ValueError(
            "rebuilt validation split's sample-ID hash does not match "
            "binding.json's split_sample_ids_sha256['validation'] -- the "
            "artifact, targets file, or build arguments have drifted since "
            "training; refusing to explain against a different split than "
            "the model was actually evaluated on."
        )
    return splits


def _infer_token_dim(state_dict: dict[str, torch.Tensor], *, is_method_adapter: bool) -> int:
    key = "token_embedding.weight" if is_method_adapter else "token.weight"
    if key not in state_dict:
        raise ValueError(f"checkpoint has no {key!r} tensor to infer token_dim from")
    return int(state_dict[key].shape[1])


def build_model(binding: dict[str, Any], state_dict: dict[str, torch.Tensor]):
    """Reconstruct the exact architecture binding.json/best.pt describe.

    Only tensor-shape-determining hyperparameters matter for a correct
    eval-mode forward pass (dropout is a no-op in eval; most 'native' settings
    like GSAT's temperature or ProtGNN's cluster/separation weights are
    training-objective-only). The one exception found and handled explicitly:
    GraphCare's `recency_decay` is a plain float baked directly into its
    forward computation, not a saved parameter -- recovered from
    method_config['effective_settings'] like every other native setting, not
    assumed safe to default. GCHM's degree_histogram affects the PNA scaler
    identically; recovered from the checkpoint's own saved buffer, not
    re-derived, since binding.json never persists the raw histogram.
    """
    method = binding["method"]
    conv = binding.get("conv")
    node_dim, edge_dim = binding["node_dim"], binding["edge_dim"]
    num_classes = binding["num_classes"]
    num_tokens = binding["vocabulary_size"]
    num_relations = binding["num_relations"]
    num_triples = binding["num_meta_relations"]
    hidden, layers = binding["hidden"], binding["layers"]

    if method == "clinical_gnn" and conv in BLACK_BOX_CONVS:
        degree_histogram = state_dict.get("degree_histogram")
        if conv == "gchm" and degree_histogram is None:
            raise ValueError("checkpoint has no saved degree_histogram buffer")
        model = ClinicalGNN(
            num_tokens=num_tokens, node_dim=node_dim, edge_dim=edge_dim,
            num_classes=num_classes, hidden=hidden, layers=layers, dropout=0.0,
            token_dim=_infer_token_dim(state_dict, is_method_adapter=False),
            use_edge_payload=binding.get("edge_payload", True),
            use_message_passing=binding.get("message_passing", True),
            conv=conv, num_triples=num_triples, payload_dim=PAYLOAD_WIDTH,
            heads=binding.get("heads") or 4, degree_histogram=degree_histogram,
            modulation=binding.get("modulation") or "multiplicative",
        )
        model.load_state_dict(state_dict)
        return model, False

    if method == "clinical_gnn" and conv in ("gchm_v2", "gchm_v3"):
        # Allowlisted, not blocklisted: `settings` also carries derived/
        # already-explicit fields (avg_deg_log, edge_payload, message_passing,
        # hidden/layers/dropout/num_relations/token_dim) that either aren't
        # real constructor kwargs or are already passed explicitly above --
        # an allowlist fails loudly on a future settings-dict change instead
        # of silently passing something stale through.
        settings = dict(binding["method_config"]["effective_settings"])
        token_dim = int(settings.pop("token_dim"))
        degree_histogram = state_dict.get("degree_histogram")
        if degree_histogram is None:
            raise ValueError("checkpoint has no saved degree_histogram buffer")
        extra_keys = (
            ("modulation", "aggregation", "readout", "hub_gate") if conv == "gchm_v2"
            else ("readout", "wide", "jk", "edge_dropout")
        )
        cls = GCHMv2 if conv == "gchm_v2" else GCHMv3
        model = cls(
            num_tokens=num_tokens, node_dim=node_dim, edge_dim=edge_dim,
            num_classes=num_classes, num_relations=num_relations,
            degree_histogram=degree_histogram, hidden=hidden, layers=layers,
            dropout=0.0, token_dim=token_dim,
            use_edge_payload=binding.get("edge_payload", True),
            use_message_passing=binding.get("message_passing", True),
            **{k: settings[k] for k in extra_keys if k in settings},
        )
        model.load_state_dict(state_dict)
        return model, False

    if method not in GRAPHXAI_METHODS:
        raise NotImplementedError(
            f"method {method!r} is not supported by this runner (supported: "
            f"{sorted(GRAPHXAI_METHODS)} and clinical_gnn gchm_v2/gchm_v3); its adapter "
            "does not accept `external_edge_mask`, which GNNExplainer requires")
    architecture = binding["method_config"].get("architecture", {})
    token_dim = architecture.get("token_dim")
    if token_dim is None:
        token_dim = _infer_token_dim(state_dict, is_method_adapter=True)
    effective_settings = dict(binding["method_config"].get("effective_settings", {}))
    model = build_method(
        method, num_tokens=num_tokens, node_dim=node_dim, edge_dim=edge_dim,
        num_classes=num_classes, hidden=hidden, layers=layers, dropout=0.0,
        token_dim=int(token_dim), num_triples=num_triples,
        args=Namespace(edge_direction=binding["edge_direction"], **effective_settings),
    )
    model.load_state_dict(state_dict)
    return model, True


@torch.no_grad()
def replay_check(model, splits, binding: dict[str, Any], *, is_method_adapter: bool,
                 device="cpu", batch_size: int | None = None) -> None:
    """Refuse to proceed unless a fresh validation pass reproduces the exact
    recorded prediction hash. This is the one check that makes every other
    reconstruction assumption above verifiable rather than merely plausible:
    if anything upstream was wrong, this fails loudly, closed, before a single
    explanation is generated.

    batch_size defaults to binding["batch_size"] (the training run's own
    value), not an arbitrary one -- confirmed necessary, not just tidy: at
    least one model's normalisation layers (GSAT's InstanceNorm, found by
    this exact check failing on a 32-subject validation fold batched
    differently than training did) are sensitive to how graphs are grouped
    into batches, not only to each graph's own content. A different batch
    size can validly change the numeric answer even with byte-identical
    weights, so replaying training's own batching exactly is the only way
    this check means what it claims to mean.
    """
    if batch_size is None:
        batch_size = binding.get("batch_size", 64)
    model = model.to(device).eval()
    loader = DataLoader(splits["validation"], batch_size=batch_size)
    logits = []
    for batch in loader:
        batch = batch.to(device)
        if is_method_adapter:
            logits.append(model(batch, epoch=0).logits.cpu())
        else:
            logits.append(model(batch).cpu())
    proba = torch.softmax(torch.cat(logits), dim=1).numpy()
    digest = proba_digest(proba)
    expected = binding["selected_validation"]["prediction_sha256"]
    if digest != expected:
        raise ValueError(
            "REPLAY CHECK FAILED: reloading best.pt and re-running validation "
            f"produced prediction hash {digest}, not the recorded {expected}. "
            "Refusing to generate explanations from a model that cannot be "
            "shown to be the one that was actually trained and evaluated."
        )


def build_wrapper(binding: dict[str, Any], model):
    """The same wrapper classes built and verified earlier this project,
    dispatched by method/conv -- no new wrapper logic here."""
    method = binding["method"]
    conv = binding.get("conv")
    kwargs = dict(
        node_dim=binding["node_dim"], edge_dim=binding["edge_dim"],
        num_tokens=binding["vocabulary_size"], num_triples=binding["num_meta_relations"],
        num_relations=binding["num_relations"],
    )
    if method == "graphcare":
        return GraphCareGraphXAIWrapper(model, **kwargs)
    if method == "clinical_gnn" and conv in BLACK_BOX_CONVS:
        return ClinicalGNNGraphXAIWrapper(model, method=conv, **kwargs)
    if method == "clinical_gnn":
        return ClinicalGraphXAIWrapper(model, method=conv, is_method_adapter=False, **kwargs)
    return ClinicalGraphXAIWrapper(model, method=method, is_method_adapter=True, **kwargs)


def _single_graph_batch(data):
    """One PyG Data -> the (x, edge_index, batch_index) a wrapper's
    set_context/forward expect for a single-graph explanation."""
    from torch_geometric.data import Batch
    return Batch.from_data_list([data])


def _fidelity_v2_for_source(wrapper, x, batch, importance, target_class, *,
                            curve_points: int) -> dict[str, Any]:
    """item #4's metrics for one importance vector: a single fidelity+/-
    point at the shared top-20%-of-removable-nodes policy, plus the
    deletion/insertion AUC curves the plan's [new] bullet asks for instead of
    trusting one threshold."""
    from core.explain import fidelity_v2 as fv2

    importance = np.asarray(importance, dtype=float)
    plus = fv2.fidelity_plus_v2(
        wrapper, x, batch.token, batch.edge_index, batch.node_type, importance, target_class
    )
    minus = fv2.fidelity_minus_v2(
        wrapper, x, batch.token, batch.edge_index, batch.node_type, importance, target_class
    )
    deletion = fv2.deletion_curve(
        wrapper, x, batch.token, batch.edge_index, batch.node_type, importance,
        target_class, num_points=curve_points,
    )
    insertion = fv2.insertion_curve(
        wrapper, x, batch.token, batch.edge_index, batch.node_type, importance,
        target_class, num_points=curve_points,
    )
    return {
        "fidelity_plus": plus, "fidelity_minus": minus,
        "deletion_curve": deletion, "insertion_curve": insertion,
    }


def explain_subject(wrapper, data, *, method: str, conv: str | None,
                    steps: int, epochs: int, num_relation_columns: int,
                    fidelity_v2_curve_points: int = 3) -> dict[str, Any]:
    from core.explain.graphxai_standardized import explain_algorithms
    from core.explain import fidelity_v2 as fv2
    from core.tensorize import reversible_edge_pairs

    batch = _single_graph_batch(data)
    x = wrapper.set_context(batch)
    # Plan item 2: passing only x and edge_index is explaining a different model
    # unless the wrapper provably reproduces the real model on this very graph.
    faithful, wrapper_max_diff = wrapper.verify(x, batch.edge_index)
    if not faithful:
        raise ValueError(
            f"wrapper does not reproduce the model on sample {data.sample_id}: "
            f"max |logit difference| = {wrapper_max_diff:.3g}"
        )
    batch_index = torch.zeros(x.size(0), dtype=torch.long)
    graphxai = explain_algorithms(
        wrapper, x, batch.edge_index, batch=batch_index, steps=steps, epochs=epochs,
        reversible_edge_pairs=reversible_edge_pairs(batch.edge_relation),
        node_reduction='mean',
    )
    gnn = graphxai.get("GNNExplainer", {})
    if gnn.get("status") == "failed" and str(gnn.get("error", "")).startswith(
            "GNNExplainer edge mask gradient disconnected"):
        # d logit / d edge_mask is None: the mask never reaches the output at all
        # (e.g. a model with message passing switched off). Same verdict the plan
        # asks for as an all-zero derivative: unsupported, not a failed run.
        gnn["status"] = "unsupported"
        gnn["unsupported_reason"] = (
            "d logit / d edge_mask is None: the mask does not reach this model's "
            "output, so GNNExplainer cannot explain it")
    for payload in graphxai.values():
        if payload.get("status") == "success" and payload["provenance"].get(
                "zero_predictive_edge_gradient"):
            payload["status"] = "unsupported"
            payload["unsupported_reason"] = (
                "d logit / d edge_mask is exactly zero (or the graph has no edges): "
                "the mask does not influence this prediction, so this is not an "
                "explanation of it and is not reported as a result")
            payload.pop("node_explanation", None)
    builtin_importance, builtin_unavailable = None, None
    try:
        builtin_importance = [
            float(v) for v in wrapper.builtin_node_importance(x, batch.edge_index)
        ]
    except NotImplementedError:
        pass
    except RuntimeError as exc:
        # The model itself produced no explanation for this graph (e.g. ProtGNN
        # found no connected subgraph). Recorded, never turned into a fake one.
        builtin_unavailable = str(exc)
    builtin_detail = (wrapper.builtin_detail()
                      if builtin_importance is not None and hasattr(wrapper, "builtin_detail")
                      else None)
    with torch.no_grad():
        prediction = int(wrapper(x, batch.edge_index).argmax(-1).item())

    # item #4: full-evidence-removal fidelity for every importance source
    # that actually produced one -- the built-in explanation (if any) and
    # each GraphXAI algorithm that succeeded. A source's own failure (e.g. a
    # 'failed' GraphXAI status) must not block fidelity for the others,
    # exactly the same isolation principle as explain_algorithms() itself.
    fidelity_v2: dict[str, Any] = {}
    sources = dict(graphxai)
    if builtin_importance is not None:
        sources["builtin"] = {"status": "success",
                              "node_explanation": {"node_importance": builtin_importance}}
    for name, payload in sources.items():
        if payload.get("status") != "success":
            continue
        try:
            fidelity_v2[name] = _fidelity_v2_for_source(
                wrapper, x, batch, payload["node_explanation"]["node_importance"],
                prediction, curve_points=fidelity_v2_curve_points,
            )
        except Exception as exc:  # noqa: BLE001 - one source's failure must not lose the rest
            fidelity_v2[name] = {"status": "failed", "error": str(exc)}

    # GraphXAI improvement plan item #5, second half: relation-type vs
    # numeric-payload contribution to this graph's own prediction (model-level,
    # not tied to any one explainer -- a source's success/failure above is
    # irrelevant here).
    try:
        if conv == "hgt":
            edge_attr_contribution = {
                "status": "not_applicable",
                "reason": "hgt reads relation identity from edge_triple, not from edge_attr's "
                          "relation one-hot, so zeroing that block would under-report it"}
        else:
            edge_attr_contribution = fv2.relation_vs_payload_contribution(
                wrapper, x, batch.edge_index, batch.edge_attr, num_relation_columns, prediction,
            )
    except Exception as exc:  # noqa: BLE001 - must not block the rest of the record
        edge_attr_contribution = {"status": "failed", "error": str(exc)}

    return {
        "schema": "medgnn.clinical_graph_v2_explanation_record",
        "schema_version": 1,
        "method": method,
        "conv": conv,
        "sample_id": str(data.sample_id),
        "subject_id": str(data.subject),
        "true_class_id": int(data.y.view(-1)[0].item()),
        "prediction_class_id": prediction,
        "wrapper_verification": {"max_abs_logit_diff": wrapper_max_diff, "tolerance": 1e-5},
        "builtin_node_importance": builtin_importance,
        "builtin_available": builtin_importance is not None,
        "builtin_unavailable_reason": builtin_unavailable,
        "builtin_detail": builtin_detail,
        "graphxai": graphxai,
        "fidelity_v2": fidelity_v2,
        "fidelity_v2_intervention": {"version": fv2.INTERVENTION_CONTRACT_VERSION,
                                      "definition": fv2.INTERVENTION_CONTRACT},
        "edge_attr_contribution": edge_attr_contribution,
    }


def run_explanations(run_dir, output_dir, *, max_subjects: int | None = None,
                     steps: int = 32, epochs: int = 50,
                     fidelity_v2_curve_points: int = 3) -> dict[str, Any]:
    """The full pipeline: load -> rebuild split -> replay-check -> explain
    every (or the first max_subjects) validation subject -> write output.
    Raises before writing anything if the replay check fails."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"explanation output directory must be empty: {output_dir}")

    bundle = load_run_bundle(run_dir)
    binding = bundle.binding
    splits = rebuild_validation_split(binding)
    model, is_method_adapter = build_model(binding, bundle.state_dict)
    replay_check(model, splits, binding, is_method_adapter=is_method_adapter)

    subjects = splits["validation"]
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    cohort = build_cohort_v2(
        binding, fold="validation",
        fold_sample_ids=[d.sample_id for d in splits["validation"]],
        explained_sample_ids=[d.sample_id for d in subjects],
    )
    validate_cohort_v2(cohort, binding)

    wrapper = build_wrapper(binding, model)
    method, conv = binding["method"], binding.get("conv")
    records, failures = [], []
    for data in subjects:
        try:
            records.append(
                explain_subject(wrapper, data, method=method, conv=conv,
                                steps=steps, epochs=epochs,
                                num_relation_columns=binding["num_relations"],
                                fidelity_v2_curve_points=fidelity_v2_curve_points)
            )
        except Exception as exc:  # noqa: BLE001 - one subject's failure must not lose the rest
            failures.append({"sample_id": str(data.sample_id), "error": str(exc)})

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cohort.json").write_text(json.dumps(cohort, indent=2) + "\n")
    record_files = []
    for record in records:
        filename = f"subject_{record['sample_id']}.json"
        (output_dir / filename).write_text(json.dumps(record, indent=2) + "\n")
        record_files.append(filename)
    manifest = {
        "schema": RUNNER_SCHEMA,
        "schema_version": RUNNER_SCHEMA_VERSION,
        "method": method,
        "conv": conv,
        "run_dir": str(Path(run_dir).resolve()),
        "binding_preprocessing_sha256": binding["preprocessing_sha256"],
        "validation_sample_ids_sha256": binding["split_sample_ids_sha256"]["validation"],
        "replay_verified": True,
        "cohort_schema_version": cohort["schema_version"],
        "cohort_file": "cohort.json",
        "cohort_is_full_fold": cohort["is_full_fold"],
        "requested_subject_count": len(subjects),
        "explained_count": len(records),
        "failed_count": len(failures),
        "failures": failures,
        "steps": steps,
        "epochs": epochs,
        "record_files": record_files,
        "source_code": recursive_source_hashes(Path(__file__).parent),
    }
    (output_dir / "explanation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fidelity-v2-curve-points", type=int, default=3)
    args = parser.parse_args()
    manifest = run_explanations(
        args.run_dir, args.output, max_subjects=args.max_subjects,
        steps=args.steps, epochs=args.epochs,
        fidelity_v2_curve_points=args.fidelity_v2_curve_points,
    )
    print(json.dumps({k: manifest[k] for k in
                      ("method", "conv", "explained_count", "failed_count")}))


if __name__ == "__main__":
    _cli()
