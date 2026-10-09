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
from core.explain import audit as audit_mod
from core.explain import fidelity_v2 as fv2
from core.explain import ig_clinical
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

GRAPHXAI_METHODS = frozenset({"gsat", "protgnn", "graphcare", "cei_gnn_v3"})
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
    preprocessing: dict[str, Any] | None = None


FOLD_BLOCK = {"validation": "selected_validation", "dev": "selected_dev"}


def load_run_bundle(run_dir, fold: str = "validation") -> RunBundle:
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
    if fold not in FOLD_BLOCK:
        raise ValueError(f"fold must be one of {sorted(FOLD_BLOCK)}, not {fold!r}")
    if binding.get(FOLD_BLOCK[fold]) is None:
        raise ValueError(
            f"binding.json has no {FOLD_BLOCK[fold]} block (this run never evaluated the "
            f"{fold} fold -- e.g. a --selection-fold dev run with --final-eval none keeps "
            "validation closed; try fold='dev'); nothing to replay or explain against."
        )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    return RunBundle(run_dir=run_dir, binding=binding, state_dict=state_dict,
                     preprocessing=json.loads(prep_path.read_text()))


def rebuild_validation_split(binding: dict[str, Any], fold: str = "validation"):
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
    if fold not in splits:
        raise ValueError(f"rebuilt splits have no {fold} fold")
    recorded = binding.get("split_sample_ids_sha256", {}).get(fold)
    actual = sample_ids_sha256(d.sample_id for d in splits[fold])
    if recorded != actual:
        raise ValueError(
            f"rebuilt {fold} split's sample-ID hash does not match "
            f"binding.json's split_sample_ids_sha256[{fold!r}] -- the "
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


def _build_cei(binding: dict[str, Any], state_dict: dict[str, torch.Tensor]):
    """CEI-GNN v3 through the shared registry from its own recorded options (arm, k,
    v3_state file, ...), then proven to be the recorded configuration: the rebuilt
    adapter's run_config() must equal binding.json's method_config exactly."""
    config = binding["method_config"]
    arch = config["architecture"]
    model = build_method(
        "cei_gnn_v3", num_tokens=binding["vocabulary_size"], node_dim=binding["node_dim"],
        edge_dim=binding["edge_dim"], num_classes=binding["num_classes"],
        hidden=binding["hidden"], layers=binding["layers"], dropout=0.0,
        token_dim=int(arch["token_dim"]), num_triples=binding["num_meta_relations"],
        args=Namespace(edge_direction=binding["edge_direction"],
                       method_options=dict(config["effective_settings"]),
                       seed=binding.get("seed")),
    )
    model.load_state_dict(state_dict)
    if model.run_config() != config:
        raise ValueError(
            "rebuilt CEI-GNN's configuration differs from binding.json's method_config "
            "(check that the recorded v3_state file is the one on disk); refusing to "
            "explain a model that cannot be shown to be the one that was trained")
    return model


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
    if method == "cei_gnn_v3":
        return _build_cei(binding, state_dict), True
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
                 device="cpu", batch_size: int | None = None, fold: str = "validation") -> None:
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
    loader = DataLoader(splits[fold], batch_size=batch_size)
    logits = []
    for batch in loader:
        batch = batch.to(device)
        if is_method_adapter:
            logits.append(model(batch, epoch=0).logits.cpu())
        else:
            logits.append(model(batch).cpu())
    proba = torch.softmax(torch.cat(logits), dim=1).numpy()
    digest = proba_digest(proba)
    expected = binding[FOLD_BLOCK[fold]]["prediction_sha256"]
    if digest != expected:
        raise ValueError(
            f"REPLAY CHECK FAILED: reloading best.pt and re-running {fold} "
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
    if method == "cei_gnn_v3":
        from core.explain.cei_bridge import CEIGraphXAIWrapper

        return CEIGraphXAIWrapper(model)
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


def _stable_seed(base: int, sample_id: str) -> int:
    """Deterministic per-subject seed, independent of Python's salted hash()."""
    digest = hashlib.sha256(f"{int(base)}:{sample_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def select_explained_subjects(validation, *, max_subjects: int | None = None,
                              sample_patients: int | None = None, sample_seed: int = 1234):
    """The visits to explain. `sample_patients=N` draws N PATIENTS (hash-ordered by
    seed, so the same patients are drawn for every method) and keeps every one of
    their validation visits in training order; the headline metrics are
    patient-equal, so the sampling unit is the patient. `max_subjects` then caps the
    list (a prefix), mostly for smoke runs."""
    subjects = list(validation)
    if sample_patients is not None:
        patients = sorted({str(d.subject) for d in subjects},
                          key=lambda s: hashlib.sha256(f"{int(sample_seed)}:{s}".encode()).hexdigest())
        chosen = set(patients[:sample_patients])
        subjects = [d for d in subjects if str(d.subject) in chosen]
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    return subjects


def _fidelity_v2_for_source(wrapper, x, batch, importance, target_class, *,
                            curve_points: int, reference=None) -> dict[str, Any]:
    """The v2 metrics for one importance vector: fidelity+/- at the shared top-20%
    of eligible nodes (capped at the explanation's own support), deletion/insertion
    curves, sparsity on the same eligible nodes, and -- when a random reference was
    computed for this graph -- how far each number sits from random removal."""
    importance = np.asarray(importance, dtype=float)
    args = (wrapper, x, batch.token, batch.edge_index, batch.node_type, importance, target_class)
    plus = fv2.fidelity_plus_v2(*args)
    minus = fv2.fidelity_minus_v2(*args)
    deletion = fv2.deletion_curve(*args, num_points=curve_points)
    insertion = fv2.insertion_curve(*args, num_points=curve_points)
    result = {
        "fidelity_plus": plus, "fidelity_minus": minus,
        "deletion_curve": deletion, "insertion_curve": insertion,
        "sparsity": fv2.sparsity_over_eligible(batch.node_type, importance),
    }
    if reference is not None:
        result["vs_random"] = reference.compare(
            fidelity_plus=plus, fidelity_minus=minus,
            deletion_curve=deletion, insertion_curve=insertion)
    return result


def _normalise_statuses(graphxai: dict[str, Any]) -> None:
    """Map the plan's 'mask does not reach the output' verdicts onto 'unsupported'."""
    gnn = graphxai.get("GNNExplainer", {})
    if gnn.get("status") == "failed" and str(gnn.get("error", "")).startswith(
            "GNNExplainer edge mask gradient disconnected"):
        # d logit / d edge_mask is None: the mask never reaches the output at all
        # (e.g. a model with message passing switched off).
        gnn["status"] = "unsupported"
        gnn["unsupported_reason"] = (
            "d logit / d edge_mask is None: the mask does not reach this model's "
            "output, so GNNExplainer cannot explain it")
    for payload in graphxai.values():
        if payload.get("status") == "success" and payload.get("provenance", {}).get(
                "zero_predictive_edge_gradient"):
            payload["status"] = "unsupported"
            payload["unsupported_reason"] = (
                "d logit / d edge_mask is exactly zero (or the graph has no edges): "
                "the mask does not influence this prediction, so this is not an "
                "explanation of it and is not reported as a result")
            payload.pop("node_explanation", None)


def _clinical_ig_payloads(wrapper, x, batch, batch_index, target_class: int, ig: dict[str, Any]):
    """IG from the pre-registered clinical baseline (and, if requested, the real-patient
    reference variant), each with its completeness error recorded."""
    from core.explain.explanation_contract import build_node_explanation

    edge_index = batch.edge_index
    baseline = ig_clinical.not_recorded_baseline(x, batch.node_type, ig["layout"],
                                                 **ig.get("feature_spec", {}))
    attribution, completeness = ig_clinical.integrated_gradients(
        wrapper, x, edge_index, baseline, target_class, ig["steps"])
    importance = ig_clinical.node_importance(attribution).numpy()

    def payload(importance, provenance, standard):
        if not np.any(importance):
            return {"status": "unsupported", "provenance": provenance,
                    "unsupported_reason": "every attribution is exactly zero"}
        node_explanation = (build_node_explanation(wrapper, x, edge_index, batch=batch_index,
                                                   node_importance=importance)
                            if standard else {"node_importance": [float(v) for v in importance]})
        return {"status": "success", "provenance": provenance, "node_explanation": node_explanation}

    standard = ig["explain_predicted"]
    out = {"main": payload(importance, {
        "implementation": "core.explain.ig_clinical", "baseline": "not_recorded",
        "steps": ig["steps"], "step_selection": ig["step_selection"],
        "tolerance": ig["tolerance"],
        "completeness_met": completeness["completeness_error_rel"] <= ig["tolerance"],
        **completeness}, standard)}
    if ig.get("bank") is not None and ig["reference_draws"] > 0:
        averaged, summary = ig_clinical.expected_gradients(
            wrapper, x, edge_index, batch.token, batch.node_type, ig["bank"], baseline,
            target_class, ig["steps"], draws=ig["reference_draws"], seed=ig["seed"])
        out["refs"] = payload(ig_clinical.node_importance(averaged).numpy(), {
            "implementation": "core.explain.ig_clinical", "baseline": "real_patient_refs",
            "steps": ig["steps"], **summary}, standard)
    return out


def _collect_importances(wrapper, batch, x, *, steps: int, epochs: int, ig, prediction: int,
                         target_class: int | None = None) -> dict[str, Any]:
    """Every explanation source for one graph: the three GraphXAI explainers (IG
    replaced by the clinical-baseline version when `ig` is given) and the model's own
    built-in explanation where one exists."""
    from core.explain.graphxai_standardized import explain_algorithms
    from core.tensorize import reversible_edge_pairs

    batch_index = torch.zeros(x.size(0), dtype=torch.long)
    graphxai = explain_algorithms(
        wrapper, x, batch.edge_index, batch=batch_index, steps=steps, epochs=epochs,
        reversible_edge_pairs=reversible_edge_pairs(batch.edge_relation),
        node_reduction="mean", target_class=target_class,
    )
    _normalise_statuses(graphxai)
    target = prediction if target_class is None else int(target_class)
    if ig is not None:
        ig_run = dict(ig, explain_predicted=target_class is None)
        payloads = _clinical_ig_payloads(wrapper, x, batch, batch_index, target, ig_run)
        vendored = graphxai.get("IntegratedGradExplainer")
        if ig.get("keep_zero_baseline") and vendored is not None:
            graphxai["IntegratedGradExplainerZeroBaseline"] = vendored
        graphxai["IntegratedGradExplainer"] = payloads["main"]
        if "refs" in payloads:
            graphxai["IntegratedGradExplainerRealPatientRefs"] = payloads["refs"]

    builtin, unavailable, detail = None, None, None
    try:
        builtin = [float(v) for v in wrapper.builtin_node_importance(
            x, batch.edge_index, target_class=target_class)]
    except NotImplementedError:
        pass
    except RuntimeError as exc:
        # The model itself produced no explanation for this graph (e.g. ProtGNN
        # found no connected subgraph). Recorded, never turned into a fake one.
        unavailable = str(exc)
    if builtin is not None and hasattr(wrapper, "builtin_detail") and target_class is None:
        detail = wrapper.builtin_detail()
    return {"graphxai": graphxai, "builtin": builtin, "builtin_unavailable": unavailable,
            "builtin_detail": detail}


def _source_vectors(collected: dict[str, Any]) -> dict[str, np.ndarray]:
    vectors = {name: np.asarray(p["node_explanation"]["node_importance"], dtype=float)
               for name, p in collected["graphxai"].items()
               if p.get("status") == "success" and "node_explanation" in p}
    if collected["builtin"] is not None:
        vectors["builtin"] = np.asarray(collected["builtin"], dtype=float)
    return vectors


def _evaluate_sources(wrapper, x, batch, vectors: dict[str, np.ndarray], target: int, *,
                      curve_points: int, random_repeats: int, random_seed: int):
    """v2 metrics for every source, all calibrated against ONE random-removal
    reference per graph (same code path, so any out-of-distribution effect of removing
    evidence is present on both sides of the comparison)."""
    removable_count = int(fv2.removable_node_mask(batch.node_type).sum())
    ks: set[int] = set()
    for vector in vectors.values():
        support = fv2.eligible_support(batch.node_type, vector)
        if support:
            ks.add(min(fv2.resolve_k(removable_count), support))
            ks.update(fv2.curve_ks(removable_count, support, curve_points))
    reference, reference_info = None, None
    if random_repeats > 0 and ks:
        try:
            reference = fv2.random_reference(
                wrapper, x, batch.token, batch.edge_index, batch.node_type, target,
                sorted(ks), repeats=random_repeats, seed=random_seed)
            reference_info = {"repeats": random_repeats, "seed": random_seed, "ks": reference.ks}
        except Exception as exc:  # noqa: BLE001 - calibration must not block the metrics
            reference_info = {"status": "failed", "error": str(exc)}
    results: dict[str, Any] = {}
    for name, vector in vectors.items():
        try:
            results[name] = _fidelity_v2_for_source(
                wrapper, x, batch, vector, target, curve_points=curve_points, reference=reference)
        except Exception as exc:  # noqa: BLE001 - one source's failure must not lose the rest
            results[name] = {"status": "failed", "error": str(exc)}
    return results, reference_info


def _audit_subject(wrapper, batch, x, base_vectors, prediction, *, steps, epochs, ig,
                   audit: dict[str, Any]) -> dict[str, Any]:
    """Weight-randomisation and stability controls for one graph (plan item 8)."""
    out: dict[str, Any] = {}
    seed = audit["seed"]
    if audit.get("weight_randomisation", True):
        try:
            random_wrapper = audit["make_wrapper"](audit_mod.randomise_weights(audit["model"], seed))
            random_x = random_wrapper.set_context(batch)
            with torch.no_grad():
                random_prediction = int(random_wrapper(random_x, batch.edge_index).argmax(-1))
            collected = _collect_importances(random_wrapper, batch, random_x, steps=steps,
                                             epochs=epochs, ig=ig, prediction=random_prediction)
            out["weight_randomisation"] = {
                "prediction_changed": random_prediction != prediction,
                "sources": audit_mod.compare_sources(base_vectors, _source_vectors(collected),
                                                     batch.node_type),
            }
        except Exception as exc:  # noqa: BLE001 - a control must not sink the record
            out["weight_randomisation"] = {"status": "failed", "error": str(exc)}
    draws = []
    try:
        for draw in range(audit.get("stability_draws", 0)):
            x_perturbed, measured = audit_mod.perturb_values(
                batch.x.float(), audit["sigma"], seed + 1 + draw)
            perturbed = batch.clone()
            perturbed.x = x_perturbed
            perturbed_x = wrapper.set_context(perturbed)
            with torch.no_grad():
                perturbed_prediction = int(wrapper(perturbed_x, batch.edge_index).argmax(-1))
            if perturbed_prediction != prediction:
                draws.append({"prediction_changed": True, "measured_nodes_perturbed": measured})
                continue
            collected = _collect_importances(wrapper, perturbed, perturbed_x, steps=steps,
                                             epochs=epochs, ig=ig, prediction=prediction)
            draws.append({"prediction_changed": False, "measured_nodes_perturbed": measured,
                          "sources": audit_mod.compare_sources(
                              base_vectors, _source_vectors(collected), batch.node_type)})
    except Exception as exc:  # noqa: BLE001
        draws.append({"status": "failed", "error": str(exc)})
    finally:
        wrapper.set_context(batch)   # leave the wrapper on the original graph
    out["stability"] = {"sigma": audit.get("sigma"), "draws": draws}
    return out


def explain_subject(wrapper, data, *, method: str, conv: str | None,
                    steps: int, epochs: int, num_relation_columns: int,
                    fidelity_v2_curve_points: int = 3, ig: dict[str, Any] | None = None,
                    random_repeats: int = 10, random_seed: int = 1234,
                    audit: dict[str, Any] | None = None,
                    true_class_explanations: bool = True) -> dict[str, Any]:
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
    with torch.no_grad():
        prediction = int(wrapper(x, batch.edge_index).argmax(-1).item())
    true_class = int(data.y.view(-1)[0].item())
    seed = _stable_seed(random_seed, str(data.sample_id))

    collected = _collect_importances(wrapper, batch, x, steps=steps, epochs=epochs, ig=ig,
                                     prediction=prediction)
    vectors = _source_vectors(collected)
    # Full-evidence-removal fidelity for every source that produced a vector. A
    # source's own failure must not block the others (same isolation principle as
    # explain_algorithms()).
    fidelity_v2, random_info = _evaluate_sources(
        wrapper, x, batch, vectors, prediction, curve_points=fidelity_v2_curve_points,
        random_repeats=random_repeats, random_seed=seed)

    try:
        if method == "cei_gnn_v3":
            edge_attr_contribution = {
                "status": "not_applicable",
                "reason": "CEI-GNN's graph metadata (edge_attr, relation, triple) is fixed inside "
                          "its bound wrapper, so the relation-vs-payload ablation cannot be applied"}
        elif conv == "hgt":
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
    try:
        dependence = audit_mod.edge_dependence(wrapper, x, batch.edge_index, prediction)
    except Exception as exc:  # noqa: BLE001
        dependence = {"status": "failed", "error": str(exc)}

    record = {
        "schema": "medgnn.clinical_graph_v2_explanation_record",
        "schema_version": 2,
        "method": method,
        "conv": conv,
        "sample_id": str(data.sample_id),
        "subject_id": str(data.subject),
        "true_class_id": true_class,
        "prediction_class_id": prediction,
        "explained_target": "predicted_class",
        "wrapper_verification": {"max_abs_logit_diff": wrapper_max_diff, "tolerance": 1e-5},
        "builtin_node_importance": collected["builtin"],
        "builtin_available": collected["builtin"] is not None,
        "builtin_unavailable_reason": collected["builtin_unavailable"],
        "builtin_detail": collected["builtin_detail"],
        "graphxai": collected["graphxai"],
        "fidelity_v2": fidelity_v2,
        "fidelity_v2_intervention": {
            "version": fv2.INTERVENTION_CONTRACT_VERSION,
            "definition": fv2.INTERVENTION_CONTRACT,
            "eligible_node_kinds": list(fv2.EVIDENCE_NODE_KINDS),
            "protected_node_kinds": list(fv2.STRUCTURAL_NODE_KINDS)},
        "random_reference": random_info,
        "edge_attr_contribution": edge_attr_contribution,
        "edge_dependence": dependence,
    }

    if true_class_explanations and true_class != prediction:
        # Plan item 8: for a misclassified patient also explain the TRUE class, so a
        # reader can see what the model attended to versus what it should have.
        try:
            true_collected = _collect_importances(
                wrapper, batch, x, steps=steps, epochs=epochs, ig=ig, prediction=prediction,
                target_class=true_class)
            true_vectors = _source_vectors(true_collected)
            true_fidelity, true_random = _evaluate_sources(
                wrapper, x, batch, true_vectors, true_class,
                curve_points=fidelity_v2_curve_points, random_repeats=random_repeats,
                random_seed=seed)
            record["true_class_explanations"] = {
                "target_class": true_class,
                "graphxai": {n: {k: v for k, v in p.items() if k != "node_explanation"} |
                             ({"node_importance": p["node_explanation"]["node_importance"]}
                              if "node_explanation" in p else {})
                             for n, p in true_collected["graphxai"].items()},
                "builtin_node_importance": true_collected["builtin"],
                "builtin_unavailable_reason": true_collected["builtin_unavailable"],
                "fidelity_v2": true_fidelity, "random_reference": true_random,
            }
        except Exception as exc:  # noqa: BLE001
            record["true_class_explanations"] = {"status": "failed", "error": str(exc)}

    if audit is not None:
        record["audit"] = _audit_subject(wrapper, batch, x, vectors, prediction, steps=steps,
                                         epochs=epochs, ig=ig, audit=dict(audit, seed=seed))
    return record


def _prepare_clinical_ig(bundle, wrapper, subjects, splits, *, tolerance: float,
                         candidates, pilot: int, reference_draws: int,
                         keep_zero_baseline: bool, seed: int) -> dict[str, Any]:
    """Pre-registered IG setup for one run: the baseline layout, and the step count
    chosen from the completeness error on a few pilot graphs (never a default)."""
    layout = (bundle.preprocessing or {}).get("node_feature_layout")
    if not layout:
        raise ValueError("preprocessing.json has no node_feature_layout; cannot build the IG baseline")
    spec = wrapper.ig_feature_spec() if hasattr(wrapper, "ig_feature_spec") else {}
    errors: dict[int, list[float]] = {}
    for steps in candidates:
        errors[steps] = []
        for data in subjects[:max(1, pilot)]:
            batch = _single_graph_batch(data)
            x = wrapper.set_context(batch)
            with torch.no_grad():
                target = int(wrapper(x, batch.edge_index).argmax(-1))
            baseline = ig_clinical.not_recorded_baseline(x, batch.node_type, layout, **spec)
            _, completeness = ig_clinical.integrated_gradients(
                wrapper, x, batch.edge_index, baseline, target, steps)
            errors[steps].append(completeness["completeness_error_rel"])
        if max(errors[steps]) <= tolerance:
            break
    chosen = ig_clinical.choose_steps(errors, tolerance)
    # Real-patient references are rows of the numeric feature matrix; a model whose
    # explained input is a composite (CEI: numeric + token vectors + type vectors) has
    # no such rows to draw from, so it gets the primary baseline only.
    bank = (ig_clinical.ReferenceBank.from_graphs(splits["train"][:2000])
            if reference_draws > 0 and not spec else None)
    return {"layout": layout, "steps": chosen["steps"], "tolerance": tolerance,
            "feature_spec": spec,
            "step_selection": chosen, "bank": bank, "reference_draws": reference_draws,
            "keep_zero_baseline": keep_zero_baseline, "seed": seed}


def dated_output_dir(base, method: str, conv: str | None = None) -> Path:
    """A NEW, timestamped directory name under `base` (plan item 10: never reuse)."""
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"explanations_{method}{'_' + conv if conv else ''}_{stamp}"
    return Path(base) / name


DEFAULT_EXPLAINED_PATIENTS = 500   # decided by the user; the same draw for every method


def run_explanations(run_dir, output_dir, *, fold: str = "validation",
                     max_subjects: int | None = None,
                     sample_patients: int | None = None, sample_seed: int = 1234,
                     steps: int = 32, epochs: int = 50,
                     fidelity_v2_curve_points: int = 3,
                     clinical_ig: bool = True, ig_tolerance: float = ig_clinical.DEFAULT_TOLERANCE,
                     ig_step_candidates=ig_clinical.DEFAULT_STEP_CANDIDATES,
                     ig_pilot_subjects: int = 3, ig_reference_draws: int = 0,
                     keep_zero_baseline_ig: bool = True,
                     random_repeats: int = 10, random_seed: int = 1234,
                     audit_subjects: int = 0, stability_draws: int = 2,
                     stability_sigma: float = 0.05,
                     true_class_explanations: bool = True) -> dict[str, Any]:
    """The full pipeline: load -> rebuild split -> replay-check -> explain the chosen
    visits of `fold` ("validation", or "dev" for runs that keep validation closed) ->
    write output. Raises before writing anything if the replay check fails. `steps` is the budget of the vendored (zero-baseline) IG; the clinical
    IG chooses its own step count from the completeness error."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"explanation output directory must be empty: {output_dir}")

    bundle = load_run_bundle(run_dir, fold)
    binding = bundle.binding
    splits = rebuild_validation_split(binding, fold)
    model, is_method_adapter = build_model(binding, bundle.state_dict)
    replay_check(model, splits, binding, is_method_adapter=is_method_adapter, fold=fold)

    subjects = select_explained_subjects(
        splits[fold], max_subjects=max_subjects,
        sample_patients=sample_patients, sample_seed=sample_seed)
    cohort = build_cohort_v2(
        binding, fold=fold,
        fold_sample_ids=[d.sample_id for d in splits[fold]],
        explained_sample_ids=[d.sample_id for d in subjects],
    )
    cohort["sampling"] = {"sample_patients": sample_patients, "sample_seed": sample_seed,
                          "max_subjects": max_subjects,
                          "patients_explained": len({str(d.subject) for d in subjects})}
    validate_cohort_v2(cohort, binding)

    wrapper = build_wrapper(binding, model)
    method, conv = binding["method"], binding.get("conv")
    ig_config = None
    if clinical_ig and subjects:
        ig_config = _prepare_clinical_ig(
            bundle, wrapper, subjects, splits, tolerance=ig_tolerance,
            candidates=tuple(ig_step_candidates), pilot=ig_pilot_subjects,
            reference_draws=ig_reference_draws, keep_zero_baseline=keep_zero_baseline_ig,
            seed=random_seed)
    audit_config = {"model": model, "make_wrapper": lambda m: build_wrapper(binding, m),
                    "stability_draws": stability_draws, "sigma": stability_sigma,
                    "seed": random_seed}

    records, failures = [], []
    for index, data in enumerate(subjects):
        try:
            records.append(
                explain_subject(
                    wrapper, data, method=method, conv=conv, steps=steps, epochs=epochs,
                    num_relation_columns=binding["num_relations"],
                    fidelity_v2_curve_points=fidelity_v2_curve_points, ig=ig_config,
                    random_repeats=random_repeats, random_seed=random_seed,
                    audit=audit_config if index < audit_subjects else None,
                    true_class_explanations=true_class_explanations)
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
    ig_summary = None if ig_config is None else {
        "baseline": "not_recorded", "steps": ig_config["steps"],
        "tolerance": ig_config["tolerance"], "step_selection": ig_config["step_selection"],
        "real_patient_reference_draws": ig_config["reference_draws"],
        "kept_zero_baseline_reference": ig_config["keep_zero_baseline"]}
    config = {
        "steps_vendored_ig": steps, "gnnexplainer_epochs": epochs, "node_reduction": "mean",
        "fidelity_v2_curve_points": fidelity_v2_curve_points,
        "intervention_contract_version": fv2.INTERVENTION_CONTRACT_VERSION,
        "eligible_node_kinds": list(fv2.EVIDENCE_NODE_KINDS),
        "protected_node_kinds": list(fv2.STRUCTURAL_NODE_KINDS),
        "clinical_ig": ig_summary,
        "random_reference": {"repeats": random_repeats, "seed": random_seed},
        "audit": {"subjects": audit_subjects, "stability_draws": stability_draws,
                  "stability_sigma": stability_sigma},
        "true_class_explanations_for_misclassified": true_class_explanations,
    }
    manifest = {
        "schema": RUNNER_SCHEMA,
        "schema_version": RUNNER_SCHEMA_VERSION,
        "method": method,
        "conv": conv,
        "run_dir": str(Path(run_dir).resolve()),
        "binding_preprocessing_sha256": binding["preprocessing_sha256"],
        "fold": fold,
        "fold_sample_ids_sha256": binding["split_sample_ids_sha256"][fold],
        "replay_verified": True,
        "cohort_schema_version": cohort["schema_version"],
        "cohort_file": "cohort.json",
        "cohort_is_full_fold": cohort["is_full_fold"],
        "binding_file": "binding.json",
        "requested_subject_count": len(subjects),
        "explained_count": len(records),
        "failed_count": len(failures),
        "failures": failures,
        "steps": steps,
        "epochs": epochs,
        "clinical_ig": ig_summary,
        "record_files": record_files,
        "source_code": recursive_source_hashes(Path(__file__).parent),
    }
    (output_dir / "explanation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Plan item 10: the run's own binding.json -- what was explained, by which model
    # and code, with which explainer settings, each pinned by a hash.
    run_path = Path(run_dir)
    binding_out = {
        "schema": "medgnn.clinical_graph_v2_explanation_binding",
        "schema_version": 1,
        "model_run": {
            "run_dir": str(run_path.resolve()),
            "binding_json_sha256": sha256(run_path / "binding.json"),
            "checkpoint_sha256": sha256(run_path / "best.pt"),
            "preprocessing_sha256": binding["preprocessing_sha256"],
            "method": method, "conv": conv, "edge_direction": binding["edge_direction"],
            "seed": binding.get("seed"), "selection_fold": binding.get("selection_fold"),
            "final_eval": binding.get("final_eval"),
            "protocol_overrides": binding.get("protocol_overrides"),
        },
        "cohort": {
            "file": "cohort.json", "sha256": sha256(output_dir / "cohort.json"),
            "fold": cohort["fold"], "fold_sample_ids_sha256": cohort["fold_sample_ids_sha256"],
            "explained_count": cohort["explained_count"], "is_full_fold": cohort["is_full_fold"],
            "sampling": cohort["sampling"],
        },
        "explainer_config": config,
        "records": {"count": len(records), "manifest": "explanation_manifest.json",
                    "manifest_sha256": sha256(output_dir / "explanation_manifest.json")},
        "source_code": manifest["source_code"],
    }
    (output_dir / "binding.json").write_text(json.dumps(binding_out, indent=2, sort_keys=True) + "\n")
    return manifest


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--output", type=Path, help="a new, empty directory")
    where.add_argument("--output-base", type=Path,
                       help="create a new dated directory under this path")
    parser.add_argument("--fold", choices=sorted(FOLD_BLOCK), default="validation",
                        help="fold to explain; use dev for runs that keep validation closed")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--sample-patients", type=int, default=DEFAULT_EXPLAINED_PATIENTS,
                        help="explain N patients (all their visits in the fold), the same draw for "
                             f"every method; default {DEFAULT_EXPLAINED_PATIENTS}; 0 = every patient")
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fidelity-v2-curve-points", type=int, default=3)
    parser.add_argument("--no-clinical-ig", action="store_true",
                        help="keep only the vendored all-zero-baseline IG")
    parser.add_argument("--ig-tolerance", type=float, default=ig_clinical.DEFAULT_TOLERANCE)
    parser.add_argument("--ig-reference-draws", type=int, default=0,
                        help="real-patient-baseline IG robustness check (costs this many IG passes)")
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--audit-subjects", type=int, default=0,
                        help="run weight-randomisation + stability controls on the first N explained visits")
    args = parser.parse_args()
    output = args.output
    if output is None:
        run_binding = json.loads((args.run_dir / "binding.json").read_text())
        output = dated_output_dir(args.output_base, run_binding["method"], run_binding.get("conv"))
    manifest = run_explanations(
        args.run_dir, output, fold=args.fold, max_subjects=args.max_subjects,
        sample_patients=args.sample_patients or None, sample_seed=args.sample_seed,
        steps=args.steps, epochs=args.epochs,
        fidelity_v2_curve_points=args.fidelity_v2_curve_points,
        clinical_ig=not args.no_clinical_ig, ig_tolerance=args.ig_tolerance,
        ig_reference_draws=args.ig_reference_draws, random_repeats=args.random_repeats,
        audit_subjects=args.audit_subjects,
    )
    print(json.dumps({k: manifest[k] for k in
                      ("method", "conv", "explained_count", "failed_count")} | {"output": str(output)}))


if __name__ == "__main__":
    _cli()
