# CEI-GNN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver an independent clinical evidence-interaction GNN, actual GraphXAI execution and the approved four-run development pilot.

**Architecture:** Encode numeric/token/type node inputs independently of ProtGNN. Class-specific signed node votes and mask-aware typed endpoint-interaction edge messages combine with a sentinel-normalized additive readout. One exact continuous-input core serves training and GraphXAI.

**Tech Stack:** Existing Python 3.9 runtime, torch 2.8.0, torch-geometric 2.6.1, pytest 8.4.2, vendored GraphXAI. No new packages.

**Spec:** `comparison/standardized/clinical_graph_v2/methods/cei/docs/superpowers/specs/2026-09-27-evidence-interaction-gnn-design.md` (written spec approved in session).

## Global Constraints

- No candidate dependence on ProtGNN code/weights/predictions/prototypes. Shared adapter, tensor validation and runner infrastructure are permitted.
- Do not edit existing core adapters, global defaults, canonical split, artifact files or vendored algorithms.
- No test-fold tensors, predictions or evaluation. Pilot uses `--selection-fold dev --final-eval none`.
- Data contract: existing max6/Top-10, forward all-edge input; sample seed 1234, matched development 10k train/5k dev, sqrt_inverse weighting.
- Four authorized training runs only: candidate 256/128 two-epoch smoke; ProtGNN 40-epoch dev; CEI-GNN 40-epoch dev; CEI-GNN explicit-product-off 40-epoch dev. No unapproved tuning trials or full multi-seed validation comparison.
- Match patience=40 across development arms, retain ProtGNN's warm-up/projection schedule. Compare actual source and input bindings before reading metrics.
- No new output may overwrite an occupied directory. Medical data, checkpoints, predictions, cohort identities and sample-level explanations stay local and untracked.
- TDD: assertion-based RED before production changes, GREEN, then refactor. An import/collection error alone is not RED. Record red/green commits and commands; supervisor replays them.
- Each meaningful unit gets its own branch, reviewed and pushed by the supervisor. Workers never push or alter the main checkout.
- Existing synthetic baseline checks: 11 passed (plugin API + real GraphXAI); no claim that the entire repository suite passes.

## Shared interface lock

`EvidenceInteractionAdapter` is defined in `methods/plugin_cei_gnn.py`, registered as `REGISTER = {'cei_gnn': EvidenceInteractionAdapter}`. Constructor matches the shared runner's existing keyword signature: `num_tokens, node_dim, edge_dim, num_classes, hidden, layers, dropout, token_dim, num_triples, args`.

It exposes:

```python
continuous_inputs(batch) -> torch.Tensor  # [N, node_dim + token_dim + hidden]
forward_continuous(features, edge_index, metadata, *, return_parts=False)
# metadata is original PyG Data/Batch. return_parts=False -> [B,C] logits tensor.
# True -> dict with logits [B,C], node_contributions [N,C],
# edge_contributions [E,C], bias [C]. All stay differentiable.
forward(batch, *, epoch) -> MethodOutput
```

The continuous columns are ordered numeric x, token embedding, node-type embedding. Numeric x is read from the supplied features in the predictive core, never from original metadata. Fixed metadata is validated and used only for graph membership and edge attributes/IDs. `layers=1` only in v1, initial rank=16, use_interactions=True. Unknown options fail closed.

`run_config()` contains method, adaptation_version=`clinical_graph_v2_cei_gnn_v1`, architecture dimensions including total/active parameter counts, and effective_settings with `interaction_rank` and `use_interactions`. The existing runner stores `best.pt` as a state_dict and a `binding.json`; reconstruction must use both, not expect a checkpoint dict with optimizer/config.

Task 1 owns the model files. Task 2 owns the wrapper/records files. Neither edits the other's code. They may develop in parallel against this lock with synthetic interface fixtures; integrated checks are mandatory before either is considered complete. Task 3 owns protocol/report files and consumes both only after integration. A protocol validator may be developed before model completion because it validates synthetic manifests without importing the candidate.

### Task 1: Independent model and proof of predictive mechanics

**Files:** Create `comparison/standardized/clinical_graph_v2/methods/plugin_cei_gnn.py`, `methods/cei_gnn.py`, `tests/test_plugin_cei_gnn.py`.

**Consumes:** Shared ClinicalMethodAdapter/MethodOutput/read_clinical_batch, NODE_KINDS, relation_count. Exact common constructor above.
**Produces:** Locked adapter interface and model state_dict. No actual clinical training by the worker.

- [ ] Add a registration test and run it before implementation:

```python
def test_cei_is_independent_registered_method():
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY
    assert 'cei_gnn' in METHOD_REGISTRY, 'independent evidence-interaction method absent'
```

Run `python3 -m pytest tests/test_plugin_cei_gnn.py::test_cei_is_independent_registered_method -q`; expect an assertion failure, not an import error. Commit test only with a red prefix. Add the minimal registered class, then grow one behavioral test/code slice at a time.

- [ ] Implement the local numeric/token/type encoder and exact continuous core. Test that ordinary and continuous forwards agree in eval mode, that changing supplied numeric/embedding channels changes predictions, and that original metadata x does not silently supply the perturbed values.

```python
features = model.continuous_inputs(graph)
torch.testing.assert_close(model(graph, epoch=0).logits,
    model.forward_continuous(features, graph.edge_index, graph), rtol=0, atol=0)
```

- [ ] Implement ordered low-rank products and the independent node/edge heads. Use the spec equations verbatim. A PyG MessagePassing edge message contains `[gate*vote, gate]`; the explanation hook masks both blocks once. Aggregate to nodes then graphs. Preserve parallel edges and ordering. The +1 sentinel prevents zero denominators.

```python
parts = model.forward_continuous(features, graph.edge_index, graph, return_parts=True)
reconstructed = parts['bias'] + parts['node_contributions'].sum(0) + parts['edge_contributions'].sum(0)
torch.testing.assert_close(parts['logits'][0], reconstructed)
```

- [ ] Add nondegenerate product/receiver/context-gradient tests and same-class product-off control. Confirm equal sorted parameter shapes and total parameter counts; separately report inactive parameters. Do not claim the whole nonlinear edge head becomes additive merely because explicit q is disabled.
- [ ] Exercise all-ones/zero/fractional masks. For fractional masks use an explicit algebraic numerator/denominator reference and a deliberately squared-mask mutant; all-ones/zero tests alone are insufficient. Verify zero mask equals node-only logits+bias and edge gradients stay connected even when finite zero.
- [ ] Exercise graph batching, node/edge permutation with metadata remapped, edgeless and one-node graphs, invalid types/ranges/nonfinite tensors/cross-graph edges, unsupported depth, unknown options and finite gradients into every active block. Keep test helpers in the owned test file.
- [ ] Run `python3 -m pytest tests/test_plugin_cei_gnn.py tests/test_clinical_method_plugins.py -q -p no:cacheprovider`, report per-slice RED/GREEN evidence and commit the implementation. No training or artifact loads.

### Task 2: Exact GraphXAI wrapper, checkpoint and dev-only record contract

**Files:** Create `comparison/standardized/clinical_graph_v2/cei_graphxai.py`, `tests/test_cei_graphxai.py`.

**Consumes:** Locked Task 1 interface, existing `explain_algorithms`, checkpoint state_dict + binding. During parallel development use a local synthetic protocol fixture; integrated tests must then use the real adapter.
**Produces:** `ClinicalGraphXAIWrapper(adapter, graph)` with `forward(features, edge_index, batch=None) -> logits`; `explain_graph(adapter, graph, *, steps=32, epochs=50) -> dict`; strict model reconstruction and sample-keyed dev-cohort validation/export. No code changes outside owned files.

- [ ] Write fail-first tests through a dynamic lookup that calls `pytest.fail('clinical dev-only GraphXAI wrapper absent')` if the new module is missing, avoiding collection errors. Prove numeric and embedding channel interventions reach the fixture's actual predictor.
- [ ] Implement wrapper that retains original fixed edge metadata, validates exact edge list/order and graph batch identity, and calls `forward_continuous` without re-embedding masked inputs or detaching supplied features. Never infer categories with argmax. Reject changed topology rather than using misaligned payloads.

```python
features = adapter.continuous_inputs(graph).detach()
wrapper = ClinicalGraphXAIWrapper(adapter, graph).eval()
torch.testing.assert_close(wrapper(features, graph.edge_index, batch=graph.batch),
                           adapter(graph, epoch=0).logits, rtol=0, atol=0)
```

- [ ] Invoke actual helper `explain_algorithms(wrapper, features, graph.edge_index, batch=batch, steps=4, epochs=3)` in synthetic tests and require all three named success records, mask-gradient status, finite objectives and mask cleanup. Also test one-node/edgeless input.
- [ ] Reconstruct from exact `binding['method_config']['architecture']`, `effective_settings`, runner settings and `best.pt` (state_dict). Use `torch.load(..., weights_only=True, map_location='cpu')`; refuse incompatible adaptation/schema/settings/shape/source bindings. Synthetic checkpoint replay must prove equality, then supervisor performs real replay.
- [ ] New clinical cohort contract: accept only sample IDs from a provided frozen dev roster, bind source/graph/checkpoint/cohort hashes and class order, refuse test/validation or ambiguous subject-only identifiers. The old `load_explanation_cohort` is NEVER called. Keep record provenance explicit: node gradients condition on edge metadata, feature zeroing is not event deletion, no causal attribution claim.
- [ ] Output export uses fresh final directories and journaling outside the final directory. A failed explainer must remain an explicit failure, propagate nonzero status and prevent a completed manifest. Test occupied-directory and incomplete-record refusal.
- [ ] Run `python3 -m pytest tests/test_cei_graphxai.py tests/test_real_graphxai.py -q -p no:cacheprovider`; integrate with real Task 1 only after its files land; commit red/green evidence. No worker training/real records.

### Task 3: Frozen pilot protocol, control parity and actual execution

**Files:** Create `comparison/standardized/clinical_graph_v2/cei_pilot.py`, `tests/test_cei_pilot.py`; public aggregate report under `docs/` only after verification. Private artifacts remain in an ignored run directory under `comparison/standardized/`.

**Consumes:** Existing train parser/run and binding outputs; real model and GraphXAI adapter; exact task-1/task-2 source revision after review.
**Produces:** Fail-closed stage plan/CLI, shared-binding comparison and replay evidence, measured run results and a bounded next-budget proposal (no automatic final matrix).

- [ ] Write tests with small synthetic manifests before implementation. Required behaviors: exact four stage plans, dev-only flags, sample seed separate from model seed, new path refusal, per-arm completed status, identical common input/source hashes, only explicit treatment fields may differ, changed nested source file changes binding, no validation/test success claim.

```python
left = {'artifact_graphs_sha256': 'a', 'source_code': {'nested/model.py': 'one'},
        'split_sample_ids_sha256': {'train': 't', 'dev': 'd'}, 'label_order': ['A','B']}
right = dict(left, artifact_graphs_sha256='b')
with pytest.raises(ValueError, match='artifact_graphs_sha256'):
    assert_common_bindings(left, right)
```

- [ ] Implement stage planning with explicit argv lists (never shell-split a string of flags). CLI defaults to print-only; `--execute` permits the fixed stages only. No tuning loop. Compare all common bindings including weights, source hashes and preprocessing, excluding explicitly enumerated method/schedule/source-independent output fields, never removing unknown keys indiscriminately.
- [ ] Verify legacy method construction/logits unchanged and new model isolated. Run targeted integrated synthetic suite. Independent reviewer must assess model/wrapper/protocol diff and report spec compliance + correctness before clinical execution.
- [ ] Run candidate smoke once with 256 train, 128 dev, 2 epochs. Reconstruct checkpoint and replay stored dev probabilities; assert finite optimization. Fail closed if any mismatch. Timing is wiring-only.
- [ ] Run the three 40-epoch dev arms once each, sequentially, from frozen source: control, candidate, product-off. All use train-limit=10000, dev-limit=5000, sample-seed=1234, seed=1234, top-k-labels=10, edges=all, edge-direction=forward, weights=sqrt_inverse, selection-fold=dev, final-eval=none, patience=40. Retain existing ProtGNN defaults for its method-specific warm-up/projection.
- [ ] Verify complete results, matching bindings, exact sample/probability alignment and checkpoint replay. Aggregate only compatible complete arms. Export a ten-example dev-only GraphXAI checkpoint pilot with 32 IG steps / 50 GNNExplainer epochs and explicit partial/failure handling.
- [ ] Report actual scores/time/parameters plus single-seed uncertainty limits; no validation/test performance claim. Price future runs from measured times and request new budget approval rather than launching them.
- [ ] Commit/push reviewed source and aggregate report in separate meaningful branches; verify exact remote refs. Update ProjectOS decision/work evidence and preserve local failed runs.

## Self-review and execution policy

Interfaces above are the binding handoff, not guessed from future child output. Task 1/2 own disjoint files and share the exact continuous core contract; Task 3 only consumes that contract after integration. Parallel worker development is allowed by the user's explicit independent-worktree rules; only one heavy suite/training job runs at once. Default execution choice is subagent-driven with main-agent verification, without another process-choice question. No user action is needed to run commands, open pages or prepare files.
