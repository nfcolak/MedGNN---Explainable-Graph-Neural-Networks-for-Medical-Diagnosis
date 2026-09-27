# CEI-GNN: Class-specific Evidence Interaction Graph Network

Date: 2026-09-27
Status: design direction and bounded testing/training pilot approved in chat; written specification pending user review. No candidate implementation or training has run.
Base: `f94e4076b0f1425eaae7c418a15eb5876a19ab4c` (`feature/round2-methods`).
Design branch: `docs/evidence-interaction-design`.

## 1. Goal, authorization and exclusions

Implement a genuinely independent clinical graph model, provisionally named **CEI-GNN**, with class-specific signed node and relation-interaction evidence. Exercise actual GraphXAI algorithms against its exact predictive computation. Seek an improvement over the local clinical ProtGNN adapter under the same input contract; do not promise an empirical win or claim publication-level novelty.

The user requested: “Sifirdan kendin yeni bir yontem yaz. GraphXAI ile uyumlu calisabilmeli ve PROTGNN skorunu da gecmeli.” The user then replied “Onay” to the proposed independent evidence-interaction direction, a limited testing/training pilot, and pricing a full comparison only after the pilot.

Authorized scope includes candidate-specific automated tests and a bounded pilot after the written-spec review gate. Existing synthetic baseline tests can establish the starting state. It does NOT authorize a full seed/tuning matrix, held-out test evaluation, graph regeneration, new data sources, modification of historical artifacts, global training defaults, or publication of medical records. Implementation planning and candidate code follow written-spec approval.

## 2. Verified starting evidence

- Historical three-seed ProtGNN validation macro-F1: 0.633731 / 0.632080 / 0.633290 for seeds 1234 / 2025 / 7; mean 0.6330336667, sample SD 0.0008548277.
- Historical ProtoNode: 0.630561 / 0.626818 / 0.627649; mean 0.6283426667. It lost on every paired seed. Adding node prototypes is not the proposed mechanism.
- These historical runs used validation for checkpoint selection. Their values are orientation, not the threshold for a dev-selected challenger or an untouched-test result.
- Incumbent parameters: 399,884. Recorded end-to-end times: 190.1 / 206.0 / 208.8 seconds, mean 201.6333 seconds. These are historical timings, not a promise of candidate runtime.
- Source result directories: `comparison/standardized/clinical_runs_v3_challenger_sample10k_max6_top10_20260927/{protgnn,protonode}_seed{1234,2025,7}/result.json` in the main checkout.
- Source review: ProtGNN uses `_RelationPayloadLayer`, graph-mean embeddings, prototypes, clustering/separation and projection. CEI-GNN must not import its encoder or consume its outputs/checkpoints/prototypes.
- The clinical plugin seam exists: `methods/plugin_<name>.py`, `REGISTER`, `ClinicalMethodAdapter`, explicit `runner_defaults`, method-local options. No core method replacement.
- `shared/lib/graphxai_standardized.py` invokes vendored GradExplainer, IntegratedGradExplainer and GNNExplainer. The legacy cohort loader in `shared/lib/explanation_contract.py` requires canonical TEST subjects and cannot be reused for this pilot.
- The base synthetic plugin/real-GraphXAI checks passed: 11 tests, 14 dependency deprecation warnings, 2.58 seconds. Command: `python3 -m pytest tests/test_clinical_method_plugins.py tests/test_real_graphxai.py -q -p no:cacheprovider`. No clinical training/evaluation took place in these checks.
- Installed runtime inspected: Python executable `/Library/Developer/CommandLineTools/usr/bin/python3`, torch 2.8.0, PyG 2.6.1, pytest 8.4.2. Do not install/upgrade dependencies or merge the separate GraphCare environment.
- Context drift check reports HIGH drift in clinical graph documents. This specification relies on inspected source and result artifacts rather than treating the older context prose as current.

## 3. Frozen input and score contract

Use the existing `clinical_graph_v3_membership_max6_20260923` graph artifact and `first_recorded_lab_all_visits_v2_targets_local_v2_max6/targets.csv`, both below the main checkout's `comparison/standardized/event_inputs/`. Pass absolute artifact paths from an isolated worktree; do not duplicate or modify the artifacts.

Bindings inherited from the inspected historical run:

| Field | Value |
|---|---|
| graph SHA-256 | `1656d121743b0e999980159d255ab996fead6e61264518484b3cc565d3cbb608` |
| membership SHA-256 | `30a0b7521872a3ac68d90fd623f22daa0ebaeca5a73fb39b5ee6aa5e1c375c6f` |
| historical train sample-ID SHA-256 | `70fe7de4f257cc46ef850502f8c3af13b72571ab530212856c8cf905e8d8e944` |
| historical validation sample-ID SHA-256 | `287973edfea71e36731eddaed887928b7bf0553f7a8b684187336c789b379173` |
| historical preprocessing SHA-256 | `9a9ab7a4df8b9a9d6b6c56bc53335f94716df8041d2acd7428e7b512710c5ecd` |
| sample seed / training rows | 1234 / 10,000 |
| retained labels | train-derived Top-10, identical ordered labels in both arms |
| edges | all, forward, exact supplied directed edge ordering and payloads |
| class weights | `sqrt_inverse`, fitted identically on each arm's same training rows |
| validation rows | 4,254 for any subsequently authorized final comparison |
| temporal_clean / test_evaluated | false / false |

For the pilot select a 5,000-row patient-disjoint dev subset from unused TRAIN patients using the existing `--selection-fold dev --dev-limit 5000 --sample-seed 1234 --final-eval none` path. Keep model-init seed separate from sample seed. Verify train/dev subject disjointness and exact IDs before comparing scores. Both arms use the same preprocessing, dev IDs and source tree; persist/check all actual hashes, not merely these historical reference values.

The current runner builds validation tensors even with `--final-eval none`; this option suppresses validation evaluation, not all validation-data I/O. Report this accurately. Do not use validation labels, predictions or scores for pilot decisions. The model must never receive test-fold tensors or produce test predictions. The artifact contains multiple folds; whole-file hashing/streaming and fold metadata checks are not test evaluation.

Primary metric: existing visit-level `metrics.macro_f1`, preserving the historical class universe. Secondary: patient-equal macro-F1, accuracy, balanced accuracy, all-class precision/recall/F1 and false positives. Do not silently switch to the patient-equal number as the primary score.

Input limitations persist: untimestamped triage assigned arrival time, historical diagnosis availability assumed at prior encounter completion, and lab storetime as a proxy. Neither a new architecture nor a higher score proves clinical validity or temporal cleanliness.

## 4. Alternatives and selected mechanism

1. **Selected: explicit node evidence plus typed edge-interaction evidence.** Independent small encoders, class-specific signed contributions, multiplicative endpoint interactions and differentiable mask-aware message aggregation. Direct graph-level readout of edge messages avoids relying solely on evidence reaching a particular hub. Main risk: one-hop edge interactions may miss longer-range combinations.
2. **Prototype refinement.** Rejected for this first pilot: the previous node/graph prototype challenger failed all three validation seeds, and this would be another modification of the incumbent rather than the requested independent mechanism.
3. **Global attention/transformer pooling.** Not selected: the current Round2 code already explores related readouts, while global pathways complicate edge explanation accounting. It remains a possible later alternative, not an authorized second search family.

The structural justification is scoped: inspect actual directed endpoints and payloads, not only a topology name. Existing source emits visit-to-evidence relations that a source-to-target hub-only readout cannot use in reverse. This design instead scores actual endpoint pairs and sums their messages into graph evidence; it does not silently add reverse or all-pairs edges. Before training, summarize train-only graph size, degree and relation-frequency distributions from the pilot's encoded graphs. Do not claim that this establishes optimality or a graph-information advantage over a tabular model.

## 5. Architecture and exact predictive accounting

### Inputs and local encoding

Consume the existing `ClinicalBatch` fields without changing fitted vocabularies or dimensions: `x`, `token`, `node_type`, `edge_index`, `edge_attr`, `edge_relation`, `edge_triple`, `batch_index`, graph count. Keep metadata and graph membership fixed during feature interpolation. Validate indices, finite numeric values and cross-graph edge isolation.

Node representation is an independently implemented shared MLP over numeric node features, token embedding and node-type embedding, followed by LayerNorm/GELU. No imported ProtGNN layer. Relation context combines learned relation/triple embeddings with projected numeric edge features. Parameters are shared across relations; rare relations do not receive independent full message matrices.

For an ordered edge e=(s,t), construct the explicit low-rank interaction:

    q_e = tanh(U h_s) * tanh(V h_t) * sigmoid(R k_e)

The endpoint projections U and V are distinct. Retain the additive endpoint/context terms as well:

    d_e = concat(h_s, h_t, k_e, q_e)

A small edge MLP maps d_e to class-specific signed votes v_ec and sigmoid gate weights g_ec. A separate node head maps h_i to signed votes u_ic and sigmoid gates a_ic. Use logits/votes, not probabilities, for this accounting. Apply dropout inside learned representations/factors, not to final aggregated class totals.

### Mask-aware message passing and graph logits

Implement the edge head as a PyG `MessagePassing` module on the exact original edge list. Each edge message contains two class-width blocks:

    message_e = concat(g_e * v_e, g_e)

The standard PyG explanation hook multiplies this whole message by m_e exactly once. Aggregate to targets, then sum these incoming totals by graph. Therefore every edge-dependent numerator AND normalization term is masked; a masked-out edge cannot remain active only through the denominator. No separate unmasked global edge summary is allowed.

Let S and T denote graph totals of masked gate-weighted votes and masked gates. The class logit is:

    z_Gc = b_c + sum_i(a_ic * u_ic) / (1 + sum_i a_ic)
                 + S_Gc / (1 + T_Gc)

Equivalently export per-node contributions A_ic and per-edge contributions B_ec using those SAME denominators, such that:

    z_Gc == b_c + sum_i A_ic + sum_e B_ec

The fixed +1 denominator is a zero-evidence sentinel: it makes an empty edge set finite and avoids an unconstrained raw sum growing with graph size. It also means duplicating identical evidence is not exactly invariant, but may saturate; no claim that multiplicity is perfectly preserved. Check logit scales on varied synthetic graph sizes and actual train-only size buckets before optimizing.

This is one graph-message block with expressive local MLPs, not a deep prototype GNN. Initially `layers=1`; reject unsupported depth rather than silently treating it as an unrelated hyperparameter. Initial method-local defaults: hidden=128, dropout=0.3, lr=0.00179, weight_decay=0.000043, batch_size=128, epochs=40, patience=10, min_delta=0.005; token dimension follows explicit runner configuration, initial interaction rank=16. Gradient clipping value=2.0. Early-stopping gate follows the common 40-epoch pilot schedule in section 8. Record total AND active parameter counts before training; do not call this parameter- or compute-matched to ProtGNN.

The numerical decomposition is exact accounting at a given input, not a causal attribution: changing an input changes gates and denominators. Supporting/contradicting means signed contribution relative to the fitted logit bias, not a clinical causal claim. GraphXAI independently probes prediction sensitivity.

### Mechanism controls

`use_interactions=false` zeros q_e while retaining the same modules and sorted parameter shapes. The additive endpoint/edge path remains. Report that some interaction parameters become inactive. This is a same-capacity mechanism ablation, not proof of equal active capacity.

A separate zero-edge-mask probe removes ALL edge numerator and denominator contributions and must equal the node-only expression plus bias. It does not delete nodes. Ordered-edge reversal is a test of direction sensitivity, not an authorized graph preprocessing change.

## 6. Interfaces and file ownership

New method implementation (separate meaningful implementation branch after approval):

- `comparison/standardized/clinical_graph_v2/methods/plugin_cei_gnn.py`: adapter registration, method settings, runner hooks.
- `comparison/standardized/clinical_graph_v2/methods/cei_gnn.py`: independent encoders, masked message block, core computation and contribution accounting. Keep modules focused, no changes to core adapters.
- `tests/test_plugin_cei_gnn.py`: synthetic model/registry/mechanism tests.

GraphXAI integration (a separate dependent branch):

- `comparison/standardized/clinical_graph_v2/cei_graphxai.py`: exact predictive wrapper, checkpoint reconstruction, fail-closed dev-only explanation cohort and record writer.
- `tests/test_cei_graphxai.py`: wrapper equivalence, algorithm executions, mask and cohort guards.

Protocol/report integration (separate dependent branch):

- `comparison/standardized/clinical_graph_v2/cei_pilot.py`: bounded stage manifest, frozen settings, matched binding checks and measured timing report. Reuse the shared runner, no global default edits.
- `tests/test_cei_pilot.py`: no-overwrite, scope, source/ID binding refusal and replay tests using synthetic fixtures.

Exact common adapter methods remain `forward(batch, *, epoch) -> MethodOutput`, `on_epoch_start(epoch, train_loader)`, `optimizer_groups(args)`, `run_config()`. New core exposes `continuous_inputs(batch)` and `forward_continuous(features, edge_index, metadata, *, return_parts=False)`; ordinary forward and explanation wrapper call this single core. `return_parts` exports logits, node contributions, edge contributions and class bias. Method loss is a differentiable zero initially; cross-entropy/class weighting remain the shared runner's responsibility.

Do not change existing GraphXAI vendored source, shared fidelity defaults, canonical split, core adapter code, saved artifacts, or unrelated Round2 code to accommodate the candidate. If integration requires such a change, stop and present a narrower correction with its own tests/branch.

## 7. GraphXAI integration and acceptance

### Exact-input wrapper

Expose `wrapper(features, edge_index, batch=...) -> logits` for one fixed clinical graph. Features concatenate the model's continuous numeric node channels, token embedding vectors and node-type embedding vectors. At the unperturbed input, this is an exact change of interface, not a surrogate model. Features used by the core must come from the supplied tensor; never re-fetch the original unmasked token embedding inside forward or convert interpolated features back to IDs via argmax.

Relation IDs, triple IDs, numeric edge payloads and original node/edge order remain fixed metadata. Thus Grad/IG explain continuous node evidence including embedding coordinates, CONDITIONAL on the recorded graph and edge payloads; they do not explain the categorical lookup operation or numeric edge-payload coordinates. GNNExplainer additionally probes edge presence through the genuine mask hook. Disclose these boundaries in every result manifest; do not call this attribution of every input field.

Validate the edge list/order against the bound graph. These three graph-level algorithms use a fixed graph; unsupported topology mutations must fail rather than reusing misaligned edge features. Numeric/embedding feature zeroing is an intervention in continuous representation space, not literal deletion of a clinical event. Record this as the fidelity baseline. Retained type/routing metadata is conditional structure, not silently claimed removed evidence.

### Execution evidence

Use actual vendored `GradExplainer`, `IntegratedGradExplainer`, `GNNExplainer` through the existing compatibility helper. No custom gradient algorithm may be relabelled GraphXAI. Synthetic bounded budgets: 4 IG steps / 3 mask-optimization epochs. Trained-checkpoint pilot: 32 IG steps / 50 mask-optimization epochs, one deterministically selected dev example per retained class (10 records), chosen by sample identity without selecting easy predictions or nonzero gradients. This is an integration pilot, not cohort-wide explanation quality evidence.

A new clinical dev-only explanation manifest must use sample IDs (visits), not assume one graph per subject. Bind graph/membership/checkpoint/source hashes, ordered class names, fold, seed, exact cohort identity and per-algorithm status. Verify membership in the frozen dev set; reject any validation/test sample. Keep sample-level records local and do not commit them to the public repository. Never route through the legacy TEST-only cohort builder.

Required tests/evidence:

1. Exact ordinary-forward/wrapper logits and checkpoint reload probabilities on CPU; no hidden prediction head bypass.
2. Bias + node + edge contributions reconstruct every class logit within explicit floating tolerance, including signed votes.
3. Endpoint cross-difference is nonzero under nondegenerate constructed weights; removing q removes the explicit product. Do not infer global separability from one random fixture.
4. Batch-vs-single graph equivalence, permutation invariance with metadata remapped, asymmetric directed-edge fixture, repeated/parallel edges, finite edgeless and one-node graphs.
5. All-ones edge mask equals ordinary output; all-zero mask equals node-only expression. Fractional masks match a manual ONCE-masked numerator/denominator computation. All-ones/all-zero alone are insufficient to catch squared masks.
6. Gradients reach token/numeric inputs, edge payloads, both endpoint projections, relation context and both class heads; use nondegenerate fixtures. Source edits must not disconnect the edge mask. Finite zero gradients are reported, not replaced with different examples.
7. GNNExplainer edge entropy/objective stays finite on edgeless inputs; provenance says edge-gradient verification not applicable, never true.
8. All three actual algorithms emit complete finite records. Failure of one is an incomplete run and nonzero exit; journal records before final export so an interruption preserves completed work.
9. Deliberately inject forbidden-fold membership, reordered edge metadata, changed source hash, occupied output directory and incompatible checkpoint settings: every guard must refuse; clean controls must pass.
10. No legacy method logits/parameters change as a side effect. Existing relevant synthetic regression checks remain green. Use TDD assertion failures, followed by the implementation, and verify red replay; collection/import errors alone do not prove behavior.

## 8. Bounded pilot and stopping policy

After written-spec approval and implementation verification, execute ONLY these training items, in isolated new output directories; no parallel heavy training jobs:

| Item | Arm / scope | Count | Decision use |
|---|---|---:|---|
| wiring smoke | CEI-GNN, 256 train / 128 dev, 2 epochs, seed 1234 | 1 | wiring/finite loss/reload only, never performance evidence |
| development control | unchanged ProtGNN, 10k train / 5k dev, 40 epochs, seed 1234 | 1 | matched incumbent on new source tree |
| development candidate | CEI-GNN, identical 10k/5k, 40 epochs, seed 1234 | 1 | preliminary screening |
| interaction ablation | same CEI-GNN class with product disabled, identical 10k/5k, 40 epochs, seed 1234 | 1 | preliminary mechanism screening |

The 40-epoch development cells have no early stopping before the last epoch (explicit CLI patience override 40 for both arms); select their checkpoint on dev macro-F1. This retains the incumbent warm-up/projection schedule and avoids declaring a candidate better than a prematurely stopped ProtGNN. Record optimizer steps, actual epochs, training seconds, total seconds and active parameters; equal epochs are not equal compute. The small smoke uses its own split/preprocessing and is excluded from all comparisons.

Before either matched development metric is interpreted, require exact equality of source-file hashes, graph/membership/target hashes, sampled train/dev ID hashes, label order, preprocessing and weighting; use an explicit allowlist only for intended method-specific settings. Run all matched arms from one source revision. A historical-control score is never substituted. Post-change source drift invalidates pending comparisons and requires a new namespace, not overwritten results.

Stop immediately on leakage/fold guard failure, nonfinite optimization, disconnected mask, wrapper mismatch, artifact drift, incomplete records, or replay mismatch. Do not spend the remaining development cells after a wiring failure. A weak or negative result is reported as such; it does not authorize more variants or hyperparameter trials. A small single-seed dev lead is inconclusive and not a forecast of validation superiority.

Time accounting is measured rather than guessed. Historical ProtGNN averages 3.3606 minutes/run in a different selection protocol. Candidate and ablation costs are unknown until the pilot. The pilot report must provide per-item actual seconds, then price every additional tuning trial, final seed, ablation and explanation cohort separately. Full comparison requires a new approval.

## 9. Later comparison: proposed, not authorized by this pilot

Freeze candidate settings before evaluating validation. Re-run ProtGNN under the same dev-selected protocol, not the historical validation-selected checkpoint rule. Initial proposed final seed set: 1234 / 2025 / 7, one fixed training sample. Validate once per final run; no test evaluation.

A higher three-seed arithmetic mean of per-seed macro-F1 is the directional objective. A supported superiority claim additionally requires a positive paired patient-cluster bootstrap interval on the seed-averaged SCORE difference; compute per-seed metrics inside each resample and average those differences. Do not average probabilities and call the resulting ensemble the single-model mean. Show each seed and paired difference, state that three seeds give limited optimization-variance evidence, and label a positive mean with overlapping uncertainty as inconclusive. Historical reuse of the validation cohort also limits a clean generalization claim.

Report the same-capacity interaction ablation separately; a candidate win alone does not establish the interaction mechanism caused it. A meaningful-edge claim would additionally require an appropriately constrained rewiring control, which is outside this pilot.

## 10. Related work, originality and remaining risks

This is an original implementation proposal in this repository, not proof of a new published algorithm. Additive graph models already exist (GNAN, arXiv:2406.01317); learned gates and multiplicative interactions also have prior art. The proposed differentiator to evaluate is their precise combination: independent typed endpoint products, class-specific signed node/edge accounting and one-mask predictive/explanation equivalence on this clinical contract. A publication novelty audit is not complete.

Primary sources consulted for API/related-work grounding:

- GraphXAI official repository, `mims-harvard/GraphXAI`, README and this repository's vendored algorithm source.
- PyG 2.6.1 official `torch_geometric/nn/conv/message_passing.py`, especially `explain_message` (mask application).
- *The Intelligible and Effective Graph Neural Additive Networks*, arXiv:2406.01317 (related additive model family, not this implementation).

Risks: direct edge pairs may be insufficient for the task; normalized gating may underuse count signal; exact contribution accounting is not causal faithfulness; node-gradient explanations condition on fixed edge information; temporal assumptions remain; candidate runtime and accuracy are not measured. Every one is reported rather than hidden by a successful import or a synthetic test.

## 11. Review/implementation handoff

Written-spec approval is the next gate. After it: write the implementation plan, implement test-first in isolated worktrees, independently review code, run the bounded pilot, verify all artifacts, and propose the next measured budget. Keep each meaningful unit on its own pushed branch. Never push local medical input, predictions, checkpoints or explanation records to the public remote.
