# Clinical Graph v2 Method Adapters Design

## Goal

Add method-faithful ProtGNN, GSAT, and GraphCare-style model paths to the repaired multi-visit `clinical_graph_v2` benchmark. All methods consume the same versioned graph samples, patient-disjoint folds, ordered targets, train-fitted preprocessing, and shared evaluation implementation. The existing ClinicalGNN GCHM-PNA path remains unchanged.

This is an implementation design, not authorization to materialize graphs/labels or run training. Those actions remain separately gated.

## Chosen approach

Add isolated model/loss adapters under `comparison/standardized/clinical_graph_v2/methods/` and dispatch them through the existing shared training/data contract. Keep method-specific model behavior in its own module; keep cohort loading, split integrity, prediction saving, metrics, manifests, and output isolation in the common runner.

Rejected alternatives:

- Converting the clinical graph to the legacy ProtGNN/GSAT/GraphCare inputs would discard or alter the repaired numeric node and edge-payload channels.
- Folding all three methods into one generic backbone with name-based flags would obscure their defining mechanisms and make method claims difficult to audit.

These are adaptations to the `clinical_graph_v2` input contract, not claims of bitwise reproduction of the upstream implementations.

## Scope and non-goals

In scope:

- New model factories, method-specific losses/schedules, and shared-runner dispatch.
- A versioned node-to-visit membership channel required by GraphCare's visit attention.
- Manifest/source hashing and compatibility guards for the new input and method code.
- Documentation of architecture-specific deviations, parameter counts, and method-native schedules.

Out of scope:

- Building or repairing any graph, target sidecar, or metadata artifact.
- Training any method, launching benchmark matrices, evaluating the test fold, or claiming measured performance.
- Changing GCHM-PNA's implementation or defaults.
- Claiming that GraphXAI explainers are integrated merely because a model is ProtGNN/GSAT/GraphCare-like.
- Writing or running tests before the user explicitly opens the testing phase.

## Shared input contract

The common tensor view retains the current primitives from `tensorize.py`: dense clinical node features `x`, `edge_index`, relation-plus-payload `edge_attr`, `token`, `node_type`, `edge_relation`, `edge_triple`, and `edge_payload`. The model adapters may encode these primitives differently, but must not substitute a legacy cohort, labels, or graph topology.

Add a separate, manifest-bound `visit_membership.jsonl` sidecar with contract version `clinical_visit_membership_v1`. It is keyed by `sample_id` and contains sample-local visit ordinals ordered oldest-to-newest, with the index visit last, sparse visit↔node membership pairs, and an explicit `global_node_mask`; it contains no patient or raw stay identifiers. The sidecar includes each prior visit considered by the graph producer, even when that visit has no representable nodes. The tensorizer materializes per-sample membership for model use. Multiple memberships are allowed for graph nodes shared across visits. Temporal memberships are derived from source encounter lineage, never guessed from node order or a timestamp bucket.

Membership rules:

- Index-visit complaints, vitals, measurements, and the index-visit node belong to the index visit.
- Historical measurement nodes belong to the source stay returned by the event index; the event query must return that stay lineage.
- Historical diagnosis nodes inherit their recorded prior-stay occurrences, not merely their recurrence count.
- Shared analyte/concept nodes may belong to every visit in which their source observations occur.
- The patient context node is global and is consumed through the global-context path, not copied into every visit. Static knowledge nodes inherit membership from their clinical anchor; genuinely global nodes are explicitly marked in `global_node_mask`, not represented as missing visits.
- An unexplained missing membership for a visit-specific node is an error. Global membership is explicit, not a catch-all for missing lineage.

The current producer filters prior measurements by stay but omits stay from returned event records (`clinical_graph_v2/store.py`); the graph stores diagnosis recurrence counts but not occurrence stays (`graph.py`). The producer must preserve those source memberships before serialization. The tensorizer reads the sidecar and emits the visit-membership tensor and global-node mask; the method batch adapter maps them to each model's representation. Raw identifiers remain metadata only and are never copied into predictive features.

Bind the sidecar path, digest, and `clinical_visit_membership_v1` in the artifact manifest, and bump the tensor preprocessing version from `clinical_inputs_v2` to `clinical_inputs_v3`. Historical graph files without this sidecar cannot be silently upgraded. New method runs reject absent/mismatched sidecars and old preprocessing/checkpoint bindings. A later, separately authorized rebuild must create a new artifact path; the existing graph schema/topology need not change because membership is carried as a separate sidecar.

## Method adapters

### ProtGNN

Preserve class-specific prototype vectors, prototype-distance activations, the prototype classifier, and the cluster/separation objective. Preserve explicit warm-up and projection phases rather than silently reducing ProtGNN to its GCN classifier. Prototype projection candidates come from the training fold only. Adapt message passing so relation identity and numeric edge payload can reach the graph representation; report the change from the legacy GCN backbone as an explicit adaptation. Save prototype count, loss weights, phase schedule, and parameter counts in the run binding.

### GSAT

Preserve the GIN predictor, stochastic binary-concrete/Gumbel attention, attention-weighted message path, KL information-bottleneck loss, and its `r` curriculum. Extend message construction to consume the shared relation and payload channels. Apply the attention intervention exactly once; deterministic evaluation uses the non-sampled attention path. Report any readout or normalization deviations from the local GSAT implementation.

### GraphCare-style adapter

Preserve BAT-style relation-aware message passing, visit-conditioned alpha/beta attention, and graph plus direct-clinical-evidence pooling. Fuse token/type embeddings with the existing continuous node/context features; encode relation identity together with numeric edge payload rather than dropping the latter. Use the versioned visit membership to construct per-visit node evidence.

Avoid a dense vocabulary-to-vocabulary alpha projection when adapting visit attention to the larger clinical token vocabulary. Use a parameter-bounded node/visit attention adapter: score each node against each visit's pooled state, normalize alpha over visits in which that node is present, compute beta from the visit state with the configured recency decay, and sum masked alpha×beta weights back to node messages. Route explicitly global nodes through a separate global-context path. Record this as an architecture deviation and report its parameter count. The result is a GraphCare-style clinical-graph adaptation, not an upstream checkpoint-compatible GraphCare run.

## Training, evaluation, and run artifacts

Add a top-level `--method` selector (`clinical_gnn` remains the default) and leave `--conv edge_conditioned|hgt|gchm` behavior intact for that method. Existing commands without `--method` keep the current `edge_conditioned` default. If a non-`clinical_gnn` method is selected, an explicitly supplied `--conv` is rejected instead of silently ignored. Shared code continues to own the canonical labels, subject-disjoint split checks, train-only preprocessing, class weighting (`sqrt_inverse` default), validation macro-F1 selection, visit-level predictions, patient-equal metrics, and held-out test exclusion. Method-specific code owns only its forward pass, extra objective terms, optimizer groups, and schedule.

Every run is isolated and no-overwrite. Its binding must include:

- method and architecture/adaptation version;
- graph, target, class-order, split sample-ID, visit-membership sidecar, and preprocessing hashes/versions;
- recursive source hashes for every imported method module (the current top-level-only source hash is insufficient);
- seed, class weights, method-specific losses/schedules, parameter counts, optimizer/training budget, selected epoch, and validation prediction binding;
- `test_evaluated=false` until separately authorized final evaluation.

Do not compare scores until artifact generation is separately approved and compatible bindings, realized graph inputs, preprocessing, label order, and split identities are verified. Different method-native schedules are recorded and exposed; equal epoch counts alone are not described as equal compute.

## Failure behavior

Fail before training when the membership sidecar is absent, contradictory, out of range, attached to an unknown source stay during production, or inconsistent with the graph's sample lineage. Do not reconstruct it from node order or silently mark uncertain nodes as global. Fail closed on graph/input/checkpoint version mismatch, missing/empty source bindings, or incompatible method configuration. Preserve all pre-existing outputs and the dirty working tree; choose a fresh run path rather than overwriting.

## Deferred verification plan

Only after an explicit testing-phase authorization:

1. Verify a synthetic multi-visit graph's exact sidecar and tensor memberships, including a repeated diagnosis/analyte, a shared knowledge node, the index visit, and explicit global context.
2. Verify missing and conflicting lineage is rejected, while legitimate global nodes remain accepted; verify raw patient/stay IDs never enter feature tensors.
3. Verify batching does not mix visit memberships or graph nodes across subjects.
4. Verify each model returns finite `[batch, classes]` logits and gradients reach its defining mechanism; verify GSAT masks act once, GraphCare visit attention respects memberships, and ProtGNN projection/objective phases are reachable.
5. Verify new run bindings include recursive source and input hashes and that legacy graph/checkpoint contracts are rejected.
6. Run the focused new tests and then the relevant existing `clinical_graph_v2` suite. Synthetic plumbing evidence is not benchmark-performance evidence.

No verification command, graph/label generation, or training is authorized by this design approval alone.

## Acceptance criteria

1. Existing default ClinicalGNN and all three existing `--conv` choices retain their current code path and behavior.
2. `--method` exposes ProtGNN, GSAT, and GraphCare-style adapters while routing all through one dataset/split/evaluation/run-contract layer.
3. Each method retains and records its named defining mechanism and its explicit input/backbone deviations.
4. Every adapter receives the same graph, node, relation, payload, label-order, and fold contract; GraphCare also receives the versioned source-derived visit-membership channel.
5. The manifest-bound visit sidecar is source-bound, sample-local, non-identifying, multi-membership capable, explicitly separates global context, and fails closed when incomplete for visit-specific nodes.
6. New manifests bind the membership sidecar and all method code recursively and reject historical artifacts/checkpoints that lack the new contracts.
7. No graph/label artifact is regenerated, no training/benchmark is launched, and no performance claim is made in this implementation milestone.
