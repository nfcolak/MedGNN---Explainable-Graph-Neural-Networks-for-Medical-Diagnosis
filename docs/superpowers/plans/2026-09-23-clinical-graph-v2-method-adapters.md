# Clinical Graph v2 Method Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add method-faithful ProtGNN, GSAT, and GraphCare-style adapters to the repaired multi-visit `clinical_graph_v2` pipeline without changing the existing ClinicalGNN/GCHM-PNA path.

**Architecture:** Keep cohort loading, tensorization, folds, class weights, validation selection, metrics, and run bindings in the shared runner. Put each method's defining architecture and auxiliary objective in an isolated module under `clinical_graph_v2/methods/`; carry GraphCare's visit structure through a separately hashed, source-derived sidecar and the versioned tensor contract.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, JSONL artifact sidecars, pytest (focused synthetic tests explicitly authorized; broader suite remains gated).

**Spec:** `docs/superpowers/specs/2026-09-23-clinical-graph-v2-method-adapters-design.md`

## Global Constraints

- All methods consume the same versioned graph samples, patient-disjoint folds, ordered targets, train-fitted preprocessing, and shared evaluation implementation.
- The existing ClinicalGNN GCHM-PNA path remains unchanged.
- Do not build or repair any graph, target sidecar, or metadata artifact in this implementation milestone.
- Do not train any method, launch benchmark matrices, evaluate the test fold, or claim measured performance.
- Do not write or run tests without explicit authorization; current approval covers only the plan's focused synthetic tests.
- `visit_membership` is derived from source encounter lineage, never guessed from node order or a timestamp bucket.
- Visit IDs and patient/stay IDs remain metadata and are never predictive feature columns or sidecar values.
- Historical graph files without the sidecar cannot be silently upgraded; a separately authorized rebuild must use a new artifact path.
- Record method-native budgets and schedules; equal epoch counts alone are not equal-compute evidence.
- Every new run binding sets `test_evaluated=false` until separately authorized final evaluation.
- Preserve the existing dirty working tree. Do not install packages, commit, or push.

---

## File Structure

- `comparison/standardized/clinical_graph_v2/store.py`: add source stay lineage to returned measurement records.
- `comparison/standardized/clinical_graph_v2/graph.py`: derive visit/node membership while source lineage is in scope; keep the existing `build_graph(...) -> graph` API and add `build_graph_with_visit_membership(...) -> (graph, sidecar_record)` for the producer.
- `comparison/standardized/clinical_graph_v2/build.py`: write and read back one sidecar row per graph, bind its version/path/hash/count in the artifact manifest.
- `comparison/standardized/clinical_graph_v2/contracts.py`: validate sidecar contract, digest, row/sample alignment, and membership bounds.
- `comparison/standardized/clinical_graph_v2/tensorize.py`: bump `PREPROCESSING_VERSION` to `clinical_inputs_v3`; decode the sidecar into sparse visit-membership tensors and an explicit global-node mask; define PyG batching offsets for visit and node indices.
- `comparison/standardized/clinical_graph_v2/methods/base.py`: shared adapter output and training hooks; no method-specific architecture.
- `comparison/standardized/clinical_graph_v2/methods/__init__.py`: method registry/factory for `protgnn`, `gsat`, and `graphcare`.
- `comparison/standardized/clinical_graph_v2/methods/protgnn.py`: relation/payload-aware prototype encoder, prototype losses, warm-up, and train-only projection.
- `comparison/standardized/clinical_graph_v2/methods/gsat.py`: GIN predictor, one stochastic attention intervention, KL information bottleneck, and `r` curriculum.
- `comparison/standardized/clinical_graph_v2/methods/graphcare.py`: BAT-style relation-aware messages, sparse visit-conditioned alpha/beta attention, direct clinical evidence pooling, and explicit global context.
- `comparison/standardized/clinical_graph_v2/train.py`: add `--method`, dispatch, common loss/evaluation, per-method schedules, and complete run bindings while preserving existing defaults.
- `comparison/standardized/clinical_graph_v2/tabular_control.py`: validate/bind the v3 sidecar contract when consuming the same artifact; keep its tabular feature projection unchanged.
- `comparison/standardized/clinical_graph_v2/aggregate.py`: include new method/input/visit-contract fields in run compatibility checks and aggregate records.
- `comparison/standardized/clinical_graph_v2/README.md`: document invocation, architecture deviations, the new contract, and the separate rebuild/training gate.
- Authorized focused synthetic tests: `tests/test_clinical_visit_membership.py` and `tests/test_clinical_method_adapters.py`.
- Read-only architecture references: `protgnn_analysis/models/GCN.py`, `protgnn_analysis/my_mcts.py`, `protgnn_analysis/config.py`, `protgnn_analysis/train.py`, `gsat_analysis/models/gin.py`, `gsat_analysis/models/gsat.py`, `graphcare_analysis/model.py`, `graphcare_analysis/run.py`, and `external/GraphCare/graphcare_/model.py`. Do not route the new benchmark through their legacy data loaders.

## Interfaces Locked by This Plan

A sidecar row has exactly these concepts and no raw stay/patient identifiers:

```json
{
  "contract_version": "clinical_visit_membership_v1",
  "sample_id": "<existing sample id>",
  "visit_ordinals": [0, 1, 2],
  "membership_pairs": [[0, 0], [1, 0], [2, 1]],
  "global_node_mask": [false, false, true]
}
```

`membership_pairs` contains `[visit_ordinal, graph_node_index]`; visits are chronological oldest-to-newest with the index visit last. `global_node_mask` has one boolean per graph node. Each node must have one or more visit pairs or be explicitly global; a node may have several visit pairs.

`clinical_graph_v2/methods/base.py` defines:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class MethodOutput:
    logits: torch.Tensor
    auxiliary_loss: torch.Tensor
    diagnostics: dict[str, float]

class ClinicalMethodAdapter(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch, *, epoch: int) -> MethodOutput: raise NotImplementedError
    @abstractmethod
    def on_epoch_start(self, epoch: int, train_loader) -> None: raise NotImplementedError
    @abstractmethod
    def optimizer_groups(self, args) -> list[dict]: raise NotImplementedError
    @abstractmethod
    def run_config(self) -> dict: raise NotImplementedError
```

For PyG batches, each sample stores `visit_membership_index` with shape `[2, pairs]`, `num_visits` with shape `[1]`, and `global_node_mask` with shape `[nodes]`. A `ClinicalGraphData.__inc__` override increments membership row 0 by that graph's visit count and row 1 by its node count. The batch adapter derives each visit's owning graph from concatenated `num_visits`; it does not infer it from node order.

---

### Task 1: Preserve source-derived node-to-visit provenance

**Files:**
- Modify: `comparison/standardized/clinical_graph_v2/store.py`
- Modify: `comparison/standardized/clinical_graph_v2/graph.py`
- Modify: `comparison/standardized/clinical_graph_v2/build.py`
- Modify: `comparison/standardized/clinical_graph_v2/contracts.py`

**Interfaces:**
- `ClinicalStore.measurements(stay, cutoff)` and `ClinicalStore.analyte_history(...)` return each measurement's actual source `stay` alongside the existing fields.
- Add `build_graph_with_visit_membership(store, sample, knowledge, vocabulary, diagnoses=None, max_nodes=20000, max_edges=200000) -> (graph: dict, sidecar_record: dict)`. Keep `build_graph(...) -> dict` as a compatibility wrapper returning only the graph.
- Sidecar record keys and membership semantics are defined in “Interfaces Locked by This Plan.”

- [ ] **Step 1: Build the chronological visit ordinal map from source encounters.** In `build_graph_with_visit_membership`, use the existing `store.prior_visits(subject, index_start)` order and append the index stay last. Retain a row for every prior visit even if it contributes no graph nodes. Reject any occurrence stay that is absent from that ordered source list.

```python
prior_stays = store.prior_visits(sample['subject_id'], index['start'])
ordered_stays = [*prior_stays, sample['stay_id']]
stay_to_ordinal = {stay: ordinal for ordinal, stay in enumerate(ordered_stays)}
if len(stay_to_ordinal) != len(ordered_stays):
    raise ValueError('Duplicate source stay in visit order')
```

- [ ] **Step 2: Retain measurement provenance.** Add the query's `stay` value to measurement dictionaries in `store.py`; do not derive it from timestamps or node ordering.
- [ ] **Step 3: Accumulate membership at node creation and reuse.** Mark index complaints, vitals, the index-visit node, and index measurements with the index ordinal. Mark historical measurements with their returned source stay. Mark diagnosis nodes with every stay in `record['occurrences']`. When a shared analyte/concept node is reused, union the source visit memberships. Mark the patient node global. Give knowledge nodes the memberships of their clinical anchor. Do not serialize raw stay IDs.

```python
node_visits: dict[str, set[int]] = defaultdict(set)
patient_node_id = node(('patient',), 'patient', 'patient',
                      age=arrival.get('age'), gender=arrival.get('gender'),
                      race=arrival.get('race'),
                      arrival_transport=arrival.get('arrival_transport'))
global_nodes: set[str] = {patient_node_id}

def add_membership(node_id, stay):
    try:
        node_visits[node_id].add(stay_to_ordinal[stay])
    except KeyError as exc:
        raise ValueError('Node source stay is outside the sample visit lineage') from exc
```

- [ ] **Step 4: Emit a separate sidecar row.** Keep graph JSON topology unchanged. Write each graph and its sidecar row in lockstep to `graphs.jsonl` and `visit_membership.jsonl`; preserve sample ordering and sample IDs.
- [ ] **Step 5: Validate the producer readback.** Extend `verify_artifact` to read both streams together and reject mismatched sample IDs/counts, unordered or missing visit ordinals, out-of-range node indices, conflicting provenance, visit-specific nodes with no membership, or nodes incorrectly marked global. Bind `clinical_visit_membership_v1`, sidecar filename, SHA-256, and row count in the new artifact manifest. Keep historical artifacts invalid rather than synthesizing a sidecar.
- Verification coverage is owned by Task 8: source-stays map to ordered visit ordinals, distinguishing repeated diagnosis/analyte membership, an empty prior visit, the index-visit node, and global patient context.

### Task 2: Version the sidecar and tensor batching contract

**Files:**
- Modify: `comparison/standardized/clinical_graph_v2/contracts.py`
- Modify: `comparison/standardized/clinical_graph_v2/tensorize.py`
- Modify: `comparison/standardized/clinical_graph_v2/train.py`
- Modify: `comparison/standardized/clinical_graph_v2/tabular_control.py`

**Interfaces:**
- Add `iter_graphs_with_membership(graphs_path, membership_path, limit=None) -> Iterator[tuple[dict, dict]]`; zip the two JSONL streams and validate each matching `sample_id` without loading the whole artifact into memory.
- Change `encode_graph(graph, prep, visit_membership, edge_mode='all', drop_relations=(), rewire_relations=(), rewire_seed=0)` to require the validated sidecar record and return `ClinicalGraphData` with `visit_membership_index`, `num_visits`, and `global_node_mask`.
- `PREPROCESSING_VERSION` becomes `clinical_inputs_v3`; the sidecar is mandatory for all v3 consumers, even when a particular model does not use visit attention.

- [ ] **Step 1: Add fail-closed sidecar validation.** Validate required fields, exact sample identity, visit ordinals equal to `range(n)`, pair indices within the graph's node/visit bounds, a correctly sized boolean global mask, and the rule that every node is either visit-member or explicitly global. Reject missing, duplicate, malformed, or old-contract rows.
- [ ] **Step 2: Add explicit batching offsets.** Introduce `ClinicalGraphData(Data)` and override `__inc__('visit_membership_index', ...)` to return the two offsets `[[num_visits], [num_nodes]]`; leave all existing edge-index batching behavior to PyG. Keep `global_node_mask` aligned with concatenated graph nodes.

```python
class ClinicalGraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'visit_membership_index':
            return value.new_tensor([[int(self.num_visits.item())], [self.num_nodes]])
        return super().__inc__(key, value, *args, **kwargs)
```

- [ ] **Step 3: Tensorize the sidecar without turning IDs into features.** `encode_graph(graph, prep, visit_membership, ...)` maps sidecar node positions to encoded nodes, creates sparse membership pairs and visit counts, and copies only the boolean global mask. Do not append visit ordinals, sample IDs, patient IDs, or stay IDs to `x` or `token`.
- [ ] **Step 4: Update every tensorizer call site.** Update `train.py` and `tabular_control.py` to validate and pass the sidecar. The tabular control may ignore membership values after validation; its feature columns and preprocessing remain unchanged. Continue fitting vocabularies/scalers and class weights on train rows only.
- Verification coverage is owned by Task 8: `test_pyg_batch_offsets_visit_and_node_indices_separately` batches graphs with different node and visit counts and asserts membership indices and graph ownership remain in bounds and isolated.

### Task 3: Define the adapter API and implement ProtGNN

**Files:**
- Create: `comparison/standardized/clinical_graph_v2/methods/__init__.py`
- Create: `comparison/standardized/clinical_graph_v2/methods/base.py`
- Create: `comparison/standardized/clinical_graph_v2/methods/protgnn.py`

**Interfaces:**
- `build_method(name, *, num_tokens, node_dim, edge_dim, num_classes, hidden, layers, dropout, token_dim, num_triples, args) -> ClinicalMethodAdapter` is the registry entry point.
- `MethodOutput` and `ClinicalMethodAdapter` use the exact signatures in “Interfaces Locked by This Plan.”

- [ ] **Step 1: Add the shared method output and training hooks.** Store logits, the differentiable auxiliary-loss scalar, and JSON-safe diagnostics separately. `run_config()` returns method name, adaptation version, native schedule, objective coefficients, and mechanism settings; register ProtGNN in the factory and let Tasks 4–5 add their adapters to it. Do not add a second data loader or evaluation implementation.
- [ ] **Step 2: Implement a clinical relation/payload-aware prototype encoder.** Consume `x`, `edge_index`, `edge_attr`, and the existing type/relation tensors. Produce graph-level embeddings, one class-specific set of prototype vectors, prototype-distance activations, and a bias-free prototype classifier. Preserve the cluster and bounded separation objectives; include their configured weights in `run_config()`.

```python
distances = torch.cdist(graph_embeddings, prototype_vectors).square()
activations = torch.log((distances + 1.0) / (distances + epsilon))
logits = prototype_classifier(activations)
prototype_class_ids = torch.arange(num_prototypes, device=labels.device) // prototypes_per_class
correct = prototype_class_ids[None, :] == labels[:, None]
cluster_loss = distances.masked_fill(~correct, torch.inf).min(dim=1).values.mean()
wrong_min = distances.masked_fill(correct, torch.inf).min(dim=1).values
separation_loss = torch.relu(margin - wrong_min).mean()
correct_class_connection_mask = prototype_class_identity.T.bool()
cross_class_l1 = (prototype_classifier.weight * (~correct_class_connection_mask)).abs().sum()
auxiliary_loss = cluster_weight * cluster_loss + separation_weight * separation_loss + 5e-4 * cross_class_l1
return MethodOutput(logits, auxiliary_loss, {'cluster_loss': float(cluster_loss.detach())})
```

- [ ] **Step 3: Preserve the warm-up and projection phases.** Use the effective local ProtGNN defaults (`max_epochs=300`, `warm_epochs=10`, `proj_epochs=20`, `proj_interval=25`, `nearest_graphs=10`, `prototypes_per_class=3`, `lr=0.00179`, `batch_size=128`, `weight_decay=4.3e-5`, `patience=10`, `min_delta=0.005`; prototype `clst=0.1`, `sep=0.1`, `margin=1.0`; MCTS `rollout=3`, `min_atoms=3`, `max_atoms=6`, `expand_atoms=8`, `c_puct=5`). The local optimizer is Adam. Expose overrides and record effective values. Adapt projection to the clinical latent representation and pass only the training loader to it; never inspect validation/test graphs for projection.
- [ ] **Step 4: Return the method-native objective through `MethodOutput`.** `logits` feed the shared weighted cross-entropy; `auxiliary_loss` contains only ProtGNN prototype terms. `on_epoch_start` applies warm-up trainability and train-only projection at configured epochs.
- Verification coverage is owned by Task 8: `test_protgnn_forward_objective_and_train_only_projection` checks finite logits/prototype terms, gradients, reachable warm-up/projection stages, and projection candidates drawn only from training sample IDs.

### Task 4: Implement the GSAT mechanism over clinical edges

**Files:**
- Create: `comparison/standardized/clinical_graph_v2/methods/gsat.py`
- Modify: `comparison/standardized/clinical_graph_v2/methods/__init__.py`

**Interfaces:**
- `GSATAdapter.forward(batch, *, epoch) -> MethodOutput`; stochastic sampling follows `self.training`, while evaluation uses deterministic sigmoid attention.

- [ ] **Step 1: Implement the GIN predictor and extractor.** Preserve the local GSAT node-level extractor and graph readout convention; adapt each message to include the existing relation identity and numeric edge payload.
- [ ] **Step 2: Apply one stochastic attention intervention.** Sample binary-concrete/Gumbel attention once in training, lift node attention to edge attention as `alpha_src * alpha_dst`, and scale the GIN message path once. Do not apply the same mask again in a second predictor layer or wrapper.

```python
u = torch.empty_like(attention_logits).uniform_(1e-10, 1.0 - 1e-10)
gumbel = torch.log(u) - torch.log1p(-u)
attention = torch.sigmoid((attention_logits + gumbel) / temperature) if self.training else attention_logits.sigmoid()
edge_attention = attention.squeeze(-1)[edge_index[0]] * attention.squeeze(-1)[edge_index[1]]
```

- [ ] **Step 3: Add the information-bottleneck objective and curriculum.** Compute `KL(Bern(p) || Bern(r))`; use local defaults `temperature=1.0`, `info_loss_coef=1.0`, `init_r=0.9`, `final_r=0.7`, `decay_interval=10`, `decay_r=0.1`, `max_epochs=100`, `lr=1e-3`, `weight_decay=0`, `batch_size=128`, `patience=10`, and `min_delta=0.005`; record the effective configuration. Use the common runner's weighted CE plus the returned IB auxiliary loss.

```python
r = max(init_r - (epoch // decay_interval) * decay_r, final_r)
p = attention.squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
ib_loss = info_loss_coef * (p * (p / r).log() + (1 - p) * ((1 - p) / (1 - r)).log()).mean()
```

- [ ] **Step 4: Make evaluation deterministic.** `model.eval()` must not sample Gumbel noise; attention-weighted messages remain active, with the mask applied exactly once.
- [ ] **Step 5: Register GSAT** in the shared method factory without changing existing entries.
- Verification coverage is owned by Task 8: `test_gsat_attention_ib_and_deterministic_eval` checks finite logits/IB loss, gradients, one mask application, the `r` schedule, and repeatable eval logits.

### Task 5: Implement GraphCare-style BAT-GNN and visit attention

**Files:**
- Create: `comparison/standardized/clinical_graph_v2/methods/graphcare.py`
- Modify: `comparison/standardized/clinical_graph_v2/methods/__init__.py`

**Interfaces:**
- `GraphCareAdapter.forward(batch, *, epoch) -> MethodOutput` consumes sparse `visit_membership_index`, per-graph `num_visits`, and `global_node_mask`; it returns graph-level logits with CE-only auxiliary loss unless a method-native term is explicitly configured.

- [ ] **Step 1: Implement BAT-style relation-aware messages.** Fuse token/type embeddings with all existing continuous node features. Encode relation identity and numeric edge payload together; do not reduce the clinical graph to legacy token/relation IDs.
- [ ] **Step 2: Build visit states from sparse source membership.** Pool only nodes connected to each visit by `visit_membership_index`, preserving empty visit rows and chronological ordinal order. Represent an empty visit with a zero state and exclude it from beta normalization, but retain its ordinal so it still affects recency ranks of older visits. Derive visit-to-graph mapping from batched `num_visits`, not node positions or timestamps.
- [ ] **Step 3: Implement bounded alpha/beta attention.** Score each node against visits in which it is present; normalize alpha only over those visits. Compute beta from visit states with configured recency decay; aggregate masked alpha×beta contributions to node messages. Keep parameters bounded by hidden width and number of attention heads, not the clinical vocabulary square.

```python
from torch_geometric.utils import scatter, softmax

visit_index, node_index = batch.visit_membership_index
pair_state = torch.cat([node_state[node_index], visit_state[visit_index]], dim=-1)
scores = self.alpha_projection(pair_state).squeeze(-1)
alpha = softmax(scores, node_index, num_nodes=node_state.size(0))
visit_counts = batch.num_visits.long()
visit_graph = torch.repeat_interleave(torch.arange(visit_counts.numel(), device=node_state.device), visit_counts)
visit_starts = visit_counts.cumsum(0) - visit_counts
visit_ordinal = torch.arange(int(visit_counts.sum()), device=node_state.device) - torch.repeat_interleave(visit_starts, visit_counts)
recency_rank = visit_counts[visit_graph] - 1 - visit_ordinal
beta_logits = self.beta_projection(visit_state).squeeze(-1) - self.recency_decay * recency_rank
beta = softmax(beta_logits, visit_graph)
visit_weight = alpha * beta[visit_index]
node_context = scatter(visit_weight[:, None] * visit_state[visit_index], node_index,
                       dim=0, dim_size=node_state.size(0), reduce='sum')
```

- [ ] **Step 4: Handle global nodes and direct clinical evidence explicitly.** Route `global_node_mask` nodes through a separate global-context path. Pool graph messages together with direct clinical evidence (including numeric/context features); never treat absent visit mapping as global.
- [ ] **Step 5: Record adaptation deviations.** Record the parameter-bounded node/visit projection, direct-feature fusion, recency rule, and parameter count as GraphCare-style adaptations, not upstream checkpoint-compatible GraphCare. Use local defaults `max_epochs=100`, `patience=10`, `min_delta=0.005`, `lr=1e-3`, `weight_decay=1e-5`, `batch_size=32`, `hidden=128`, `layers=2`, `dropout=0.3`, and upstream `decay_rate=0.03`; keep common class weighting and validation selection.
- [ ] **Step 6: Register GraphCare** in the shared method factory without changing other entries.
- Verification coverage is owned by Task 8: `test_graphcare_visit_attention_respects_membership_and_global_context` checks visit-specific attention, empty visits, global routing, and gradients to BAT/alpha/beta parameters.

### Task 6: Integrate methods into the shared runner and compatibility contracts

**Files:**
- Modify: `comparison/standardized/clinical_graph_v2/train.py`
- Modify: `comparison/standardized/clinical_graph_v2/tabular_control.py`
- Modify: `comparison/standardized/clinical_graph_v2/contracts.py`
- Modify: `comparison/standardized/clinical_graph_v2/aggregate.py`

**Interfaces:**
- Add `--method {clinical_gnn,protgnn,gsat,graphcare}`, default `clinical_gnn`.
- Keep old `--conv` choices/default behavior for `clinical_gnn`; reject an explicitly supplied `--conv` for other methods.
- Adapter training calls `optimizer_groups(args)`, `on_epoch_start(epoch, train_loader)`, then `model(batch, epoch=epoch)`, and computes `weighted_cross_entropy(logits, y) + auxiliary_loss`. Shared validation uses deterministic forward and validation macro-F1 only.
- `evaluate(model, loader, device, *, epoch=0)` dispatches to `model(batch, epoch=epoch)` for adapters and the unchanged `ClinicalGNN(batch)` forward for the default method.

- [ ] **Step 1: Preserve the existing CLI behavior.** Change `--conv` parsing so omission remains `edge_conditioned` for `clinical_gnn`, while an explicitly passed `--conv` with another method fails before artifact loading or output creation. Set `--hidden`, `--layers`, `--dropout`, `--lr`, `--weight-decay`, `--batch-size`, `--epochs`, `--patience`, and `--min-delta` parser defaults to `None`, then fill missing values from the selected method profile. Existing commands without `--method` retain the current ClinicalGNN values: hidden 96, 3 layers, dropout 0.1, AdamW lr 1e-3/weight decay 1e-5, batch 64, and 12 epochs.

```python
p.add_argument('--method', choices=['clinical_gnn', 'protgnn', 'gsat', 'graphcare'], default='clinical_gnn')
p.add_argument('--conv', choices=['edge_conditioned', 'hgt', 'gchm'], default=None)
p.add_argument('--hidden', type=int, default=None)
p.add_argument('--layers', type=int, default=None)
p.add_argument('--dropout', type=float, default=None)
p.add_argument('--lr', type=float, default=None)
p.add_argument('--weight-decay', type=float, default=None)
p.add_argument('--batch-size', type=int, default=None)
p.add_argument('--epochs', type=int, default=None)
p.add_argument('--patience', type=int, default=None)
p.add_argument('--min-delta', type=float, default=None)
# After parse_args():
if args.method != 'clinical_gnn' and args.conv is not None:
    p.error('--conv is only valid with --method clinical_gnn')
defaults = {
    'clinical_gnn': dict(hidden=96, layers=3, dropout=0.1, lr=1e-3, weight_decay=1e-5, batch_size=64, epochs=12, patience=None, min_delta=None),
    'protgnn': dict(hidden=128, layers=3, dropout=0.51, lr=1.79e-3, weight_decay=4.3e-5, batch_size=128, epochs=300, patience=10, min_delta=0.005),
    'gsat': dict(hidden=128, layers=3, dropout=0.3, lr=1e-3, weight_decay=0.0, batch_size=128, epochs=100, patience=10, min_delta=0.005),
    'graphcare': dict(hidden=128, layers=2, dropout=0.3, lr=1e-3, weight_decay=1e-5, batch_size=32, epochs=100, patience=10, min_delta=0.005),
}[args.method]
for name, value in defaults.items():
    if getattr(args, name) is None:
        setattr(args, name, value)
if args.method == 'clinical_gnn' and args.conv is None:
    args.conv = 'edge_conditioned'
```

- [ ] **Step 2: Select method-native defaults only for method adapters.** Keep clinical defaults unchanged. Use ProtGNN's 300-epoch schedule with patience 10/minimum validation macro-F1 delta 0.005, GSAT's 100 epochs with patience 10/minimum delta 0.005, and GraphCare's 100 epochs with patience 10/minimum delta 0.005. Use Adam for the three adapters (the local legacy runners use Adam), with the recorded learning-rate/weight-decay/batch profiles above; keep AdamW for ClinicalGNN. If the user supplies a common override, record the effective value and the original native default.
- [ ] **Step 3: Share fold/evaluation discipline.** Reuse `build_dataset`, patient-disjoint fold checks, train-only preprocessing, `sqrt_inverse` default weights, validation macro-F1 selection, visit-level predictions, patient-equal metrics, and the held-out test exclusion. Only method adapters own forward mechanics, auxiliary objectives, optimizer groups, and method schedules.

```python
optimizer = torch.optim.Adam(model.optimizer_groups(args), lr=args.lr, weight_decay=args.weight_decay)
for epoch in range(epochs):
    model.train()
    model.on_epoch_start(epoch, train_loader)
    for batch in train_loader:
        output = model(batch, epoch=epoch)
        loss = criterion(output.logits, batch.y.view(-1)) + output.auxiliary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # Shared validation path; no test loader is constructed or opened.
    validation = evaluate(model, validation_loader, device, epoch=epoch)
```

- [ ] **Step 4: Bind every run to the v3 contract.** Include method/adaptation version, graph and target hashes, ordered labels, sample-ID split hashes, visit-sidecar digest/version, preprocessing version/hash, recursive sorted source hashes for imported method modules, seed, class weights, method objectives/schedules, total and active parameter counts, optimizer/budget, selected validation epoch/prediction binding, and `test_evaluated=false`.

```python
package_root = Path(__file__).parent
source_files = sorted(package_root.rglob('*.py'))
source_code = {p.relative_to(package_root).as_posix(): sha256(p)
               for p in source_files if '__pycache__' not in p.parts}
```

- [ ] **Step 5: Reject incompatible artifacts and runs.** Require `clinical_visit_membership_v1` and `clinical_inputs_v3`; reject old preprocessing state, missing/mismatched sidecar, empty source bindings, and incompatible method/conv combinations. Keep output-directory no-overwrite behavior.
- [ ] **Step 6: Preserve comparability metadata in downstream tools.** Make `tabular_control.py` validate/bind the sidecar even though its feature projection ignores visit tensors. Extend `aggregate.py` compatibility keys so runs with different method, sidecar, source code, label order, or preprocessing cannot be silently grouped.
- Verification coverage is owned by Task 8: parser tests check legacy defaults and explicit `--conv` rejection; manifest tests check recursive source and sidecar hashes.

### Task 7: Document invocation and honest evidence boundaries

**Files:**
- Modify: `comparison/standardized/clinical_graph_v2/README.md`

- [ ] **Step 1: Document each method's preserved mechanism and deviation.** State ProtGNN prototypes/losses/projection, GSAT GIN/stochastic attention/KL curriculum, and GraphCare BAT/visit attention/global path; identify each clinical input adaptation and parameter/budget fields.
- [ ] **Step 2: Document the v3 visit contract.** Explain the sidecar, source-lineage rules, global mask, multi-membership, historical artifact incompatibility, and need for a fresh output path on any later authorized build.
- [ ] **Step 3: Document CLI compatibility and evidence limits.** Show method/conv selection semantics, shared split/selection rules, `test_evaluated=false`, and state clearly that implementation availability is not a validated real-data run. Do not include a graph-generation or training command as an instruction to execute in this milestone.

### Task 8: Focused synthetic tests — explicitly authorized

**Files:**
- Create: `tests/test_clinical_visit_membership.py`
- Create: `tests/test_clinical_method_adapters.py`

The user has explicitly authorized this focused synthetic testing task. Keep it limited to in-memory synthetic inputs: source-ordered membership and empty visits; ambiguous/missing lineage rejection; raw identifier exclusion; mixed-size PyG batch isolation; finite forward/backward for all adapters; ProtGNN train-only projection; GSAT one-time mask/IB/eval determinism; GraphCare membership/global routing; recursive hashes; and rejection of legacy contracts. Do not run the existing clinical suite, read patient data, generate graphs/labels, or train benchmark arms.

```python
import torch
from torch_geometric.data import Batch
from comparison.standardized.clinical_graph_v2.tensorize import ClinicalGraphData

def test_sparse_visit_pairs_and_global_mask_are_sample_local():
    first = ClinicalGraphData(
        x=torch.zeros((3, 4)), edge_index=torch.empty((2, 0), dtype=torch.long),
        visit_membership_index=torch.tensor([[0, 1, 1], [0, 0, 1]]),
        num_visits=torch.tensor([2]),
        global_node_mask=torch.tensor([False, False, True]))
    second = ClinicalGraphData(
        x=torch.zeros((2, 4)), edge_index=torch.empty((2, 0), dtype=torch.long),
        visit_membership_index=torch.tensor([[0], [0]]),
        num_visits=torch.tensor([1]),
        global_node_mask=torch.tensor([False, True]))
    batch = Batch.from_data_list([first, second])
    assert batch.visit_membership_index.tolist() == [[0, 1, 1, 2], [0, 0, 1, 3]]
    assert batch.num_visits.tolist() == [2, 1]
    assert batch.global_node_mask.numel() == batch.num_nodes
```

The remaining method tests use the same synthetic-only rule: call each adapter on an in-memory PyG `Batch`, backpropagate `output.logits.sum() + output.auxiliary_loss`, and assert finite gradients on the method's defining parameters. The projection test supplies only a training loader and records consumed `sample_id` values; the GSAT test counts one attention scaling per message path and compares two eval forwards; the GraphCare test changes one sparse visit pair while keeping the graph fixed and checks the affected visit-attention outputs.

Use synthetic in-memory inputs only. Do not read patient data. Run only the focused command authorized for this task:

```bash
pytest -q tests/test_clinical_visit_membership.py tests/test_clinical_method_adapters.py
```

Expected acceptance for this implementation: the focused synthetic tests pass. Passing them establishes plumbing only, not benchmark performance or temporal availability. The broader existing suite remains unrun.

## Self-Review Against the Spec

- Source-derived, non-identifying, sample-local membership; explicit empty visits, global context, and multi-membership: Tasks 1–2.
- Version/hash binding and rejection of historical inputs: Tasks 1–2 and 6.
- ProtGNN prototypes, losses, warm-up, train-only projection, and recorded schedule: Task 3.
- GSAT GIN, stochastic attention, one intervention, KL objective, deterministic evaluation, and `r` curriculum: Task 4.
- GraphCare BAT, visit-conditioned alpha/beta, bounded attention, direct clinical evidence, and global path: Task 5.
- Shared patient-disjoint folds, preprocessing, class weights, validation selection, test exclusion, manifests, and aggregation: Task 6.
- Existing ClinicalGNN/GCHM-PNA and `--conv` behavior: Tasks 3 and 6.
- Documentation, focused-test scope, and explicit rebuild/training gates: Tasks 7–8 and Global Constraints.
- Interface consistency: producer emits the declared sidecar keys; tensorizer converts them to the declared sparse PyG fields; each adapter consumes the common batch; the runner consumes `MethodOutput` and method metadata.
- Focused tests are explicitly authorized in this implementation; graph/label generation, real training, test-fold evaluation, and the broader existing test suite remain outside scope.
