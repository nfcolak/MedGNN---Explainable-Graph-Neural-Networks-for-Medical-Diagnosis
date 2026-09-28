# CEI-GNN v2: within-visit evidence-pair interactions

Date: 2026-09-28
Status: design approved in chat by the user on 2026-09-28 ("Uygun"). Implementation plan and training are NOT yet approved.
Base: `5f23c829` (`feature/cei-integration`). Branch: `feature/cei-v2-pairs`.

## 1. Goal

Make CEI-GNN's interaction mechanism measurably useful. Success means the multiplicative pair term beats a same-shape additive control and a no-pair control on development macro-F1 across three model seeds, under a pre-registered rule (section 6). Beating ProtGNN is not the goal of this round.

## 2. Why v1's interaction did not help (evidence)

- Pilot dev macro-F1 (seed 1234): CEI 0.649207, product-off 0.649730. Product-off was higher.
- Confound: v1's edge MLP receives `concat(h_s, h_t, k_e)` and can model endpoint interactions itself, so `use_interactions=false` does not remove interaction capacity.
- Location: in the train split, 75.8% of forward edges are structural (`instance_of`, `measured_in`, `observed_vital`, ...), where the endpoint product adds little. Informative relations are 24.2%.
- Reach: clinically interesting pairs (complaint x lab, lab x vital) are both attached to a visit node and are two hops apart. v1 only scores directly connected pairs.
- Train-only scan of the first 3,000 train graphs: within-visit evidence pairs per graph have quantiles 0 / 36 / 179 / 378 / 588 (p90) / 1,044 (p99) / 2,140 (max), mean 238. v1 scores about 37 edges per graph at the median. Membership comes from the existing `visit_membership.jsonl` sidecar. No graph rebuild.

## 3. Model: `cei_gnn_v2`

New files only. v1 (`cei_gnn.py`, `plugin_cei_gnn.py`) stays byte-identical.

- `comparison/standardized/clinical_graph_v2/methods/cei_gnn_v2.py`: core network and pair builder.
- `comparison/standardized/clinical_graph_v2/methods/plugin_cei_gnn_v2.py`: adapter, `REGISTER = {"cei_gnn_v2": ...}`.

### 3.1 Node evidence (unchanged from v1)

`h_i = GELU(LayerNorm(Linear([x_i, token_emb, type_emb])))`. No message passing. Node head gives signed votes `u_ic` and sigmoid gates `a_ic`.

### 3.2 Edge evidence, made endpoint-additive

For edge `e = (s, t)` with context `k_e` (relation + triple embeddings + projected payload):

    v_e = P_s(h_s) + P_t(h_t) + P_k(k_e)        (class-width signed votes)
    g_e = sigmoid(G_k(k_e))                      (gate depends on edge context only)

`P_s`, `P_t` are separate small MLPs on one endpoint each; `P_k`, `G_k` are linear. The edge path cannot form an `h_s x h_t` term. Edge messages still go through the PyG `MessagePassing` mask hook, so GraphXAI edge masks keep working as in v1.

### 3.3 Within-visit pair evidence (new)

Pair set `Q`: all unordered node pairs `{i, j}`, `i < j`, that share a visit ordinal in `visit_membership_index` and whose node kinds are both in `{complaint, measurement, vital}`. Patient, visit, analyte, diagnosis and knowledge nodes are excluded. A node that belongs to several visits contributes pairs in each; duplicate pairs are counted once. Pairs never cross graphs. Pair indices are built deterministically per batch from `visit_membership_index` and `node_type`, with no data-dependent sampling or cap.

Shared projection `U: hidden -> r` (rank `r = 16`), `z_i = tanh(U h_i)`:

| `pair_mode` | Pair feature `q_ij` | Role |
|---|---|---|
| `product` | `z_i * z_j` | candidate |
| `additive` | `z_i + z_j` | same-shape control: same pairs, same parameters, no multiplicative term |
| `off` | pair term removed | no-pair control |

All modes are symmetric in `i, j`. Pair head: `w_ijc = W q_ij` (signed votes, `W` linear, class-width). Pair gates depend only on the unordered kind pair, not on node states: `b_ijc = sigmoid(T[kind_i, kind_j])`, with `T` a class-width table over the 6 unordered pairs of `{complaint, measurement, vital}`. Reason: a state-dependent gate multiplied by the vote would reintroduce an `h_i x h_j` term in `additive` mode. With kind-only gates, `additive` is exactly additive in the two endpoints, and the multiplicative term is the only difference between `product` and `additive`.

### 3.4 Graph logit and exact accounting

    z_Gc = bias_c
         + sum_i a_ic u_ic / (1 + sum_i a_ic)
         + sum_e g_ec v_ec / (1 + sum_e g_ec)
         + sum_Q b_qc w_qc / (1 + sum_Q b_qc)

Per-node, per-edge and per-pair contributions are exported with the same denominators, so `z_Gc == bias_c + sum nodes + sum edges + sum pairs` (float tolerance `1e-5`). The shared normalizers couple terms within a block; this is accounting at a given input, not causal attribution. In `off` mode the pair block contributes exactly 0.

All three modes register identical parameter names and shapes. `off` reports `U`, `W`, `T` as inactive in `run_config`. `layers` must be 1.

### 3.5 GraphXAI

`continuous_inputs` / `forward_continuous(features, edge_index, metadata)` keep v1's signatures. Edge masks cover edges only. Pairs are not graph edges, so GraphXAI cannot attribute them; pair contributions come only from the model's own accounting. The real 10-example GraphXAI acceptance is not re-run in this round; synthetic wrapper-parity tests are.

## 4. Training settings

Same runner and contract as the v1 pilot: existing `clinical_graph_v3_membership_max6_20260923` artifact, Top-10 train-derived labels, `--edges all --edge-direction forward --weights sqrt_inverse`, `--train-limit 10000 --dev-limit 5000 --sample-seed 1234`, `--selection-fold dev --final-eval none`, `--epochs 40 --patience 40`, batch 128. Method defaults copied from v1 (`hidden=128, dropout=0.3, lr=1.79e-3, weight_decay=4.3e-5, grad clip 2.0`). No tuning. Model seeds 1234, 2025, 7; sample seed fixed at 1234 so all arms see identical train/dev IDs.

Known limit: v1 selected epoch 37/40 and was still improving. The fixed 40-epoch budget is kept for comparability and reported as a limit.

## 5. Study runner

New `comparison/standardized/clinical_graph_v2/cei_v2_study.py`, following `cei_pilot.py`:

- Exactly 10 stages: `v2_smoke` (256/128/2, product, seed 1234), then `{product, additive, off} x {1234, 2025, 7}` at 10000/5000/40.
- Default is print-only plan plus ETA from the smoke. Execution requires `--execute` and empty stage output dirs under a new dated root `comparison/standardized/clinical_runs_cei_v2_pairs_20260928/`.
- Reuse `cei_pilot` validation ideas: full binding checks, identical non-treatment bindings across arms, identical parameter shapes across modes, dev-only metrics, `test_evaluated=false`, no `validation.npz`, exact replay of saved dev probabilities from `best.pt`.
- Validation tensors are built by the unchanged loader but never evaluated. No test tensors are loaded.

## 6. Pre-registered decision rule

Primary metric: visit-level dev macro-F1 of the selected checkpoint. The interaction is declared useful only if ALL hold:

1. `product > additive` on each of the three seeds.
2. Seed-mean `product` > seed-mean `additive` and > seed-mean `off`.
3. Paired patient-cluster bootstrap (1,000 resamples, seed 2026; each resample recomputes every arm-seed macro-F1 on the same resampled patients, then averages the delta over seeds) gives a 95% interval for `product - additive` with lower bound > 0.

Otherwise the report says the benefit is not established. Secondary, reported but not decisive: `product - off` interval, patient-equal macro-F1, per-class F1/precision/recall, false positives, pair counts, parameters, seconds.

## 7. Testing (TDD, synthetic only)

Failing tests first, in `tests/test_plugin_cei_gnn_v2.py` and `tests/test_cei_v2_study.py`:

- Pair builder: exact pair sets on hand-built multi-visit graphs; excluded kinds; no cross-graph pairs; correct after PyG batching; empty-pair graphs finite.
- Accounting identity holds in all modes; `off` pair block is exactly zero.
- Edge path has no endpoint cross term: mixed second difference of the edge block w.r.t. `h_s`, `h_t` is zero.
- For one pair with the kind-pair gate fixed, `additive` mode has zero mixed difference w.r.t. `h_i`, `h_j`; `product` mode does not.
- Parameter names/shapes identical across modes; unknown options and `layers != 1` rejected.
- Edge mask zero removes the edge block exactly; wrapper logits equal model logits.
- Study plan: exactly 10 stages, correct argv, refuses non-empty outputs, never passes test options.
- Existing v1 and plugin tests still pass.

## 8. Cost

10 training runs. v1 recorded about 125 s per 40-epoch run; pairs add about 6x more scored items per graph, so the rough estimate is 20-40 minutes of training in total. The smoke run fixes the ETA before the 9 full runs, and the full runs need a separate explicit user approval.

## 9. Out of scope

Test fold; validation-fold scoring; graph rebuilds or new data; tuning; changing v1, ProtGNN or shared runner defaults; ProtGNN re-runs; real GraphXAI acceptance on v2; merging to main.
