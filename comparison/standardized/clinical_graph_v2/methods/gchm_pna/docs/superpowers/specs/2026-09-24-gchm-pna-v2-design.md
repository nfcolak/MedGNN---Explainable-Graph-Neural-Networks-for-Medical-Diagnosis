# GCHM-PNA v2 and a matched-budget protocol — design

## Goal

Make GCHM-PNA the strongest method in the `clinical_graph_v2` comparison **if it can be**, in a way that
survives review. The protocol must also be able to report that it is not the strongest.

Starting point: `clinical_runs_v3_adapters_sample10k_max6_top10_20260924`. Settings: 10,000 training visits,
top-10 labels, one seed, 4,254 validation visits from 3,472 patients.

| Rank | Method | Macro-F1 | 95% patient CI | Params |
|---|---|---|---|---|
| 1 | ProtGNN | 0.6325 | 0.617–0.647 | 399,884 |
| 2 | XGBoost | 0.6313 | 0.617–0.646 | — |
| 3 | GCHM-PNA v1 | 0.6168 | 0.603–0.631 | 556,938 |
| 4 | GraphCare | 0.6137 | 0.599–0.628 | 120,306 |
| 5 | GSAT | 0.5937 | 0.579–0.609 | 200,046 |

The paired difference ProtGNN − GCHM is +0.016, with a 95% interval of +0.005 to +0.027.

## Measured causes

These were measured on the real artifact, with no training involved.

1. **Evidence never reaches the hub.**
   - Every visit→evidence relation is emitted in one direction only: `reports_complaint`, `observed_vital`,
     `measured_in` and `instance_of`.
   - The only relation into the index visit is `index_visit_of`, which comes from the patient.
   - Under source→target message passing, complaint, vital and lab information reaches the index visit in 0% of
     graphs at any depth. GCHM's hub gate therefore never sees the evidence it was designed to weigh.
2. **PNA is mostly inert.**
   - 71% of nodes receive at most one message. For those nodes mean = min = max and std = 0.
   - v1's 12 aggregator×scaler blocks and 13d→d update were mostly redundant width.
   - v1 was the largest arm. It peaked at epoch 13 and then drifted down while training loss kept falling.
3. **Unequal and untuned settings.** v1 used generic `clinical_gnn` defaults (dropout 0.1). The adapters used
   their published defaults. No arm was tuned for this task.
4. **The protocol flatters every GNN.** The epoch was selected on the reported validation fold, with a single
   seed.

## Design

### 1. Shared edge view: `edge_direction`

`tensorize.encode_graph(..., edge_direction='bidirectional')` appends one typed reverse edge for every kept edge
of the 9 relations whose inverse the producer does not emit:

- `reports_complaint`, `measured_in`, `instance_of`, `observed_vital`
- `baseline_of`, `trajectory_of`
- `medical:member_of`, `medical:assesses`, `medical:measures`

The following are excluded because they are already explicit inverse pairs or already symmetric:

- `has_visit` / `index_visit_of`
- `has_prior_diagnosis` / `recurrence_of`
- `co_complaint`, `comorbid_with`

Invariants:

- The node tensors and the whole forward edge block are byte-identical to `forward`.
- The fitted preprocessing state, and its hash, are unchanged. `edge_feature_layout(prep, direction)` records
  the realised layout.
- Reverse relation ids are 15–23. The reverse triple id is `T + t` for fitted triple `t`, and the unseen id
  stays 0.
- A reverse edge carries its forward edge's payload. It adds routing, not information.
- Every adapter sizes its relation table from the runner's view (`methods.base.relation_count`), so every rival
  can use the same structure.

### 2. GCHM-PNA v2 (`clinical_graph_v2/gchm_v2.py`, `--conv gchm_v2`)

- **Hub gate:** the message is `content ⊙ σ(W_r x_i + e_rel + W_h h_hub(graph))`. The receiver, the relation
  type and the current index-visit state jointly scale each message. This is multiplicative, which keeps GCHM's
  second-order idea.
- **Compact PNA:** mean/min/max/std are aggregated, then projected 4d→d. They are then scaled by learned
  per-channel identity/amplification/attenuation weights, initialised to identity and normalised by the
  train-fold average log-degree.
- **Readout:** `[sum, mean, hub state]`.
- **Budget:** hidden 92. The model has 379,786 parameters on the bidirectional view and 374,818 on the forward
  view, both below ProtGNN. The runner records the realised count.
- **Defaults:** dropout 0.3 and weight decay 1e-4, chosen as a response to v1's measured overfitting. This
  choice was informed by v1's validation curve and is disclosed as a caveat.

Ablation switches, each removing exactly one mechanism:

| Switch | What it removes |
|---|---|
| `--modulation additive` | Multiplicative gate. The replacement is a capacity-matched additive gate. |
| `--aggregation sum` | PNA. Its parameter count is recorded. |
| `--no-hub-gate` | The hub term only. |
| `--readout pool` | The hub readout. |
| `--edge-direction` | Swaps between the forward and bidirectional edge views. |

### 3. Selection without touching validation

`train.py` / `tabular_control.py` flags: `--selection-fold dev --dev-limit N [--final-eval none] [--sample-seed S]`.

- `dev` is drawn with `random.Random("dev-<sample_seed>")` from labelled TRAIN-fold rows. Every patient in the
  drawn training sample is excluded, and the dev split is patient-disjoint from validation by fold construction.
  Preprocessing is fitted on the sample only.
- GNN epochs, adapter early stopping and XGBoost round counts are selected on dev.
- Validation is evaluated exactly once, at the dev-selected checkpoint. With `--final-eval none` it is never
  read.
- `--sample-seed` fixes one training sample while `--seed` varies initialisation.
- The historical validation-selected path is unchanged and remains the default.

### 4. Protocol (`gchm_v2_protocol/protocol.py`)

| Stage | Runs | Rule |
|---|---|---|
| preflight | — | `mechanism_check` (existing and v2 checks) and `identity_snapshot --compare` must pass. The comparison covers incumbent state hashes, encoded tensors and parameter counts, plus the edge-view invariants and the premise that evidence reaches the hub only in the bidirectional view. |
| pilot | 7 | 1 epoch and 300 visits, in its own namespace. This is a wiring check only. |
| tune | 30 | 6 trials for every arm, selected on dev; validation is never read. GNNs: 3 hyper-parameter points × both edge views. XGBoost: depth {4, 6, 8} × learning rate {0.05, 0.1}, with round count selected on dev. |
| select | — | The best dev macro-F1 per arm, with ties going to the earlier trial. Frozen once into `selection.json`. |
| final | 15 | 5 arms × seeds {1234, 2345, 3456} on one fixed sample. XGBoost is `--match-run`-checked against the v2 run of the same seed. |
| ablate | 15 | v2 with one mechanism removed, for each of 4 mechanisms. Plus v1, untouched. |
| report | — | Matched-binding audit, seed means, patient bootstrap (2,000 resamples) of paired seed-averaged differences, and the verdict. |

Guards:

- `protocol_lock.json` freezes the settings, trials, win rule and source-tree hashes at the first real stage.
  Any later drift refuses to run.
- Cell states and an append-only event log are kept. Completed cells are revalidated and skipped. Partial
  outputs are archived, never overwritten. Failed cells need `--retry-failed`.
- A lock ensures one runner per output root.

**Win rule, registered before any run:** GCHM-PNA v2 is the best arm iff both of the following hold.

- Its mean validation macro-F1 over the final seeds is the highest of all arms.
- The 95% patient-cluster bootstrap interval of the seed-averaged difference to the runner-up lies above zero.

Every other outcome is reported as it is. Changing the design afterwards means a fresh protocol root, and the
failed attempt stays in the record.

## Fairness notes

- XGBoost is tuned on the forward view. On its tabular projection a reverse edge only duplicates columns, so it
  gains no information.
- Equal trial counts are not equal compute. Per-run seconds are recorded.
- One training sample: the seeds measure optimisation variance, not sampling variance.
- Validation is not the test fold. The test fold stays closed until separately authorised.

## Out of scope

Test-fold evaluation, full-data (non-sampled) training, 30-class runs, and new graph artifacts.
