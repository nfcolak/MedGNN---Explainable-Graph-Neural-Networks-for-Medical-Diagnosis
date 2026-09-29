# CEI-GNN v3 extensions: larger model, XGBoost+CEI hybrid, validation-tuned offsets, comorbid/bidirectional edges

Date: 2026-09-29
Status: design only. Pre-registered. No implementation, preprocessing, training or fold scoring is authorized by this document (v3 spec §11.3: G2 OPEN for tests, G3 NOT approved).
Base checkout: `agent/spec-cei3` at `7bc40e6e28550079967b4635c4292ba278a4eee6`, plus v3 spec §11 (this branch).
Parent spec: `docs/superpowers/specs/2026-09-29-cei-gnn-v3-value-encoding-design.md` (v3). Everything not restated here is inherited from v3 unchanged: data, screen fold (§5), decision rule (§6.2), gates (§8, §11.3), source-hash rule (§7), TDD (§9).

## 1. Scope and claim structure

Four improvement items (user decisions U5–U8, 2026-09-29) are built on top of **v3 arm C** (v2 additive pair mode + PLE + not-measured block, frozen K per v3 §11.2). Each item is one or two arms; each arm is trained with the same 10,000-row TRAIN sample, the same OLD-dev checkpoint selection, seeds 1234/2025/7, and scored once per frozen checkpoint on the v3 screen fold. Each arm has its own claim: "arm X beats v3 arm C on this screen", decided by the v3 §6.2 rule with the multiplicity policy of §7 below. No arm is a validation, test, or clinical claim. Item 3 (hybrid) additionally has "beats XGBoost alone on this screen". None of the arms is combined into one "best CEI" after results are seen.

Fixed control: arm C's three frozen checkpoints (the winning-K checkpoints of v3 §11.2). Every extension arm binds the v3 K-freeze hash and C's three checkpoint SHA-256 values.

Dimensions used for every count below (recorded v2 full-run binding, `additive_seed1234/binding.json` of the run reported in `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/.worktrees/_runs/run-v2-full-3/report.md`): vocabulary 391, node_dim 63, edge_dim 22, classes 10, triples 16, relations 15 (forward), token_dim 32, pair_rank 16, node types 8, hidden 128, layers 1; recorded total 92,300 parameters (`docs/cei-gnn-v2-pair-study-result-2026-09-29.md` line 16). Analytic counts in this spec reproduce that 92,300 exactly from `cei_gnn_v2.py` lines 101–119 (`/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/.worktrees/_runs/spec-ext-1/param_counts.py`). v3 arm C adds the PLE projection `(K+1) × hidden` and the absence tables `2 × |universe| × 10` on top; those v3 counts are not yet known and must be read from the C binding, so all parameter numbers here are **v2-schema counts plus stated deltas**, not final v3 totals.

## 2. Item 2 — larger CEI model (U5)

### 2.1 What the code supports today

- `PairEvidenceNetwork.__init__` takes no `layers` argument (`comparison/standardized/clinical_graph_v2/methods/cei_gnn_v2.py` lines 90–91). The node encoder is one `Linear(node_dim + token_dim + hidden, hidden)` followed by `LayerNorm` (lines 103–104) and one GELU + dropout in the forward pass (lines 178–179). There is no message passing; node, edge and pair votes are all computed from that single-layer `h`.
- The v2 adapter rejects any depth other than 1: `plugin_cei_gnn_v2.py` lines 35–36 raise `ValueError("CEI-GNN v2 supports layers=1 only")`. The v1 GraphXAI bridge also refuses `layers != 1` (`cei_graphxai.py` lines 191–192).
- Width is free: `hidden` sizes every block (lines 100–118), and the runner accepts `--hidden` (`train.py` line 1074).

So `layers > 1` is **not supported** by `cei_gnn_v2.py`. Per v3 §7, v2 files stay byte-identical; the change lands in the new v3 method (`cei_gnn_v3`) only.

### 2.2 What must change (v3 core interface obligation)

The v3 network takes an `encoder_depth ≥ 1` constructor argument (default 1 = v2/v3-C behaviour, bit-identical on identical weights). Depth `d > 1` appends `d − 1` node-local residual blocks `h ← h + Dropout(GELU(LayerNorm(Linear_w×w(h))))` after the v2 encoder and before `node_head`, edge and pair votes. Node-local means: no neighbour aggregation, so the CEI exact logit decomposition (v3 §4.3) is unchanged; each block adds `w² + w` (Linear) `+ 2w` (LayerNorm) parameters. The v3 adapter maps the runner's `--layers` to `encoder_depth` and no longer raises for `layers > 1`; the v3 GraphXAI bridge must accept the bound depth (v1's `layers == 1` check at `cei_graphxai.py:191` is v1-only and stays).

### 2.3 Fixed arms (no size search on the screen)

Two arms, fixed now. Both: same K, data, seeds, epochs (40), optimizer, checkpoint selection as C; only the named dimension differs.

| Arm | Definition relative to C | v2-schema parameter delta (analytic) | v2-schema total |
|---|---|---|---|
| **E2w** (width) | C with `--hidden 256`, `encoder_depth 1` | +177,792 | 270,092 (2.93× C's 92,300) |
| **E2d** (depth) | C with `--hidden 128`, `encoder_depth 2` | +16,768 | 109,068 (1.18×) |

Justification. (i) The v2 CEI at 92,300 parameters is small next to the frozen ProtGNN reference of 399,884 (`train.py` lines 94–95); hidden 256 (270,092 + v3 blocks) stays below that reference and is the largest power-of-two width that does, while hidden 320 (395,852 before v3 blocks) would not once PLE/absence are added. (ii) The v2 selected epochs for the additive arm were 17, 24 and 30 of 40 and no arm selected the last epoch (`docs/cei-gnn-v2-pair-study-result-2026-09-29.md` lines 89–95), so the 40-epoch budget is not truncating training; the size question is capacity, not schedule, and the 40/patience-40 budget is kept for parity. (iii) Width scales every evidence block (node, edge, pair, PLE projection); depth adds only node-local nonlinearity. They answer different questions, so both are run rather than one blend. If the user wants a single arm, §10 lists the pre-registered OLD-dev selection alternative (same protocol as v3 §11.2: seed-mean dev macro-F1 of dev-selected checkpoints, tie → smaller model, frozen and hashed before the screen is read).

Active/inactive: both arms have all blocks active as in C; report total/active/inactive from the binding; never claim capacity parity with C.

### 2.4 Memory and runtime (estimate)

E2w doubles the width of every `[nodes, hidden]` and `[edges, hidden]` intermediate (`edge_source`/`edge_target` first layers, lines 109–110) and of the pair projection input; the pair intermediates `[Q, 16]` are unchanged (rank fixed). Estimate: ≤ 2× the v2 per-stage node/edge activation memory; the v2 measured peak RSS 2.63 GiB (v3 §6.3) is not a bound. Time: unmeasured; the v2 per-stage mean is 125.38 s (v3 §6.3), and E2w is expected to be slower by a factor between 1 and 3 (matrix work scales with `w` to `w²`); E2d adds one `w×w` product per node, expected ≤ 1.2×. Both must be measured in the approved smoke and reported, not guessed. The v3 §4.5 memory preflight applies at batch 128.

### 2.5 Decision rule and multiplicity

E2w vs C and E2d vs C: each the v3 §6.2 three-condition rule, family-corrected as in §7 (m = 6). Claim wording: "wider (deeper) CEI v3 beats CEI v3 C on this screen". A win does not identify whether width or depth is the mechanism unless both win.

### 2.6 TDD obligations (item 2)

1. `encoder_depth=1` reproduces C's forward bit-for-bit on identical weights; `encoder_depth=2` changes only `h` and keeps exact logit reconstruction within 1e-5 (v3 §4.3).
2. Parameter accounting: analytic deltas above equal `parameter_count(model)` differences; `run_config()` records `hidden`, `encoder_depth`, totals.
3. The v3 adapter accepts `layers=2` and the v2 adapter still raises for it (v2 byte-identical); bindings reject an E2 arm whose K-freeze hash differs from C's.

## 3. Item 3 — jointly used XGBoost + CEI hybrid (U6)

### 3.1 XGBoost control as it exists (`tabular_control.py`)

- Feature build: `fit_features` fits exactly the GNN preprocessing on the TRAIN sample IDs (lines 69–80, `fit_preprocessing`); `projection_layout` (lines 88–103) and `project_encoded` (lines 106–134) turn each encoded graph (`encode_graph`) into per-token count/mean summaries, node-kind counts, context sums, per-relation count/mean payload, and triple counts; `build` (lines 137–176) walks `graphs.jsonl`, skips test rows (line 155), encodes train/validation/dev rows (fold codes line 148) and returns `X, y, folds` plus optional `sample_ids/subjects` metadata.
- Class weights: `class_weights(y[tr], a.weights, num_classes)` (line 350), implemented in `train.py` lines 506–518 (`sqrt_inverse` = `sqrt(N / (C · count_c))`), applied as row weights in the train `DMatrix` (line 352).
- Booster: params `multi:softprob`, `tree_method=hist`, `max_depth` 6, `learning_rate` 0.1, `subsample` 0.9, `colsample_bytree` 0.8, `reg_lambda` 2.0, `min_child_weight` 3.0, `seed=a.seed` (lines 354–358); `xgb.train(params, dtr, num_boost_round=a.rounds)` (line 359); optional dev round selection every 25 rounds, ties to the smaller count (lines 361–372); validation is scored by default unless `--final-eval none` (lines 197, 379).
- Sample parity: `--train-limit` replays the same seeded TRAIN draw as `train.py` (lines 52–66, 264; `train.py` lines 313–319).

### 3.2 Chosen scheme: XGBoost first, cross-fitted margins as a fixed logit offset, CEI trained on the residual

Direction choice. The reverse (CEI first, XGBoost on CEI residuals via `base_margin`) would need out-of-fold CEI checkpoints, i.e. five extra 40-epoch CEI trainings per seed; XGBoost boosters fit in seconds to minutes, so cross-fitting XGBoost is the cheap side. Fitting the offset by cross-fitting (not joint gradient training) keeps the tree model a fixed function and keeps CEI's exact evidence decomposition intact: the hybrid logit is

    logits_Gc = m_Gc + bias_c + Σ node + Σ edge + Σ pair + Σ absence      (all CEI terms as v3 §4.3)

where `m_G ∈ R^10` is the XGBoost raw margin (`predict(..., output_margin=True)`, `multi:softprob`) for graph G, a constant with respect to CEI parameters (no gradient flows into it; scale fixed at 1, see §10). `return_parts` gains one key `base_margin` (shape `[graphs, classes]`). Training loss: the runner's weighted cross-entropy (`train.py` lines 713–715) on the summed logits; checkpoint selection on OLD-dev macro-F1 of the summed logits. The CEI part is trained fresh with the offset present; C's checkpoints are not reused.

Protocol per seed s ∈ {1234, 2025, 7}:

1. **Rows.** A new builder (not `tabular_control.build`, which cannot represent a `screen` fold, lines 148, 159–160, and encodes validation rows by default) encodes only the 10,000 TRAIN-sample rows, the 5,000 OLD-dev rows and the 5,000 screen rows with `encode_graph` + `project_encoded` on the SAME preprocessing state as arm C (equality of `preprocessing_state(prep)` asserted, as `--match-run` does at lines 312–314). Validation and test rows are never encoded here.
2. **Full booster X_s.** Trained on all 10,000 train rows with the §3.1 params, `seed=s`, `sqrt_inverse` weights. Round count `R_s` selected on OLD dev from the grid 25, 50, …, 300 by dev macro-F1, ties → smaller (the `tabular_control` policy, lines 364–370). `X_s` is the **XGBoost-alone reference row** and the margin source for dev and screen rows.
3. **Out-of-fold boosters.** The 10,000 train rows are split into 5 patient-grouped folds: sort distinct subjects, shuffle with `random.Random(f"hybrid-oof-{s}")`, assign round-robin. For fold k, booster `X_s^{(−k)}` is trained on the other four folds' rows with `R_s` rounds and the same params/seed; the margins of fold-k rows come from `X_s^{(−k)}`. Every train row therefore receives a margin from a booster that never saw that row or that patient.
4. **Margin table.** One float32 array per fold: train rows (OOF), dev rows (from `X_s`), screen rows (from `X_s`), keyed by ordered sample ID; SHA-256 of each array, of `X_s` (`model.ubj`), of the fold assignment and of `R_s` are bound. Margins are raw and uncentred (softmax is shift-invariant per row).
5. **CEI hybrid arm H.** Arm C's architecture and K, trained with the margin table as a graph-level input added at the logit sum only. The node feature tensor `x`, tokens, edges and membership are byte-identical to C's inputs (asserted by hash). Dev checkpoint selection as C.
6. **Screen.** H's frozen checkpoint scored once with the screen margins; `X_s` scored once on the screen rows (probabilities from `predict` at `R_s`).

Leakage guarantees (each is a test in §3.6): no dev, screen, validation or test row enters any booster fit (dev is used only to pick `R_s`, the same selection role as epoch selection); train rows use OOF margins only; XGBoost output is never a node/edge feature of CEI (the margin enters only at the logit sum); the screen margins are computed from `X_s` frozen before any screen tensor is read. Known caveat, stated not hidden: train rows see 8,000-row-booster margins while dev/screen rows see the 10,000-row booster, a standard cross-fitting mismatch.

### 3.3 Parameter accounting

H's trainable parameters equal C's (same schema, same K); the offset scale is a fixed constant, so H's total/active/inactive counts are C's. The booster is reported separately: `R_s × 10` trees, `max_depth 6`, and the total leaf count read from the saved model. These are not comparable to parameter counts; H must be reported as "CEI v3 C + XGBoost(R_s rounds) offset", never as a parameter-matched CEI.

### 3.4 Memory and runtime (estimate)

CEI stage: C's cost plus one `[graphs, 10]` tensor per batch — negligible. XGBoost: 6 boosters per seed (1 full + 5 OOF), 18 in total, each up to 300 rounds × 10 classes on 10,000 × |`projection_layout`| features with `hist`. No recorded timing exists for this configuration in the repository; it is unmeasured and must be reported from the smoke. Feature-matrix memory: 10,000 + 5,000 + 5,000 rows × |layout| float32; |layout| is fixed by the preprocessing vocabulary (391 tokens × 7 columns plus kinds, context, 15 relations × 8, 17 triples) and is reported from the run.

### 3.5 Decision rules and wording

Two pre-registered contrasts, each the v3 §6.2 three-condition rule, paired on the same screen rows and same seed index:

- **H vs C** (family of §7, m = 6): wording "the XGBoost+CEI hybrid beats CEI v3 C on this screen".
- **H vs X** (its own family, m = 1, uncorrected 95% CI): X_s and H_s pair by seed; wording "the hybrid beats XGBoost alone on this screen".

Forbidden wording (U6): "the GNN caught up", "CEI matches XGBoost". The hybrid is its own row; C's and X's rows stay as they are. A secondary, non-decisive report gives the share of |m| in the total absolute logit mass per patient (analogous to the v3 §6.2 absence-share statistic), so readers can see how much of H is the tree offset.

### 3.6 TDD obligations (item 3)

1. Fold assignment is patient-grouped, deterministic per seed, covers all 10,000 rows exactly once; every train row's margin comes from a booster whose training IDs exclude that row's subject.
2. Booster fits reject any dev/screen/validation/test sample ID in their training set; margin arrays are hashed and replayed exactly; `R_s` comes from the dev grid with the smaller-rounds tie rule.
3. The hybrid network's node/edge/pair/absence inputs are byte-identical to C's for the same batch; `return_parts["base_margin"]` reconstructs logits within 1e-5; with an all-zero margin table H's forward equals C's forward on identical weights.
4. Screen scoring refuses to run without bound margin hashes and refuses a margin table whose row IDs differ from the screen ID hash; the X reference row is scored from the same `model.ubj` hash.

## 4. Item 5 — validation-tuned per-class logit offsets (U7)

### 4.1 Mechanism

For a frozen checkpoint with screen logits `z ∈ R^{n×10}`, an offset vector `δ ∈ R^10` changes the prediction to `argmax_c (z_c + δ_c)`. `δ` does not change the model, its checkpoint, or the logit decomposition; it is a decision-rule change, reported as a separate arm **O** = "C + δ".

### 4.2 Fitting protocol (validation fold; opened by U7 for this item only)

- **Checkpoints.** The three frozen arm-C checkpoints (winning K). Offsets are fitted on **C only** (not on extension arms; §10 lists the alternative and why it is rejected: "best extension arm" cannot be chosen without reading the screen).
- **Validation scoring.** The full 4,254-row validation fold (v2 binding `counts.validation = 4254`) is tensorized and scored once per C checkpoint to obtain validation logits. This is the first and only validation scoring in the v3 programme. Test is never loaded.
- **Target.** Validation visit-level macro-F1 over the fixed ten labels, `zero_division=0` (`train.py` lines 375–377 semantics, applied to argmax of `z + δ`).
- **Optimizer (fixed).** Coordinate ascent. Grid `Δ = {−2.0, −1.9, …, +2.0}` (41 values, step 0.1). Start `δ = 0`. One sweep visits classes in label order 0..9; for each class, set `δ_c` to the grid value maximising validation macro-F1 with all other coordinates fixed; ties → the value with the smallest |δ_c|, then the smaller value. Exactly 5 sweeps, no early stop, no restarts, no other grid. One `δ` per checkpoint (per seed).
- **Freeze.** Each `δ` is stored with its checkpoint SHA-256, the validation ordered sample-ID hash, validation logits SHA-256, grid, sweep count and the resulting validation macro-F1 (reported as a tuning score, not a result); the record's SHA-256 is bound before any screen read.

### 4.3 Screen comparison

Arm O = the same three C checkpoints with their `δ` applied to the stored screen logits (no re-inference). Contrast **O vs C**, paired on identical checkpoints and screen rows, v3 §6.2 rule with §7 correction (m = 6). Wording: "validation-tuned per-class offsets improve CEI v3 C's screen macro-F1". A win says the class decision thresholds were mis-set for macro-F1 under the training weights, not that the model learned more.

### 4.4 Consequence for fold roles (binding)

From the first item-5 validation scoring onward, **validation is a tuning fold** of this programme. It can no longer be reported as a held-out result for **any** arm (A, B, C, E2, H, E6 or O), because a reported model choice (`δ`) has been fitted on it. Any later held-out claim needs the closed test fold and a separate explicit approval; nothing here requests that. The screen remains the only decision fold; the OLD dev fold remains the selection fold.

### 4.5 Parameter, memory and runtime

Zero trainable parameters (10 fitted constants per checkpoint). Runtime: 3 validation inferences (about 4,254 rows each) plus 3 × 5 × 10 × 41 macro-F1 evaluations on cached logits — seconds; memory negligible. Estimate, unmeasured.

### 4.6 TDD obligations (item 5)

1. The fitter accepts only validation-labelled rows (rejects train/dev/screen/test IDs), runs exactly 5 sweeps over the fixed grid in label order, is deterministic, and its tie rule is exercised.
2. The freeze record hashes checkpoint, validation IDs, logits and `δ`; replay recomputes `δ` bit-for-bit.
3. Screen application: `δ = 0` reproduces C's screen predictions exactly; the O scorer refuses an unbound or mismatched `δ`; the test fold loader is never called (guard test).

## 5. Item 6 — `comorbid_with` and the bidirectional edge view (U8)

### 5.1 How the graph and CEI v2 treat relations today

- Producer: `comorbid_with` links two prior-diagnosis nodes coded in the same past encounter and is emitted **in both directions** (`graph.py` lines 254–259: `edge(left, right, 'comorbid_with', True)` and `edge(right, left, ...)`), with no payload (no `delta`, `interval_hours`, `last_seen_hours`, `prior_encounters`). It is an INFORMATIVE relation (`__init__.py` lines 34–36).
- Tensorization: the bidirectional view appends a `rev:` edge only for `REVERSIBLE_RELATIONS` (`tensorize.py` lines 41–43), after the forward block (lines 348–358); `comorbid_with` is explicitly excluded because it is already symmetric (line 39). Relation vocabulary grows 15 → 24 and triples 16 → 31 (lines 76–93); `edge_attr` width grows from 22 to 31 (relation one-hot + 7 payload columns, line 359).
- CEI v2 edge block: per-edge context = relation embedding + triple embedding + payload projection (`cei_gnn_v2.py` lines 187–188); vote = `edge_source(h[src]) + edge_target(h[dst]) + edge_context_vote(context)` (line 190); gate = `sigmoid(edge_context_gate(context))` (line 191), a function of context only; messages summed to target nodes by `EdgeEvidenceAggregator` (`methods/cei_gnn.py` lines 10–23, `flow="source_to_target"`, `aggr="add"`), then per graph with denominator `1 + Σ gates` (lines 192–195, 204). There is no message passing: `h` never receives neighbour information.

Consequences. (a) A `comorbid_with` edge already produces two votes per unordered pair (one per direction), each **endpoint-additive** in `h_i` and `h_j` with a gate that is constant for every comorbid edge in the dataset (identical relation, triple and all-missing payload). The model cannot express "diagnosis i together with j" beyond the sum of the two diagnoses' own transforms. (b) The bidirectional view does nothing to `comorbid_with`. (c) For CEI, the bidirectional view's stated purpose — routing for message passing (`tensorize.py` lines 34–40) — does not apply; it adds one endpoint-swapped vote per reversible edge with its own `rev:` relation/triple embeddings, i.e. edge-vote symmetrisation, not information.

Aggregate count over the fixed 10,000-row TRAIN sample (read-only, counts only; script and output in `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/.worktrees/_runs/spec-ext-1/count_comorbid.py` and `count_comorbid.out`): 4,359 graphs have a prior-diagnosis node, 3,534 have two or more, **3,307 graphs (33.1%) have at least one `comorbid_with` edge**; 34,712 directed `comorbid_with` records (7.1% of the 489,471 forward-view edge records) = 17,356 unordered pairs; the largest single graph has 102 directed records (51 pairs). The reversible relations sum to 393,001 records, so the bidirectional view raises the edge count to about 882,472 (1.80×) over the sample.

### 5.2 Arm E6a — bidirectional view

Definition: C with `--edge-direction bidirectional`; everything else identical (K, data, seeds, selection). The relation table, triple table and payload projection grow with the view (`plugin_cei_gnn_v2.py` line 43 sizes relations from `relation_count`, `methods/base.py` lines 26–32). v2-schema delta: +4,224 parameters (9 × 128 relation rows + 15 × 128 triple rows + 9 × 128 projection columns), total 96,524 before v3 blocks; all active. Preprocessing state is unchanged (forward triples are reused, `tensorize.py` lines 85–93), so the same knot table, item universe and K apply. Memory/runtime estimate: edge intermediates scale with 1.80× edges; time expected between 1.0× and 1.8× the C stage; unmeasured.

GraphXAI. The edge mask has one entry per edge of the arm's own tensorization (`cei_gnn_v2.py` lines 153–155); for E6a the list is `[forward block | reverse block]` (`tensorize.py` line 350: appended strictly after the forward block so forward ids are stable). Therefore the v3 §4.5 rule ("preserve original edge order, list and mask shape exactly") applies to the bidirectional list: mask length = F + R; a forward-length mask is rejected; the exporter records `edge_direction`, F and R; a reverse edge is attributed as its own item (relation `rev:*`) and is never merged into its forward edge — a forward+reverse sum may be reported only as a derived, labelled quantity. Masking a forward edge to zero does not mask its reverse twin.

### 5.3 Arm E6b — comorbid pair term

Definition: C plus one new evidence block over **unordered comorbid pairs**. From the forward edge list, take every edge with relation `comorbid_with`, map to the unordered pair `(min(src,dst), max(src,dst))` and deduplicate (both directions are present, §5.1). For each pair `(i, j)` in graph G:

    q_ij   = tanh(P h_i) ⊙ tanh(P h_j)              P: hidden × 16, no bias
    v_ij   = V q_ij + b_V                           V: 16 × 10
    g_c    = sigmoid(γ_c)                           γ: one gate logit per class (shared by all comorbid pairs)
    comorbid_Gc = Σ_pairs g_c v_ijc / (1 + Σ_pairs g_c)

added to the logits as a fifth block alongside node/edge/pair/absence. Empty pair set → block exactly zero, denominator one. `return_parts` adds `comorbid_contributions`, `comorbid_pairs`, `comorbid_gates`, `comorbid_denominator`; reconstruction within 1e-5. The block is not an edge: it lives outside the GraphXAI edge mask like the absence block (v3 §4.5), and a zero edge mask removes only the edge block. The product form is chosen because the missing capability is an interaction (§5.1(a)); the v2 within-visit product result (`interaction_useful=false`) concerned complaint/measurement/vital pairs, not diagnosis pairs, and does not transfer. v2-schema delta: 2,048 + 170 + 10 = **+2,228 parameters**, total 94,528 before v3 blocks. Memory: ≤ 51 pairs per graph in the sample (vs. up to 5,741 within-visit pairs, v3 §4.5), negligible. Time ≈ C stage; unmeasured.

E6a and E6b are separate arms; a combined "both" arm is not run (it would be a third contrast and cannot be attributed).

### 5.4 Decision rules

E6a vs C and E6b vs C: v3 §6.2 rule with §7 correction (m = 6). Wording: "the bidirectional edge view (the comorbid pair term) improves CEI v3 C on this screen". Secondary, non-decisive: the patient-mean absolute-evidence share of the comorbid block (E6b), defined as in v3 §6.2 for the absence block, and the fraction of screen graphs with a non-empty comorbid pair set.

### 5.5 TDD obligations (item 6)

1. Pair extraction: both directions collapse to one unordered pair; non-comorbid relations are ignored; pairs never cross graphs; a graph without comorbid edges yields an exact-zero block and denominator one.
2. E6b reconstruction with the new keys; existing keys and shapes unchanged; parameter delta equals the analytic +2,228; block is outside the edge mask (zero mask leaves it intact).
3. E6a: bidirectional tensorization of a fixture graph yields F + R edges with the reverse block appended after the forward block; mask length F + R accepted, length F rejected; K-freeze and preprocessing hashes equal C's.

## 6. Common arm table

| Arm | Base | Difference from C | Trained? | Screen scorings | Control(s) |
|---|---|---|---|---|---|
| C | v3 | — (winning K) | v3 §11.2 | 3 | — |
| E2w | C | hidden 256 | 3 stages | 3 | C |
| E2d | C | encoder_depth 2 | 3 stages | 3 | C |
| H | C | + XGBoost margin offset | 3 stages + 18 boosters | 3 | C, X |
| X | — | XGBoost alone, 10k sample, dev-selected rounds | 3 boosters (shared with H) | 3 | reference only |
| O | C | + validation-fitted δ on C's checkpoints | none | 3 (from stored logits) | C |
| E6a | C | `--edge-direction bidirectional` | 3 stages | 3 | C |
| E6b | C | + comorbid pair block | 3 stages | 3 | C |

All arms: seeds 1234/2025/7, the 10,000-row TRAIN sample of sample seed 1234, OLD-dev 5,000-row checkpoint selection, 40 epochs, patience 40, Adam lr 1.79e-3, weight decay 4.3e-5, batch 128, dropout 0.3, `sqrt_inverse` weights, `--edges all`, Top-10 (v3 §6.1), and the frozen K and knot table of v3 §11.2.

## 7. Multiplicity policy (fixed before any result)

Six primary contrasts share one control (C) on one screen: E2w, E2d, H, O, E6a, E6b vs C. Conditions 1 and 2 of v3 §6.2 (per-seed strict win; seed-mean win) are unchanged. Condition 3 is Bonferroni-corrected with m = 6: from the same 1,000 paired patient-cluster bootstrap resamples with seed 2026 (identical patient draws for every arm and seed), sort the 1,000 averaged deltas and take the `max(1, ⌊1000 · 0.05 / (2 · 6)⌋)` = **4th smallest** as the lower bound; the decision requires it to be strictly above zero. The uncorrected 95% lower bound (25th smallest) is reported next to it as "nominal". Wording: an arm passing all three corrected conditions "beats C on this screen (family-corrected, m = 6)"; an arm passing only with the nominal bound is reported as "nominal win, not family-corrected" and is not a positive decision. H vs X is a separate one-contrast family (m = 1, 25th smallest). No contrast may be added to or removed from the family after any screen result is seen; if an arm is not run, m stays 6 (conservative). This is a deliberate choice of a crude but transparent correction over none; 1,000 resamples make the corrected tail coarse (4 samples), which is accepted and stated rather than fixed by changing the pre-registered resample count.

## 8. Combined run plan, stage count and time

Order (each step completes before the next starts; all outputs to new paths):

1. v3 §11.2 steps (i)–(iv): knot tables, 9 C stages (K grid), K-freeze record, A and B stages — 15 CEI stages.
2. Item 3 XGBoost: per seed, full booster X_s with dev round selection, then 5 OOF boosters; margin tables hashed — 18 booster fits, 3 dev scorings of X.
3. Extension CEI stages, dev-selected, in any order or in parallel subject to memory: E2w ×3, E2d ×3, H ×3, E6a ×3, E6b ×3 — 15 CEI stages.
4. Freeze: all checkpoints hashed; item-5 validation scoring of C's three checkpoints, δ fitted and frozen.
5. Screen scored once per frozen checkpoint: A, B, C, E2w, E2d, H, E6a, E6b (24 inferences) + X (3 booster predictions) + O (3, from C's stored screen logits) = 30 screen rows; then the decision analysis once.

Totals. CEI 40-epoch stages: 15 + 15 = **30**. Baseline time from the nine recorded v2 stage times (mean 125.38 s, range 105.1–138.4 s, v3 §6.3): 30 × 125.38 = **3,761.4 s (62.7 min)**, range 30 × 105.1 = 3,153 s (52.6 min) to 30 × 138.4 = 4,152 s (69.2 min). Not included, because unmeasured: the v3 PLE/absence overhead on every stage, E2w's width factor (1–3× on 3 stages), E6a's edge factor (1–1.8× on 3 stages), 18 booster fits, 6 XGBoost/validation scorings, 30 screen scorings and orchestration. A rough upper envelope using the stated factors on those six stages alone adds up to (2 × 3 + 0.8 × 3) × 125.38 = 8.4 × 125.38 = 1,053.2 s (17.6 min) to the mean baseline, giving 62.7 + 17.6 = 80.3 min at the mean stage time and 69.2 + 8.4 × 138.4 / 60 = 88.6 min at the slowest recorded stage time, before the v3 overhead and the other overheads listed; this is an estimate, and the approved smoke must replace it with measured numbers before the full run is launched. Memory: apply the v3 §4.5 preflight per arm; E2w and E6a are the two arms whose worst-batch bound must be recomputed.

## 9. Implementation unit split (for the testing phase; G3 still closed)

Interface obligations the **v3 core unit** must ship first (so extension units own only new files): (a) `encoder_depth` constructor argument and `--layers` mapping (§2.2); (b) a graph-level `base_margin` input added at the logit sum with `return_parts["base_margin"]` (§3.2); (c) an evidence-block hook so a new block module can register `(contributions, denominator, parts_keys)` without editing the network file (§5.3); (d) `edge_direction` passed through unchanged to the v3 adapter (already available via `relation_count`). Module names below are placeholders under `comparison/standardized/clinical_graph_v2/`; the v3 implementation plan fixes the final names.

| Unit | Owns (new files only) | Red/green steps (≤ 3) | Depends on | Parallel with |
|---|---|---|---|---|
| X1 size | `cei_v3_ext/arms_size.py`, `tests/test_cei_v3_ext_size.py` | 1 depth-1 parity + reconstruction; 2 parameter deltas + run_config; 3 v2 adapter still rejects layers>1, K-hash guard | v3 core (a) | X2–X5 |
| X2 hybrid | `cei_v3_ext/hybrid_xgb.py`, `cei_v3_ext/hybrid_rows.py`, `tests/test_cei_v3_ext_hybrid.py` | 1 fold assignment + leakage guards; 2 booster/round selection + margin hashing; 3 zero-margin parity + reconstruction + screen guards | v3 core (b) | X1, X3–X5 |
| X3 offsets | `cei_v3_ext/offsets.py`, `tests/test_cei_v3_ext_offsets.py` | 1 fitter (fold guard, grid, sweeps, ties); 2 freeze/replay hashing; 3 δ=0 parity + scorer guards + no-test guard | none (works on stored logits) | X1, X2, X4, X5 |
| X4 bidirectional | `cei_v3_ext/arms_bidirectional.py`, `tests/test_cei_v3_ext_bidirectional.py` | 1 F+R tensorization order + parameter delta; 2 mask length F+R accepted / F rejected; 3 reverse-edge attribution kept separate | v3 core (d) | X1–X3, X5 |
| X5 comorbid | `cei_v3_ext/comorbid_block.py`, `tests/test_cei_v3_ext_comorbid.py` | 1 pair extraction (dedupe, no cross-graph, empty → zero); 2 block maths + new parts keys + +2,228 delta; 3 outside edge mask | v3 core (c) | X1–X4 |
| X6 study | `cei_v3_ext/study.py` (plan, bindings, screen scorer, family-corrected analysis), `tests/test_cei_v3_ext_study.py` | 1 plan/binding refusal (arm drift, K-hash, overwrite, validation/test guards); 2 screen-once-per-checkpoint + margin/δ binding; 3 bootstrap family correction (4th smallest, m fixed) | X1–X5 | — |

Each unit's `red:` commit must fail on an assertion (not an import); each `green:` commit is the minimal code. No unit edits v1/v2 files, `cei_v2_study.py`, or the v3 core files after the v3 core lands.

## 10. Open decisions for the user (each with a recommendation)

1. **Item 2: two fixed size arms (E2w, E2d) or one dev-selected?** Recommendation: both fixed arms (6 stages, ~12.5 min baseline). Alternative: pre-register an OLD-dev selection between them exactly like v3 §11.2 (seed-mean dev macro-F1 of dev-selected checkpoints; tie → smaller model) and score only the winner, which lowers m to 5.
2. **Hybrid offset scale: fixed 1 or a learned per-class scalar?** Recommendation: fixed 1. A learned scale adds 10 parameters, can silently down-weight the trees, and blurs the "tree offset + CEI evidence" reading; if it is wanted, it must be a separate arm.
3. **Offsets (item 5) on C only, or on every extension arm as well?** Recommendation: C only. Fitting on all arms would score validation for 8 arms and add 7 non-decisive contrasts; fitting on "the best" arm is post-hoc and rejected.
4. **Multiplicity: Bonferroni on condition 3 with m = 6 (§7), or no correction?** Recommendation: correct as in §7 and report the nominal bound alongside.
5. **XGBoost rounds: dev-selected per seed (§3.2 step 2) or fixed 300?** Recommendation: dev-selected; it mirrors CEI's epoch selection and the existing `tabular_control` policy.
6. **E6b pair form: product (§5.3) or additive?** Recommendation: product; the additive form is already what the edge block computes (§5.1(a)) and would test nothing new.

None of these opens a real run; G3 remains closed and must be requested separately with a fresh output root and the memory preflight.
