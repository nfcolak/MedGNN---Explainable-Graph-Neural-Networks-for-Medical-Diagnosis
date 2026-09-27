# Round2 methods: verified research and screening audit

Audit date: 2026-09-27. Inspected code: `f94e4076b0f1425eaae7c418a15eb5876a19ab4c` (`feature/round2-methods`).

## Decision summary

The five additions are **project-specific adaptations of established architectures**, not demonstrated inventions or exact source-paper reproductions. The existing single-seed development screen is complete. TokenFusion has the highest observed candidate score, followed by GraphGPS, but a strict same-source comparison against ProtGNN is blocked by an older control runner. None of the five plugin explanation routines currently establishes faithful, target-specific explanation of the complete prediction.

This is a research/documentation deliverable, not a repair or an experiment. No model tests, inference, training, cache generation, raw-patient inspection or held-out-test evaluation was performed in this audit. No model source, scientific binding, historical output or method identifier was changed. Historical “55 passed” evidence was not rerun and is not a current verification claim.

## 1. What the names legitimately mean

All code anchors below are relative to `comparison/standardized/clinical_graph_v2/` at the inspected commit.

| Internal ID | Defensible description | Mechanism boundary |
|---|---|---|
| `gmt` | Set-Transformer-style pooling on a clinical relation/payload GNN, inspired by GMT | Learned-seed pooling, self-attention and second pooling are present. Its K/V projections are linear; original GMT §3.2 Eq. (6) constructs graph-aware K/V with GNNs. Do not transfer GMT-specific theoretical guarantees to this implementation. |
| `gps` | GraphGPS-style interleaved local-message/global-attention GNN with random-walk encoding | Preserves the local/global combination. Uses a clinical local operator, dense within-graph attention and an additive projection of eight-step undirected RWSE. Do not inherit the paper's scalable-attention complexity claim for this dense implementation. |
| `vnode` | OGB-style virtual-node clinical GNN with a custom concatenated readout | Repeated virtual-node injection/update is present. Post-convolution pooling for virtual-node updates and concatenating node mean with the virtual state differ from the inspected OGB example. |
| `tokenfusion` | Wide/deep clinical GNN plus count-based linear and optional count-MLP heads | Uses a shared graph encoder and transformed token counts. The internal name is not the CVPR 2022 TokenFusion vision method. |
| `labelattn` | CAML-style classwise attention readout over clinical graph nodes | Adapts code-specific attention from clinical text; uses multiclass cross-entropy and an additional mean path rather than reproducing CAML's complete text/multilabel model. |

Primary-method grounding:

- GMT's graph-dependent pooling construction is in §3.2; the local simplification is explicitly disclosed in `methods/plugin_gmt.py:1-6,37-39`.[2]
- GraphGPS's recipe combines positional/structural encoding, local message passing and global attention; local implementation routes are `methods/plugin_gps.py:110-176`.[19]
- OGB's pinned constructor and forward establish the virtual-node reference; local injection/update/readout is `methods/plugin_vnode.py:114-133`.[23]
- Wide & Deep supplies the joint linear/deep mechanism, and CAML supplies code-specific attention; local readouts are `methods/plugin_tokenfusion.py:106-131` and `methods/plugin_labelattn.py:110-134`.[4][18]
- The published CVPR TokenFusion replaces uninformative tokens using inter-modal information, a different mechanism from this project's count/GNN model.[10]

“Not found in the consulted papers” does not establish novelty. These adaptations need mechanism ablations and a wider prior-art search before a publication-novelty claim. Shared use of `_RelationPayloadLayer` does **not** make the experiment readout-only: GPS interleaves global attention with local messages, and virtual nodes alter intermediate node states. Widths also differ.

## 2. Saved development-screen evidence

Task: max-six-visits eligibility, TRAIN-derived Top-10 labels, 10,000 training samples, 5,000 unused-TRAIN development samples, model/sample seed 1234. Validation count 4,254 is bound but was not evaluated in these runs. Recorded patient-disjointness is a manifest claim; this audit did not re-derive it from individual records.

| Method | Dev macro-F1 | Selected epoch | Epochs completed | Recorded seconds |
|---|---:|---:|---:|---:|
| TokenFusion (internal label) | 0.642403 | 15 | 31 | 131.4 |
| GraphGPS-style | 0.641107 | 21 | 31 | 443.9 |
| Virtual node | 0.639877 | 27 | 35 | 204.0 |
| ProtGNN historical control | 0.638844 | 15 | 31 | 200.9 |
| Label attention | 0.633944 | 21 | 31 | 223.8 |
| GMT-inspired | 0.630065 | 27 | 32 | 217.5 |

All six result files report completion. Supervisor checks independently confirmed standalone/embedded binding equality, score equality with the first maximum in history, selected epoch, epoch-log/history equality and a completed log event followed by `EXIT=0`. GraphGPS finished at epoch 31, selecting epoch 21; it is not still running.

### Comparison gates

- Recorded graph/membership fingerprints, target/preprocessing bindings, ordered labels, sample hashes, seed, input view and dev-selection settings agree across all six arms. This is metadata parity, not independent raw-artifact rehashing or temporal-validity proof.
- All five candidate source inventories agree and their 31 recorded files match the inspected tree.
- The ProtGNN control has an older 26-file inventory. `methods/__init__.py`, `methods/base.py` and `train.py` hashes differ; the five plugin files are absent from its inventory. `methods/protgnn.py` matches. Behavioral equivalence of the older runner/base is **not established**.
- Common stopping limits are a 40-epoch cap, patience 10, min-delta 0.005 and stopping start index 20. The saved runs ended before the cap. Selected epoch alone cannot establish convergence, continuing improvement or a binding epoch cap.
- Shared declared optimizer settings include Adam, batch size 128, learning rate 0.00179 and weight decay 0.000043. Candidates use dropout 0.3; the historical control uses 0.51. Widths and capacities differ: GMT 76/377094 parameters, GPS 80/331802, TokenFusion 64/148484, virtual node 104/380266, label attention 128/399604, control 128/399884. Equal stopping limits are not equal compute or equal capacity.
- The source artifact manifest marks `temporal_clean=false`; shared input metadata does not remove its availability/proxy limitations.

Thus the table is a descriptive, single-seed, dev-selected screen. It establishes neither a generalization win over ProtGNN nor a faithful-explanation result. Previous validation-selected ProtoNode/ProtGNN runs are a separate evidence set and are not pooled here.

### Private local evidence locations

The existing run root is `comparison/standardized/clinical_runs_v3_challenger_sample10k_max6_top10_20260927/`. Candidate leaves are `screen/a_gmt`, `screen/a_gps`, `screen/a_tokenfusion`, `screen/a_vnode`, `screen/a_labelattn`; the control is `tune/t4_protgnn_dev`. The audit used each leaf's `binding.json`, `result.json` and adjacent `.log`, plus the referenced graph manifest. These medical research artifacts remain local and are deliberately not bundled with this report. No patient identifiers, predictions, checkpoints or raw run files are published here.

## 3. Confirmed code findings

| ID | Finding and source anchor | Impact / boundary |
|---|---|---|
| C1 | `methods/base.py:201-206`, reached through `train.py:149-156,174-175`, maps every explicit boolean string outside `1,true,yes,on` to false. | A misspelling such as `tru` silently disables an option. One shared configuration issue, not five separate defects. Effective configuration is recorded; this does not prove an existing run used a typo. |
| E1 | GMT `explain()` averages first-pooling attention (`plugin_gmt.py:176-185`), while prediction also uses later pooling/classifier layers and the default direct mean path (`166-172`). | Attention diagnostic, not whole-predictor or target-class attribution. |
| E2 | GPS explanations average received attention (`plugin_gps.py:209-228`); prediction combines local/global/residual paths and mean readout (`164-192`). | Does not account for all predictive paths or class-specific contributions. |
| E3 | Virtual-node explanations use nonnegative cosine alignment (`plugin_vnode.py:144-150`); prediction classifies both final node mean and virtual vector (`130-133`). | Alignment diagnostic, not prediction decomposition. |
| E4 | TokenFusion explains using unsigned wide weights summed across classes and divided over occurrences (`plugin_tokenfusion.py:142-159`). Prediction uses transformed counts and deep/wide/tab logits (`106-131`). | Not even a signed class-specific decomposition of the transformed wide path, much less the entire predictor. |
| E5 | Label attention returns the maximum attention over classes (`plugin_labelattn.py:144-157`), ignoring classifier weights and its optional mean branch (`110-134`). | Class-agnostic visualization, not an explanation of a chosen diagnosis. |
| D1 | Label-attention entropy sums over nodes then averages classes without averaging graphs (`plugin_labelattn.py:136-140`). | Batch-dependent diagnostic aggregation. The inspected runner does not persist adapter diagnostics, so this is not evidence of damaged saved prediction metrics. |
| E6 | The inspected runner evaluates forward logits (`train.py:521-534`) and trains on logits/auxiliary loss (`876-895`); AST inspection found no `explain()` calls. | Prediction scores are not explanation-fidelity evidence. No checkpoint-backed explanation evaluation was established. |

These are source-level findings; no new numerical runtime witness was executed. The review did not identify a demonstrated normal-run cross-patient routing defect. “No defect found” is not an empirical correctness certificate.

## 4. Corrections applied to sub-agent claims

The supervisor did not adopt the raw reports verbatim:

1. **OGB initialization:** the report called OGB's virtual node randomly initialized. The pinned reference explicitly zero-initializes it at line 164. Zero initialization is not a MedGNN/reference difference.[23]
2. **Epoch-cap claim:** the report inferred that selected epochs 21–27 meant all arms were improving at a 40-epoch cap. Saved histories show completion after 31–35 epochs for these arms. The inference is rejected.
3. **Readout-only claim:** common message-layer code does not imply identical encoders when GPS inserts attention between layers and virtual nodes modify node inputs. Mechanistic isolation requires additional controls.
4. **Initialization versus training:** a zero initial virtual vector does not remain identically zero throughout training; it is trainable. Initial-state observations cannot describe every first layer of every trained forward pass.
5. **Explanation criteria:** unchanged macro-F1 after path removal does not prove zero predictive contribution. Signed per-class logit accounting, magnitude ranking and causal input attribution are distinct. Gradient correlation is a diagnostic, not a gold-standard faithfulness certificate.
6. **“Invented” wording:** replace that label with “addition relative to the inspected reference.” Neither a code mismatch nor absence from a small source set proves literature novelty.
7. **Normalization/RWSE:** the GMT attention block normalizes after residual addition, not “pre-norm.” GPS RWSE symmetrization is explicit in its function docstring, while relation/direction information remains available to its local message branch. Do not call it global information removal from the model.

## 5. Next stage — proposed, not executed

1. In an explicitly authorized testing/implementation phase, fix boolean parsing with assertion-failing red evidence and verify the green behavior. Do not change historical bindings.
2. Establish a same-source ProtGNN control before stronger comparisons. Retain the same input/sample/selection contract and explicitly declare remaining capacity/dropout differences.
3. If continuing predictive screening, compare shortlisted candidates across matched seeds. If the research objective is self-explanation, specify target class, every prediction path, intervention semantics and random/size-matched controls before wiring an explanation evaluator. Exact encoded-state accounting alone is not causal clinical explanation.
4. Keep held-out test evaluation closed. Do not change architecture or perform large retraining merely because a reviewer called a design choice a defect.

Measured planning inputs, not promised future durations: one recorded control run 200.9 s; TokenFusion 131.4 s; GPS 443.9 s. Two additional recorded-duration equivalents for TokenFusion and control total 664.6 s, excluding a same-source seed-1234 control replay, any validation-only stage and any explanation work. Early stopping, contention and launch overhead make this a planning scenario, not a time guarantee. No run is scheduled by this document.

## 6. Review provenance and verification limits

| Agent | Measured model/provider | Effort | Role |
|---|---|---|---|
| r2-literature | stealth/space-bunny-alpha / nous | medium | GMT/GPS/virtual-node primary-source audit |
| r2-originality | stealth/space-bunny-alpha / nous | medium | Wide/deep and label-attention attribution/novelty audit |
| r2-review-graph | gpt-6-sol / openai-codex | high | Independent graph-method source review |
| r2-review-heads | gpt-6-sol / openai-codex | high | Independent prediction-head source review |
| r2-evidence | gpt-6-astra / openai-codex | medium | Aggregate artifact/binding verification |

All ran through Hermes with pinned roles; launcher metadata reports no requested/actual model or effort mismatches. Codex billing metadata says `subscription_included`; Nous billing mode is unreported and no monetary estimate is asserted. Supervisor: current conversation's GPT-6 Astra, responsible for the independent source/artifact checks and corrections above.

Local raw reports and supervisor verification records are retained under the ignored `.worktrees/_runs/` directory. They are not published because they are working records containing local paths and uncorrected reviewer claims. This document is the reviewed synthesis. Its aggregate numbers are a human-readable research summary, not copied medical run outputs. No automated model tests or training claims accompany this documentation-only change.

## Sources

[2] https://arxiv.org/pdf/2102.11533
[4] https://arxiv.org/abs/1606.07792
[10] https://arxiv.org/abs/2204.08721
[18] https://aclanthology.org/N18-1100
[19] https://arxiv.org/html/2205.12454v4
[23] https://raw.githubusercontent.com/snap-stanford/ogb/856f6f51fd53496783e7ad9d7035cd6ce0393516/examples/graphproppred/mol/conv.py
