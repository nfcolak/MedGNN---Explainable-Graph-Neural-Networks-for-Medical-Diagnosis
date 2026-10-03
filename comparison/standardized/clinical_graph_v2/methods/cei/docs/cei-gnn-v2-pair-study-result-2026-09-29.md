# CEI-GNN v2 within-visit pair study — 2026-09-29

## Summary

This dev-only exploratory screen asked whether the rank-16 multiplicative within-visit pair head improves dev macro-F1 over matched additive and pair-off arms.
Decision: `interaction_useful=false`; benefit not established. This is not evidence of no benefit.
Checkpoint selection and the decision both use the same dev fold, so these results are exploratory, not confirmatory.

## Study setup

- Source revision: `e238d097`.
- Cohort/task: the existing `clinical_graph_v3_membership_max6_20260923` cohort, patients with at most 6 total visits, and the 10 most frequent diagnoses by train-fold frequency (Top-10; other labels dropped). Task: visit-level diagnosis classification.
- Fixed sample seed: 1234. Each arm used 10,000 train visits and 5,000 dev visits (4,841 patient clusters); model seeds were 1234, 2025, and 7.
- Three arms: `product` computes `tanh(Uh_i) * tanh(Uh_j)` for eligible within-visit evidence pairs; `additive` computes `tanh(Uh_i) + tanh(Uh_j)` for the same pairs; `off` removes the pair term. Evidence node kinds are complaint, measurement, and vital; pairs do not cross visits or graphs.
- Each of the nine full runs trained for 40 epochs; the checkpoint was selected by dev macro-F1. Validation rows were tensorized but not scored; test rows were parsed but not tensorized.
- Each arm has 92,300 total parameters. Product and additive have 92,300 active and 0 inactive; off has 90,022 active and 2,278 inactive parameters.

## Pre-registered decision rule

Spec §6, with amendment A10 in §6.1:

> Primary metric: visit-level dev macro-F1 of the selected checkpoint. The interaction is declared useful only if ALL hold:
> 1. `product > additive` on each of the three seeds.
> 2. Seed-mean `product` > seed-mean `additive` and > seed-mean `off`.
> 3. Paired patient-cluster bootstrap (1,000 resamples, seed 2026; each resample recomputes every arm-seed macro-F1 on the same resampled patients, then averages the delta over seeds) gives a 95% interval for `product - additive` with lower bound > 0.
>
> Otherwise the report says the benefit is not established.

Reconciliation of the stated rule and its implementation:

| Condition | Spec | Analysis implementation | Match |
|---|---|---|---|
| Product vs additive by seed | Strictly greater on each seed | `all(product > additive)` | Yes |
| Product vs additive mean | Product mean strictly greater | Included in strict mean-above-both-controls check | Yes |
| Product vs off | Mean-above-off remains required; A10 says directional, with off interval secondary | Strict mean comparison; no product-off interval in decision | Yes, directional per A10 |
| Bootstrap | Patient-cluster paired interval for product−additive; lower bound > 0 | Checks lower endpoint of 95% interval | Yes |
| Ties | Do not pass strict comparisons | Strict `>` throughout | Yes |

ADR-013's wording (“beats additive and off in every seed”) is stricter on product-vs-off than spec §6/A10, which requires a product mean above off and treats product-vs-off as directional. The result fails under both readings: product is below additive on two seeds, its mean is below additive, and the decisive product−additive interval includes zero; it also does not beat off on every seed.

## Results

Primary metric: dev macro-F1.

| Arm | Seed 1234 | Seed 2025 | Seed 7 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| product | 0.649578 | 0.647191 | 0.647217 | 0.647995 ± 0.001371 |
| additive | 0.647950 | 0.650263 | 0.649069 | 0.649094 ± 0.001156 |
| off | 0.647470 | 0.647716 | 0.644120 | 0.646435 ± 0.002009 |

Paired patient-cluster bootstrap; point estimates and intervals are macro-F1 units; 1,000 resamples, RNG seed 2026. Intervals are conditional on these three fixed seeds and selected checkpoints.

| Contrast | Bootstrap point | 95% CI | Decision role |
|---|---:|---:|---|
| product − additive | −0.001099 | [−0.004715, 0.002709] | Decisive |
| product − off | +0.001560 | [−0.002816, 0.005586] | Secondary, directional only |

| Decisive condition | Result |
|---|---|
| Product > additive on every seed | FAIL (seed deltas: +0.001628, −0.003072, −0.001853) |
| Product mean > additive mean and off mean | FAIL (0.647995 < 0.649094; 0.647995 > 0.646435) |
| Product−additive bootstrap 95% lower bound > 0 | FAIL (−0.004715) |
| Overall `interaction_useful` | `false` — benefit not established |

## Secondary, non-decisive outputs

Class-level means average the seedwise per-class deltas across the 10 classes. False positives are aggregate counts across classes, by arm and seed.

| Contrast | Mean Δrecall | Mean Δprecision | Classes with improved F1 | Median ΔF1 |
|---|---:|---:|---:|---:|
| product vs additive | +0.000863 | −0.005174 | 4/10 | −0.003362 |
| product vs off | +0.002515 | +0.000528 | 7/10 | +0.003945 |

| Arm | False positives, seeds 1234 / 2025 / 7 |
|---|---:|
| product | 1,829 / 1,843 / 1,844 |
| additive | 1,825 / 1,848 / 1,829 |
| off | 1,850 / 1,855 / 1,859 |

Pair-count bands are descriptive, non-decisive. Per rule, any subgroup with fewer than 10 patients is reported as `<10` and its metrics are suppressed.

| Pairs per graph | Graphs | Patients | Product mean macro-F1 | Additive mean macro-F1 | Off mean macro-F1 | Product−additive |
|---|---:|---:|---:|---:|---:|---:|
| 0 | <10 | <10 | suppressed | suppressed | suppressed | suppressed |
| 1–179 | 2,512 | 2,467 | 0.644689 | 0.645350 | 0.640675 | −0.000661 |
| 180–1,044 | 2,454 | 2,415 | 0.613056 | 0.612759 | 0.613928 | +0.000298 |
| >1,044 | 32 | 32 | 0.508627 | 0.554719 | 0.507172 | −0.046092 |

Selected epochs below are zero-based indices as stored in the bindings and normalized by the validator; “epoch N of 40” gives the corresponding one-based epoch number. No arm selected the last epoch.

| Arm | Seed 1234 | Seed 2025 | Seed 7 |
|---|---|---|---|
| product | 26 (epoch 27 of 40) | 18 (epoch 19 of 40) | 23 (epoch 24 of 40) |
| additive | 16 (epoch 17 of 40) | 23 (epoch 24 of 40) | 29 (epoch 30 of 40) |
| off | 29 (epoch 30 of 40) | 28 (epoch 29 of 40) | 25 (epoch 26 of 40) |

Recorded stage wall seconds:

| Arm | Seed 1234 | Seed 2025 | Seed 7 |
|---|---:|---:|---:|
| product | 135.9 | 132.1 | 132.3 |
| additive | 138.4 | 132.9 | 138.2 |
| off | 107.4 | 106.1 | 105.1 |

## Verification

The independent validator verdict was **VERIFIED**. The library validator completed successfully and reported common pair counts across arms. It performed 166 independent boolean checks, all passed; checked identical dev rows and labels across arms, source hashes, exact checkpoint replay (9/9), and arm parity. The output-root manifest covered 86 files and was unchanged across analysis and validation.

`test_evaluated=false` is recorded in the embedded binding in each `result.json`. Validation never scored: it was tensorized but not evaluated, and the validator found no validation/test prediction files. The validator performed only its authorized read-only dev replay; it did not score validation or test.

## Execution record

The three-mode smoke completed before the full study. The full run completed all nine arm-by-seed stages: 1,572.99 seconds wall time (reported as 1,573 s), maximum RSS 2,823,782,400 bytes (about 2.6 GiB), exit code 0.

Two earlier launch attempts stopped before any full stage was trained: one due to a monitoring-contract blocker and one due to a pre-flight command error. Neither produced full-stage artifacts. An earlier agent printed the smoke journal (256 train / 128 dev samples, 2 epochs) into its own tool output; those smoke numbers were never reported as performance results and are not part of the decision.

## Limitations

- Checkpoint selection and decision use the same dev fold; this is exploratory, not confirmatory.
- Bootstrap intervals are conditional on three fixed seeds; seeds are not resampled.
- Product `tanh(Uh_i) * tanh(Uh_j)` is bounded in [−1, 1], while additive `tanh(Uh_i) + tanh(Uh_j)` is bounded in [−2, 2]. The scale/gradient confound means a product gain could not be attributed purely to non-additive capacity; no scale-matched arm was added.
- Validation was tensorized but never scored; test was parsed but never tensorized or scored.
- Pair-memory gates rest on the train-fold scan and are not enforced in code.
- Product-vs-off is directional only; off is a structural ablation with fewer active parameters, not a parameter-matched control.
- GraphXAI edge masks do not attribute within-visit pairs; pair contributions are model accounting, not causal importance.
- The result applies only to this cohort, these seeds, and these selected checkpoints.

## Not done / next decisions

- No validation-fold evaluation; that requires separate explicit user approval.
- No test evaluation.
