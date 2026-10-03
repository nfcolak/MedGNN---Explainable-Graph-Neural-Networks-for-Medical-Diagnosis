# GCHM concept-dropout pilot — seed 1234 (measured)

## Verdict

**The candidate did not improve the model. The incumbent GCHM-sqrt stays.**
Do not extend to seeds 1235/1236.

## Scope actually executed

One decision variable: `concept_dropout` 0.0 → 0.15. Artifact, split, label
order, seed (1234), epochs (30), batch size (128), sqrt-inverse loss weighting,
optimizer, learning rate, weight decay and star topology frozen.

| Run | Method | Output |
| --- | --- | --- |
| Wiring | `gchm_concept_dropout` | `native_runs/_wiring/gchm_concept_dropout` (`--limit 512`, 1 epoch, **not benchmark evidence**) |
| Control | `gchm` | `native_runs/gchm_dropout_pilot_v1/control_seed1234` |
| Candidate | `gchm_concept_dropout` | `native_runs/gchm_dropout_pilot_v1/dropout_seed1234` |

Both full runs: `status=completed`, exact validation replay, `test_evaluated=false`.
Fold 2 was never read.

Analysis: `python -m comparison.standardized.gchm_dropout_pilot_v1.analyse`
→ `comparison/standardized/gchm_dropout_pilot_v1/report.json`.

## Parity proof (from bindings, not prose)

- Exactly one binding field differs between the two runs: `method`.
- All **90** source-file hashes are byte-identical across both runs.
- Parameter count identical: **179,486** in both. No capacity was added.
- Contract fingerprint `2a2566e6…dcd9f7c0` (pinned) in both.
- Cohort: 59,607 train / 7,448 validation, ordinal hashes equal in both.

## Control reproduces the incumbent exactly

The control run was launched under the new code to prove the edit did not move
the incumbent. Against `native_runs/gchm_sqrt_v2/seed1234`:

- All 30 epochs' validation metrics identical.
- Selected epoch 7 in both.
- Selected-epoch validation logits **bitwise equal** (max abs diff 0.0).

The `best.pt` SHA differs only because the binding metadata embedded in the
checkpoint records a different output path/command. The weights and predictions
are the same.

## Headline result (validation, n = 7,448)

| Metric | Control (gchm) | Candidate (dropout 0.15) | Δ |
| --- | --- | --- | --- |
| Selected macro-F1 | **0.561467** | 0.550005 | **−0.011462** |
| Final macro-F1 | 0.535263 | 0.528378 | −0.006885 |
| Selected balanced acc | 0.596764 | 0.585776 | −0.010988 |
| Selected accuracy | 0.622046 | 0.608888 | −0.013158 |
| top3 / top5 acc | 0.821026 / 0.888158 | 0.821831 / 0.893663 | +0.0008 / +0.0055 |
| Selected epoch | 7 | 14 | +7 |
| Overfit gap (selected − final) | 0.026204 | 0.021627 | −0.004577 |
| Total false positives | 2,815 | 2,913 | **+98** |

## The mechanism worked; the objective did not

Concept dropout regularized exactly as designed — and that is measurable:

- Training loss is uniformly higher across all 30 epochs (final 1.2191 vs
  1.0041), so the model genuinely cannot lean on single concept identities.
- The best epoch moved from 7 to 14: overfitting onset is delayed.
- The overfit gap shrank from 0.0262 to 0.0216.

But the regularization lowers the ceiling more than it recovers. The candidate's
best epoch (0.5500) is below the control's best (0.5615) and below the control's
own epoch-3 value (0.5505). It never crosses the control's peak at any epoch.

## Precision goal: not met at the macro level

The stated target was **raise precision while preserving rare-class recall**.
Measured across all 30 classes, both moved the wrong way:

- Mean Δrecall: **−0.0110**; mean Δprecision: **−0.0095**.
- 10 rarest classes: mean Δrecall **−0.0161**, mean Δprecision **−0.0196**,
  total false positives **+63**.
- Only **11 of 30** classes improved in F1; median ΔF1 **−0.0074**.

The two classes named in the matched GCHM/XGBoost report did improve sharply —
and this is precisely the cherry-picking risk:

| Class | n | Recall | Precision | FP | ΔF1 |
| --- | --- | --- | --- | --- | --- |
| NSTEMI | 72 | 0.792 → 0.750 | 0.528 → **0.635** | 51 → **31** | +0.0546 |
| Unsp intestnl obst | 118 | 0.763 → 0.636 | 0.462 → **0.573** | 105 → **56** | +0.0273 |

But the cost landed elsewhere, on classes that were not being watched:

| Class | n | Recall | Precision | FP | ΔF1 |
| --- | --- | --- | --- | --- | --- |
| Epilepsy | 97 | 0.804 → 0.794 | 0.821 → 0.658 | 17 → **40** | −0.0929 |
| Sepsis | 40 | 0.550 → 0.600 | 0.275 → 0.198 | 58 → **97** | −0.0685 |
| End stage renal disease | 43 | 0.512 → 0.442 | 0.386 → 0.345 | 35 → 36 | −0.0522 |
| Heart failure | 85 | 0.518 → 0.494 | 0.379 → 0.321 | 72 → 89 | −0.0489 |
| Lower respiratory disease | 369 | 0.393 → 0.485 | 0.578 → 0.482 | 106 → **192** | +0.0160 |

Concept dropout redistributed false positives rather than removing them. It made
the model less certain about sharply concept-identified conditions (Epilepsy's
precision collapse is the clearest case: that class depends on a specific
medication/diagnosis cue, which is exactly what the dropout destroys) and pushed
the freed probability mass onto broad respiratory/septic classes.

## What this rules out, and what it does not

Established by this pilot:

- 0.15 concept dropout at seed 1234 does not improve macro-F1; it costs 0.0115.
- The overfitting the three-seed history showed is real but is **not** primarily
  caused by concept-identity memorization — removing that dependence reduces the
  gap by only 0.0046 while lowering both endpoints.

Not established:

- Whether a smaller rate (0.05) would land differently. This run tested one
  prespecified value, not a rate sweep.
- Whether any regularizer helps. Only this one was measured.
- Nothing about seeds 1235/1236, and nothing about test performance.

## Limitations

- Single seed. A one-seed loss is weaker evidence than a one-seed win would have
  needed, but the direction is consistent across the entire 30-epoch curve, not
  a single epoch.
- Checkpoint selection and reporting share the validation split; no held-out
  estimate of the selection gain.
- Per-class counts are point estimates with no confidence intervals and no
  multiple-comparison correction.
- `test_evaluated=false` on both runs.
- Upstream `temporal_clean=false` limitations of the pinned artifact are
  unchanged and still apply.

## Standing comparison

The measured incumbent option is unchanged: the three-seed GCHM-sqrt ensemble at
macro-F1 **0.5706**, at the cost of running three models. This pilot does not
affect it.
