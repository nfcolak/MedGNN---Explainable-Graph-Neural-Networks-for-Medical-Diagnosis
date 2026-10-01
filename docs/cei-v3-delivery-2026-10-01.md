# CEI-GNN v3 delivery report (2026-10-01)

Aggregate-only report. All numbers are computed from the five saved result roots and frozen in
[`cei-v3-evidence-2026-10-01.json`](cei-v3-evidence-2026-10-01.json) (key-to-source table under `provenance`,
source file hashes under `source_aggregate_sha256`). No patient-level data, predictions, checkpoints or private bindings are included.

## Scientific conclusion

- v3 vs v2 (arm C vs arm A): **benefit not demonstrated on the primary screen** (-0.0002, 95% CI [-0.0047, +0.0042]).
- C vs ProtGNN is a **validation second-look result only**: the validation fold was opened after the screen was inconclusive, and the contrasts below were not corrected for multiple comparisons. It is not a broad winner claim.
- C vs XGBoost: not demonstrated on screen or validation. No XGBoost victory is claimed.
- A non-significant interval is not evidence of equivalence; no statistical-equivalence claim is made anywhere in this report.
- Held-out test fold: never accessed.

## Task and cohort

- Cohort: MIMIC-IV-ED max6 (patients with more than 6 total visits excluded); Top-10 diagnoses by TRAIN-fold frequency; other labels dropped, not merged.
- Samples: 10,000 train / 5,000 dev (checkpoint selection on dev macro-F1); screen 5,000 rows / 4,840 distinct subjects; validation 4,254 rows / 3,472 distinct subjects.
- Training seeds 1234, 2025, 7 (one model per arm and seed; folds: train, dev, screen, validation; test closed). Screen subsampling seed 20260929.
- Arms: A = v2 additive control, B = PLE only, C = PLE (K = 4) + absence, P = ProtGNN, X = XGBoost control. Metric key as saved: `weighted_macro_f1`.
- Not comparable with the native 30-class benchmark (331-slot star input) or any other training size; those results stay separate.

## Selection policy (frozen before screen)

- K grid [4, 8, 16], arm C only, statistic: unweighted mean over seeds of selected_dev.metric_value. Dev seed means: K4 0.6493, K8 0.6482, K16 0.6482; K = 4 selected (tie rule not applied).
- Class weighting: dev-selected sqrt-inverse (unchanged). XGBoost selection from screen or validation: False; validation reselection: False; validation retraining: False.

## Macro-F1 seed means

| Arm | Screen | Validation |
|---|---:|---:|
| A | 0.6420 | 0.6342 |
| B | 0.6439 | 0.6375 |
| C | 0.6418 | 0.6395 |
| P | 0.6371 | 0.6303 |
| X | 0.6371 | 0.6317 |

## Paired contrasts, Screen (5,000 rows)

Conditional bootstrap (1000 resamples, linear quantiles, seed 2026, paired by subject, conditional on these three trained seeds; not a seed-population interval). Per-seed deltas in order 1234 / 2025 / 7. Rule = all of: wins every seed, mean above, CI lower bound above 0.

| Contrast | Delta | 95% CI | Per-seed delta | Rule |
|---|---:|---|---|---|
| C - A | -0.0002 | [-0.0047, +0.0042] | +0.0013 / -0.0050 / +0.0031 | fail |
| B - A | +0.0019 | [-0.0024, +0.0063] | +0.0013 / -0.0013 / +0.0057 | fail |
| C - P | +0.0046 | [-0.0017, +0.0107] | +0.0083 / +0.0034 / +0.0023 | fail |
| A - P | +0.0049 | [-0.0007, +0.0110] | +0.0070 / +0.0084 / -0.0008 | fail |
| A - X | +0.0049 | [-0.0024, +0.0123] | +0.0089 / +0.0074 / -0.0016 | fail |
| B - X | +0.0068 | [-0.0005, +0.0139] | +0.0103 / +0.0061 / +0.0041 | fail |
| C - X | +0.0047 | [-0.0025, +0.0117] | +0.0102 / +0.0024 / +0.0015 | fail |
| P - X | +0.0001 | [-0.0079, +0.0079] | +0.0020 / -0.0010 / -0.0008 | fail |

## Paired contrasts, Validation (4,254 rows)

Conditional bootstrap (1000 resamples, linear quantiles, seed 2026, paired by subject, conditional on these three trained seeds; not a seed-population interval). Per-seed deltas in order 1234 / 2025 / 7. Rule = all of: wins every seed, mean above, CI lower bound above 0.

| Contrast | Delta | 95% CI | Per-seed delta | Rule |
|---|---:|---|---|---|
| A - P | +0.0040 | [-0.0026, +0.0102] | +0.0100 / +0.0031 / -0.0012 | fail |
| B - A | +0.0032 | [-0.0017, +0.0080] | +0.0073 / +0.0003 / +0.0021 | fail |
| C - A | +0.0052 | [+0.0006, +0.0098] | +0.0074 / -0.0015 / +0.0097 | fail |
| C - P | +0.0092 | [+0.0024, +0.0153] | +0.0174 / +0.0016 / +0.0085 | pass |
| A - X | +0.0025 | [-0.0055, +0.0103] | +0.0125 / -0.0027 / -0.0022 | fail |
| B - X | +0.0058 | [-0.0022, +0.0136] | +0.0198 / -0.0024 / -0.0001 | fail |
| C - X | +0.0078 | [-0.0002, +0.0155] | +0.0199 / -0.0042 / +0.0075 | fail |
| P - X | -0.0014 | [-0.0100, +0.0068] | +0.0025 / -0.0058 / -0.0010 | fail |

Reading: on the screen no contrast clears the rule. On validation only C - P clears it; C - A has a CI above 0 but C loses seed 2025, so the rule fails; C - X has a CI lower bound just below 0 and C loses seed 2025.

## Absence evidence

Arm C screen absence share (historical): 1.67% / 1.83% / 1.81% for seeds 1234 / 2025 / 7. This is a **row (visit) mean**, not a patient mean; the patient mean has not been computed. A code fix is forthcoming; historical outputs are not repaired and are not rewritten. Arm B (no absence) is above C on the screen (0.6439 vs 0.6418), so no benefit of absence evidence is shown.

## GraphXAI explanation comparison (C K4 vs ProtGNN)

- 500 screen visits / 498 distinct subjects (10 classes x 50), seeds 1234, 2025, 7, 15,000 records; no training; validation and test not evaluated.
- Native ProtGNN explanation: unavailable (no native prototype-similarity node-importance API), so no native C vs native ProtGNN comparison exists.
- Bootstrap is conditional on three checkpoints. Faithfulness to each model only: not clinical correctness, not causality.

| Explainer | Metric | C - P | 95% CI | C better (seeds of 3) |
|---|---|---:|---|---:|
| GradExplainer | fidelity_minus (lower better) | -0.4272 | [-0.4499, -0.4043] | 3 |
| GradExplainer | fidelity_plus (higher better) | -0.0009 | [-0.0183, +0.0159] | 1 |
| GradExplainer | sparsity (higher better) | +0.1255 | [+0.1130, +0.1373] | 3 |
| IntegratedGradExplainer | fidelity_minus (lower better) | -0.4703 | [-0.4935, -0.4479] | 3 |
| IntegratedGradExplainer | fidelity_plus (higher better) | +0.0033 | [-0.0129, +0.0189] | 2 |
| IntegratedGradExplainer | sparsity (higher better) | +0.2318 | [+0.2217, +0.2407] | 3 |
| GNNExplainer | fidelity_minus (lower better) | -0.1778 | [-0.2012, -0.1547] | 3 |
| GNNExplainer | fidelity_plus (higher better) | -0.1651 | [-0.1830, -0.1462] | 0 |
| GNNExplainer | sparsity (higher better) | -0.0010 | [-0.0027, +0.0007] | 1 |
| Random | fidelity_minus (lower better) | -0.0943 | [-0.1145, -0.0746] | 3 |
| Random | fidelity_plus (higher better) | -0.2016 | [-0.2188, -0.1824] | 0 |
| Random | sparsity (higher better) | +0.0000 | [+0.0000, +0.0000] | 0 |

Grad and IntegratedGrad favour C on fidelity-minus and sparsity (3/3 seeds). Fidelity-plus is not uniformly better: indistinguishable for Grad/IG and in favour of ProtGNN for GNNExplainer (C better in 0 of 3 seeds).

## Not executed / not approved

- Extension families E2w, E2d, E6a, E6b, O: **not executed** (`extensions_executed` false in the core decision record).
- Arm O validation fitting: **refused** (not approved). Long-run matrix: **unapproved** until a measured train/dev budget is shown. Test fold: closed.

## Expected interfaces (pending integrator confirmation)

Concurrent work packages are expected to add a preservation/backup helper, a parameterized runner with a preflight/plan mode, and an extension CLI (`cei_v3_ext`, `cei_v3_preflight.py`). Their names, flags and behaviour are **not verified here**; read-only plan/help modes must write nothing and deserialize no graphs. Nothing in this report depends on them being merged.

## Provenance

| Evidence JSON key | Source (logical path) |
|---|---|
| `task.train_rows / dev_rows / split_hashes.train_ids_sha256 / dev_ids_sha256 / screen_ids_sha256 / task.screen_rows / screen_distinct_subjects` | `core/screen_record.json` |
| `task.validation_rows / validation_distinct_subjects / split_hashes.validation_ids_sha256 / selection_policy.validation_*` | `validation/decision.json` |
| `selection_policy.k_*` | `core/k_selection.json` |
| `selection_policy.xgboost_selection_from_screen_or_validation` | `xgboost/decision.json` |
| `macro_f1_seed_means / macro_f1_per_seed` | `xgboost/decision.json (cross-asserted equal to core, protgnn, validation decision.json for shared arms)` |
| `contrasts.screen.C_vs_A / B_vs_A` | `core/decision.json` |
| `contrasts.screen.C_vs_P / A_vs_P` | `protgnn/decision.json` |
| `contrasts.screen.*_vs_X / contrasts.validation.*_vs_X` | `xgboost/decision.json (decisions.<split>)` |
| `contrasts.validation.C_vs_P / C_vs_A / B_vs_A / A_vs_P` | `validation/decision.json (decisions)` |
| `absence_share.per_seed` | `core/C_K4_seed<seed>/screen/screen_result.json (field absence_share_mean only)` |
| `graphxai.*` | `graphxai/summary.json` |
| `not_run.extensions_executed` | `core/decision.json` |

Logical roots: core = `clinical_runs_cei_v3_20260930`; protgnn = `clinical_runs_cei_v3_vs_protgnn_20260930`; validation = `clinical_runs_cei_v3_validation_20260930`; xgboost = `clinical_runs_cei_v3_vs_xgboost_20260930_retry2`; graphxai = `cei_v3_graphxai_screen500_20260930`. Saved results are git-ignored and live in linked worktrees; immutable outputs embed their original absolute paths, which are not rewritten (hashes would break).

Historical context: [`graphxai-500-results.md`](graphxai-500-results.md), [`cei-gnn-v2-pair-study-result-2026-09-29.md`](cei-gnn-v2-pair-study-result-2026-09-29.md).

