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
- Class weighting: fixed sqrt-inverse policy (unchanged), with checkpoints selected on dev macro-F1. XGBoost selection from screen or validation: False; validation reselection: False; validation retraining: False.

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

Arm C screen absence share (historical): 1.67% / 1.83% / 1.81% for seeds 1234 / 2025 / 7. This is a **row (visit) mean**, not a patient mean; the historical patient mean has not been computed. The implemented correction retains the legacy row field and adds explicit row/equal-patient means, an aggregation version and a file-hashed per-visit proof with replay validation in the core and extension consumers. Existing targeted CEI tests verify the implementation; historical outputs are neither recomputed nor rewritten. Arm B (no absence) is above C on the screen (0.6439 vs 0.6418), so no benefit of absence evidence is shown.

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

- Extension families E2w, E2d, E6a, E6b: **not executed at full scale** (`extensions_executed` false in the core decision record, which is unchanged); only the bounded train/dev smoke below was run. O: not executed.
- Arm O validation fitting: **refused** (not approved). Full long-run matrix: **unapproved and unrun**; a measured train/dev smoke budget now exists (below) but is not an approval. Test fold: closed.

## Verified interfaces and protected archive usage

Run from the merged checkout. Set `DATA_ROOT` to the original private repository-layout input root and `ARCHIVE` to the separately preserved `cei-v3-20261001` archive, **outside all worktrees**. These are operator-supplied absolute paths, not public data. The archive holds `core`, `protgnn`, `validation`, `xgboost` and `graphxai` children; its private `preservation_manifest.json` maps historical paths only at verified file-open boundaries. Original bindings, source hashes and checkpoint bytes are never rewritten. Archive verification reports 291 files / 231136374 bytes and zero mismatches; manifest SHA-256: `b0a1118f420d3cce6d892fcc22877e867b08276555c5e6f57140e31eb71bf522`.

Actual `--help` and metadata plans for all five CLIs below passed against that external archive. The four comparisons each verified 12 bindings; plans created no output, loaded no prediction arrays and deserialized no graphs. Before/after hash snapshots of the archive, original evidence roots and tracked source matched. Choose fresh ignored output paths, never inside an input/archive root.

```bash
# Read-only archive verification; no copying or historical mutation.
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 \
  -m comparison.standardized.clinical_graph_v2.cei_v3_preserve --verify "$ARCHIVE"

# Verified comparison metadata interfaces; no --execute here.
for cli in cei_v3_vs_protgnn cei_v3_validation cei_v3_vs_xgboost cei_v3_graphxai; do
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 \
    -m "comparison.standardized.clinical_graph_v2.$cli" --help
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 \
    -m "comparison.standardized.clinical_graph_v2.$cli" \
    --data-root "$DATA_ROOT" --results-root "$ARCHIVE" \
    --path-map "$ARCHIVE/preservation_manifest.json" \
    --output-root "comparison/standardized/clinical_runs_plan_only_recovery_$cli" --plan
done

# Extension plan (no --execute): four sequential bounded stages, 12 full stages.
# OUTPUT_ROOT must be an absolute, fresh, ignored path in the merged checkout.
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 \
  -m comparison.standardized.clinical_graph_v2.cei_v3_ext.run --help
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 \
  -m comparison.standardized.clinical_graph_v2.cei_v3_ext.run \
  --core-root "$ARCHIVE/core" \
  --artifact "$DATA_ROOT/comparison/standardized/event_inputs/clinical_graph_v3_membership_max6_20260923" \
  --targets "$DATA_ROOT/comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_local_v2_max6/targets.csv" \
  --canonical "$DATA_ROOT/comparison/canonical_split.json" \
  --path-map "$ARCHIVE/preservation_manifest.json" --output-root "$OUTPUT_ROOT" \
  --python /usr/bin/python3 --device cpu --threads 2 \
  --memory-ceiling-gib 8 --stage-timeout-seconds 1800
```

The exact extension argv parses against the merged training entrypoint and retained control metadata passes replay checks. A **separate approved smoke unit**, not this integration, may append `--execute smoke` to the extension command: E2d/E6b/E6a/E2w sequentially, seed 1234, 256 train / 128 dev / 2 epochs, frozen K=4, fixed sqrt-inverse weights. It uses selected train/dev graph reads only; no screen, validation or test scoring, no cache rebuild. That smoke has now been run (see the next section).

`m=5` remains mandatory even with O omitted. Do not pass `--include-o` or create an O approval: validation fitting is refused. `--execute full` fails closed without a distinct approval bound to the completed measured smoke journal, unchanged source/input identity and AC power. A historical scoring-pass approval does not authorize fitting. Long-run execution remains unapproved.

**Plan success is not reproduction or execution preflight.** Historical comparison `--execute` remains intentionally fail-closed: retained source hashes differ for `cei_v3_ext/study.py`, `cei_v3_run.py`, `cei_v3_study.py`, `cei_v3_vs_protgnn.py` and `train.py`. Exact historical replay from this merged source is unsupported; a matching isolated historical-source environment would require separately authorized execution and has not been supplied or verified here. No validation or GraphXAI scoring was launched. XGBoost no longer requires an arbitrary agent branch name; clean committed source plus the existing bound scientific source/input/fold/hash checks remain required. No source parity is silently waived.

## Smoke and verification (2026-10-01)

Approved bounded train/dev smoke, run from source commit `12c7bf3c` (RSS-monitor fallback). Aggregate receipt: [`cei-v3-smoke-2026-10-01.json`](cei-v3-smoke-2026-10-01.json). Private evidence (journal, logs, failed attempt) lives in the private archives `$PRIVATE_ARCHIVE/cei-v3-20261001` and `$PRIVATE_ARCHIVE/cei-v3-smoke-20261001/{completed,failed-attempt}`; they are not part of this repository.

- E2d, E6b, E6a, E2w each ran sequentially: seed 1234, 256 train / 128 dev rows, 2 real epochs, CPU, 2 threads, 8 GiB ceiling, completed and verified checkpoint (8 real epochs in total, journal return code 0). Held-out test, validation and screen graphs were not deserialized; no validation fitting; O excluded.
- One earlier attempt failed because the sandbox denied `ps` for RSS monitoring. A native `psutil` fallback (the only source change, in `cei_v3_ext/run.py`) fixed it; the failed attempt is preserved.
- Existing smoke checks: 372 passed. Combined 12-file existing suite on the merged branch: 381 passed. Archive verification: 291 files / 231136374 bytes, zero mismatches.
- The smoke is a runtime/plumbing check, not a scientific result; the main experiment evidence above is unchanged.

Measured budget for a future full run (three seeds per arm, 40 epochs, 10,000 train / 5,000 dev), minutes:

| Arm | Three seeds |
|---|---:|
| E2d | 29.17 |
| E6b | 27.76 |
| E6a | 29.41 |
| E2w | 32.08 |
| Total (12 trainings) | 118.41 |

This is a linear scaling of a small two-epoch CPU smoke (setup held constant, first epoch includes warmup, combined-row scale 39.0625), **not a promise**. The extension screen encoding/inference, paired bootstrap/decision aggregation and full-data setup tail are unmeasured, and O is excluded. Larger graph/batch composition, early stopping and peak-memory scaling are unmeasured.

Source-scoped limitations:

- The smoke binds to source `12c7bf3c`; the later documentation/test merge did not change `clinical_graph_v2` Python source (hash inventory compared). Historical source-bound execution approvals are incompatible with it, and any full approval must bind this completed smoke and unchanged source/input identity.
- Full execution has not been run and is not approved. Arm O validation fitting remains refused. Test fold never opened.
- The historical patient-mean absence share has not been recalculated, and exact historical replay is not supported on this source.

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

Logical roots: core = `clinical_runs_cei_v3_20260930`; protgnn = `clinical_runs_cei_v3_vs_protgnn_20260930`; validation = `clinical_runs_cei_v3_validation_20260930`; xgboost = `clinical_runs_cei_v3_vs_xgboost_20260930_retry2`; graphxai = `cei_v3_graphxai_screen500_20260930`. Saved results are protected in the verified external archive; original worktree outputs are retained unchanged. Private immutable outputs embed their original absolute paths, which are mapped at I/O boundaries, not rewritten (hashes would break).

Historical context: [`graphxai-500-results.md`](../../../../../../docs/graphxai-500-results.md), [`cei-gnn-v2-pair-study-result-2026-09-29.md`](cei-gnn-v2-pair-study-result-2026-09-29.md).
