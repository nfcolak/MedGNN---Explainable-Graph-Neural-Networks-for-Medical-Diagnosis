# CEI-GNN bounded development pilot — 2026-09-28

## Result and limits

The independent CEI-GNN candidate scores higher than the matched ProtGNN control in this single-seed development pilot, but superiority is not established. The paired patient-bootstrap interval includes zero. The explicit-product-off ablation scores slightly higher still; no benefit from the explicit interaction product is demonstrated here.

This is development-set checkpoint selection, not independent validation or test performance. No full multi-seed comparison was run. Historical validation macro-F1 values must not be substituted into this table.

## Frozen contract

- Source: `5f23c8297294172b7cc4ba482377eead772debbe` (`feature/cei-integration`).
- Existing max6/Top-10 clinical graph artifact; no graph rebuild or input modification. Forward/all-edge view, sqrt_inverse class weighting, model/sample seed1234.
- 10,000 training visits, 5,000 development visits from 4,841 development patients; train/dev patients disjoint.
- Four authorized training runs only: CEI256/128/two-epoch smoke, then the three40-epoch arms below. All full arms batch128, patience40, 3,160 optimizer steps (derived from observed epoch count and runner batch loop).
- Existing method-native schedules/settings preserved; equal epochs and optimizer steps do not imply equal compute.
- Validation tensors are materialized by the unchanged loader but never evaluated; no test tensors/predictions/evaluation.
- Input manifest `temporal_clean=false`: triage timing, storetime-as-availability, historical diagnosis availability and non-clinically-reviewed knowledge edges remain limitations. No clinical utility/causal claim.

## Measured results

| Arm | Dev macro-F1 | Accuracy | Balanced accuracy | Patient-equal macro-F1 | Parameters (active) | Selected epoch | Recorded seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| protgnn_control | 0.643315 | 0.627000 | 0.649068 | 0.641669 | 399884 (399884) | 15 | 222.4 |
| cei_candidate | 0.649207 | 0.630000 | 0.647898 | 0.646061 | 111906 (111906) | 37 | 127.7 |
| cei_product_off | 0.649730 | 0.632000 | 0.648406 | 0.647799 | 111906 (103698) | 37 | 123.9 |

Smoke recorded21.6seconds and2epochs; not a performance comparison. Times above are the runner's `total_seconds`, not end-to-end orchestration time.

## Paired uncertainty and all-class accounting

1,000 patient-cluster bootstrap resamples, seed2026, preserving all visits within each sampled patient. Intervals condition on these selected dev checkpoints and do not measure seed variability or correct model-selection optimism.

| Comparison (first minus second) | Macro-F1 delta | 95% conditional interval | Classes with higher F1 | Mean precision delta | Mean recall delta | Median class F1 delta | Total FP delta |
|---|---:|---|---:|---:|---:|---:|---:|
| cei_candidate − protgnn_control | +0.005892 | [-0.002508, +0.014268] | 7/10 | +0.012151 | -0.001170 | +0.007672 | -15 |
| cei_candidate − cei_product_off | -0.000523 | [-0.004995, +0.003932] | 5/10 | -0.000334 | -0.000507 | +0.000268 | +10 |
| cei_product_off − protgnn_control | +0.006415 | [-0.001785, +0.014863] | 8/10 | +0.012484 | -0.000663 | +0.007661 | -25 |

The product-off arm keeps identical registered parameter shapes and total count;8,208 factor parameters are inactive. Its nonlinear edge head remains active, so this is not a purely additive-model ablation.

## Real GraphXAI acceptance

- Trained candidate checkpoint:10 deterministic development examples, one per true class, selected by minimum SHA256(sample_id), never by correctness/gradient.
- Real `GradExplainer`, `IntegratedGradExplainer` (32steps), `GNNExplainer` (50epochs):30/30 successful explanation records. Fixed edge metadata conditioning and zero-feature-versus-node-deletion limitations are included in every export.
- Wrapper logits exactly match model logits on all10 examples (maximum absolute difference0). Finite explanation values validated; vendored finite-objective guard raises on invalid GNNExplainer optimization.
- Complete export bound to checkpoint, source, actual graph and membership files, seed, class order and frozen dev roster. Incremental durable journal retained. Actual vendor/helper source snapshots saved locally.
- This verifies a ten-example compatibility pilot, not full-cohort explanation quality or cross-method explanation parity.

## Verification and failure history

-68 targeted synthetic checks passed,14 pre-existing dependency deprecation warnings. Whole repository suite not claimed.
- Independent model review approved; final integrated review initially found6 blockers hidden by the60-test suite (runner schema, schedule comparison, stale arm path, source root, export identity, native-default reconstruction). All6 corrected and final scoped review approved5f23c829.
- Supervisor replayed all6 final test-only RED commits against faulty parents. Eight model mutants and three XAI provenance guard mutants were killed. Earlier test-after-code cases remain process deviations; mutation evidence is not retroactive TDD.
- All4 actual saved checkpoints replayed stored dev probabilities, labels and sample IDs exactly; all3 compared arms passed common binding checks before metrics were read.
- Executable-source and input byte identity unchanged across every stage;8 protected existing input/checkpoint files unchanged after all execution. Existing ProtGNN synthetic state/logits/config fingerprint unchanged.
- No existing artifacts overwritten; no tuning runs or extra training seeds performed.

## Next budget boundary — not executed

The small dev lead is a screening tie, not an expected final win. Recommended next evidence: repeat all three dev arms at two additional fixed seeds before choosing whether to retain the explicit product.

| Proposed work | Runs | Recorded seconds/run | Estimated training minutes |
|---|---:|---:|---:|
| ProtGNN, two additional dev seeds |2|222.4|7.4133|
| CEI, two additional dev seeds |2|127.7|4.2567|
| Product-off, two additional dev seeds |2|123.9|4.1300|
| Total additional dev evidence |6|—|15.8000|

Loading/replay/report overhead is additional and not included in this training-time estimate. A later frozen three-seed validation comparison of ProtGNN and one selected CEI design would require6new training runs:17.505minutes estimated from these measured times, plus final validation/replay overhead. Neither phase is authorized or launched by this pilot. Test remains closed.

## Source branches and local evidence

- Model: `feature/cei-independent-model` at `4befe0adf6a00bee990eb6da9687bf7db3fba7b4`.
- GraphXAI: `feature/cei-graphxai` at `ed708558a7332ac4f02c4436a24ffc5f4d15c261`.
- Protocol: `feature/cei-pilot-protocol` at `8106ba1e4e342714dda2f9d5459d57cab04a0991`.
- Full reviewed RED/GREEN history: `feature/cei-integration` at5f23c829. Published unit snapshots are byte-identical to reviewed files. No main-branch merge.
- Private artifacts (ignored): `comparison/standardized/clinical_runs_cei_pilot_20260928/`, including stage binding/result/dev predictions/checkpoints/replay, `aggregate-analysis.json`, `acceptance-verification.json`, `cei_dev_graphxai/`, vendor snapshots.
- Review/test evidence: `.worktrees/_runs/cei-supervisor/` and `.worktrees/_runs/cei-final-rereview/report.md`. No subject IDs, medical graphs, checkpoints or per-example explanations are published.

## Execution rulings

1. Parallel model and XAI development in disjoint worktrees under a locked interface; protocol training waited for integration. Risk/cost if wrong: integration rework, not altered prior artifacts.
2. Use the approved subagent-driven implementation without an additional process-choice prompt. Risk/cost if wrong: workflow preference rework, no expanded experiment scope.
3. Replace the disposable parent-replay shell invocation with a Python harness when the shell stopped on unmatched failure text. No global tool edit. Risk/cost if wrong: repeat tests, no source/data change.
4. Develop the protocol as a third disjoint task while the supervisor audits artifacts; actual execution waits for frozen shared source. Risk/cost if wrong: integration rework.
