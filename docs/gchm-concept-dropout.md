# GCHM-sqrt: concept-dropout challenger

## Status

**Measured and rejected.** The seed-1234 pilot was executed; the candidate lost
0.0115 macro-F1 against the control. See
`docs/gchm-concept-dropout-pilot-results.md` for the measured outcome. The
existing GCHM-sqrt remains the incumbent and the code path below stays in the
repository as a registered, inert method (`concept_dropout=0.0` for every other
method, so nothing else is affected).

## Evidence behind the candidate

Read-only inspection of `comparison/standardized/native_runs/gchm_sqrt_v2/seed*/history.json`
and the corresponding completed manifests gives:

| Seed | Best epoch (zero-based) | Best validation macro-F1 | Final macro-F1 | Training loss at best | Final training loss |
| --- | --- | --- | --- | --- | --- |
| 1234 | 7 | 0.561467 | 0.535263 | 1.268742 | 1.004101 |
| 1235 | 10 | 0.563653 | 0.538320 | 1.206631 | 1.004758 |
| 1236 | 9 | 0.552641 | 0.538584 | 1.228181 | 1.004479 |

Mean selected macro-F1: 0.559253667; mean final macro-F1: 0.537389.
This pattern is consistent with overfitting, but does not establish its cause.
Historical `gchm_sqrt_v1` and `gchm_sqrt_v2/seed1234` are not independent seeds.
The concept-pair challenger did not beat the incumbent on seed 1234 (0.552250
versus 0.561467); extra interaction capacity is not automatically beneficial.

## One-variable intervention

New registry method: `gchm_concept_dropout`, paired with `--loss sqrt_inverse`.
The method name alone does not select the loss; the shared runner still defaults to CE.

During training, independently zero each concept occurrence's complete input
embedding with probability 0.15. Retained embeddings use inverted-dropout scaling.
The probability is a prespecified candidate, not an empirically optimal value.
This discourages dependence on individual concept identities and their conjunctions,
without increasing model capacity. Whether it helps must be measured.

- Hub encoding retains its existing behavior; the new dropout never touches hub rows.
- No new parameters, edges, input channels, or concept-pair mixer.
- Existing message passing, PNA degrees, layer dropout, optimizer, learning rate,
  weight decay, split, labels, sqrt-loss weighting and epoch budget are unchanged.
- Evaluation disables the new dropout; graph explanation edge masks still flow
  through the existing message-passing path.
- This is input-content regularization, not node/edge deletion. A zeroed concept
  can receive hub messages later and the original degrees remain intact.
- Existing methods use probability zero, avoiding an additional random draw.
- Existing checkpoints and run artifacts are not overwritten. The runner binds
  the registered method and model source hashes into each new run.

Changing these sources means old run bindings no longer match the current tree;
use their saved source snapshots for historical replay, not `--resume` under new code.

## Deferred verification and decision gate

After explicit testing-phase permission:

1. Verify probability range, row-wise dropout, untouched hub encoding, gradients,
   hub-only graphs, and evaluation determinism.
2. Verify probability-zero baseline equivalence and unchanged parameter/state-dict
   structure; check explicit and PyG edge masks and no-message interventions.
3. Exercise the registered method with the real runner in isolated bounded wiring
   scope. This is not evidence of benchmark improvement.
4. With full-cohort scope approval, run the challenger first at seed 1234, 30 epochs,
   batch 128, sqrt-inverse loss, same pinned artifact and patient order. Save under
   a new `comparison/standardized/native_runs/gchm_concept_dropout_v1/seed1234` directory.
5. Check selected and final validation macro-F1, accuracy, balanced accuracy, class
   metrics, manifest bindings, cohort order and exact checkpoint replay. Do not use
   test labels for tuning. Do not change rate/epochs after inspecting this pilot.
6. Only extend to seeds 1235/1236 with approval and a promising pilot. Compare paired
   seeds and stability; do not replace the incumbent based on a single peak epoch.

No acceptance threshold or full-run authorization is implied by this document.
