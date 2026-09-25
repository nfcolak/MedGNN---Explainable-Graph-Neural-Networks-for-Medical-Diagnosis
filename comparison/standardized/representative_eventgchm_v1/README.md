# Approved local representative EventGCHM run

Isolated runner for **one seed (1234), exactly 6,000 training graphs and all 9,582 validation graphs, 10 epochs**. No early stopping, hyperparameter sweep, final test, XGBoost launch, remote access, or graph regeneration.

```sh
cd .
PYTHONPATH="$PWD" /Library/Developer/CommandLineTools/usr/bin/python3 -m comparison.standardized.representative_eventgchm_v1.run --execute
```

The output `comparison/standardized/event_training/representative_6000_fullval_v1` must not exist. The command refuses to overwrite it. This is documentation of the single approved invocation, not authorization to launch another attempt.

## Frozen policy

- Allocate 60 training visits per ordered class, then the remaining 4,200 proportionally to **original training class counts**, using deterministic largest remainders and availability caps.
- Within each class, shuffle subjects with a selection RNG seeded 1234; process scarce classes first and globally unseen subjects first. Select a random visit per subject before considering extra visits. Assert that the actual selection achieves 6,000 distinct patients, the global maximum possible, rather than silently accepting a lower-diversity approximation.
- Keep every natural validation visit. Assert inherited train/validation patient separation.
- Never select by predictions, loss, or graph size. No silent graph truncation.
- Reuse unchanged native `EventGCHM`, native graph reader and training-only tensorizer. Fit vocabulary, normalization and reverse-edge degree histogram only on the chosen 6,000 graphs.
- AdamW, learning rate 0.001, weight decay 0.00001; microbatch 1, accumulation 8. Class weights are mean-normalized inverse square roots of selected-training class counts. Normalize accumulated weighted CE by the entire accumulation group's target-weight sum, not each individual microbatch.
- CPU only, Torch threads/inter-op threads and BLAS environment limits all 1, niceness 19. Checkpoint model, optimizer, fitted adapter, training order/cursor, class weights and Python/NumPy/Torch RNG.
- Select the best checkpoint on validation macro-F1 with all 30 fixed classes. Reload it and replay all validation probabilities before completion.
- SIGINT/SIGTERM request interruption at an optimizer/validation boundary and preserve `last.pt`; this runner does not provide an automatic resume CLI.

## Evidence and limitations

`binding.json` binds original input manifests, target/cohort hashes, selected metadata and unchanged original source files. Selected graph records and an independent seed-81234 1,000-visit training reference are freshly SHA-verified using read-only existing SQLite offsets. The previously recorded whole-graph hash is bound, but the 18-GB file is **not freshly fully scanned/hashed**.

`distribution_profile.json` compares all-training exact index/target metadata with the selection. Payload history/current measurements/missingness/knowledge comparisons use the selected graphs and independent 1,000-visit reference. Patient diversity and minimum class quotas deliberately change the natural visit distribution; distribution matching is not claimed. Zero missingness of represented event fields follows producer eligibility and is not evidence of complete clinical laboratory coverage.

`metrics.json`, `per_class.csv`, `history.json`, `validation.npz`, `replay_verification.json` and `report.md` are the real-run results. Patient-equal metrics assign each visit weight `1 / validation_visits_for_patient`. Accuracy uncertainty uses 500 independent seeded patient-cluster bootstrap resamples, retaining all visits of each sampled patient. Intervals exclude seed and checkpoint-selection uncertainty; macro-F1 intervals are not estimated.

Validation is repeatedly used for checkpoint selection, not final-test evidence. The source remains `temporal_clean=false`; recorded storetime is a database availability proxy and knowledge relations are not clinically reviewed. Different input/cohort results cannot establish superiority over prior benchmark methods.

The original full and small runs are preserved and source/artifact hashes are checked before/after. Output directories/files are private (700/600). No unit/pytest or unrelated test suite is part of this approved production training execution.
