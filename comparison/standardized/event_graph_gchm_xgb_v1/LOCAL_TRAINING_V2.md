# Local EventGCHM run v2

This is one local EventGCHM adaptation run, not a baseline matrix or a performance claim. No VPS/SSH/upload. Existing `labels.py`, `runner.py`, baseline source edits, v1 target/features artifacts and immutable event input are not changed.

## Labels

`local_labels_v2.py` creates a separate `first_recorded_lab_all_visits_v2_targets_local_v2` binding. Diagnosis and crosswalk CSVs are read as strings (ICD-9 leading zeros preserved). Literal positive integer patient/stay identifiers are validated before any conversion, every shared stay is cross-checked against its diagnosis subject, and labels join on **both subject_id and stay_id**.

The 30 ordered classes are frozen by `native_inputs/protgsat_snapshot_v1/contract.json` and must match the graph manifest. The inherited `merge_ed.py` category-title ontology is reconstructed only on the identical, hash-pinned original diagnosis/crosswalk/merge-policy sources recorded by the pre-existing v1 binding. There is no new event-cohort category vote or top-30 reselection. The reconstructed map is persisted as `frozen_category_map.json`. This historical ontology is frequency-derived, **not a train-only learned ontology**.

`merge_ed.py` first orders unique mapped diseases by diagnosis sequence and later explicitly drops rows with nonempty disease_2. Accordingly exactly one unique selected frozen-class disease is labelled. Out-of-ontology diagnoses do not create new classes. No selected disease => target -1; multiple selected diseases => target -1. These rows remain in the binding with a reason, rather than being silently dropped. This reproduces the single-selected-disease target policy, not all old complete-case/demographic filtering. No subject-only target lookup is used.

The new binding contains 185,888 rows: 97,618 labelled, 64,764 without a selected class, and 23,506 with multiple selected classes. Train/validation/test labelled counts are 77,930 / 9,582 / 10,106. A one-to-one readback against the existing v1 targets found no changed target among the 185,888 exact sample/subject/stay triples; the corrections strengthen binding and literal-code safety rather than alter this source's observed targets.

## Training

`local_train_v2.py` uses 64-hidden/16-relation/3-layer EventGCHM, AdamW lr 0.001, weight decay 0.00001, seed 1234, 30 epochs. Selection uses validation macro-F1 over all 30 classes; there is no test prediction path. The scaler, clinical token/unit vocabulary, relation vocabulary and PNA degree histogram use **only labelled training graphs**. The degree histogram is `bincount(in_degree)`, not a vector of node degrees. Identifier/provenance fields are not tensor inputs. Historical diagnoses already legitimately present in the immutable graph are not newly added target covariates.

Local CPU uses 2 intra-op threads / 1 inter-op thread, microbatch 1 and 8-graph gradient accumulation. This bounds memory with large event graphs and leaves capacity for the already-running independent XGBoost job. MPS availability is recorded but accelerator execution is not claimed. Weighted CE uses sum reduction divided by the accumulation group's total target weight: mean reduction on a single graph would cancel its class weight.

Preparation streams the 18GB graph JSONL, stores a resumable SQLite offset/digest index outside the input root, verifies the full SHA, and stores a train-only adapter/degree state. A single lazy reader is used; no worker rescans the entire input. Interrupted index creation resumes its committed suffix; interrupted unfinished scaler fitting restarts its fit, while a finished bound preprocessing artifact is reused.

`last.pt` is atomically saved after update 1, every 25 updates or 60 seconds, epoch boundaries, and a handled stop. It contains model and optimizer state, exact epoch permutation and next cursor, Python/NumPy/Torch RNG, loss/history/selection state, class weights, adapter and source/input/preprocessing hashes. SIGTERM/SIGINT request a stop after the current optimizer update; no partial accumulated gradients are checkpointed. Resume refuses different bindings and restores those states. A stop during validation resumes validation from the same completed training epoch. `best.pt` is created only after actual validation selection.

## Commands (repository root)

```sh
python3 -u -m comparison.standardized.event_graph_gchm_xgb_v1.local_labels_v2 \
  --output comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_local_v2

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 -u -m comparison.standardized.event_graph_gchm_xgb_v1.local_train_v2 \
  --output comparison/standardized/event_training/local_first_lab_v2 \
  --binding-root comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_local_v2 \
  --epochs 30 --threads 2 --accumulation 8 --execute
```

For an interrupted same run append `--resume` to the training command. Do not run a second process against a live output directory. Inspect `process.json` and the tracked process first. Do not edit hash-bound source files while the run is active.

Authorised label preparation and real training are executed; unit/pytest suites are deferred by the user's phase policy. First updates/checkpoints are execution evidence, not scientific performance evidence. Input remains `temporal_clean=false`, and lab storetime remains an availability proxy.
