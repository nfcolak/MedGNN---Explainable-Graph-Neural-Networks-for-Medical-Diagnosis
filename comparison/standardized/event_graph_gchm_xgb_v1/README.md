# Event-graph GCHM/XGBoost comparison

This comparison uses `comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2` as the sole covariate source.
The graph JSONL is immutable and remains unlabelled. Targets are attached in a separate artifact from raw
`diagnosis.csv` by exact `stay_id`, using the existing `disease_1` 30-class merge policy and label order.
Only stays with one final disease label are supervised; unmatched/multi-disease stays remain with `target=-1`.

## Methods

- **EventGCHM**: event-graph adaptation of the existing GCHM receiver-conditioned gate. It consumes typed
  patient/visit/event/concept/knowledge nodes, numeric event covariates, temporal edge attributes and the
  train-fitted tensorizer state. The legacy 331-slot native GCHM is not fed this artifact unchanged.
- **XGBoost graph projection**: deterministic graph-level count/presence/value/time features from the same
  JSONL. Vocabularies and feature columns are fitted on the train fold only; subject IDs, stay IDs and source
  provenance are never features.

Both methods use the same subject-level split, class order, target rows and validation-only checkpoint selection.
The test fold is not opened by the runner before an explicit final-evaluation phase. The artifact remains
`temporal_clean=false` because storetime is an availability proxy rather than proven clinician visibility.

## Preparation and execution

The following commands are intentionally deferred until the user opens the explicit `testing phase`:

```bash
python -m comparison.standardized.event_graph_gchm_xgb_v1.labels
python -m comparison.standardized.event_graph_gchm_xgb_v1.features
python -m comparison.standardized.event_graph_gchm_xgb_v1.runner --all --execute
```

The first command creates the separate target binding. The second fits the XGBoost feature vocabulary on
training graphs and materializes the matched matrix. The third trains both methods for the frozen seed and
class-weight cells and writes validation-only comparison reports under `runs/`.
