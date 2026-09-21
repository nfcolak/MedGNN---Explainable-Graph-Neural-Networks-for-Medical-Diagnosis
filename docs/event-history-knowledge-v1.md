# Event/history/knowledge graphs — v1

## Status and boundary

Production implementation. The user separately authorized **real graph artifact
construction and full artifact readback**, without opening model testing/training.
The first-recorded-lab production entrypoint below performs that authorized work.
Unit tests, model forwards, training and evaluation remain deferred. No performance
or clinical validity is claimed. Old benchmark files, splits, checkpoints and dirty
working-tree files are not modified by this feature.

This is a **new input contract**, not a replacement cache for the existing native
reference comparison. Explicit encounter IDs and prediction cutoffs are required;
the builder never guesses an index visit from a subject-only legacy row.

## Representation

- One graph per supplied sample / index encounter / cutoff.
- Multiple visits of the same patient are explicitly allowed as separate prediction
  samples, each with its own cutoff and target. All supplied eligible visit rows
  are retained; no first-visit or subject-level deduplication is performed.
- All visits of a patient must remain in one train/validation/test fold. A visit
  cannot belong to multiple patients. The manifest records distinct patient and
  visit counts, sample counts, and patients with multiple index visits per cohort.
- Patient → visit → individual measurement or reconciliation record → concept.
- Completed previous encounters retain their own events; visits overlapping the
  index visit are excluded from the prior-history layer.
- Consecutive distinct event-time groups are connected inside a visit. Simultaneous
  events are not given an arbitrary sequential order.
- Repeated measurements connect only within the same concept **and exact unit**.
- Source-backed medical relations connect observed concepts to shared knowledge
  nodes. The knowledge node is not the patient's diagnosis.
- Every event retains its value, unit, time relative to cutoff, availability time
  relative to cutoff, timing basis and source-row provenance. Metadata IDs are not
  model features.
- An event-free index patient is retained, never silently discarded.
- Graph budgets cause failure rather than silently truncating the patient history.

Clinical timing order is not causality. A medication reconciliation record is
neither a confirmed administration nor the time a patient began taking a drug.
No medication dose, onset, diagnosis timestamp or symptom detail is invented.

## Raw inputs and eligibility

The new builder imports neither `merge_ed.py` nor the aggregate laboratory
extractor. It streams the existing source CSVs with pandas chunks into an
output-local SQLite index instead of loading the complete lab file into memory.

| Source | Default use | Constraint |
|---|---|---|
| `edstays.csv` | Explicit visit lineage/history | Index subject/stay must agree; cutoff lies inside the index visit |
| `labevents.csv` | Individual allowlisted measurements | Finite numeric value, nonempty exact unit, valid charttime and storetime, storetime >= charttime, exactly one matching ED interval |
| `vitalsign.csv` | Excluded by default | Optional `charttime-proxy` mode; recording availability is unverified |
| `medrecon.csv` | Excluded by default | Optional `charttime-proxy`; reconciliation only, not administration |
| `triage.csv` | Excluded | No event/availability timestamp in inspected header |
| `diagnosis.csv`, `pyxis.csv`, legacy wide tables | Excluded | No target-derived or post-decision shortcuts |

Both event time and availability time must be at or before the explicit cutoff.
The index visit's future discharge time is used only for source interval checks,
not emitted as a feature. Prior-visit completion is required before index arrival.
Temperature conversion applies once to raw vital records; lab units are not
converted or combined. Negative/nonfinite source anomalies do not become guessed
clinical values; finite numeric outliers are retained without undocumented clipping.

Even strict-storetime mode remains `temporal_clean=false` and
`early_triage_eligible=false`: storetime is a database proxy, encounter metadata
availability is not proven, and the task's diagnosis availability has not been
independently established. These flags must not be relaxed just because a run
finishes.

## Explicit cohort contract

Supply a CSV with:

- `sample_id`: opaque unique sample identity (metadata only).
- `subject_id`, `stay_id`: exact decimal identifiers from raw data. No repair of
  float-looking identifiers, no subject-only encounter matching.
- `cutoff`: explicit timezone-naive ISO timestamp in the same shifted MIMIC clock.
- `split`: `train`, `validation`, or `test`. All samples of one subject must remain
  in one split. Duplicate stay/cutoff samples are rejected.
- Optional `target`: nonnegative integer into the supplied ordered label list.

Labelled cohorts require `--labels` pointing to a JSON array of unique class names.
Labels are output targets only, not graph-building inputs. There is no automatic
new split and no claim that supplied rows equal the original canonical cohort.

The approved production cutoff is the first nonempty recorded laboratory result,
including all results available at that instant. This does not establish target
lineage: the legacy all-visits label table lacks stay/time identifiers, so the
production artifact is deliberately unlabelled. Never attach labels by subject or
visit ordinal.

## Knowledge layer

`comparison/standardized/event_graph_v1/knowledge_seed.csv` is an initial,
source-checked lab relationship file, not a clinically reviewed ontology. It
contains panel membership, kidney-function assessment and blood-glucose
measurement relationships grounded in the linked MedlinePlus pages. Each row
records its URL, retrieval version, reviewer status and supporting text.

`reviewed_by=hermes-source-check-not-clinical-review` intentionally does **not**
claim expert review. MIMIC item IDs use the repository's existing closed lab
allowlist. Knowledge identities are local research identifiers, not purported
LOINC/SNOMED codes. No patient labels or disease-conditioned selection is used.

Custom knowledge CSV columns: `source_token,relation,target_token,source_uri,
source_version,reviewed_by`; optional `evidence` is retained in edge provenance.
Allowed source namespaces: `lab:`, `vital:`, `med:`. Destination namespace:
`knowledge:`. Relations: `is_a`, `member_of`, `measures`, `assesses`, `may_affect`,
`interacts_with`. The first version expands only observed concept → knowledge;
there is no arbitrary multi-hop KG expansion.

Full-mode construction fails if no supplied medical relation matches any graph.
An explicit `--event-only` mode omits knowledge and is labelled accordingly.
Per-fold coverage counts disclose event-empty samples, prior-history coverage and
knowledge coverage; missing information is not silently replaced with fake edges.

## Entrypoint (documented, not run)

Run from the repository root after supplying the real cohort and ordered labels:

```sh
python3 -m comparison.standardized.event_graph_v1.build \
  --raw-root 'data/Original CSVs' \
  --cohort /absolute/path/to/approved_encounter_cutoffs.csv \
  --labels /absolute/path/to/ordered_labels.json \
  --decision-policy 'Describe the approved prediction task and cutoff policy' \
  --knowledge comparison/standardized/event_graph_v1/knowledge_seed.csv \
  --timing-policy strict-storetime \
  --output comparison/standardized/event_inputs/approved_policy_v1
```

Without `--execute`, only the command contract is printed; no raw scan or output
creation occurs. Add `--execute` only with explicit production authorization or in the execution/testing phase. The output
path must not exist. Failures remain marked `failed`; partial files must never be
consumed as completed artifacts. Use a new output path for a fresh attempt.

Outputs:

- `graphs.jsonl`: graph-local IDs, features, typed edges, optional targets, coverage.
- `events.sqlite`: sensitive local raw-event index, including subject/stay lineage.
- `manifest.json`: status, policies, source/code hashes, coverage, counts, graph and
  index fingerprints, exact label order, limitations.
- `source_snapshot/`: copied builder implementation and source-path index.
- `knowledge.csv`: exact supplied medical relation source when enabled.

Source hashing and raw laboratory streaming are substantial IO. Progress prints
scan counts and graph counts, not private patient values. No ETA is fabricated.
Directories use mode 0700 and files 0600; these are sensitive local artifacts, not
public exports. No Keychain or credential system is involved.

## Model integration

`event_graph_analysis/` supplies a train-fitted tensor adapter and a typed-edge
GNN for this schema. It is opt-in, not wired into the legacy comparison matrix.
Graph IDs/source-row references are provenance only. Token/relation vocabularies
and numeric scaling belong to the training fold, not validation/test. Exact units
remain distinct. Numeric values, timing and typed relations feed the model rather
than only concept presence.

No trained checkpoint, benchmark score, GraphXAI integration or existing-method
input parity is delivered by this implementation. Those require a separate phase.

### Completed-artifact model API

`EventGraphArtifact(root)` rejects incomplete manifests, fingerprint mismatches,
duplicate samples and fold-count discrepancies. It lazily loads individual JSONL
records and checks each record against its indexed fingerprint on access.

- `artifact.fit_adapter()` returns JSON-safe state bound to the graph fingerprint,
  ordered labels and full train count. It fits only the training fold.
- `artifact.dataset('train', state)` and `artifact.dataset('validation', state)`
  return PyTorch datasets of PyG graphs using the same frozen preprocessing.
- `artifact.make_model(state)` constructs the untrained `EventGraphGNN` using the
  complete manifest label order, including classes absent from training samples.
- Test dataset/graph access requires an explicit `allow_test=True`. None is
  requested by default.

Persist the returned adapter state and `model.config` alongside future trained
weights. A matching embedding width alone does not establish vocabulary identity.
The tensorizer pairs tokens with exact units, reserves unknown ID zero, and emits
zero scaled value for units/tokens without a fitted numeric scale while retaining
observedness. Known z-scores are clipped at the documented configurable default of
10; source graph values remain unclipped. Times and deltas use signed log1p.
Reverse relations preserve edge identity with `inv:` prefixes and negate time
deltas. IDs, source references and targets do not enter model covariates.

Implementation source review was performed; no adapter fit, tensor dataset materialization,
model forward or training was executed. Graph production is separately authorized below. The editor could not resolve
Torch/PyG in its analysis environment; this is not evidence of runtime dependency
availability or failure. Resolve the actual interpreter in the testing phase.

## Deferred testing checklist — not run

- Cutoff filtering on both timestamps, late storetime, missing availability,
  ambiguous lab-to-visit joins, invalid IDs and overlapping history.
- Same tabular summary / different event order produces distinct edge/time payloads.
- Simultaneous-event behavior, unit separation, event-empty patients and budget errors.
- Training-only vocabulary/scalers, unknown values, target/ID independence, serialization.
- Model batch isolation, gradients, finite empty-edge behavior, edge-type/time use.
- Real bounded extraction, tensorization and model forward before any full extraction.
- Manifest/source binding checks and validation-only train/reload path.
- Same-input graph-free sequence/set control, degree-preserving random-edge control,
  time-order and knowledge-relation ablations before graph-benefit claims.
- Explicit canonical lineage and clinical cutoff review before comparisons with old runs.

## Authorized first-recorded-lab all-visits production

`comparison.standardized.event_graph_v1.first_lab` is a separate production
orchestrator using the existing graph materializer, not a change to legacy merges
or splits. The generic `build` entrypoint above retains its 26-item allowlist;
the new entrypoint requires explicit `--lab-scope all-numeric-known-unit`.

```sh
python3 -u -m comparison.standardized.event_graph_v1.first_lab \
  --raw-root 'data/Original CSVs' \
  --canonical comparison/canonical_split.json \
  --knowledge comparison/standardized/event_graph_v1/knowledge_seed.csv \
  --lab-scope all-numeric-known-unit \
  --output comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2 \
  --execute
```

- Retain the canonical **subject population and subject folds** (0=train,
  1=validation, 2=test). Expand to all eligible raw ED visits; no cohort size cap,
  subject deduplication, guessed visit labels or old-graph parity claim.
- Associate a lab using subject plus charttime inside exactly one valid closed ED
  interval. Ambiguous and unmatched associations are separately counted. A cutoff
  requires a nonempty `value` or `valuenum`, valid `charttime <= storetime`, and
  `intime <= charttime <= storetime <= outtime`. SQL upsert MIN retains the earliest
  available result across all raw-file chunks, regardless of numeric scope.
- Include finite numeric `valuenum` events for **all lab item IDs** with nonempty,
  non-placeholder units. Preserve exact units, values, repeated measurements and
  original source-row provenance. "Known" means explicitly present in the source,
  not clinically standardized against a unit ontology. This is **not all laboratory
  result types**: no categorical features are inferred from free-text results.
- A categorical/unknown-unit first result still sets the cutoff. Retain visits with
  no representable numeric index event; separately count index-empty and wholly
  event-empty graphs. Events with later availability cannot enter the index graph.
- Retain completed prior visits only when their finish precedes index arrival;
  their individual events also require both timestamps at/before sample cutoff.
  A prior event stored after its own visit may enter only later eligible graphs.
- Strict storetime only: vitals, medication reconciliation and untimestamped triage
  are not read or hashed because they are not used. No model preprocessing/training
  is performed; `raw_to_model_train_only`, `temporal_clean` and benchmark parity
  remain false. Targets and task labels are absent. The original class order is
  retained only as reference metadata.
- Intrinsically invalid ED intervals are data-ineligible, not repaired. Preserve
  exact source strings in SQLite `excluded_visits` and count excluded visits plus
  affected canonical subjects. The initial v1 production attempt failed on two
  end-before-start visits affecting two train-fold subjects; its failed manifest,
  snapshot and partial index are preserved. The revised v2 uses a fresh directory.
- Eligibility counters reconcile raw canonical visits into eligible visits,
  no-lab-charttime-in-window, no-eligible-recorded-result-in-window, missing-end,
  or invalid-interval exclusions. Lab counters are staged (later numeric/unit
  counters concern already time/lineage-eligible nonempty results); missing numeric
  and unknown unit counts can overlap and must not be summed as disjoint rows.
- Stream the large lab CSV once into SQLite; commit each chunk and persist progress
  plus counts. `prepared_index.json` binds the complete index and generated cohort.
  On materialization failure, `--resume-prepared` reuses only the complete index
  after full source, code, cohort and SQLite hash checks. An incomplete scan is not
  resumable. Existing partial JSONL is renamed, never silently overwritten.
- All histories are retained. Event/edge budgets fail rather than truncate; the
  explicit defaults are 1,000,000 events and 20,000,000 edges per graph. They are
  safety guards, not cohort subsampling limits.
- Before `status=completed`, read back **every** graph and reconcile its visit list,
  event multiset (source/value/exact unit/event and availability times), cutoff,
  subject fold, absent target and graph counts against the prepared index/cohort.
  Rehash every used raw source and implementation source after construction.

Additional outputs: `cohort.csv`, exact `canonical_split.json`,
`ingest_checkpoint.json`, `prepared_index.json`, `progress.jsonl`, and
`verification.json`. Consult the completed manifest and readback report for actual
counts and fingerprints; incomplete/failed outputs must not be consumed.

### Completed production result

The authorized v2 command exited **0**, with `manifest.json: status=completed`
and `verification.json: status=verified`. All 185,888 graphs were read back; no
cutoff, future-visit, event-payload or subject-fold violation was found. A separate
post-completion read independently reconciled the cohort/folds, raw-visit totals,
all recorded artifact hashes, source copies and private file permissions.

- Canonical population: 74,511 subjects; 244,847 raw visits.
- Eligible: 60,273 patients, 185,888 index visits/graphs; 32,827 patients have
  multiple eligible index visits. 14,238 canonical subjects have no eligible visit.
- Train / validation / test: **148,188 / 18,354 / 19,346** graphs.
- Excluded visits: 56,028 without in-window lab charttime; 2,929 without an eligible
  recorded result in-window; 2 invalid end-before-start visits. No visit was capped.
- Retained event-empty graphs: 2,142. Index-event-empty graphs: 8,504 (some have
  historical events). Prior-visit coverage: 132,934; knowledge coverage: 124,997.
- Index: 6,326,971 numeric events spanning 411 lab item tokens. Materialized graphs
  contain 34,890,445 nodes and 117,216,101 edges, including 1,101,878 medical edges.
- JSONL size: 18,358,584,329 bytes. Graph SHA-256:
  `5e16d2660b0910ab2ee25b6e4eddfced86594e27b0c423197c20ce28bcbbdfcb`.
- SQLite SHA-256:
  `d7e48ca96093d24102802e06f521145679cfca179e208309f89d8197c46cec9b`.
- Manifest SHA-256:
  `c98943834ddc9e5427abeb575d7e1da865c200ebf87c6dcf3ed1a4de2ca74ee7`.

The completed artifact is
`comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2/`;
`delivery_report.json` contains the aggregate handoff. **Targets remain absent.**
No unit/smoke tests, model fitting, training, evaluation or clinical validity
assessment was performed. The initial failed v1 artifact remains separate.
