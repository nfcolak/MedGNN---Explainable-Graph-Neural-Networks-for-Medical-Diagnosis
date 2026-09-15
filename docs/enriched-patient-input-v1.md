# Versioned enriched patient input (source-snapshot v1)

## Delivered scope

A new identifier-free input artifact and an opt-in numerical patient-hub path are implemented. **This is a retrospective source-snapshot diagnostic, not an initial-triage model, temporally clean benchmark, or clinical deployment claim.** No performance improvement is claimed.

- Artifact: `comparison/standardized/enriched_inputs/source_snapshot_v1/{inputs.npz,manifest.json,source_snapshot/}`.
- Implementation: `comparison/standardized/enriched_input_v1/{spec,features,audit,build,model,run}.py`.
- Canonical cohort and order retained: **74,511 records: 59,607 train / 7,448 validation / 7,456 test**, with the original 30 ordered targets and 192 medication/complaint concepts.
- New payload: **127 patient-specific channels**: 32 continuous measurements with 32 missingness indicators, 30 historical-category indicators with 30 missingness indicators, and fixed F/M/unknown source-sex coding.
- Effective PyG feature width: **320**, comprising the original 193 identity channels and 127 numerical/binary hub channels. Concept IDs, node types, edges and graph membership are unchanged.
- Missing records are retained, including **552 train / 59 validation / 58 test concept-empty, hub-only graphs**.
- Original merged CSV, split, concept NPZ, existing PNA implementation, existing full runner, and prior checkpoints/results were not overwritten. The original PNA source still exactly matches its historical full-run snapshot.

## Feature schema and actual coverage

All counts below refer to the complete canonical cohort, not an explanation subset. The exact ordered allowlist, per-field statistics, coverage by fold, raw-source hashes and availability labels are stored in the manifest. Feature discovery does not accept arbitrary `lab_*` or `hx_*` columns.

| Block | Audited source | Observed records | Meaning and units |
|---|---|---:|---|
| Age | `age`, merged from `patients.anchor_age` | 74,498 | Anchor age in years; **not a reconstructed age at the index encounter**. Thirteen missing values are retained. |
| Source sex | `gender_F`, `gender_M` | 74,511 | 39,622 F / 34,889 M / 0 unknown in this artifact. Missing, conflicting or unrecognized coding maps to a separate unknown channel. This is not an inference about gender identity. |
| Systolic BP | `sbp` | 74,511 | Initial triage value, mmHg. |
| Diastolic BP | `dbp` | 74,511 | Initial triage value, mmHg. |
| Heart rate | `heartrate` | 74,511 | Initial triage value, beats/min. |
| Temperature | `temperature` | 74,511 | Initial triage value, Celsius; the source already applied Fahrenheit conversion. No second conversion is applied to model inputs. |
| Oxygen saturation | `o2sat` | 74,511 | Initial triage value, percent. |
| Laboratory results | 26 allowlisted `lab_*` means | 50,794 with at least one result | Whole-stay source summaries, **not first/early results**. See timing and unit limitations below. |
| Historical diagnoses | 30 audited historical category columns | 18,738 with at least one reconstructed positive | Indicators reconstructed from strictly prior, completed ED stays. Documentation timing remains unverified. |

The initial-vital coverage is partly a consequence of the original source's missing-vital exclusion; this new builder does not perform that exclusion. Respiratory rate and acuity are used only to corroborate encounter fingerprints, not added as model inputs. BMI, race, transport, pain, acuity, visit/medication counts, medication classes, whole-stay vital trends and laboratory abnormal flags are not added.

### Laboratory fields

Each field has its individual item ID and the explicit availability state `whole_stay_mean_result_availability_unknown` in `spec.py` and the manifest.

| Field | MIMIC item ID | Observed records |
|---|---:|---:|
| Albumin | 50862 | 14,863 |
| Alkaline phosphatase | 50863 | 15,764 |
| ALT | 50861 | 15,638 |
| Anion gap | 50868 | 49,342 |
| AST | 50878 | 15,792 |
| Bicarbonate | 50882 | 49,368 |
| Bilirubin | 50885 | 15,531 |
| BUN | 51006 | 49,669 |
| Calcium | 50893 | 14,913 |
| Chloride | 50902 | 49,383 |
| Creatinine | 50912 | 49,661 |
| Glucose | 50931 | 49,362 |
| Hematocrit | 51221 | 49,556 |
| Hemoglobin | 51222 | 49,528 |
| INR | 51237 | 21,804 |
| Lactate | 50813 | 18,506 |
| Magnesium | 50960 | 15,003 |
| MCV | 51250 | 49,519 |
| Phosphate | 50970 | 14,709 |
| Platelet count | 51265 | 49,482 |
| Potassium | 50971 | 49,349 |
| PTT | 51275 | 21,660 |
| RBC count | 51279 | 49,524 |
| Sodium | 50983 | 49,385 |
| Troponin T | 51003 | 3,082 |
| WBC count | 51301 | 49,538 |

`shared/data_prep/extract_ed_labs.py` restricts **charttime** to the complete ED stay and averages repeated values. It does not use result **storetime**, preserve `valueuom`, or establish availability before diagnosis. The raw `labevents.csv` exists and contains both fields, but the 18 GB source was not rescanned or rebuilt in this bounded task. Consequently, laboratory units are explicitly marked **source numeric unit unverified**; conventional units are not guessed or advertised as harmonized. Near-confirmatory laboratory measurements such as troponin change the information setting and must not be treated as evidence of superior early prediction.

No prediction time was specified and the diagnosis table has no documentation timestamp. No arbitrary minute cutoff, first-lab selection, or retrospective result-availability claim is introduced. Proper availability-safe laboratory extraction and unit verification remain separate work.

### Encounter lineage and historical-diagnosis audit

The original merge drops `stay_id`, `intime` and `outtime`, then selects a subject's first row after sorting by subject—not necessarily their first chronological visit. A subject-only join is therefore insufficient.

The new audit matches the exact subject plus initial triage fingerprint, then all available laboratory means, without using the current target or current ICD values in the matching key:

- Initial triage alone yields **74,507 unique matches**.
- Laboratory-summary corroboration raises this to **74,510 unique matches**.
- **One ambiguous record is retained**, without choosing an arbitrary visit; its entire historical block is missing.
- These are uniquely corroborated clinical fingerprints, not a claim that the original file retained encounter lineage.

Existing `hx_*` values were not simply copies of current targets: among **44,902 uniquely matched first raw visits**, none had a positive source history. Nevertheless, the legacy algorithm accumulates diagnoses in visit-start order and lacks a strict end/availability guard. **47 source-positive history values lack corroboration from an earlier completed visit.**

The model indicators are reconstructed independently of source history bits using `prior.intime < index.intime`, `prior.outtime < index.intime` and a different stay ID. Current-only, simultaneous-start and unfinished prior encounters cannot supply a positive history value. Unsafe source flags are removed from both values **and missingness**; flipping the original flags cannot change the reconstructed payload. A zero means no supported prior recorded category, not absence of disease. Only unavailable/ambiguous encounter lineage creates history missingness.

The 30 source headers are frozen, representative names for **three-character diagnosis categories**, not fine-grained phenotypes. For example, the header `hx_end_stage_renal_disease` represents category N18 and must not be interpreted as independently verified end-stage renal disease; `hx_hypertensive_urgency` represents I16. The manifest records every category-to-header mapping. This preserves the audited historical-source-summary meaning rather than inventing precise clinical phenotypes.

Earlier visit completion is an event-time check, **not proof that the diagnosis was recorded before the index encounter**. History remains `historical_source_summary_timing_unverified`. The source cohort, medication/complaint filtering and historical label vocabulary also retain upstream whole-source limitations. Therefore:

```text
preprocessing_fit_train_only = true
raw_to_model_train_only = false
temporal_clean = false
early_triage_eligible = false
```

## Preprocessing and leakage guards

- Numeric mean imputation and population standard deviations are fit using observed **canonical training rows only**. Constant/all-missing training fields use scale one; all-missing numeric fields use zero as the fallback center.
- Missingness is captured before imputation. No canonical patient is dropped.
- Initial vital bounds reproduce the audited source bounds. Nonfinite measurements become missing. All exported model values are finite float32.
- Binary historical indicators remain 0/1 and do not receive a numerical z-score. Sex uses a fixed documented F/M/unknown coding, with no holdout-fitted category discovery.
- Original medication/complaint vocabulary and presence arrays are reused exactly. The builder separately checks their association with the source rows, exact target/fold arrays, source hash and canonical split binding.
- Current diagnosis, current ICD columns, disposition, LOS, future visit counts and unrequested demographic aliases cannot enter the closed feature allowlist. Current labels are read only for unchanged target/order verification.
- The artifact binds feature specification, ordered names, scaler/imputer parameters, source/split/NPZ hashes and ordered-subject hash. No patient identifiers are stored in the model NPZ or aggregate reports.

## Model architecture: hub identity versus numerical state

The existing PNA implementation is untouched. The opt-in subclass replaces only the input encoder when numerical width is positive:

```text
initial node state = identity_projection(original one-hot ID)
                   + numeric_projection(hub payload; zero on concept leaves)

initial states -> unchanged residual PNA message passing
               -> optional unchanged medication–complaint pair-content residual
               -> unchanged patient-hub readout -> classifier
```

The same numerical projection is used in plain and interaction PNA. This remains a GNN, not a separate tabular classifier or ensemble. Numerical fields do not create node IDs, change immutable type metadata or alter graph membership.

With numerical width zero, parameter keys and baseline predictions are exactly compatible. With positive width and an all-zero numerical payload, the original path's predictions are also exactly reproduced under the same initial base weights. This is a compatibility control, **not** a claim that an all-zero vector represents a real patient.

### Explanations

All-zero edge masks remove concept messages but deliberately leave numerical hub information. That residual is legitimate patient self-information, not hidden concept leakage. Edge-only explanations consequently cannot explain the entire enriched predictor.

`explain_numeric` runs the actual vendored **GradExplainer, IntegratedGradExplainer and GNNExplainer**. It additionally retains signed per-channel Grad/IG attributions using the vendor's feature-aggregation callback, rather than replacing GraphXAI with a custom gradient algorithm. Node types remain fixed under feature perturbations. Integrated gradients uses the vendor's all-zero mathematical baseline; its sex/missingness pattern is not a validated clinical counterfactual.

## Integration boundaries

| Method | Delivered | Not claimed |
|---|---|---|
| Plain / interaction PNA | Shared enriched NPZ loader, separate numerical hub encoder, bounded real training, saved checkpoint, exact logit/metric replay | Full-data 30-epoch enriched benchmark or improvement |
| ProtGNN | Opt-in `build_comparator('protgnn', concepts, numeric_dim)` plus identical `numeric_graph` payload; actual forward/backward through the existing common-input path | Completed warm-up/prototype projection curriculum or enriched benchmark checkpoints |
| GSAT | Opt-in `build_comparator('gsat', concepts, numeric_dim)` plus identical payload; actual forward/backward through the existing common-input path | Completed information-bottleneck curriculum or enriched benchmark checkpoints |
| GraphCare | Original categorical embedding/defaults preserved; enriched payload exported independently for future integration | Numerical hub consumption or four-model input parity; requesting this unsupported adapter raises explicitly |

The all-row check verifies the shared numerical graph representation without changing the old topology. ProtGNN/GSAT separately consumed **eight real canonical training records**, with finite nonzero numerical gradients and changed logits after numeric ablation. Different learned encoders/readouts do not imply hidden-state equality.

## Commands

Run from the repository root. `python3` is the existing interpreter used here (Python 3.9 with Torch 2.8); no new dependency installation was required. `requirements.txt` recommends Python 3.10/3.11 for a fresh environment. The GraphCare regression uses its separate existing environment.

```bash
# Side-effect-free output planning (default action is also dry-run).
python3 -m comparison.standardized.enriched_input_v1.build \
  --output-dir comparison/standardized/enriched_inputs/source_snapshot_v1 --dry-run

# Creates a NEW input only. Already executed; repeating this exact target fails.
python3 -m comparison.standardized.enriched_input_v1.build \
  --output-dir comparison/standardized/enriched_inputs/source_snapshot_v1 --execute

# Explicit source locations can be supplied with --legacy-root, --source,
# --split and --raw-root. Missing raw lineage is disclosed; history becomes missing.

# Bounded smoke. Already executed in these directories; choose a NEW output
# directory for another authorized attempt. No automatic overwrite/resume.
python3 -m comparison.standardized.enriched_input_v1.run \
  --input-root comparison/standardized/enriched_inputs/source_snapshot_v1 \
  --output-dir comparison/standardized/enriched_runs/pna_smoke_v1/interaction \
  --architecture interaction --steps 4 --train-limit 128 --val-limit 32 \
  --batch-size 32 --execute

# Use --architecture plain and a distinct output directory for plain PNA.
# Without --execute, the input is validated and no run directory is created.

# Read-only exact replay.
python3 -m comparison.standardized.enriched_input_v1.run \
  --replay comparison/standardized/enriched_runs/pna_smoke_v1/interaction
python3 -m comparison.standardized.enriched_input_v1.run \
  --replay comparison/standardized/enriched_runs/pna_smoke_v1/plain

python3 -m pytest tests -q
.venv-graphcare/bin/python -m pytest graphcare_analysis/test_zero_concept_model.py -q
```

The enriched runner intentionally caps steps at 100, training records at 4,096 and validation records at 1,024. It is **not a full-training entrypoint**. Full enriched experiment orchestration and GraphCare integration remain pending. The existing fixed 30-epoch runner continues to consume its old bound input.

Example opt-in comparator consumption (no training loop or result claims):

```python
from comparison.standardized.enriched_input_v1.build import load_enriched
from comparison.standardized.enriched_input_v1.run import graph_subset
from comparison.standardized.enriched_input_v1.model import build_comparator
import numpy as np

bundle = load_enriched('comparison/standardized/enriched_inputs/source_snapshot_v1')
rows = np.flatnonzero(bundle['folds'] == 0)[:8]
graphs = graph_subset(bundle, rows)
model, device = build_comparator('gsat', len(bundle['contract']['names']),
                                bundle['hub_numeric'].shape[1])
```

## Executed verification, not performance evidence

- **426 tests passed, two existing tests skipped** in `python3 -m pytest tests -q`; **eight GraphCare tests passed** in its own interpreter. New behavior was introduced through observed failing tests and then passing implementations. The suites include source mapping, forbidden fields, holdout-outlier independence, unknown sex, missing inputs, history timing/current-visit exclusion, no-overwrite/dry-run, real train/replay, baseline compatibility, node permutation and batch isolation.
- All **74,511** enriched graphs were checked against original identity features, node types and edge indices. Every numerical leaf slot was zero and every hub payload matched the exported row.
- All **32 continuous-field** training statistics and transformed values were independently recomputed from canonical source rows.
- Both PNA variants completed **four optimizer steps on 128 train records** and evaluated **32 validation records**. Checkpoint reconstruction reproduced saved arrays and shared metrics exactly: **maximum logit difference 0.0**.
- Numerical projection gradient L1 summed across the four updates: plain **97.50179290771484**, interaction **106.84799766540527**. Numeric-ablation maximum logit differences: plain **0.7803680896759033**, interaction **0.6264700293540955**. These are connectivity/sensitivity checks, not clinical effect sizes or performance gains.
- All three named GraphXAI algorithms succeeded on mixed-concept and hub-only **synthetic** inputs for both reconstructed PNA models. Saved explanations contain synthetic inputs only.
- The old concept-only smoke checkpoint was replayed on its original 64 validation-smoke records with identical arrays/metrics and maximum logit difference **0.0**.
- **242 protected file hashes remained unchanged**, including the original source, split, common inputs and prior result/checkpoint artifacts. `git diff --check` and targeted compilation passed. No commit, push or full training was performed.
- Existing environment warnings were observed: LibreSSL/urllib3 and matplotlib/pyparsing deprecations, plus expected sklearn class-coverage warnings for tiny smoke cohorts. They are not hidden or treated as model-quality evidence.

Evidence: [final gates](enriched-input-evidence/final-gates.json), [aggregate verification](enriched-input-evidence/verification.json), [verification script](enriched-input-evidence/verify.py), [main test XML](enriched-input-evidence/regression.xml), [GraphCare test XML](enriched-input-evidence/graphcare-regression.xml). Training checkpoints, predictions, histories, manifests and code snapshots are in the two isolated smoke directories above. The new NPZ is 5,576,079 bytes.
