# PNA quality, fewer-class tasks, and candidate information improvements

## Executive conclusion

The medication–complaint interaction extension did not improve the completed PNA comparison. The evidence points to limited clinical information, uneven class difficulty and imbalance, and additional overfitting in the interaction model—not simply an insufficiently sophisticated GNN.

**The strongest next information experiment is to add decision-time-eligible patient measurements, not another interaction mechanism.** Start with age and initial vital signs; evaluate selected laboratory results as a separate, later-decision task if their result-availability times can be established. Better complaint detail and genuinely prior medical history are additional candidates. None of these individual feature blocks has yet demonstrated a PNA improvement.

A prior same-family logistic-regression diagnostic improved validation macro-F1 from **0.472141 to 0.552493** with a larger demographic/numeric feature bundle. This is evidence that the larger bundle contains predictive signal, **not** evidence that it is safe at triage, that any particular block caused the improvement, or that PNA will gain the same amount. The larger bundle contains timing-sensitive information and its fit reached the iteration limit.

## 1. Scope and verified model results

- Canonical cohort: **59,607 training / 7,448 validation / 7,456 test**.
- This analysis uses training and validation only. No new training or test inference was performed for the diagnostic analysis.
- Completed PNA comparison: star topology, seed 1234, 30 epochs per model, batch size 128, Adam learning rate 0.001, weight decay 0.00001, unweighted cross-entropy.
- Shared input: **104 medication + 88 complaint binary concepts**, encoded as 193-dimensional one-hot node features including a constant patient hub. No patient-specific numerical hub payload.
- Checkpoint selection: first maximum validation macro-F1 under the shared metric convention.

| Model | Selected epoch | Validation macro-F1 | Balanced accuracy | Accuracy | Parameters |
|---|---:|---:|---:|---:|---:|
| Standard PNA | 7 | 0.472846 | 0.464540 | 0.580559 | 145,758 |
| Interaction PNA | 19 | 0.470246 | 0.454446 | 0.574382 | 153,950 |

Interaction minus standard PNA macro-F1: **−0.002600**. Both runs completed 13,980 optimizer steps. Selected and final checkpoints were reloaded and reproduced the saved full-validation arrays exactly. One seed and unequal parameter counts limit conclusions about general superiority.

Source: [completed benchmark report](pna-performance-results.md), `comparison/standardized/pna_experiments/full_ce_seed1234_v1/{results.json,interaction/,plain/}`.

## 2. Why is quality limited?

### 2.1 The observable representation cannot distinguish some patients

Binary medication and complaint presence does not preserve dose, severity, onset, temporal order, or physiological measurements. There are **1,349 conflicting training signature groups containing 19,391 patients**: the exact same observed concept set occurs with different targets.

This demonstrates ambiguity in the available representation. It does **not** prove incorrect labels, establish a population/test performance ceiling, or quantify how much each missing feature would help. Multiple diagnoses can be compatible with similar observed symptoms.

The prior diagnosis audit also found **4,632 of 7,448 validation patients** with an exact concept combination absent from training. This is combination novelty, not proof of a new disease or distribution shift.

### 2.2 Some classes are learned well; others are barely detected

Selected standard-PNA checkpoint, full 30-class validation evaluation:

| Target | Validation support | Class F1 |
|---|---:|---:|
| Alcohol abuse with intoxication | 344 | 0.825291 |
| GI bleed | 302 | 0.781132 |
| Epilepsy | 97 | 0.773196 |
| Hypokalemia | 126 | 0.217687 |
| End stage renal disease | 43 | 0.000000 |
| Sepsis | 40 | 0.000000 |

Macro-F1 gives each class equal weight. Failure on rare or poorly separated targets therefore matters even when frequent classes produce reasonable accuracy. Training support ranges from **322 to 7,224** patients. Epilepsy performs well despite a smaller support than several weak classes, so class imbalance alone does not explain the ranking.

The most frequent standard-PNA confusion is skin/soft-tissue infection predicted as limb injury/pain (**111 validation patients**). This suggests a useful target for richer symptom/severity information, but is not evidence of annotation error.

### 2.3 Missing complaint information marks a difficult subgroup

| Standard-PNA validation subgroup | Patients | Accuracy |
|---|---:|---:|
| No retained complaint | 890 | 0.266292 |
| At least one retained complaint | 6,558 | 0.623208 |
| No retained medication | 2,190 | 0.671689 |
| Both medication and complaint | 4,427 | 0.592501 |
| Hub only | 59 | 0.118644 |

These subgroups have different class mixtures; their score differences are **associations, not causal feature-removal experiments**. In particular, fewer concepts are not automatically worse. The interaction block can create cross-type content only when both types exist: **4,427 of 7,448 validation patients**.

### 2.4 Additional interaction capacity fits training better than validation

All measurements below were recomputed in evaluation mode with fixed checkpoint weights, across the complete training and validation folds. They are not online training-loss averages.

| Model / checkpoint | Epoch | Train CE | Val CE | Train macro-F1 | Val macro-F1 |
|---|---:|---:|---:|---:|---:|
| Interaction / selected | 19 | 1.326111 | 1.541218 | 0.520032 | 0.470246 |
| Interaction / final | 30 | 1.226034 | 1.636591 | 0.544305 | 0.455655 |
| Standard / selected | 7 | 1.444399 | 1.534247 | 0.487140 | 0.472846 |
| Standard / final | 30 | 1.298852 | 1.574851 | 0.529463 | 0.466484 |

Training improves while validation deteriorates, particularly for the interaction model. This supports overfitting in the later epochs. It does not prove that parameter count alone caused the problem. Both models still have limited training performance, so overfitting is not a complete explanation of low absolute quality.

## 3. What happens if we predict fewer diseases?

### Method: a conditional diagnostic, not a new trained model

Choose the top K targets by **training support**, with canonical index as the tie-breaker. Do not choose classes by their validation F1. Retain validation patients whose true label belongs to these K targets, then restrict the existing model's output candidates to K.

This uses known true-label membership to define a closed-set cohort. It assumes the patient belongs to a selected target and says nothing about detecting excluded diseases. No patients, labels, or predictions in the original benchmark are changed.

Selected **standard-PNA** checkpoint:

| Retained targets | Validation patients | Coverage of original validation | Conditional accuracy | Conditional macro-F1 |
|---|---:|---:|---:|---:|
| 30 | 7,448 | 100.0% | 0.580559 | 0.472846 |
| 20 | 6,735 | 90.4% | 0.615887 | 0.560916 |
| 15 | 6,128 | 82.3% | 0.657637 | 0.637355 |
| 10 | 5,248 | 70.5% | 0.679497 | 0.673573 |
| 5 | 3,432 | 46.1% | 0.771270 | 0.767473 |

**These are not forecasts of retrained 20/15/10/5-class performance.** Much of the increase comes from removing difficult classes and changing the macro-F1 denominator. On the ten-class subset, even the unchanged 30-output predictions achieve macro-F1 **0.656803**; restricting candidate outputs raises it to **0.673573**. The original 30-class score is not a matched-cohort comparator for that increase.

Interaction-PNA conditional macro-F1 at K=10 is **0.669303**, still below standard PNA in that setting.

The ten most frequent training targets are limb injury/pain, back/spine pain, cardiovascular risk factor, head injury, diabetes mellitus, UTI/pyelonephritis, skin/soft-tissue infection, lower respiratory disease, alcohol abuse with intoxication, and GI bleed. These include broad categories and risk factors, not a uniform set of clinician-adjudicated disease phenotypes.

### An alternative that retains every patient: K targets plus Other

Sum the original probabilities of excluded classes into an Other output, keeping all **7,448 validation patients**. This is probability aggregation without retraining or recalibration, and changes the target task.

| Explicit targets + Other | Accuracy | Macro-F1 |
|---|---:|---:|
| 20 + Other | 0.584855 | 0.541547 |
| 15 + Other | 0.600967 | 0.589516 |
| 10 + Other | 0.640172 | 0.610223 |
| 5 + Other | 0.747046 | 0.647483 |

An Other prediction does not identify the underlying disease. At small K, a large Other class can make accuracy look attractive while sacrificing diagnostic detail. A 15-plus-Other experiment is a reasonable candidate for a deliberately narrower task, not an established optimum or a clinical recommendation. Final target selection should be based on intended use, not on dropping inconvenient classes.

## 4. Which information should we add?

### Direct evidence for an information improvement, with an important limitation

A previous logistic-regression comparison used the same 7,448 validation patients:

| Input view | Validation macro-F1 | Accuracy |
|---|---:|---:|
| Shared medication/complaint concepts | 0.472141 | 0.573845 |
| Concepts + native demographic/numeric bundle | 0.552493 | 0.624731 |

The richer bundle includes demographic channels and numerical lab/history/medication-class/vital-summary channels. These were added **together**, so no individual block's contribution is identified. The native fits reached their 150-iteration limit with convergence warnings. Critically, some values are stay-wide or not availability-time constrained. This result must not be advertised as a leakage-safe gain or a predicted PNA gain.

### Prioritized feature hypotheses

The following priorities are engineering hypotheses grounded in the current errors and available preprocessing code, not demonstrated per-feature improvements or clinical diagnostic rules.

| Priority / feature block | Candidate information | Why it may help this task | Required scope and availability check |
|---|---|---|---|
| **First baseline expansion: age and initial vital signs** | Age; first eligible temperature, heart rate, respiratory rate, oxygen saturation, systolic/diastolic BP; missingness flags | Adds patient-specific physiology absent from the concept-only hub; may help separate severity/hemodynamic/respiratory patterns | Verify selected encounter and recording availability. Use eligible initial observations, not whole-stay mean/min/max/std. Age must refer to the index encounter. |
| **High-potential later-decision block: selected lab results** | Potassium, creatinine, BUN, glucose, hemoglobin/hematocrit, WBC; lactate and troponin only where appropriate to the declared task | Matches weak targets such as hypokalemia, renal disease, anemia, infection-related targets and NSTEMI more directly than medication presence | Verify result availability before prediction. These are not triage inputs by default. Some measurements are near-confirmatory for target labels, changing the task to post-investigation classification. |
| **Richer presenting complaint** | Original initial complaint text or structured onset, duration, severity, location/laterality and relevant negation when actually recorded | Binary vocabulary compresses distinctions; may help confusions such as infection versus injury/pain | Use only initial text available at the decision time. Exclude assessment, final diagnosis and discharge text. Audit completeness; do not invent missing symptom fields. |
| **Genuinely prior history** | Previously documented kidney disease, diabetes, heart failure and other relevant history; prior values such as baseline creatinine if available | Could distinguish an acute presentation from pre-existing disease and contextualize current measurements | Require documentation before the index decision, not simply an earlier visit start. Never include current-encounter diagnosis codes as history. |
| **Medication detail, lower initial priority** | Documented dose, frequency, route, timing and relevant medication classes | Could add context lost by presence-only encoding | Verify actual coverage and availability; home-medication reconciliation may be recorded after arrival. More rare medications alone have not shown a reliable gain in the previous experiments. |

The existing preprocessing code explicitly contains vital fields and lab mappings, including potassium, creatinine, BUN, glucose, hemoglobin, WBC, lactate and troponin. This establishes that extraction logic exists—not that every canonical patient has a timely eligible value. Detailed symptom onset/severity and prior baselines require separate source-coverage checks.

There is no justified numerical forecast for adding an individual block. For example, the existing native-input LR comparison improved class F1 for NSTEMI **0.339623 → 0.759124**, hypokalemia **0.208589 → 0.506122**, and sepsis **0.160000 → 0.416667**. These are **bundled-input LR comparisons**, not isolated effects of troponin, potassium, or lactate and not PNA results.

## 5. How to add information without producing an unfair or leaky score

1. **Define the prediction moment first.** Separate an initial-assessment model from a model after initial laboratory results become available. Report their scores as different information settings. No specific minute-based cutoff is assumed here.
2. **Recover encounter lineage and availability.** The current merged snapshot drops selected-stay/time lineage and cannot establish timing from a subject-only join. If an encounter cannot be reliably recovered, report that blocker rather than infer it.
3. **Do not silently reuse unsafe summaries.** Existing labs average the full ED stay; vital trends use the full stay; total ED visit counts are not cut off at the index time. These cannot be relabeled as early inputs.
4. **Construct identical eligible features for every model.** For PNA, continuous values plus missingness flags can be encoded on the patient hub while retaining message passing. Give the same information to comparator models. Do not attribute a richer PNA input's advantage to the interaction block.
5. **Fit transformations on training data only.** Vocabulary, imputation, normalization and any feature filtering must use the training fold; preserve validation distribution and class ordering. Do not drop missing-value patients merely to improve scores.
6. **Test one information block at a time.** First standard PNA with the current input versus age/initial vitals; then separate complaint/history and eligible-lab ablations. Hold the 30-class task fixed during feature tests so label reduction is not confounded with information gain.
7. **Preserve the existing benchmark.** Create new input/output versions and explicit contracts; do not overwrite the current 192-concept cache or original results. Keep the original split. If temporal validation is later needed, create a separately named evaluation protocol.
8. **Select on validation and confirm stability.** Use macro-F1, balanced accuracy and per-class precision/recall, report missingness coverage, and repeat promising matched comparisons across seeds. Keep test data untouched during feature selection.
9. **Revisit explanations after changing the hub.** Numeric patient features create a deliberate self-feature prediction path. All-zero edge masks will no longer remove all patient-specific information. Evaluate feature attributions as well as edge attributions; do not claim that edges alone explain the entire predictor.

The present snapshot remains `temporal_clean=false` and `raw_to_model_train_only=false`. New features and a GNN modification do not fix upstream cohort-selection, vocabulary or timing limitations automatically.

## 6. Evidence and reproducibility

- [Class-scope aggregate evidence](pna-quality-evidence/class_scope.json): both models, K=30/20/15/10/5, unchanged-versus-restricted output diagnostics, coverage, K-plus-Other alternatives, per-class errors and input/prediction hashes. No subject-level rows.
- [Checkpoint-gap aggregate evidence](pna-quality-evidence/checkpoint_gaps.json): four reconstructed checkpoints, all train/validation measurements, exact saved-validation replay, input/code bindings and protected-file checks. No optimizer updates or test graphs.
- [PNA performance results](pna-performance-results.md).
- [Model performance diagnosis](model-performance-diagnosis.md), especially sections 1–4 and 8: bundled-input LR comparisons, signatures, class imbalance and target/availability caveats.
- [Performance improvement review](performance-improvement-review.md), sections 2 and 4: the 132 additional native hub channels and the prior LR input comparison.
- `shared/data_prep/extract_ed_labs.py`: lab mappings and whole-stay aggregation.
- `shared/data_prep/merge_ed.py`: initial vital fields, stay-wide vital summaries, history and cohort preparation.
- `pna_analysis/model.py`: current concept-only encoder and interaction mechanism.

**Status:** English analysis and aggregate evidence saved. No new feature pipeline, label remapping, training run, source-data modification, commit or push was performed as part of saving this report.
