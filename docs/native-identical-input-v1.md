# Exact native input for all four GNN methods — v1

## Delivered scope

The current comparison entrypoint is `python3 -m comparison.standardized.train_identical`.
It trains ProtGNN, GSAT, GraphCare, PNA plain and PNA interaction on one immutable
artifact. Default execution has **59,607 train / 7,448 validation / 7,456 test
members**; test members are retained in the artifact but never loaded for training
or validation selection. No full 30-epoch experiment was run for this integration.

The executed proof used the **same first 16 training and 16 validation ordinals**,
seed 1234, batch size 8, two epochs for every variant. It stopped after epoch one,
then restored each optimizer/RNG checkpoint and finished epoch two (2 → 4 updates).
Every selected checkpoint was reloaded; saved validation logits matched bitwise.
These are wiring checks, **not benchmark performance or improvement evidence**.

## Which native reference, exactly?

`gsat_analysis.train.get_dataset` is the same imported function as
`protgnn_analysis.load_dataset.get_dataset`. Both canonical star entrypoints resolve
`mimic_intra_patient_disease` through the same native cache:

```
data/graphs/star/protgnn/hetero_merged_ed_noLOS_prev10_pmi2_miss_disease_std_ds95b055d53c5f_split33f6cd71399a_recipe55d186a48d5d/data.pt
```

Its current native feature width is **331**, not 320. Its patient payload is
**11 demographic + 121 numeric = 132 channels**, not the earlier enrichment's 127.
The noncanonical historical 415-slot cache is a different legacy task/input and is
not the canonical benchmark reference. No existing cache was regenerated.

The native 331-slot layout, including redundant identity indicators, is retained:

| Zero-based slots | Native block |
|---|---|
| 0–3 | patient, med, icd, chiefcomplaint type indicators; ICD type is unused |
| 4–107 | 104 training-selected medication identity indicators |
| 108–195 | 88 training-fitted chief-complaint identity indicators |
| 196–198 | native value, abnormal flag, missing flag; last two are zero in this canonical view |
| 199–209 | 11 unchanged demographic indicators |
| 210–330 | 121 train-normalized patient numeric fields |

There are 193 categorical identities (192 concepts plus hub). Fixed templates
losslessly expand them into the native first 199 slots. All patients keep one hub;
**669 hub-only patients** remain in their original folds. Edges are the exact native
ordered bidirectional star edges; no extra concepts, diagnoses, neighbors or KG
edges are added for GraphCare. Packed NPZ contains only `templates`, `node_ids`,
`node_ptr`, `edges`, `edge_ptr`, `hub`, `y`, `folds`—no subject/stay identifiers.
Graph ordinals in bounded evidence are positions, not source identifiers.

The 30 ordered labels and every one of the 331 feature names are in `contract.json`.
`disease_*`, `symptom_*`, `icd_codes`, disposition, LOS and identifiers are forbidden
as feature columns. Current target-derived diagnosis nodes are absent. History
columns retain their **historical native upstream caveats**, described below.

## Exact fitting and equality proof

The builder does not independently preprocess each method. It reads the actual
native PyG cache, reconstructs all 121 numeric columns using canonical training
rows, and compares every fitted pandas mean/sample-standard-deviation (`ddof=1`)
exactly with native metadata. It then verifies **all 74,511 full native x tensors,
ordered edge tensors and labels**, exports the compact representation, reads it
back, and repeats the complete native digest. Missing numeric values are imputed
to native z=0; constant numeric columns follow the native zero rule. No new
missingness indicator is invented; lost raw missingness cannot be recovered by
equality. Demographics are not re-normalized.

Actual pre-encoder adapters were audited for every graph in each of the five
variants. GraphCare's actual categorical IDs plus continuous arguments are decoded
back to native x; its direct-EHR and visit memberships are checked against the
same concepts, excluding the hub. All five native tensor digests match:

```
f6c4c5f16a1f7a943b526dc56d939c93f10a35a99ce52246e461e2af57d8bf0c
```

This establishes **input information equality**, not parameter-count, computation,
learned-embedding, explanation-baseline or predictive-performance equality.

## Real consumers and training mechanisms

- **ProtGNN:** native 331-input GCN/prototype model; warm-up freezes the last layer
  for native warm epochs, then unfreezes it. Cluster, separation and L1 auxiliary
  terms remain. Native projection starts at zero-based epoch 20 and repeats every
  25 epochs, with up to 10 class-matching training candidates per prototype.
- **GSAT:** native 331-input GIN, stochastic attention, information loss and
  r curriculum 0.9 → 0.8 → 0.7 at epochs 0/10/20 remain.
- **GraphCare:** categorical embedding and original BAT alpha/beta, relation
  attention and joint readouts remain. `numeric_encoder: Linear(132, hidden,
  bias=False)` adds each node's numeric block immediately after `lin(node_emb)`
  and before the first BAT layer. Non-hub numeric values are zero. No batch
  modifies `node_emb.weight`; no separate numeric direct-EHR bypass is added.
  With zero numeric state, logits exactly recover the original categorical model.
- **PNA:** native 331-input linear encoder, residual PNA message passing and hub
  readout; plain and pair-content variants share this exact representation.
  Its degree histogram is fitted on all 59,607 training graphs, even for a bounded
  optimization check. Existing `pna_analysis` sources/checkpoints are untouched.

The production runner uses CPU in both interpreter environments, fixed epoch
budget and validation macro-F1 selection (not method-specific early stopping).
Initialization seed, batch size, training cohort/order and validation cohort are
shared. A dedicated epoch-seeded sampler prevents architecture RNG consumption
from changing patient order. Model capacities and method-native learning rates
are not claimed equal. PNA in this new runner uses Adam lr=0.001, weight decay=0;
this is an explicit new training contract, not replay of the old PNA optimizer.

`--loss ce` means shared unweighted classification CE. `--loss sqrt_inverse`
uses full-training-label square-root inverse-frequency weights. `--loss native`
uses inverse-frequency classification weights for ProtGNN and unweighted CE for
the other methods. **None removes model-specific auxiliary objectives.** Epoch,
model, optimizer, RNG and selected-checkpoint state are saved; interrupted partial
epochs replay from the last committed epoch. Resume rejects differing code,
input, source, split, feature/scaler/label/topology contract, seed or cohort.

### Explicit projection-science scope

Native MCTS leaves a root with `num_nodes <= min_atoms` unscored at P=0. The new
runner scores that one available whole-graph coalition and uses its native
embedding; larger graphs call unchanged native MCTS. No patients are dropped and
no legacy MCTS file is patched. This is a documented projection policy extension,
not a claim of bitwise training equivalence with that historical small-graph bug.
Tests execute real terminal-root scoring and GSAT curriculum/info gradients;
a fixture exercises the scheduled projection branch in the production trainer.
The two-epoch integrated proof itself does not reach epoch-20 projection.

## Explanation access

The new GraphXAI wrapper binds immutable native IDs and node types independently
of perturbed continuous features. GraphCare explanations therefore traverse the
numeric encoder rather than reusing its old categorical-only wrapper. Actual
vendored **GradExplainer and IntegratedGradExplainer** ran on the same one-graph
bounded probe for all five variants, with finite, nonzero hub numeric attribution.
This is not a full explanation cohort/fidelity study; GNNExplainer was not executed
for this integration. GraphCare categorical metadata remains fixed during IG,
whereas PyG models' continuous x includes identity templates; their baseline
semantics are explicitly **not claimed identical**. Edge-only interventions retain
observed hub values and GraphCare's existing direct-EHR categorical branch.

## Upstream temporal and clinical limitations — retained, not repaired

- Native `n_ed_visits` is a whole-subject count and may contain future visits.
- Native `hx_*` source flags are retained exactly. The archived raw-lineage audit
  found **47 positive values unsupported by a prior-ended stay** and one unresolved
  index row. Its earlier 127-field enrichment removed those values; **this exact
  native reproduction does not**. Documentation-time eligibility remains unverified.
- Labs/abnormal flags are whole-stay summaries; source extraction discarded result
  availability (`storetime`) and units (`valueuom`). No unit harmonization or early
  availability is asserted; raw labevents were not rescanned here.
- `vs_*` extrema/std use the stay, not just initial triage. Age/BMI source anchors
  and documentation times are not recovered by normalization.
- Source medication/complaint filtering and selected-stay lineage may already
  embody whole-cohort or retrospective decisions; fitting cache transforms on
  training rows does not undo them.

`reference_vs_enriched.json` preserves the previous aggregate timing audit and its
manifest hash, plus exact field-name differences. Its prior audit is evidence
about the same source snapshot, **not a new claim of temporal cleaning**. This
artifact must not support clinical early-prediction or leakage-free claims.

## Commands

From the repository root:

```bash
# Full cohort, actual model instantiation, no run directory written.
python3 -m comparison.standardized.train_identical \
  --output comparison/standardized/native_runs/full_v1 --dry-run

# Full training is opt-in; NOT executed as part of this integration.
python3 -m comparison.standardized.train_identical \
  --output comparison/standardized/native_runs/full_v1 \
  --epochs 30 --batch-size 128 --seed 1234 --loss ce --execute

# Reproduce the bounded stop/resume proof in a NEW namespace.
python3 -m comparison.standardized.train_identical \
  --output comparison/standardized/native_runs/my_check \
  --epochs 2 --limit 16 --batch-size 8 --stop-after 1 --execute
python3 -m comparison.standardized.train_identical \
  --output comparison/standardized/native_runs/my_check \
  --epochs 2 --limit 16 --batch-size 8 --resume --execute
python3 -m comparison.standardized.train_identical \
  --output comparison/standardized/native_runs/my_check \
  --epochs 2 --limit 16 --batch-size 8 --replay --execute

# Choose methods without changing data:
# --methods protgnn gsat graphcare pna
# --methods pna pna_interaction
```

The launcher selects `.venv-graphcare/bin/python` for GraphCare. Its direct worker
entrypoint is `comparison.standardized.native_reference_v1.run`; every worker
loads and verifies the authoritative artifact independently. Old native,
common-concept-only, PNA and enriched commands are historical reproduction paths.
They do not accept the new artifact. The legacy aggregator's schema guard rejects
these new manifests; the new launcher rejects any legacy artifact fingerprint.
No old checkpoint is silently switched to the new tensors.

## Evidence and paths

All paths below are relative to the repository root:

- `comparison/standardized/native_inputs/protgsat_snapshot_v1/inputs.npz`
- `comparison/standardized/native_inputs/protgsat_snapshot_v1/contract.json`
- `comparison/standardized/native_inputs/protgsat_snapshot_v1/native_audit.json`
- `comparison/standardized/native_evidence/parity_{method}.json` (five variants)
- `comparison/standardized/native_runs/integration_v1/integration.json`
- Each method directory: `cohort.npz`, `last.pt`, `best.pt`, `run_manifest.json`,
  `history.json`, `validation_*.npz`, `replay.json`, exact `source_snapshot/`.
- `comparison/standardized/native_evidence/graphxai_{method}.json`
- `comparison/standardized/native_evidence/reference_vs_enriched.json`
- Regression logs and before/after preservation hashes in `native_evidence/`.

Contract SHA256:
`2a2566e662bf87b10d97e58700f868d8d6d6b036582ec046dd5453e6dcd9f7c0`.
NPZ SHA256:
`e8c922b68083a8fff7d83fc7b9bb85c552ee2b4d0c9ae34b596e2c1fb8fb45eb`.

Measured regression: main interpreter **446 passed, 4 skipped**; legacy GraphCare
selected native/topology/zero-concept tests **70 passed, 6 skipped**. Skips mark the
opposite model environment. Existing third-party deprecation/SSL and tiny-cohort
metric warnings are not suppressed; a scalar logging warning does not change the
training gradient. One initially mistyped GraphCare test path failed collection;
the corrected actual test command succeeded and both logs are retained.

## Exact ordered hub fields and fitted transforms

Full-precision scaler values and exact feature names are authoritative in
`contract.json`; this table rounds numeric statistics for readability. Demographic
slots are followed by alphabetically ordered native numeric slots, unchanged.

| Native x slot | Source field | Native transform |
|---|---|---|
| 199 | `gender_F` | native demographic value, unchanged |
| 200 | `gender_M` | native demographic value, unchanged |
| 201 | `race_ASIAN` | native demographic value, unchanged |
| 202 | `race_BLACK` | native demographic value, unchanged |
| 203 | `race_HISPANIC` | native demographic value, unchanged |
| 204 | `race_NATIVE` | native demographic value, unchanged |
| 205 | `race_OTHER` | native demographic value, unchanged |
| 206 | `race_WHITE` | native demographic value, unchanged |
| 207 | `transport_AMBULANCE` | native demographic value, unchanged |
| 208 | `transport_HELICOPTER` | native demographic value, unchanged |
| 209 | `transport_WALK IN` | native demographic value, unchanged |
| 210 | `age` | z-score; mean=50.4598462984, std=20.4308844782 |
| 211 | `bmi` | z-score; mean=28.9201455253, std=7.22884202928 |
| 212 | `hx_acute_kidney_failure` | z-score; mean=0.0153002164175, std=0.122745152886 |
| 213 | `hx_acute_upper_respiratory_infection` | z-score; mean=0.00770043786804, std=0.0874143541922 |
| 214 | `hx_alcohol_abuse_with_intoxication` | z-score; mean=0.016323586156, std=0.126717781226 |
| 215 | `hx_anemia` | z-score; mean=0.012565638264, std=0.111390983305 |
| 216 | `hx_anxiety_disorder` | z-score; mean=0.00840505309779, std=0.0912937457069 |
| 217 | `hx_athscl_heart_disease_of_native_coronary_artery_w_o_ang_pctrs` | z-score; mean=0.0149982384619, std=0.121546448546 |
| 218 | `hx_cellulitis_of_unspecified_part_of_limb` | z-score; mean=0.0248292985723, std=0.155605946927 |
| 219 | `hx_chronic_obstructive_pulmonary_disease_w_acute_exacerbation` | z-score; mean=0.00969684768567, std=0.0979947954513 |
| 220 | `hx_contusion_of_unspecified_part_of_head` | z-score; mean=0.00699582263828, std=0.0833486511623 |
| 221 | `hx_dehydration` | z-score; mean=0.0127334037949, std=0.112122589734 |
| 222 | `hx_end_stage_renal_disease` | z-score; mean=0.00951230560169, std=0.0970668826746 |
| 223 | `hx_essential_primary_hypertension` | z-score; mean=0.0345596993642, std=0.182663314074 |
| 224 | `hx_familial_hypercholesterolemia` | z-score; mean=0.0395087825255, std=0.194803683932 |
| 225 | `hx_gastrointestinal_hemorrhage` | z-score; mean=0.013689667321, std=0.116200201612 |
| 226 | `hx_heart_failure` | z-score; mean=0.0148136963779, std=0.120807680314 |
| 227 | `hx_hypertensive_urgency` | z-score; mean=0.0806281141477, std=0.272265431109 |
| 228 | `hx_hypokalemia` | z-score; mean=0.0172295200228, std=0.13012666037 |
| 229 | `hx_low_back_pain` | z-score; mean=0.0392571342292, std=0.194207735162 |
| 230 | `hx_major_depressive_disorder` | z-score; mean=0.0138071031926, std=0.116690597459 |
| 231 | `hx_other_chronic_pain` | z-score; mean=0.00577113426275, std=0.0757490893323 |
| 232 | `hx_other_specified_injuries_of_head` | z-score; mean=0.0128676162196, std=0.11270427575 |
| 233 | `hx_pain_in_unspecified_knee` | z-score; mean=0.0271947925579, std=0.162652020125 |
| 234 | `hx_pain_in_unspecified_limb` | z-score; mean=0.0315399198081, std=0.174773183634 |
| 235 | `hx_pneumonia` | z-score; mean=0.0226315701176, std=0.148727109986 |
| 236 | `hx_type_1_diabetes_mellitus_without_complications` | z-score; mean=0.0124482023923, std=0.11087583547 |
| 237 | `hx_type_2_diabetes_mellitus_without_complications` | z-score; mean=0.0499773516533, std=0.217900005812 |
| 238 | `hx_unspecified_asthma_with_acute_exacerbation` | z-score; mean=0.00947875249551, std=0.0968971788173 |
| 239 | `hx_unspecified_atrial_fibrillation` | z-score; mean=0.0172462965759, std=0.13018888654 |
| 240 | `hx_unspecified_open_wound_of_other_part_of_head` | z-score; mean=0.00867347794722, std=0.0927275200731 |
| 241 | `hx_urinary_tract_infection` | z-score; mean=0.0290402133978, std=0.167920673111 |
| 242 | `lab_albumin` | z-score; mean=3.95182753966, std=0.633406849692 |
| 243 | `lab_albumin_abn` | z-score; mean=0.204269321633, std=0.40318368203 |
| 244 | `lab_alk_phos` | z-score; mean=108.054439059, std=113.337623283 |
| 245 | `lab_alk_phos_abn` | z-score; mean=0.242650564119, std=0.428702542862 |
| 246 | `lab_alt` | z-score; mean=41.2175398159, std=113.999695775 |
| 247 | `lab_alt_abn` | z-score; mean=0.202961184474, std=0.402219951918 |
| 248 | `lab_anion_gap` | z-score; mean=15.4092770289, std=3.29739840805 |
| 249 | `lab_anion_gap_abn` | z-score; mean=0.0994085243571, std=0.299213539716 |
| 250 | `lab_ast` | z-score; mean=51.9280307595, std=142.72794663 |
| 251 | `lab_ast_abn` | z-score; mean=0.282860313937, std=0.450406970836 |
| 252 | `lab_bicarbonate` | z-score; mean=24.3611778139, std=3.52294195891 |
| 253 | `lab_bicarbonate_abn` | z-score; mean=0.197097330762, std=0.397811498507 |
| 254 | `lab_bilirubin` | z-score; mean=0.913531499556, std=2.17594689753 |
| 255 | `lab_bilirubin_abn` | z-score; mean=0.0875211744777, std=0.282608671125 |
| 256 | `lab_bun` | z-score; mean=19.5512728236, std=15.0671638666 |
| 257 | `lab_bun_abn` | z-score; mean=0.302456370423, std=0.459327590343 |
| 258 | `lab_calcium` | z-score; mean=9.19909052702, std=0.732724574844 |
| 259 | `lab_calcium_abn` | z-score; mean=0.143985878793, std=0.351090168558 |
| 260 | `lab_chloride` | z-score; mean=101.313172755, std=4.65669250166 |
| 261 | `lab_chloride_abn` | z-score; mean=0.13635210553, std=0.343166426453 |
| 262 | `lab_creatinine` | z-score; mean=1.13584054279, std=1.15852346858 |
| 263 | `lab_creatinine_abn` | z-score; mean=0.198552223371, std=0.39891509337 |
| 264 | `lab_glucose` | z-score; mean=124.450917096, std=63.7738852124 |
| 265 | `lab_glucose_abn` | z-score; mean=0.615956555942, std=0.486374423804 |
| 266 | `lab_hematocrit` | z-score; mean=37.8883016198, std=6.10499488435 |
| 267 | `lab_hematocrit_abn` | z-score; mean=0.396280292118, std=0.48913011351 |
| 268 | `lab_hemoglobin` | z-score; mean=12.4945240082, std=2.22752960022 |
| 269 | `lab_hemoglobin_abn` | z-score; mean=0.437078055071, std=0.496031299585 |
| 270 | `lab_inr` | z-score; mean=1.3971461503, std=0.918660211658 |
| 271 | `lab_inr_abn` | z-score; mean=0.421911018903, std=0.493878697637 |
| 272 | `lab_lactate` | z-score; mean=1.72096300805, std=0.893680674833 |
| 273 | `lab_lactate_abn` | z-score; mean=0.252113342801, std=0.434240667441 |
| 274 | `lab_magnesium` | z-score; mean=1.98469708365, std=0.298396443763 |
| 275 | `lab_magnesium_abn` | z-score; mean=0.086989220356, std=0.281831036636 |
| 276 | `lab_mcv` | z-score; mean=90.1195281325, std=6.85514007685 |
| 277 | `lab_mcv_abn` | z-score; mean=0.169882412442, std=0.375534212288 |
| 278 | `lab_phosphate` | z-score; mean=3.49933940581, std=1.01619067769 |
| 279 | `lab_phosphate_abn` | z-score; mean=0.243892057547, std=0.429446645715 |
| 280 | `lab_platelet` | z-score; mean=243.480176917, std=95.2847403283 |
| 281 | `lab_platelet_abn` | z-score; mean=0.160237914452, std=0.366831202427 |
| 282 | `lab_potassium` | z-score; mean=4.28904246947, std=0.740116640263 |
| 283 | `lab_potassium_abn` | z-score; mean=0.1233467875, std=0.328839022239 |
| 284 | `lab_ptt` | z-score; mean=32.9402855153, std=12.6696386339 |
| 285 | `lab_ptt_abn` | z-score; mean=0.238974001857, std=0.426469205034 |
| 286 | `lab_rbc` | z-score; mean=4.23016258724, std=0.736553665294 |
| 287 | `lab_rbc_abn` | z-score; mean=0.484474562557, std=0.499765221168 |
| 288 | `lab_sodium` | z-score; mean=138.354408452, std=4.03605553195 |
| 289 | `lab_sodium_abn` | z-score; mean=0.100705189995, std=0.300941775181 |
| 290 | `lab_troponin_t` | z-score; mean=0.219124236253, std=0.567862855037 |
| 291 | `lab_troponin_t_abn` | z-score; mean=0.910386965377, std=0.285684762224 |
| 292 | `lab_wbc` | z-score; mean=9.12098286061, std=5.18825943579 |
| 293 | `lab_wbc_abn` | z-score; mean=0.317685423934, std=0.465582296782 |
| 294 | `medclass_acid_suppression` | z-score; mean=0.217944201184, std=0.412852741149 |
| 295 | `medclass_antibiotic` | z-score; mean=0.072994782492, std=0.260130120231 |
| 296 | `medclass_anticoagulant` | z-score; mean=0.0690858456222, std=0.253602189512 |
| 297 | `medclass_anticonvulsant` | z-score; mean=0.134061435737, std=0.340721168686 |
| 298 | `medclass_antidepressant` | z-score; mean=0.228395993759, std=0.419802596939 |
| 299 | `medclass_antidiabetic` | z-score; mean=0.123441877632, std=0.328946493824 |
| 300 | `medclass_antihypertensive` | z-score; mean=0.365644974584, std=0.481614595409 |
| 301 | `medclass_antiplatelet` | z-score; mean=0.206821346486, std=0.405029664717 |
| 302 | `medclass_antipsychotic` | z-score; mean=0.0445081953462, std=0.206223008816 |
| 303 | `medclass_benzodiazepine` | z-score; mean=0.134027882631, std=0.34068512809 |
| 304 | `medclass_corticosteroid` | z-score; mean=0.052913248444, std=0.223862183783 |
| 305 | `medclass_diuretic` | z-score; mean=0.160987803446, std=0.367522783847 |
| 306 | `medclass_lipid_lowering` | z-score; mean=0.24784001879, std=0.431762054071 |
| 307 | `medclass_opioid` | z-score; mean=0.133155501871, std=0.339745567536 |
| 308 | `medclass_respiratory` | z-score; mean=0.160601942725, std=0.367166475029 |
| 309 | `medclass_stimulant` | z-score; mean=0.0220108376533, std=0.146719875352 |
| 310 | `medclass_thyroid` | z-score; mean=0.0926401261597, std=0.289929893959 |
| 311 | `n_ed_visits` | z-score; mean=3.28060462697, std=5.29809100165 |
| 312 | `n_medications` | z-score; mean=6.34442263493, std=6.96423750467 |
| 313 | `vs_dbp_max` | z-score; mean=82.2552720318, std=13.867550559 |
| 314 | `vs_dbp_min` | z-score; mean=67.4785176238, std=13.6849158743 |
| 315 | `vs_dbp_std` | z-score; mean=6.91696089385, std=5.94082749332 |
| 316 | `vs_heartrate_max` | z-score; mean=87.1152515644, std=17.6589276898 |
| 317 | `vs_heartrate_min` | z-score; mean=73.5565336286, std=13.9122526078 |
| 318 | `vs_heartrate_std` | z-score; mean=6.47008139984, std=6.05372096314 |
| 319 | `vs_o2sat_max` | z-score; mean=99.0772761588, std=1.33668784632 |
| 320 | `vs_o2sat_min` | z-score; mean=97.1597647927, std=2.62765508573 |
| 321 | `vs_o2sat_std` | z-score; mean=0.905931501334, std=0.999999760354 |
| 322 | `vs_resprate_max` | z-score; mean=18.6901706175, std=3.13559634133 |
| 323 | `vs_resprate_min` | z-score; mean=16.0615028436, std=1.86719117622 |
| 324 | `vs_resprate_std` | z-score; mean=1.20839399735, std=1.27836795169 |
| 325 | `vs_sbp_max` | z-score; mean=139.125371181, std=21.475675671 |
| 326 | `vs_sbp_min` | z-score; mean=120.370073985, std=18.285148858 |
| 327 | `vs_sbp_std` | z-score; mean=8.77555530391, std=7.38529713986 |
| 328 | `vs_temperature_max` | z-score; mean=36.9359240693, std=0.508560837986 |
| 329 | `vs_temperature_min` | z-score; mean=36.5738504706, std=0.388334720207 |
| 330 | `vs_temperature_std` | z-score; mean=0.194379250759, std=0.239067372542 |
