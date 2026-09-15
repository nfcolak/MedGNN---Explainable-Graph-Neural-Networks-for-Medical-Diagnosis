# Zero-concept patient support: verified production result

## Outcome

**Resolved for the standardized star/cooccur paths.** The full production canonical-subject audit passes for ProtGNN, GSAT, and GraphCare on **74,511 subjects per topology**, retaining all **669 clinical-empty subjects**. No subject filtering, fabricated concept, source edit, relaxed canonical descriptor, or disabled parity assertion was used.

| Production audit | Subjects compared | Aggregate canonical parity SHA-256 |
|---|---:|---|
| star | 74,511 | `4926789149c80b267fe44675369b455e77acf883cc85bf333ed580a146e3a7b2` |
| cooccur | 74,511 | `2ef3cd09ea92f2310ebc448fd2f3b5ba7a62263400b824aeba953e190cb1d642` |

These are **subject/class, canonical node-membership, and typed-topology parity** results, not equality of method-native feature embeddings or benchmark accuracy. GSAT consumes the same PyG dataset as ProtGNN; its actual training import/construction path was traced. The auditor compares that shared representation against actual GraphCare adapter records, checks every identity and duplicate/membership invariant, and hashes sorted subject/fingerprint pairs. The audit code and canonical self-edge rejection were not weakened or changed for this repair.

Canonical train/validation/test counts remain **59,607 / 7,448 / 7,456**. Clinical-empty counts are **552 / 59 / 58** in both topologies. Their sorted subject-set SHA-256 is `cf4278fc835e4c320ef74654e0d98587c92c31f687adf6982217ea40a45d2d80`, identical to the previously blocked production report. No subject identifiers or patient records are reproduced here.

## Explicit clinical-empty policy

- Keep every canonical subject and its label/fold. A clinical-empty record has one reserved patient hub, no clinical nodes, no materialized edges, and zero EHR-code membership. The hub is not relabeled as a clinical concept.
- GraphCare's raw direct-EHR mean is `sum(indicator * embedding) / max(sum(indicator), 1)`. Therefore a clinical-empty record has an **exactly zero raw pooled EHR embedding** and finite gradients. Its subsequent learned affine projection retains its normal bias; its hub GNN representation is still computed normally. A zero raw mean does not require zero classification logits.
- Nonempty binary EHR membership follows the identical arithmetic and RNG/dropout order as upstream. Explanation masks change embedding weights, not the membership denominator.
- Shared explanation/fidelity semantics are unchanged: attribution is over graph nodes, including the real reserved hub. A hub-only graph has one eligible node and the existing minimum-one top-k policy; it is not represented as a zero-node graph or a fabricated clinical explanation. Keeping its sole node gives zero fidelity-minus degradation; masking it remains a finite model intervention. Truly missing/malformed graph input guards remain in place.
- GSAT's singleton training batch cannot estimate BatchNorm variance. Its new checkpoint-compatible `SingletonSafeBatchNorm1d` uses existing running statistics only when the entire 2-D node batch contains one node; no running statistics/counter update occurs in that branch. Affine and upstream gradients remain enabled. Every batch with at least two nodes retains normal BatchNorm behavior exactly.

## Root causes and changed paths

1. `graphcare_analysis/adapter.py` rejected all canonical rows with no fitted concepts. Replaced only that rejection with the hub-only policy; label, identity, KG vocabulary/schema, provenance, and split guards remain.
2. Upstream `external/GraphCare/graphcare_/model.py` divided its raw EHR sum by zero. **The vendor file was not edited.** New first-party `graphcare_analysis/model.py` inherits upstream parameter construction and owns its forward integration with the denominator clamp. `graphcare_analysis/run.py::build_graphcare_model` now constructs that class.
   - Actual training: `run.main` → `build_loaders` → `build_standardized_dataset`; `build_graphcare_model` → `_forward`.
   - Actual explanation: `explainability/explain_graphcare.py::main` → the same standardized dataset builder and model factory → `GraphCareGraphXAIWrapper` → functional call of the first-party model. The existing torch-1.x stateless fallback was exercised in the real environment.
   - Strict upstream state-dict loading, logits, and all populated gradients match exactly for nonempty records in both eval and matched-RNG training mode. A deterministic AST comparison confirms that the forward differs from upstream **only** by `clamp_min(1)` on the EHR denominator.
3. The first real audit after the GraphCare fix exposed another genuine defect: PyG's builder inserted a materialized `patient:hub` self-edge for clinical-empty records. Canonical topology correctly rejected it. `protgnn_analysis/load_dataset.py` now keeps standardized hubs edgeless, while preserving legacy nonstandardized behavior. Added recipe field `clinical_empty_graph_policy=retain_patient_hub_no_edges_v1` to both cache-path and dataset recipe construction so stale caches cannot masquerade as corrected ones.
   - ProtGNN training and standardized explanation both use `get_dataset` → `IntraPatientHeteroDataset`; `GnnNets` → configured GCN; explanation uses `ProtGNNWrapper`.
   - GSAT training imports that same dataset builder and constructs `GIN`/`ExtractorMLP`/`GSAT`; explanation uses deterministic `GSAT.predict` through `GSATGraphXAIWrapper`.
4. Real hub-only GSAT training-mode forward then exposed its singleton BatchNorm exception. Fixed the local GIN normalization integration in `gsat_analysis/models/gin.py`, not by dropping the last subject/batch or disabling training mode globally.

Tests added/changed:
- `tests/test_graphcare_topology.py`: former zero-concept rejection regression is now a positive retained-hub policy test for both star and cooccur; malformed-input regressions remain.
- `graphcare_analysis/test_zero_concept_model.py`: real BAT-GNN singleton, all-empty and mixed batches; finite CE/backward; raw zero pooling; nonempty upstream exact output/gradient equivalence; mixed/alone eval equivalence; wrapper equivalence and differentiability; complete hub-only explanation and fidelity.
- `tests/test_protgnn_preprocessing.py`: real standardized builder must retain an edgeless hub and version its recipe.
- `tests/test_hub_only_models.py`: real ProtGNN with prototypes enabled/disabled and mixed/singleton batches; real node-attention GSAT; finite forward/backward; singleton BatchNorm statistics preservation and non-singleton exact equivalence; both PyG explanation wrappers on hub-only inputs.

## RED → GREEN evidence

All local evidence is under `comparison/standardized/zero_concept_verification/`.

| Stage | Actual output | Log |
|---|---|---|
| New GraphCare retained-hub test, before adapter change | `2 failed, 44 deselected` — existing zero-concept rejection | `red_adapter.log` |
| Real GraphCare model before first-party integration | `5 failed, 2 passed` — NaN logits and `(False, nan)` wrapper equivalence | `red_model.log` |
| Adapter after policy change | `46 passed in 3.57s` | `green_adapter.log` |
| Initial GraphCare model GREEN | `7 passed` | `green_model.log` |
| First production audit after GraphCare repair | exit 1: `Self-edges are not canonical: 'patient:hub'` | `audit_first.log` |
| New PyG hub-only regression before builder change | `2 failed, 3 deselected` — `(2, 1)` instead of `(2, 0)` | `red_pyg.log` |
| New real PyG model probes before GSAT fix | `1 failed, 10 passed` — BatchNorm expected more than one value per channel | `green_pyg.log` (historical filename) |
| Final maintained suite | **`354 passed, 14 warnings in 89.01s (0:01:29)`** | `maintained_tests_final.log` |
| Final real GraphCare model suite | **`8 passed in 1.87s`** | `real_model_tests_final.log` |
| Full production audit, both topologies | exit 0; counts/hashes above | `audit_final.log` |
| Compile and tracked diff whitespace checks | both exit 0 | `compile.log`, `diff_check.log` |

The initial GraphCare pytest attempt found no pytest in `.venv-graphcare`; installed pytest into that environment, then obtained the genuine RED model failures above. No model dependencies were replaced. Dependency warnings in the maintained environment are the existing urllib3/LibreSSL and matplotlib/pyparsing deprecations; they are not hidden or counted as failures.

## Actual production model and cache verification

Beyond synthetic tests and the full parity audit, `probe_production.py` ran against the actual production source/caches:

- For each topology, GraphCare selected one clinical-empty subject from each canonical fold and a nonempty companion from that fold: **6 real forward/backward probes and 3 real hub explanation probes per topology**. All were finite; wrapper maximum absolute difference was **0.0**. Random initialized real production architecture; no optimizer or checkpoint training.
- For each topology and each of ProtGNN/GSAT, **3 real singleton forward/backward and explanation probes** used one actual clinical-empty record from each fold. All were finite. The factory/default architecture was used, and GSAT wrapper verification passed. Separate maintained tests cover prototypes on/off and mixed batches.
- Compared every newly generated PyG record against its preserved old v4 cache. For both topologies, **all 73,842 nonempty records have exactly identical edges and edge types**; all nonedge tensors/metadata fields and corresponding slices are exactly equal for **all 74,511 records**. The only record-content change is removal of the one fallback hub self-edge and its type for each of the 669 clinical-empty records. No patient features, source-row indices, labels, or node membership changed.
- Aggregate detail: `production_pyg.json`, `production_graphcare.json`. The first PyG probe used the dataset class's sample-CSV default and correctly failed closed before loading; the probe was corrected to explicitly pass `csv_filename='merged_ed.csv'`. This was a verification-script correction, not a production loader bypass (`production_pyg_first.log`).

## Cache and input preservation

Regenerated **only the two scientifically invalid PyG standardized caches and their metadata**, at new recipe-qualified paths. Both valid GraphCare KGs were reused byte-for-byte. All old standardized and legacy caches remain at their original paths; no cache was overwritten or deleted.

Before/after SHA-256, file sizes, resolved paths, and symlink status matched for **17 existing files**: source, split, upstream GraphCare model, all existing `.pt` graph caches and graph metadata files. Evidence: `preservation_before.json` and `preservation_after.json`. Four new files are recorded in `generated.json`.

| Preserved input | SHA-256 |
|---|---|
| `data/merged_ed.csv` | `95b055d53c5f879e5a5784b9d44cd3e6f8b0a073798fadc6bb25e5a228eaaf8b` |
| `comparison/canonical_split.json` | `33f6cd71399ab2605948d66f80902030cc0b36ae1017bdde551f4504e4bf82b0` |
| Both `data/graphs/{star,cooccur}/graphcare/kg.pt` | `9d75126e250f0f4078c4e908615db7427ca72e1b1d4c2ac622b5d094cbef2527` |

New PyG directories, relative to `data/graphs/<topology>/protgnn/`:
- star: `hetero_merged_ed_noLOS_prev10_pmi2_miss_disease_std_ds95b055d53c5f_split33f6cd71399a_recipe55d186a48d5d/`
- cooccur: `hetero_merged_ed_noLOS_prev10_pmi2_miss_disease_std_ds95b055d53c5f_split33f6cd71399a_recipe2dad34935a98/`

| New artifact | Bytes | SHA-256 |
|---|---:|---|
| star `data.pt` | 628447815 | `19f1d49bc7a9e230329aee31a9fc682f73a83bdb75b78aba19c0ee4436b4af7d` |
| star `metadata.json` | 22958 | `22c78034b7ca68d645767c7e7c01f23b255327bcc5148bc5fb29293ceb1e953e` |
| cooccur `data.pt` | 629268935 | `9d90ba4d2ad42c919e8483367f9314ebb1e31241f47210e5b03724e30f82d819` |
| cooccur `metadata.json` | 22965 | `c6f616fe5d8a3455079ea96a41c75056a0ad489498a07369136170613d8a47cd` |

## Exact commands

Working directory for every command:
`/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN`.

Main interpreter: `/Library/Developer/CommandLineTools/usr/bin/python3`, Python 3.9.6, torch 2.8.0, PyG 2.6.1. GraphCare interpreter: `.venv-graphcare/bin/python3`, Python 3.9.6, torch 1.12.0, PyG 2.3.0.

```bash
# RED/GREEN narrow checks; stdout/stderr retained in the logs named above.
/Library/Developer/CommandLineTools/usr/bin/python3 -m pytest tests/test_graphcare_topology.py -k retains_zero -q
.venv-graphcare/bin/python3 -m pip install pytest
.venv-graphcare/bin/python3 -m pytest graphcare_analysis/test_zero_concept_model.py -q
/Library/Developer/CommandLineTools/usr/bin/python3 -m pytest tests/test_graphcare_topology.py -q
/Library/Developer/CommandLineTools/usr/bin/python3 -u -m comparison.standardized.audit_caches --structures star cooccur
/Library/Developer/CommandLineTools/usr/bin/python3 -m pytest tests/test_protgnn_preprocessing.py -k hub_only -q
/Library/Developer/CommandLineTools/usr/bin/python3 -m pytest tests/test_hub_only_models.py tests/test_protgnn_preprocessing.py -q

# This evidence harness hashed existing files, ran tests, then built/audited caches.
/Library/Developer/CommandLineTools/usr/bin/python3 -u comparison/standardized/zero_concept_verification/verify_production.py
# Its actual subprocess commands (all exit 0; exact argv/timing also in commands.json):
/Library/Developer/CommandLineTools/usr/bin/python3 -m pytest tests/ -q
/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/.venv-graphcare/bin/python3 -m pytest graphcare_analysis/test_zero_concept_model.py -q
/Library/Developer/CommandLineTools/usr/bin/python3 -u -m comparison.standardized.build_caches --execute --structures star cooccur
/Library/Developer/CommandLineTools/usr/bin/python3 -u -m comparison.standardized.audit_caches --structures star cooccur

# Actual production-record/differential probes, not training jobs:
/Library/Developer/CommandLineTools/usr/bin/python3 -u comparison/standardized/zero_concept_verification/probe_production.py pyg
.venv-graphcare/bin/python3 -u comparison/standardized/zero_concept_verification/probe_production.py graphcare

# Final gates after adding the ProtGNN hub-explanation assertion:
/Library/Developer/CommandLineTools/usr/bin/python3 -m pytest tests/ -q
.venv-graphcare/bin/python3 -m pytest graphcare_analysis/test_zero_concept_model.py -q
/Library/Developer/CommandLineTools/usr/bin/python3 -m compileall -q graphcare_analysis/model.py graphcare_analysis/adapter.py graphcare_analysis/run.py gsat_analysis/models/gin.py protgnn_analysis/load_dataset.py tests/test_hub_only_models.py graphcare_analysis/test_zero_concept_model.py comparison/standardized/zero_concept_verification
git diff --check
```

`verify_production.py` is a one-shot evidence harness: its preservation snapshot is for this execution, and rerunning it intentionally refreshes those local evidence files. Use the maintained builder/auditor directly for routine reruns rather than replacing this historical evidence bundle.

## Review, limitations, and scope

- Large pre-existing dirty worktree preserved. No checkout, stash, reset, commit, push, optimizer step, short training, full training, or benchmark-result overwrite. Git state is recorded in `git_before.txt` / `git_after.txt`.
- Deterministic self-review included upstream-forward AST comparison, exact state-dict/numerical/gradient regression tests, all-record cache differential, maintained tests, full real audit, compile, and diff checks. An independent reviewer was not available in this subagent toolset; none is claimed.
- Verified the configured primary standardized GCN/GraphCare BAT/node-attention GSAT CPU paths. Alternate nonstandardized graph structures, GSAT edge-attention mode, accelerators, and upstream's standalone scripts are not claimed as tested. Legacy noncanonical cohort filtering was intentionally not redesigned.
- No checkpoint-based full explanation-cohort run and no trained-model accuracy/faithfulness result is claimed. Real-record explanation probes use initialized production models and establish finite execution and wrapper fidelity, not clinical quality. Training remains outside authorization.
- The first-party GraphCare class intentionally owns a forward copied from upstream to avoid monkeypatching or an untracked vendor-only change. When upgrading upstream, keep the exact-equivalence regression and review this integration. Its constructor remains upstream-owned; strict checkpoint key compatibility was verified against the current upstream architecture.
