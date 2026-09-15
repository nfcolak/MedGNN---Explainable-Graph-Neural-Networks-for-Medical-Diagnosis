# Standardized GNN Benchmark Design

## Goal

Compare ProtGNN, GSAT, and GraphCare under auditable, matched experimental conditions. The benchmark must separate model effects from graph-topology effects and must not overwrite legacy results.

## Scientific contract

Every standardized run uses:

- source data: `data/merged_ed.csv`
- target: `disease_1`, 30 classes
- frozen subject-aware split: `comparison/canonical_split.json`
- fold sizes: train 59,607; validation 7,448; test 7,456
- seeds: 1234, 1235, 1236
- checkpoint selection: highest validation macro-F1
- shared test metrics: accuracy, balanced accuracy, macro-F1, micro-F1, top-3 accuracy, top-5 accuracy
- no test-set tuning
- outputs isolated by method, topology, and seed under `comparison/standardized/results/`

## Topology contract

The benchmark distinguishes a topology policy from a model-specific tensor representation.

### Common benchmark topologies

- `star`: one patient hub connected bidirectionally to every observed clinical concept; no concept-to-concept edges.
- `cooccur`: `star` plus bidirectional concept pairs whose training-fold PMI exceeds 2.0.
- `ontology`: `star` plus bidirectional concept pairs sharing the configured therapeutic class or ICD chapter.
- `full`: record-local union of `cooccur` and `ontology`; no record-external nodes.

### Method-specific topology

- `full_kg_expanded`: GraphCare-only exploratory topology that adds record-external one-hop KG neighbours. It is excluded from the primary cross-method table.

The primary benchmark requires `star` and `cooccur`. `ontology` and record-local `full` are secondary ablations.

## Representation boundary

A canonical graph manifest defines subject ID, class ID, node identities/types, and typed edges. ProtGNN and GSAT consume the PyG patient graph. GraphCare converts the same node/edge policy to its global-ID representation. Model-native embeddings and encoders may differ; graph membership and topology may not.

A validator rejects comparisons if the same subject/topology has different canonical node or edge fingerprints across methods.

## Leakage policy

Feature vocabularies, prevalence filters, scaling statistics, PMI pairs, and other learned preprocessing artifacts must be fitted on the training fold only, then frozen for validation and test. Labels, diagnosis-derived target columns, and post-diagnosis ICD target leakage are excluded from model inputs.

## Reproducibility contract

Every run writes `run_manifest.json` containing:

- schema version
- method, topology, seed
- dataset path and SHA-256
- canonical split path and SHA-256
- topology parameters
- train/validation/test counts
- class ordering
- model parameter count
- device and package versions
- git commit and dirty state
- start/end timestamps and status
- metrics path and checkpoint path

A run fails before training if the canonical split, class ordering, topology, or seed is absent.

## Runner

`comparison/standardized/run_benchmark.py` supports:

- `--methods protgnn gsat graphcare`
- `--structures star cooccur`
- `--seeds 1234 1235 1236`
- `--dry-run`
- `--limit` and `--max-epochs` for smoke verification
- `--continue-on-error`

The runner creates explicit subprocess commands; it never relies on interactive defaults. It always passes the canonical split and seed.

## Reporting

`comparison/standardized/summarize.py` reads completed manifests and metrics, rejects incompatible runs, and produces:

- `summary.csv`: one row per run
- `summary_aggregate.csv`: mean and sample standard deviation by method/topology
- `summary.md`: table-first human-readable report

The primary comparison contains only common structures with three completed seeds per method. Incomplete or method-specific runs are listed separately.

## Explainability cohort

A fixed, stratified list of canonical test subject IDs (configurable count, default 500; legacy 50 only when explicitly requested) is stored independently of model predictions. Every method explains exactly that cohort with the same top-k rule and shared fidelity/sparsity implementation.

## Compatibility

Existing checkpoints, outputs, and comparison files remain untouched. GSAT implementation code is incorporated into the current branch, but its collaborator-generated logs and absolute machine paths are not copied.

## Acceptance criteria

1. Registry exposes all three methods and distinguishes `full` from `full_kg_expanded`.
2. GraphCare `star` contains an explicit patient hub and hub edges rather than an edgeless code set.
3. Every standardized command explicitly supplies method, structure, canonical split, and seed.
4. A dry run produces the full 18-run primary matrix without launching training.
5. Synthetic topology tests prove star/cooccur/full edge semantics and GraphCare hub behavior.
6. Manifest and summarizer tests reject mismatched split/topology fingerprints.
7. A one-graph or small-limit smoke run exercises each available method path without overwriting legacy outputs.
