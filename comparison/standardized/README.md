# Standardized ProtGNN / GSAT / GraphCare benchmark

This directory owns the auditable primary matrix: three methods × two common
topologies (`star`, `cooccur`) × three seeds (`1234`, `1235`, `1236`) = 18
runs. All commands below are run from the repository root.

The scientific contract is fixed by `benchmark_config.json` and
`comparison/canonical_split.json`: `data/merged_ed.csv`, 30-class `disease_1`,
59,607/7,448/7,456 train/validation/test subjects, highest validation macro-F1
checkpoint selection, and the six shared test metrics. Standardized
preprocessing is fitted on fold 0 only. Diagnosis-derived `symptom_*` and
other diagnosis/target inputs are excluded; pre-diagnosis
`chiefcomplaint_*` inputs remain eligible. The common graph-membership subset
is patient hub + medications + chief complaints. Vital nodes are excluded from
all standardized methods because GraphCare cannot consume the identical numeric
measurements; no categorical vital-name-only substitute is created. Legacy
graphs retain their historical vital behavior.

## Pipeline entry point

```bash
# Safe defaults: help or a printed command plan, no subprocesses/writes.
python3 -m comparison.standardized.run_all --help
python3 -m comparison.standardized.run_all --dry-run
```

After separate approval for full preprocessing and training:

```bash
python3 -m comparison.standardized.run_all --execute
```

This performs preflight, **builds missing caches then audits them**, runs each
of the 18 training cells separately, actually executes the explanation runner
for every `star`/`cooccur` × `1234`/`1235`/`1236` set, then summarizes. The fixed
`explanation_subjects.json` cohort is an existing input, not silently regenerated.
A failed command blocks every downstream command, including on the next launch.

Checkpoint state is per command under `comparison/standardized/checkpoint/`;
completed child outputs are validated before reuse. Old phase-only train/explain
checkpoint entries are not trusted as proof that individual runs completed.

```bash
# Reuse validated completed cells; failed commands remain blocked.
python3 -m comparison.standardized.run_all --execute --resume

# Retry after resolving the failure; never overwrite an occupied output.
python3 -m comparison.standardized.run_all --execute --resume --retry-failed

# Only after confirming all old workers have stopped: preserve each incomplete
# run/set under attempts/, then rerun it from scratch. No artifacts are deleted.
python3 -m comparison.standardized.run_all --execute --resume --retry-failed --archive-incomplete
```

The last command is **run-level restart**, not epoch-level resume: optimizer,
scheduler and last-epoch state are not restored. Completed valid cells are never
archived. Invalid completed artifacts fail validation instead of being silently
accepted or replaced. Run only one orchestrator at a time. Use `--state` and
`--events` together for a separate checkpoint journal; those flags do not change
scientific output locations. Dry-run and execute are mutually exclusive.

## Explanation cohort size

The standardized builder defaults to **500** test subjects (`--n`); the runner,
all three method CLIs, and `run_all` default to **500** (`--cohort-size`).
One prediction-independent, deterministic, stratified cohort (selection seed 1234)
is shared by ProtGNN, GSAT and GraphCare, regardless of the model seed. No method
resamples or truncates it. Counts must be positive integers and cannot exceed the
eligible test population. Manifests report the validated actual count.

The existing `explanation_subjects.json` remains a **50-subject** artifact. It is
not regenerated or relabeled by changing defaults: a 500 request rejects it.
To reuse it, explicitly pass `--cohort-size 50`. A new 500-subject artifact must
be deliberately built at a separate path, then supplied identically via `--cohort`:

```bash
# Preparation only, after approval; this writes a NEW cohort, not explanations.
python3 -m comparison.standardized.build_explanation_cohort --n 500 \
  --output comparison/standardized/explanation_subjects_500.json
# Safe command inspection only; no model execution or output writes.
python3 -m comparison.standardized.run_explanations --topology star --seed 1234 \
  --cohort comparison/standardized/explanation_subjects_500.json --cohort-size 500 --dry-run
# Explicit legacy compatibility:
python3 -m comparison.standardized.run_explanations --topology star --seed 1234 \
  --cohort-size 50 --dry-run
```

`run_all` accepts the same `--cohort`/`--cohort-size` options and validates the
requested count on explanation-output recovery. Changed commands require a new
checkpoint journal; occupied output sets are never overwritten. Use a separate
`run_explanations --output-root` for a newly sized explanation set. Dry-run only
prints commands, so it does **not** certify an existing cohort/checkpoint.
No real GraphXAI, training, cache rebuilding, or cohort migration was performed
for this default change. Legacy non-standardized explanation scripts retain
their separate historical options.

## 1. Verify the matrix without training

```bash
python3 comparison/standardized/run_benchmark.py --dry-run
```

The command prints JSON for exactly 18 unique cells and launches no subprocess.
It must not create `comparison/standardized/results/`.

## 2. Regenerate standardized caches first

Legacy graph/KG caches are intentionally not accepted for standardized runs:
they were fitted on all rows, lack current dataset/split/recipe provenance,
and may contain diagnosis-derived symptoms. The checked-in legacy outputs are
not modified. At present, both standardized `star` and `cooccur` caches must be
regenerated before a bounded smoke or the 18 full runs.

ProtGNN and GSAT share split-aware PyG patient-graph caches. GraphCare uses a
split-aware KG payload in each topology's cache directory. The maintained cache
CLI calls `get_dataset(..., canonical_split=...)` and
`build_global_kg(save=False, split_json=..., dataset_path=...)` directly:

```bash
# Inspect only (default); neither data nor output directories are created.
python3 -m comparison.standardized.build_caches --dry-run

# Full-dataset preprocessing: execute only after approval.
python3 -m comparison.standardized.build_caches --execute
```

Existing valid caches are reused. Stale GraphCare caches and partial PyG cache
pairs cause an actionable refusal before regeneration; they are **not** silently
overwritten. Preserve affected artifacts elsewhere explicitly before retrying.
`--structures star cooccur`, `--dataset-dir` and `--canonical-split` are explicit
options; alternate paths are useful for isolated fixtures, not a change to the
primary benchmark contract. Cache generation and parity auditing are separate
steps, both included in `run_all`.

These are full-dataset preprocessing steps, not training. They can take time and
were deliberately not run on production data during usability verification.

## 3. Audit real cache-record parity

Cache existence and provenance are necessary but not sufficient. After both
topologies are regenerated, run the production-record audit:

```bash
python3 comparison/standardized/audit_caches.py --structures star cooccur
```

The command loads the actual standardized ProtGNN/GSAT PyG records and builds
the exact GraphCare records used by its loaders. For every canonical subject it
extracts canonical node identities/types and typed edges, then requires the
three method fingerprints to match. It fails closed on missing/stale metadata,
subject membership, node membership, or typed-edge differences. Success prints
one `subject_count` and aggregate `parity_sha256` per topology. It does not
write or regenerate caches.

Current honest state: this audit fails at the missing v4 standardized `star`
ProtGNN/GSAT cache. Both v4 PyG caches and both v4 GraphCare KG cache files must
be regenerated with the commands in section 2; no production parity hash has
yet been claimed.

## 4. Lightweight scientific smokes

The non-training method smokes exercise the contracts without creating result
cells:

```bash
python3 -m pytest \
  tests/test_gsat_smoke.py::test_gsat_synthetic_forward_backward \
  tests/test_gsat_smoke.py::test_graphxai_wrapper_is_deterministically_faithful -q

PYTHONPATH=.:external/GraphCare .venv-graphcare/bin/python3 \
  graphcare_analysis/test_model_smoke.py

python3 -m pytest tests/test_method_contract.py \
  tests/test_protgnn_preprocessing.py -q
```

After cache regeneration, bounded one-epoch `star` and `cooccur` method runs can
be isolated in a temporary directory so they do not occupy real benchmark
cells:

```bash
SMOKE_ROOT="$(mktemp -d)"
for STRUCTURE in star cooccur; do
  PYTHONPATH=.:external/GraphXAI-main python3 -m protgnn_analysis.train \
    --graph-structure "$STRUCTURE" \
    --canonical-split comparison/canonical_split.json --seed 1234 \
    --output-dir "$SMOKE_ROOT/protgnn/$STRUCTURE" --limit 8 --max-epochs 1

  PYTHONPATH=. python3 -m gsat_analysis.train \
    --graph_structure "$STRUCTURE" \
    --canonical_split comparison/canonical_split.json --seed 1234 \
    --out_dir "$SMOKE_ROOT/gsat/$STRUCTURE" --limit 8 --max_epochs 1

  PYTHONPATH=.:external/GraphCare .venv-graphcare/bin/python3 \
    -m graphcare_analysis.run \
    --graph_structure "$STRUCTURE" \
    --canonical_split comparison/canonical_split.json --seed 1234 \
    --out_dir "$SMOKE_ROOT/comparison/standardized/results/graphcare/$STRUCTURE/seed_1234" \
    --limit 8 --max_epochs 1
done
```

Remove `SMOKE_ROOT` after inspection. Do not point smokes at
`comparison/standardized/results/`: the runner refuses to overwrite an existing
cell, and a bounded smoke is not a full benchmark result.

## 5. Run the full 18-cell matrix

Only after both standardized topologies have been regenerated, the audit in
section 3 succeeds, and the smokes pass:

```bash
python3 comparison/standardized/run_benchmark.py --continue-on-error
```

Every unbounded/full runner invocation performs the same production cache audit
again before reserving a result cell or launching training. A missing, stale, or
mismatched cache therefore blocks full runs. `--dry-run` intentionally skips
the cache audit so the 18-command matrix remains inspectable before caches
exist; bounded `--limit`/`--max-epochs` smokes are not accepted as full-run
parity evidence.

Each cell is isolated at
`comparison/standardized/results/<method>/<topology>/seed_<seed>/` and receives
a `run_manifest.json`. The GraphCare cells use the interpreter configured in
`benchmark_config.json` (`.venv-graphcare/bin/python3`). Do not treat dry-run or
smoke evidence as benchmark metrics.

## 6. Interruption and resume policy

The runner never overwrites a cell directory. On failure it records a failed
manifest; on an interrupt it marks the current manifest failed before exiting.
Completed cells remain valid.

1. Inspect existing manifests and preserve completed cells.
2. Move a failed/interrupted cell outside `comparison/standardized/results/`
   before retrying it; keep it as attempt evidence rather than editing its
   manifest.
3. Resume only missing cells with explicit filters, for example:

```bash
python3 comparison/standardized/run_benchmark.py \
  --methods gsat --structures cooccur --seeds 1235
```

Run one filtered cell at a time when resuming. A selected cell must not already
exist. Re-running the unfiltered command while any result cell exists will fail
closed rather than overwrite it.

## 7. Summarize completed and incomplete runs

```bash
python3 comparison/standardized/summarize.py \
  --results-dir comparison/standardized/results \
  --summary-csv comparison/standardized/summary.csv \
  --summary-aggregate-csv comparison/standardized/summary_aggregate.csv \
  --summary-md comparison/standardized/summary.md
```

The primary aggregate includes a method/topology cell only when all three seeds
are complete. Dataset, canonical split, class order, fold counts, metric schema,
and topology-policy fingerprints are validated before aggregation. Failed and
running manifests are listed separately.

For a no-training summarizer check, use the temporary-manifest fixture test:

```bash
python3 -m pytest \
  tests/test_benchmark_summary.py::test_cli_writes_three_deterministic_explicit_outputs_and_separate_sections -q
```

## Verification scope and current limitation

`shared.lib.canonical_graph` now has separate production adapters for cached PyG
records and GraphCare global-ID records. Vital-bearing production-shaped tests
cover `star`, `cooccur`, node mismatch, edge mismatch, and missing audit
metadata. Full-data per-subject fingerprint evidence cannot be produced until
the v4 standardized caches above are regenerated and the audit succeeds; no
full training result or production parity hash is claimed here.
