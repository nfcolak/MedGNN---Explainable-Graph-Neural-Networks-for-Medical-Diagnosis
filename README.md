# Self-Explainable Graph Neural Networks for Medical Diagnosis

The active task is the **max6 / train-derived Top-10 clinical task** on MIMIC-IV
ED patient graphs: patients with at most 6 total visits, the 10 most frequent
diagnoses by TRAIN-fold frequency, multi-visit clinical graphs, and
validation-only model selection. The held-out fold is never loaded.

All current methods and plugins are kept: ProtGNN, GSAT, GraphCare (native
clinical adapter), GCHM-PNA v2/v3, the XGBoost tabular control, CEI-GNN
(v1/v2/v3) and the GMT / GPS / label-attention / token-fusion / virtual-node
plugins. They share one artifact, class set, split and seed.

ADR-007 and ADR-008 are proposed, not accepted or enabled.

## Start here

- [Current runbook: clinical_graph_v2](comparison/standardized/clinical_graph_v2/README.md)
  (build, train, audits, output contracts)
- [Repository map and output ownership](STRUCTURE.md)
- [CEI-GNN v3 delivery report](docs/cei-v3-delivery-2026-10-01.md)
- [Max6/Top-10 cleanup and frozen-legacy policy](docs/max6-top10-cleanup.md)
- [Optional browser graph viewer](visualizer/README.md)

Real preprocessing, cache builds and training need explicit user approval and a
new, unoccupied output directory. Tests run only in an explicitly opened testing
phase.

## Preserved rebuild chain

These files stay in place, with unchanged paths, because the current task still
depends on them:

- `comparison/canonical_split.json` (fixed class order and subject folds; never regenerate)
- `comparison/standardized/icd_mapping.py`
- `comparison/standardized/clinical_graph_v2/` (all methods, plugins, CEI, audits)
- `comparison/standardized/event_graph_v1/` (`schema`, `first_lab`, `ingest`, `graph`, `knowledge_seed.csv`)
- `comparison/standardized/enriched_input_v1/spec.py`
- `comparison/standardized/event_graph_gchm_xgb_v1/` (`labels`, `local_labels_v2`)
- `comparison/standardized/gchm_v2_protocol/` (optional, frozen; does not enable ADR-008)
- `shared/lib/` and `shared/data_prep/`; `shared/data_prep/merge_ed.py` is
  hash-pinned by `local_labels_v2.py` and must stay byte-identical
- local data/evidence (not all tracked in Git): the max6 inputs under
  `comparison/standardized/event_inputs/`,
  `comparison/standardized/native_inputs/protgsat_snapshot_v1/contract.json`,
  the historical `binding_manifest.json`, and all result directories with their
  `source_snapshot` copies and binding/manifest files

Top-10 selection still starts from the original fixed class order, so those
class-order and label records are required even though they carry a "30-class"
or "native" name. Max6/Top-10 is a different task from the 30-class legacy
benchmark; scores are not comparable.

## Frozen-legacy policy

The old 30-class native/star/cooccur model code, the tests tied to it and the
legacy ProtGNN graph exporter are retired together (see
[the cleanup record](docs/max6-top10-cleanup.md)). Their original source is
recoverable from the Git refs and local archives listed there; it is not part
of the active runtime. Historical result and evidence directories are kept in
place. Keeping a result does not mean it can be reproduced: a historical run is
reproducible only if its source hashes were verified against a ref, snapshot or
archive, and byte-exact reproduction of some max6 sidecars is not demonstrated.
Old docs that name retired commands describe history; do not run those commands.
`comparison/standardized/build_explanation_cohort.py` is retained byte-identical
only as a frozen synthetic-fixture helper for shared cohort assertions, not as a
current command. HOLD `performance_diagnosis/` and `zero_concept_verification/`
checks depend on archived legacy runtime; their sources/results stay untouched
and are not active suite entrypoints.

## Requirements

```bash
conda env create -f environment.yml
conda activate protgnn-mimic
```

or `python3 -m venv .venv` and `pip install -r requirements-lock.txt`.
`requirements.txt` has flexible ranges; the lock file and `environment.yml` are
pinned. GraphXAI stays under `external/GraphXAI-main/` (used by the CEI
explanation path). The clinical GraphCare adapter is native code in
`clinical_graph_v2/methods/graphcare.py` and does not need the separate
`.venv-graphcare/`; that environment and `external/GraphCare/` are retained
legacy dependencies and are left in place.

## More links

- [Exact-input 30-class reference (historical)](docs/native-identical-input-v1.md)
- [Raw data workflow (legacy caveats)](docs/RUN_WITH_OWN_DATA.md)
- [Earlier working-tree cleanup](docs/cleanup-working-tree.md)

## Reference

This work builds on the AAAI 2022 paper "ProtGNN: Towards Self-Explaining Graph Neural Networks".

```bibtex
@article{zhang2021protgnn,
  title={ProtGNN: Towards Self-Explaining Graph Neural Networks},
  author={Zhang, Zaixi and Liu, Qi and Wang, Hao and Lu, Chengqiang and Lee, Cheekong},
  journal={arXiv preprint arXiv:2112.00911},
  year={2021}
}
```
