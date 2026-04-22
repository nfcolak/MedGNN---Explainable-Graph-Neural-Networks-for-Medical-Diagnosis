# DS1 Patient Similarity Graph Variants

This directory stores patient-similarity graph datasets derived from `datasets/ds1/raw/ds1.csv`.

Each subdirectory contains:

- `processed/<variant>_ps_data.pt`: PyTorch Geometric `Data` object
- `raw/ds1.csv`: source CSV snapshot
- `metadata.json`: graph construction details
- `feature_columns.txt`: node feature names

## Variants

- `k5`: directed KNN graph with `k=5`
- `k10`: directed KNN graph with `k=10`; includes the existing training and GraphXAI analysis under `k10/train`
- `k20`: directed KNN graph with `k=20`
- `k30`: directed KNN graph with `k=30`
- `mutual_k10`: mutual KNN graph with `k=10`
- `mutual_k20`: mutual KNN graph with `k=20`

All variants use standardized DS1 clinical features, cosine similarity, and the raw `mortality_label` as the node label.

See `graph_variants_summary.json` for graph-level statistics.
