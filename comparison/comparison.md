# Aligned comparison on the canonical split

Every method is trained + evaluated on the **identical** canonical split
(`comparison/canonical_split.json`): same patients, same test set, same 30
classes, same metrics. Differences come from the **method**, not the data.

| Metric | Majority | XGBoost | Plain-GCN | GraphCare | ProtGNN | Best |
|---|---|---|---|---|---|---|
| accuracy | 0.1211 | 0.6540 | 0.5653 | 0.5775 | 0.5632 | XGBoost |
| balanced_acc | 0.0333 | 0.6173 | 0.5754 | 0.4454 | 0.5687 | XGBoost |
| macro_f1 | 0.0072 | 0.6066 | 0.5156 | 0.4704 | 0.5084 | XGBoost |
| micro_f1 | 0.1211 | 0.6540 | 0.5653 | 0.5775 | 0.5632 | XGBoost |
| top3_acc | 0.3362 | 0.8594 | 0.7701 | 0.7645 | 0.7696 | XGBoost |
| top5_acc | 0.4606 | 0.9218 | 0.8456 | 0.8403 | 0.8467 | XGBoost |

- **macro_f1 / balanced_acc**: per-class balance (rare-class sensitivity).
- **accuracy / micro_f1 / top-k**: overall correctness + ranking.

Majority + XGBoost come from `comparison/tabular_baseline_canonical.py`,
the prototype-free ablation from `comparison/plain_gcn_canonical.py`.
ProtGNN and GraphCare use the same ontology+PMI KG signal, so that pair
isolates the modelling approach (prototype case-based reasoning vs KG
bi-attention GNN).
