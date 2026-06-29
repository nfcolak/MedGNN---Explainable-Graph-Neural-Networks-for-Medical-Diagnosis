# Aligned comparison: ProtGNN vs GraphCare

Both trained + evaluated on the **identical** canonical split
(`comparison/canonical_split.json`): same patients, same test set, same 30
classes, same metrics. Differences come from the **method**, not the data.

| Metric | ProtGNN (A) | GraphCare (B) | Winner |
|---|---|---|---|
| accuracy | 0.5476 | 0.5775 | GraphCare |
| balanced_acc | 0.5721 | 0.4454 | ProtGNN |
| macro_f1 | 0.4963 | 0.4704 | ProtGNN |
| micro_f1 | 0.5476 | 0.5775 | GraphCare |
| top3_acc | 0.7489 | 0.7645 | GraphCare |
| top5_acc | 0.8219 | 0.8403 | GraphCare |

- **macro_f1 / balanced_acc**: per-class balance (rare-class sensitivity).
- **accuracy / micro_f1 / top-k**: overall correctness + ranking.

Both use the same ontology+PMI KG signal, so this isolates the modelling
approach (prototype case-based reasoning vs KG bi-attention GNN).
