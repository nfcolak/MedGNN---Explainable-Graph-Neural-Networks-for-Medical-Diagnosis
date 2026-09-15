"""GSAT analysis package (Method C).

Graph Stochastic Attention (Miao, Liu & Li, ICML 2022) — an inherently
interpretable GNN whose explanation is the stochastic-attention weight on each
node, learned jointly with the classifier via the graph information bottleneck.

Parallel to protgnn_analysis / graphcare_analysis: same dataset
(mimic_intra_patient_disease), same canonical split, same seed, same shared
metrics, same three GraphXAI post-hoc explainers for an apples-to-apples
explainability comparison. GSAT uses ProtGNN's split-aware per-patient graph
representation and provenance-bound standardized cache.
"""
