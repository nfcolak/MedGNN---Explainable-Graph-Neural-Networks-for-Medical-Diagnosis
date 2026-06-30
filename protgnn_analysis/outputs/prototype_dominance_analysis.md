# Prototype Dominance Analysis

Date: 2026-05-29

## Question

Why do clinical summaries almost always show `P4` and `P8` as the relevant prototypes?

## Run Context

- Model: prototype-enabled GCN / ProtGNN
- Dataset: `mimic_patient_sim`
- Task: binary classification, `HOME` vs `ADMITTED`
- Prototypes: 10 total
  - `P0`-`P4`: `HOME`
  - `P5`-`P9`: `ADMITTED`
- Prototypes per class: 5
- Test explanations inspected: 2000 graphs

## Key Findings

### 1. P4 and P8 dominate genuinely, not only because of summary formatting.

Top-1 nearest prototype over 2000 test explanations:

| Prototype | Count |
|---|---:|
| P4 | 1271 |
| P8 | 729 |

Closest prototype within each class:

| Class | Closest Prototype | Count |
|---|---|---:|
| HOME | P4 | 2000 / 2000 |
| ADMITTED | P8 | 2000 / 2000 |

This means every inspected test graph has P4 as the nearest HOME prototype and P8 as the nearest ADMITTED prototype.

### 2. The final classifier weights do not specially prefer P4 or P8.

The learned final layer weights are still symmetric:

```text
HOME logit:      P0..P4 = +1.0, P5..P9 = -0.5
ADMITTED logit:  P0..P4 = -0.5, P5..P9 = +1.0
```

So P4/P8 dominance is not caused by special last-layer weights. It is caused mainly by distance/activation: P4 and P8 are much closer to graph embeddings than the other prototypes.

### 3. Distance distributions show large prototype collapse / dead-prototype behavior.

Mean squared distance from test graph embeddings:

| Prototype | Mean Distance |
|---|---:|
| P0 | 3.717 |
| P1 | 18.036 |
| P2 | 6.160 |
| P3 | 8.969 |
| P4 | 0.856 |
| P5 | 5.960 |
| P6 | 5.294 |
| P7 | 5.282 |
| P8 | 1.311 |
| P9 | 6.615 |

Within the HOME class, P4 is always much closer than P0-P3.
Within the ADMITTED class, P8 is always much closer than P5-P7/P9.

### 4. The current training objective encourages only one prototype per class to become close.

The cluster loss uses the minimum distance over same-class prototypes:

```text
cluster_cost = mean(min(distance to correct-class prototypes))
```

This rewards at least one prototype per class being close, but it does not force all five same-class prototypes to cover different regions.

### 5. Diversity loss is computed but disabled.

The code computes a prototype diversity term `ld`, but the final objective multiplies it by zero:

```text
total_loss = loss + clst * cluster_cost + sep * sep_cost + 5e-4 * l1 + 0.0 * ld
```

Therefore there is no active penalty preventing one prototype from dominating its class.

### 6. Separation loss was set to zero in this run.

The run used:

```text
clst = 0.02
sep = 0.0
```

So there was a weak cluster pull toward correct-class prototypes, but no active separation pressure in this run.

## Interpretation

The most likely explanation is prototype under-utilization:

P4 became the central/active HOME prototype, and P8 became the central/active ADMITTED prototype. The remaining prototypes are technically present, but they sit farther from the learned graph embedding manifold and rarely influence the top explanation fields.

This is a known failure mode of prototype-style models when the objective only requires the nearest same-class prototype to be close and does not enforce prototype diversity or coverage.

## Recommended Follow-Up Experiments

1. Enable prototype diversity loss with a non-zero coefficient.
2. Use a non-zero separation weight, then compare prototype usage.
3. Track prototype usage during training, not only after training.
4. Report top-3 prototypes per class in the summary instead of only top-1.
5. Add a coverage loss or assignment balancing term so each class prototype is used by some training examples.
6. Inspect whether projection starts too late (`proj_epochs = 100`) relative to early stopping at epoch 27 in this run. If projection never starts, prototypes are never projected to representative training subgraphs.

## Important Note

This run stopped at epoch 27, while prototype projection starts at epoch 100. Therefore the MCTS prototype projection stage did not run for this checkpoint. That makes dominance by only one prototype per class more plausible.
