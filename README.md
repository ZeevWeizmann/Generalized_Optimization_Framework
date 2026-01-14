# Generalized Optimization Framework for Graph-based Semi-Supervised Learning

This repository provides a scalable implementation of the **Generalized Optimization Framework for graph-based semi-supervised learning** proposed by **Avrachenkov et al. (SIAM SDM, 2011)**.

The implementation is designed for **very large graphs** and follows the **iterative fixed-point formulation** of the method, avoiding matrix inversion or direct solution of large linear systems.

---

## Reference

K. Avrachenkov, P. Gonçalves, A. Mishenin, M. Sokol  
_Generalized Optimization Framework for Graph-based Semi-Supervised Learning_  
SIAM International Conference on Data Mining (SDM), 2011

https://www-sop.inria.fr/members/Marina.Sokol/data/articles/confs/siam.pdf

---

## Problem Setting

Given a graph with a small subset of labeled nodes (seeds), the objective is to infer labels for the remaining nodes by enforcing smoothness of the classification function over the graph while preserving fidelity to the known labels.

The method is applicable to large-scale graphs where only a fraction of nodes are labeled and classical supervised learning is infeasible.

---

## Method Description (Avrachenkov et al.)

Let:

- **W** be the adjacency (similarity) matrix,
- **D** be the diagonal degree matrix,
- **Y** be the label indicator matrix (seed labels),
- **F** be the matrix of class scores.

The framework balances two objectives:

1. smoothness of the classification function over the graph,
2. closeness of the solution to the known labels.

---

## Iterative Formulation

Instead of using the closed-form solution, this implementation relies on the following **fixed-point iteration**:

$$
F_{t+1} = (1-\alpha)\,Y + \alpha\,D^{-\sigma} W D^{\sigma-1} F_t
$$

with:

α = 2 / (2 + μ)

where:

- μ is the regularization parameter,
- σ controls the diffusion regime.

### Parameters used in this implementation

- μ = 0.5
- α = 2 / (2 + μ) = 0.8
- Fixed number of iterations: 20

---

## Diffusion Regimes

Following Avrachenkov et al., the parameter σ defines three canonical diffusion regimes:

- σ = 0  
  PageRank-based method

- σ = 1/2  
  Normalized Laplacian method

- σ = 1  
  Standard Laplacian method

These regimes correspond to **qualitatively different normalization schemes**, not simple hyperparameter tuning.

---

## Why Iterative (and not Closed Form)

The original framework admits the closed-form expression:

$$
F^{*} = \frac{\mu}{2 + \mu}
\left(
I - \frac{2}{2 + \mu} D^{-\sigma} W D^{\sigma - 1}
\right)^{-1} Y
$$

However, for very large graphs this formulation is impractical.

In this project:

- Number of nodes: 31,535,968
- Number of edges: 34,769,058

As a result:

- matrix inversion is infeasible,
- storing even sparse matrices is prohibitively expensive,
- iterative propagation over the edge list is the only scalable approach.

The fixed-point iteration converges to the same solution as the closed-form expression.

---

## Dataset and Experimental Protocol

Graph statistics:

- Total nodes: 31,535,968
- Total edges: 34,769,058

Labeled subset (“known”):

- Total labeled nodes: 235,030
  - Licit: 220,876
  - Illicit: 14,154

Class-balanced semi-supervised split:

- SEED_FRAC_LIC = 0.5
- SEED_FRAC_ILL = 0.5

Resulting in:

- Seeds (L): 117,515
  - Licit seeds: 110,438
  - Illicit seeds: 7,077
- Masked evaluation nodes (U ∩ known): 117,515
  - Licit masked: 110,438
  - Illicit masked: 7,077

All reported metrics are computed **exclusively on masked labeled nodes**.

---

## Results

Evaluation on masked labeled nodes (U ∩ known).

### Undirected Graph

#### σ = 0 (PageRank-based)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 1.00   | 0.98     | 110,438 |
| Illicit | 0.92      | 0.22   | 0.36     | 7,077   |

Accuracy: 0.95  
Confusion matrix: TN = 110,300, FP = 138, FN = 5,503, TP = 1,574

---

#### σ = 0.5 (Normalized Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 1.00   | 0.97     | 110,438 |
| Illicit | 0.89      | 0.23   | 0.36     | 7,077   |

Accuracy: 0.95  
Confusion matrix: TN = 110,236, FP = 202, FN = 5,463, TP = 1,614

---

#### σ = 1 (Standard Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 1.00   | 0.97     | 110,438 |
| Illicit | 0.86      | 0.21   | 0.34     | 7,077   |

Accuracy: 0.95  
Confusion matrix: TN = 110,188, FP = 250, FN = 5,577, TP = 1,500

---

### Directed Graph

#### σ = 0 (PageRank-based)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 0.92   | 0.93     | 110,438 |
| Illicit | 0.16      | 0.23   | 0.19     | 7,077   |

Accuracy: 0.88  
Confusion matrix: TN = 101,689, FP = 8,749, FN = 5,437, TP = 1,640

---

#### σ = 0.5 (Normalized Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.98      | 0.90   | 0.94     | 110,438 |
| Illicit | 0.31      | 0.73   | 0.43     | 7,077   |

Accuracy: 0.89  
Confusion matrix: TN = 98,903, FP = 11,535, FN = 1,940, TP = 5,137

---

#### σ = 1 (Standard Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 0.98   | 0.97     | 110,438 |
| Illicit | 0.42      | 0.21   | 0.28     | 7,077   |

Accuracy: 0.94  
Confusion matrix: TN = 108,433, FP = 2,005, FN = 5,598, TP = 1,479

---

## Implementation Notes

- Edge-list-based propagation (no adjacency matrix)
- Numba-accelerated kernels
- Supports directed and undirected graphs
- Binary classification (licit / illicit)

---

## Repository Contents

- `Generalized_Optimization_Framework.git.py` — main implementation and experiment runner
