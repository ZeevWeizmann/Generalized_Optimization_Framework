# Generalized Optimization Framework for Illicit Activity Detection in Bitcoin Transaction Graphs

This repository provides a scalable implementation of the **Generalized Optimization Framework for graph-based semi-supervised learning** proposed by **Avrachenkov et al. (SIAM SDM, 2011)**.

The implementation is designed for **very large graphs** and follows the **iterative fixed-point formulation** of the method, avoiding matrix inversion or direct solution of large linear systems.

---

## Reference

K. Avrachenkov, P. Gonçalves, A. Mishenin, M. Sokol  
_Generalized Optimization Framework for Graph-based Semi-Supervised Learning_  
SIAM International Conference on Data Mining (SDM), 2011

https://www-sop.inria.fr/members/Marina.Sokol/data/articles/confs/siam.pdf

K. Avrachenkov, P. Gonçalves, A. Mishenin, M. Sokol

_Graph Based Classification of Content and Users in BitTorrent_
BigLearn 2011 Workshop on Big Learning (co-located with NIPS), 2011
https://www-sop.inria.fr/members/Konstantin.Avratchenkov/pubs/biglearn2011_submission_14.pdf

---

## Problem Setting

Given a Bitcoin transaction graph with a small subset of labeled nodes from both classes (licit and illicit), used as seeds, the objective is to infer labels for the remaining nodes by enforcing smoothness of the classification function over the graph while preserving fidelity to the
known labels.

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
F_{t+1} = (1-\alpha)\ Y + \alpha\ D^{-\sigma} W D^{\sigma-1} F_t
$$

with:

$$
\alpha = \frac{2}{2 + \mu}
$$

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
- Masked evaluation nodes (unknown): 117,515
  - Licit masked: 110,438
  - Illicit masked: 7,077

All reported metrics are computed **exclusively on masked labeled nodes**.

---

## Results

Evaluation on masked labeled nodes (**unknown**).  
Results are grouped by seed fraction (**SEED_FRAC_LIC = SEED_FRAC_ILL**).

---

## Seed fraction = 0.5

### Undirected Graph

#### σ = 0 (PageRank-based)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 1.00   | 0.98     | 110,438 |
| Illicit | 0.92      | 0.22   | 0.36     | 7,077   |

Accuracy: 0.95

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 110,299 | 139     |
| **Illicit** | 5,502   | 1,575   |

---

#### σ = 0.5 (Normalized Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 1.00   | 0.97     | 110,438 |
| Illicit | 0.89      | 0.23   | 0.36     | 7,077   |

Accuracy: 0.95

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 110,236 | 202     |
| **Illicit** | 5,462   | 1,615   |

---

#### σ = 1 (Standard Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 1.00   | 0.97     | 110,438 |
| Illicit | 0.85      | 0.21   | 0.34     | 7,077   |

Accuracy: 0.95

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 110,175 | 263     |
| **Illicit** | 5,577   | 1,500   |

---

### Directed Graph

#### σ = 0 (PageRank-based)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 0.92   | 0.93     | 110,438 |
| Illicit | 0.16      | 0.23   | 0.19     | 7,077   |

Accuracy: 0.88

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 101,689 | 8,749   |
| **Illicit** | 5,437   | 1,640   |

---

#### σ = 0.5 (Normalized Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.98      | 0.90   | 0.94     | 110,438 |
| Illicit | 0.31      | 0.73   | 0.43     | 7,077   |

Accuracy: 0.89

| True \ Pred | Licit  | Illicit |
| ----------- | ------ | ------- |
| **Licit**   | 98,903 | 11,535  |
| **Illicit** | 1,940  | 5,137   |

---

#### σ = 1 (Standard Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 0.98   | 0.97     | 110,438 |
| Illicit | 0.42      | 0.21   | 0.28     | 7,077   |

Accuracy: 0.94

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 108,433 | 2,005   |
| **Illicit** | 5,598   | 1,479   |

---

## Seed fraction = 0.3

### Undirected Graph

#### σ = 0 (PageRank-based)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 1.00   | 0.98     | 154,614 |
| Illicit | 0.91      | 0.22   | 0.36     | 9,908   |

Accuracy: 0.95

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 154,400 | 214     |
| **Illicit** | 7,703   | 2,205   |

---

#### σ = 0.5 (Normalized Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 1.00   | 0.98     | 154,614 |
| Illicit | 0.90      | 0.23   | 0.36     | 9,908   |

Accuracy: 0.95

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 154,370 | 244     |
| **Illicit** | 7,653   | 2,255   |

---

#### σ = 1 (Standard Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 1.00   | 0.97     | 154,614 |
| Illicit | 0.91      | 0.21   | 0.34     | 9,908   |

Accuracy: 0.95

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 154,405 | 209     |
| **Illicit** | 7,842   | 2,066   |

---

### Directed Graph

#### σ = 0 (PageRank-based)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.97      | 0.80   | 0.88     | 154,614 |
| Illicit | 0.17      | 0.61   | 0.26     | 9,908   |

Accuracy: 0.79

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 124,090 | 30,524  |
| **Illicit** | 3,845   | 6,063   |

---

#### σ = 0.5 (Normalized Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.97      | 0.89   | 0.93     | 154,614 |
| Illicit | 0.25      | 0.57   | 0.35     | 9,908   |

Accuracy: 0.87

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 137,916 | 16,698  |
| **Illicit** | 4,266   | 5,642   |

---

#### σ = 1 (Standard Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.94      | 1.00   | 0.97     | 154,614 |
| Illicit | 0.64      | 0.03   | 0.05     | 9,908   |

Accuracy: 0.94

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 154,405 | 209     |
| **Illicit** | 9,842   | 66      |

---

## Seed fraction = 0.1

### Undirected Graph

#### σ = 0 (PageRank-based)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 1.00   | 0.97     | 198,789 |
| Illicit | 0.91      | 0.21   | 0.34     | 12,739  |

Accuracy: 0.95

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 198,538 | 251     |
| **Illicit** | 10,046  | 2,693   |

---

#### σ = 0.5 (Normalized Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 1.00   | 0.97     | 198,789 |
| Illicit | 0.91      | 0.22   | 0.35     | 12,739  |

Accuracy: 0.95

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 198,527 | 262     |
| **Illicit** | 9,999   | 2,740   |

---

#### σ = 1 (Standard Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.95      | 1.00   | 0.97     | 198,789 |
| Illicit | 0.91      | 0.21   | 0.34     | 12,739  |

Accuracy: 0.95

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 198,533 | 256     |
| **Illicit** | 10,077  | 2,662   |

---

### Directed Graph

#### σ = 0 (PageRank-based)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.94      | 1.00   | 0.97     | 198,789 |
| Illicit | 0.10      | 0.00   | 0.00     | 12,739  |

Accuracy: 0.94

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 198,508 | 281     |
| **Illicit** | 12,707  | 32      |

---

#### σ = 0.5 (Normalized Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.94      | 1.00   | 0.97     | 198,789 |
| Illicit | 0.10      | 0.00   | 0.00     | 12,739  |

Accuracy: 0.94

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 198,522 | 267     |
| **Illicit** | 12,710  | 29      |

---

#### σ = 1 (Standard Laplacian)

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Licit   | 0.94      | 1.00   | 0.97     | 198,789 |
| Illicit | 0.14      | 0.00   | 0.00     | 12,739  |

Accuracy: 0.94

| True \ Pred | Licit   | Illicit |
| ----------- | ------- | ------- |
| **Licit**   | 198,771 | 18      |
| **Illicit** | 12,736  | 3       |

## Implementation Notes

- **Edge-list-based propagation** (no explicit adjacency matrix construction)
- **Numba-accelerated kernels** for efficient large-scale graph diffusion
- Supports **directed and undirected graphs**
- **Binary node classification** (licit / illicit)

---

## Project Page

https://zeevweizmann.github.io/Generalized_Optimization_Framework/

---

## Source Code

https://github.com/ZeevWeizmann/Generalized_Optimization_Framework

---

## Repository Contents

- `Generalized_Optimization_Framework.py` — main implementation and experiment runner
