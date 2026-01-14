# Generalized Optimization Framework for Graph-based Semi-Supervised Learning

This repository provides a NumPy/Numba-based implementation of the **Generalized Optimization Framework for graph-based semi-supervised learning** introduced by **Avrachenkov et al. (SIAM SDM, 2011)**.

The implementation follows the **iterative formulation** of the framework and allows to study different diffusion regimes on large-scale graphs, including both **undirected** and **directed** settings.

---

## Reference

K. Avrachenkov, P. Gonçalves, A. Mishenin, M. Sokol  
*Generalized Optimization Framework for Graph-based Semi-Supervised Learning*  
SIAM International Conference on Data Mining (SDM), 2011  

https://www-sop.inria.fr/members/Marina.Sokol/data/articles/confs/siam.pdf

---

## Method Overview

The classification functions are computed using the iterative scheme:

F(t+1) = (1 − α)Y + α D^(−σ) W D^(σ−1) F(t)  
with α = 2 / (2 + μ)

where:
- W is the graph adjacency or similarity matrix
- D is the degree matrix
- Y is the label indicator matrix
- σ defines the diffusion regime

---

## Diffusion Regimes (Avrachenkov et al.)

| σ | Method |
|---|--------|
| 0 | PageRank-based method |
| 1/2 | Normalized Laplacian method |
| 1 | Standard Laplacian method |

---

## Implementation Details

- Iterative solver (no matrix inversion)
- Full-graph propagation using edge lists
- Numba-accelerated computation
- Binary classification (licit / illicit)
- Evaluation on unlabeled but known nodes only
- Supports undirected and directed graphs

---

## Dataset

- Transaction graph with licit and illicit entities
- ~117k labeled nodes used for evaluation
- Seeds sampled from known labeled nodes
- Evaluation performed on U ∩ known

---

## Results Summary

### Undirected Graph

| Method | Accuracy | Illicit Precision | Illicit Recall | Illicit F1 |
|------|----------|------------------|---------------|-----------|
| PageRank-based (σ=0) | 0.95 | 0.92 | 0.22 | 0.36 |
| Normalized Laplacian (σ=1/2) | 0.95 | 0.89 | 0.23 | 0.37 |
| Standard Laplacian (σ=1) | 0.95 | 0.85 | 0.21 | 0.34 |

### Directed Graph

| Method | Accuracy | Illicit Precision | Illicit Recall | Illicit F1 |
|------|----------|------------------|---------------|-----------|
| PageRank-based (σ=0) | 0.88 | 0.16 | 0.23 | 0.19 |
| Normalized Laplacian (σ=1/2) | 0.89 | 0.31 | 0.73 | 0.43 |
| Standard Laplacian (σ=1) | 0.94 | 0.42 | 0.21 | 0.28 |

---

## Key Observations

- Undirected diffusion yields high accuracy but poor illicit recall
- Directed diffusion is sensitive to the diffusion regime
- Normalized Laplacian provides the best trade-off for illicit detectionы
- σ defines qualitatively different diffusion mechanisms

---

## Usage

The main implementation is contained in:

Generalized_Optimization_Framework.git.py


