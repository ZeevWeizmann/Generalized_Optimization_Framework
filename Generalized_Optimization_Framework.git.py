import numpy as np
from numba import njit, prange
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_graph_and_labels
import pandas as pd
from sklearn.metrics import confusion_matrix



# ===================== Hyperparameters =====================
MU = 0.5
ALPHA = 2 / (2 + MU)
N_ITER = 20

SIGMAS = [0.0, 0.5, 1.0]
MODES = ["undirected", "directed"]

SEED_FRACS = [0.1, 0.3, 0.5]

RANDOM_SEED = 42
# ===========================================================


# ===================== Load data ============================
print("[Data] Loading graph...")
df, merged, illicit_set, licit_set, known_ids = load_graph_and_labels()

edges = df[["addr_id1", "addr_id2"]].to_numpy(np.int64)
nodes_unique = np.unique(edges)

node_to_idx = {n: i for i, n in enumerate(nodes_unique)}
idx_to_node = np.array(nodes_unique, dtype=np.int64)
N = len(nodes_unique)

src = np.array([node_to_idx[a] for a in edges[:, 0]], dtype=np.int64)
dst = np.array([node_to_idx[b] for b in edges[:, 1]], dtype=np.int64)

print(f"[Graph] Nodes={N}, Edges={len(src)}")
# ===========================================================


# ===================== Labels ===============================
y_true = -np.ones(N, dtype=np.int32)

illicit_idx = np.array(
    [node_to_idx[n] for n in illicit_set if n in node_to_idx], dtype=np.int64
)
licit_idx = np.array(
    [node_to_idx[n] for n in licit_set if n in node_to_idx], dtype=np.int64
)

y_true[illicit_idx] = 1
y_true[licit_idx] = 0
known_mask = (y_true != -1)

# ===========================================================


# ===================== Degrees ==============================
deg = np.zeros(N)
deg_out = np.zeros(N)
deg_in = np.zeros(N)

for i, j in zip(src, dst):
    deg[i] += 1
    deg[j] += 1
    deg_out[i] += 1
    deg_in[j] += 1
# ===========================================================


# ===================== Kernels ==============================
@njit(parallel=True, fastmath=True)
def propagate_undirected(src, dst, deg, F, Y, alpha, sigma, N):
    s = np.zeros((N, 2))
    F_new = np.empty_like(F)

    for k in prange(len(src)):
        i, j = src[k], dst[k]
        di, dj = max(deg[i],1.0), max(deg[j],1.0)

        s[i] += F[j] / (dj ** (1-sigma))
        s[j] += F[i] / (di ** (1-sigma))

    for i in prange(N):
        di = max(deg[i],1.0)
        F_new[i] = (1-alpha)*Y[i] + alpha*(s[i]/(di**sigma))

    return F_new


@njit(parallel=True, fastmath=True)
def propagate_directed(src, dst, deg_out, deg_in, F, Y, alpha, sigma, N):
    s = np.zeros((N, 2))
    F_new = np.empty_like(F)

    for k in prange(len(src)):
        i, j = src[k], dst[k]
        doi = max(deg_out[i],1.0)
        s[j] += F[i] / (doi ** (1-sigma))

    for j in prange(N):
        dij = max(deg_in[j],1.0)
        F_new[j] = (1-alpha)*Y[j] + alpha*(s[j]/(dij**sigma))

    return F_new
# ===========================================================


# ===================== Experiments ==========================
for mode in MODES:
    for sigma in SIGMAS:
        for seed_frac in SEED_FRACS:

            print("\n" + "="*80)
            print(f"[RUN] mode={mode.upper()} | sigma={sigma} | seed_frac={seed_frac}")
            print("="*80)

            # ---------- SEEDING ----------
            rng = np.random.default_rng(RANDOM_SEED)

            seed_ill = rng.choice(
                illicit_idx,
                max(1, int(len(illicit_idx) * seed_frac)),
                replace=False
            )
            seed_lic = rng.choice(
                licit_idx,
                max(1, int(len(licit_idx) * seed_frac)),
                replace=False
            )

            L_mask = np.zeros(N, dtype=np.bool_)
            L_mask[seed_ill] = True
            L_mask[seed_lic] = True
            U_mask = ~L_mask

            Y = np.zeros((N, 2))
            Y[seed_lic, 0] = 1.0
            Y[seed_ill, 1] = 1.0

            F = Y.copy()

            for _ in range(N_ITER):
                if mode == "undirected":
                    F = propagate_undirected(src, dst, deg, F, Y, ALPHA, sigma, N)
                else:
                    F = propagate_directed(src, dst, deg_out, deg_in, F, Y, ALPHA, sigma, N)

            y_pred = (F[:,1] > F[:,0]).astype(int)
            eval_mask = U_mask & known_mask

            yt = y_true[eval_mask]
            yp = y_pred[eval_mask]

            print(classification_report(
                yt, yp,
                target_names=["licit", "illicit"],
                zero_division=0
            ))

            # Confusion matrix: illicit / licit
            cm = confusion_matrix(yt, yp, labels=[0, 1])

            cm_df = pd.DataFrame(
                cm,
                index=pd.Index(["Licit", "Illicit"], name="True"),
                columns=pd.Index(["Licit", "Illicit"], name="Pred")
            )

            print(cm_df)

print("\n[Done] All experiments finished.")