"""
    Extra functions useful for Distns
"""
import numpy as np
import math
from numba import njit, prange, float64, vectorize, guvectorize
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors
from .distn import RegressionDistn

# pylint: disable=too-few-public-methods


def SurvivalDistnClass(Dist: RegressionDistn):
    """
    Creates a new dist class from a given dist. The new class has its implemented scores

    Parameters:
        Dist (RegressionDistn): a Regression distribution with censored scores implemented.

    Output:
        SurvivalDistn class, this is only used for Survival regression
    """

    class SurvivalDistn(Dist):
        # Stores the original distribution for pickling purposes
        _basedist = Dist
        scores = (
            Dist.censored_scores
        )  # set to the censored versions of the scores implemented for dist

        def fit(Y):
            """
            Parameters:
                Y : a object with keys {time, event}, each containing an array
            """
            return Dist.fit(Y["Time"])

    return SurvivalDistn

@njit(parallel=True, fastmath=True)
def rbf_kernel_and_grad_numba(X, gamma):
    """
    Compute the RBF kernel matrix and its gradient wrt the first argument using Numba.

    Parameters:
        X (np.ndarray): Input array of shape (n, d).
        gamma (float): RBF kernel parameter.

    Returns:
        K (np.ndarray): Kernel matrix of shape (n, n).
        dK (np.ndarray): Gradient tensor of shape (n, n, d), where
                         dK[i, j, k] = ∂/∂X[i, k] [exp(-gamma * ||X[i] - X[j]||^2)].
    """
    n, d = X.shape
    K = np.empty((n, n), dtype=np.float64)
    dK = np.empty((n, n, d), dtype=np.float64)
    for i in prange(n):
        for j in range(n):
            # Compute squared distance
            sq_dist = 0.0
            for k in range(d):
                diff = X[i, k] - X[j, k]
                sq_dist += diff * diff
            # Kernel value
            k_val = math.exp(-gamma * sq_dist)
            K[i, j] = k_val
            # Gradient wrt X[i]
            for k in range(d):
                diff = X[i, k] - X[j, k]
                dK[i, j, k] = -2.0 * gamma * diff * k_val
    return K, dK


def compute_leaf_boxes(tree: DecisionTreeRegressor, feature_dim: int):
    """
    For each leaf node in `tree`, compute the axis-aligned [min, max] box
    in R^feature_dim that the leaf covers.
    Returns:
        leaf_box: dict mapping leaf_id -> (lower_bounds[feature_dim], upper_bounds[feature_dim])
    """
    t = tree.tree_
    # node -> (lo_bounds, hi_bounds)
    boxes = {0: (np.full(feature_dim, -np.inf),
                 np.full(feature_dim,  np.inf))}
    leaf_box = {}

    def recurse(node_id):
        lo, hi = boxes[node_id]
        # if this node is a leaf
        if t.children_left[node_id] == t.children_right[node_id]:
            leaf_box[node_id] = (lo.copy(), hi.copy())
            return
        feat = t.feature[node_id]
        thr  = t.threshold[node_id]
        # left child: x[feat] <= thr
        lo_l, hi_l = lo.copy(), hi.copy()
        hi_l[feat] = min(hi[feat], thr)
        boxes[t.children_left[node_id]] = (lo_l, hi_l)
        recurse(t.children_left[node_id])
        # right child: x[feat] > thr
        lo_r, hi_r = lo.copy(), hi.copy()
        lo_r[feat] = max(lo[feat], thr)
        boxes[t.children_right[node_id]] = (lo_r, hi_r)
        recurse(t.children_right[node_id])

    recurse(0)
    return leaf_box


def leaf_volume_density_vec(tree, X):
    """
    Vectorized leaf‐volume density: p_hat[i] ∝ n_leaf(i) / vol(leaf(i)),
    with mean(p_hat)=1, but using only NumPy indexing (no Python loops).
    """
    n, d = X.shape

    # 1) which leaf each row lands in
    leaf_ids = tree.apply(X)            # shape (n,)

    # 2) count per leaf, and get unique ids
    unique, counts = np.unique(leaf_ids, return_counts=True)
    # build a dense map from leaf_id -> count
    max_id = unique.max()
    count_map = np.zeros(max_id+1, int)
    count_map[unique] = counts         # now count_map[leaf_ids] gives counts per row

    # 3) build bounding‐boxes once (you still need the recursion for that)
    #    but we can then compute volumes for only the unique leaves
    boxes = compute_leaf_boxes(tree, d)   # your existing function
    # volumes in a dense array
    vol_map = np.empty(max_id+1, float)
    for lid in unique:
        lo, hi = boxes[lid]
        span = hi - lo
        span[np.isinf(span)] = 1e6
        vol_map[lid] = np.prod(span)

    # 4) density per sample, vectorized
    p_hat = count_map[leaf_ids] / (vol_map[leaf_ids] + 1e-12)

    # 5) normalize to mean=1
    p_hat *= (n / p_hat.sum())

    return p_hat




# ---------- 1) Dirichlet-smoothed leaf density, evaluated on X_query ----------
def leaf_density_dirichlet(tree, X_train, X_query, alpha0=1.0, beta=0.5):
    """
    Compute a *relative* leaf-density weight on X_query using *training* counts.

    Steps:
      • Count train points per leaf (n_ell).
      • Dirichlet smoothing:  p_hat_ell = (n_ell + alpha0) / (N + alpha0 * L)
      • Assign p_hat(x) = p_hat_{leaf(x)} for x in X_query.
      • Robust normalize by train median and invert with exponent beta:
            r(x) = p_hat(x) / median_train
            w_leaf(x) = r(x)^(-beta)
        so sparse leaves (small p_hat) => w_leaf > 1.

    Returns
    -------
    w_leaf : (n_query,) array
    """
    # Leaves for train + query
    leaf_train = tree.apply(X_train)              # (N,)
    leaf_query = tree.apply(X_query)              # (n_query,)

    # Train leaf counts
    unique, counts = np.unique(leaf_train, return_counts=True)
    N = len(leaf_train)
    L = len(unique)

    # Dense maps for counts and probabilities
    max_id = unique.max()
    count_map = np.zeros(max_id + 1, dtype=np.int64)
    count_map[unique] = counts

    # Dirichlet smoothing on TRAIN leaves
    p_map = (count_map + alpha0) / (N + alpha0 * L)

    # Assign to query leaves (unseen leaves -> prior mass alpha0 only)
    # If some query leaf id > max_id (shouldn't happen with same tree), guard:
    safe_leaf = np.minimum(leaf_query, max_id)
    p_query = p_map[safe_leaf]

    # Robust normalization by TRAIN median of p_hat among *visited* leaves
    median_train = np.median(p_map[unique])
    r = p_query / (median_train + 1e-12)

    # Invert with exponent beta (beta=0.5 is a good default)
    w_leaf = np.power(r + 1e-12, -beta)
    return w_leaf


def knn_mahalanobis_proximity(
    X_train, X_query, k=10, aleatoric_train=None,
    median_subset=2048, ridge=1e-6,
    q_for_scale=0.90, w_floor=1e-3,
    mapping="cauchy",  # {"cauchy","exp","root-exp","sigmoid","log10","poly"}
    poly_power=1.0,
    return_debug=False
):
    """
    kNN Mahalanobis proximity with per-query aleatoric diagonal inflation and robust calibration.

    Pipeline
    --------
    1) Global train covariance Σ with scale-aware ridge λ: eigendecompose Σ+λI = U diag(e) U^T.
    2) For each point, add *scalar* s I on the diagonal, where s is the mean aleatoric
       of its k train neighbors (or 0 if not provided):
           Σ_q = (Σ + λ I) + s_q I = U diag(e + s_q) U^T.
    3) Energy m^2(x) = mean_j ||x - x_j||^2_{Σ_q^{-1}} over its k neighbors.
    4) Robust center/scale via LOO train energies:
           med = median(m^2_train), qv = quantile_{q_for_scale}(m^2_train),
           z   = max(0, m^2 - med) / max(qv - med, eps).
    5) Map z -> (0,1] with chosen monotone mapping (default: Cauchy 1/(1+z)), then clip.

    Parameters
    ----------
    X_train : (n_train, d)
    X_query : (n_query, d)
    k : int, number of neighbors (uses k+1 and drops self for train LOO)
    aleatoric_train : None or (n_train,) nonnegative vector of per-train-point aleatoric variances
    median_subset : int or None, subsample size for LOO calibration; None = all
    ridge : float, scale-aware ridge factor on Σ (λ = ridge * tr(Σ)/d)
    q_for_scale : float in (0.5, 1), robust scale quantile
    w_floor : float, lower clip for proximity
    mapping : {"cauchy","exp","root-exp","sigmoid","log10","poly"}
        - "cauchy":     w = 1 / (1 + z)                  (heavy-tailed, tuning-free)
        - "exp":        w = exp(-0.5 * z)               (Gaussian-like, can be sharp)
        - "root-exp":   w = exp(-sqrt(z))               (softer than Gaussian)
        - "sigmoid":    w = 1 / (1 + exp(z - 1))        (0.5 at z=1)
        - "log10":      w = 1 / (1 + log10(1 + z))      (very gentle)
        - "poly":       w = 1 / (1 + z**poly_power)     (Cauchy is poly_power=1)
    poly_power : float, power for "poly" mapping
    return_debug : bool, if True also returns dict with internals

    Returns
    -------
    w : (n_query,) proximity in [w_floor, 1]
    debug (optional) : dict with "m2_q","med","scale","z","neighbors" if return_debug=True
    """
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    n_train, d = X_train.shape
    k = min(k, max(1, n_train - 1))

    # --- covariance + scale-aware ridge ---
    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    Sigma = (Xc.T @ Xc) / max(1, n_train - 1)
    lam = ridge * (np.trace(Sigma) / d + 1e-12)
    # eigendecomp of (Σ + λI)
    e, U = np.linalg.eigh(Sigma + lam * np.eye(d))
    e = np.maximum(e, 0.0)  # numerical safety

    # helper: mean Mahalanobis^2 in the eigenbasis for a batch of diffs and per-point s
    def mean_maha2_in_U(diffsU, s_vec):  # diffsU: (B, k, d), s_vec: (B,)
        denom = e[None, None, :] + s_vec[:, None, None]
        return np.sum(diffsU**2 / denom, axis=2).mean(axis=1)

    # --- kNN on train for LOO calibration and for queries ---
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X_train)

    # --- LOO calibration on a subset ---
    if median_subset is None or n_train <= median_subset:
        idx_cal = np.arange(n_train)
    else:
        rng = np.random.default_rng(0)
        idx_cal = rng.choice(n_train, size=median_subset, replace=False)

    dists_base, ind_base = nn.kneighbors(X_train[idx_cal])
    ind_base = ind_base[:, 1:]  # drop self
    diffs_base = X_train[idx_cal, None, :] - X_train[ind_base, :]  # (m,k,d)
    diffs_base_U = diffs_base @ U

    if aleatoric_train is not None:
        aleatoric_train = np.asarray(aleatoric_train, dtype=float).reshape(-1)
        if aleatoric_train.size == 1 and aleatoric_train.shape[0] != n_train:
            aleatoric_train = np.full(n_train, float(aleatoric_train))
        if aleatoric_train.shape[0] != n_train:
            raise ValueError("aleatoric_train must have length n_train or be scalar.")
        s_base = np.mean(aleatoric_train[ind_base], axis=1)  # (m,)
    else:
        s_base = np.zeros(len(idx_cal))

    m2_base = mean_maha2_in_U(diffs_base_U, s_base)  # (m,)
    med = float(np.median(m2_base))
    qv = float(np.quantile(m2_base, q_for_scale))
    scale = max(qv - med, 1e-12)

    # --- queries ---
    dists_q, ind_q = nn.kneighbors(X_query)
    # keep first k neighbors (no self in general)
    ind_q = ind_q[:, :k]
    diffs_q = X_query[:, None, :] - X_train[ind_q, :]  # (n_q, k, d)
    diffs_q_U = diffs_q @ U

    if aleatoric_train is not None:
        s_q = np.mean(aleatoric_train[ind_q], axis=1)  # (n_q,)
    else:
        s_q = np.zeros(X_query.shape[0])

    m2_q = mean_maha2_in_U(diffs_q_U, s_q)  # (n_q,)

    # robust center & scale → z
    z = np.maximum(0.0, m2_q - med) / scale

    # mapping z → (0,1]
    if mapping == "cauchy":
        w = 1.0 / (1.0 + z)
    elif mapping == "exp":
        w = np.exp(-0.5 * z)
    elif mapping == "root-exp":
        w = np.exp(-np.sqrt(z))
    elif mapping == "sigmoid":
        w = 1.0 / (1.0 + np.exp(z - 1.0))
    elif mapping == "log10":
        w = 1.0 / (1.0 + np.log10(1.0 + z))
    elif mapping == "poly":
        w = 1.0 / (1.0 + np.power(z, poly_power))
    else:
        raise ValueError(f"unknown mapping '{mapping}'")

    w = np.clip(w, w_floor, 1.0)

    if return_debug:
        return w, {
            "m2_q": m2_q,
            "median": med,
            "scale": scale,
            "z": z,
            "neighbors": ind_q,
        }
    return w

def knn_mahalanobis_weight(
    X_train, X_query, k=10, aleatoric=None, ridge=1e-6,
    median_subset=2048, mapping="cauchy"
):
    """
    kNN Mahalanobis weight with per-query aleatoric inflation of the train covariance.

    Steps
    -----
    1) Empirical Sigma from X_train; precompute eigendecomp Sigma = V diag(s) V^T.
    2) For each query point, use Sigma_i = Sigma + (ridge_scale + s2_i) * I
       where s2_i = aleatoric[i] if provided (else 0).
       Compute mean Mahalanobis^2 to its k nearest training neighbors.
    3) Calibrate via the LOO median of the *same* energy (no aleatoric) and scale:
           \tilde m^2 = d * m^2 / median_train
    4) Map to (0,1] with a heavy-tailed function (default Cauchy): w = 1/(1+tilde m^2).

    Returns
    -------
    w_maha : (n_query,) array in (0,1]
    """
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    n_train, d = X_train.shape
    k = min(k, max(1, n_train - 1))

    # ---- covariance & eigendecomposition ----
    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    Sigma = (Xc.T @ Xc) / max(1, n_train - 1)

    # scale-aware ridge
    ridge_scale = ridge * (np.trace(Sigma) / d + 1e-12)

    # eigendecomposition once (stable for SPD)
    s, V = np.linalg.eigh(Sigma)  # Sigma = V diag(s) V^T
    s = np.maximum(s, 0.0)        # numerical safety

    # helper: Mahalanobis^2 with Sigma + tau*I using the same eigenvectors
    # m2(diff; tau) = || (diag(s) + tau I)^(-1/2) V^T diff ||^2
    def maha2_with_tau(diffs, tau):
        # diffs: (..., d)
        Z = diffs @ V                         # rotate into eigenbasis
        denom = (s + tau)[None, ...]          # broadcast eigenvalues + tau
        M2 = np.sum((Z * Z) / denom, axis=-1) # sum over dimensions
        return M2

    # ---- kNN indices on train ----
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X_train)

    # ---- baseline LOO median (no aleatoric, just ridge) ----
    if median_subset is None or n_train <= median_subset:
        idx_base = np.arange(n_train)
    else:
        rng = np.random.default_rng(0)
        idx_base = rng.choice(n_train, size=median_subset, replace=False)

    _, ind_base = nbrs.kneighbors(X_train[idx_base])
    ind_base = ind_base[:, 1:]  # drop self
    diffs_base = X_train[idx_base, None, :] - X_train[ind_base, :]  # (m, k, d)
    m2_base = maha2_with_tau(diffs_base, tau=ridge_scale).mean(axis=1)  # (m,)
    median_train = np.median(m2_base) + 1e-12

    # ---- query energies with per-query aleatoric inflation ----
    _, ind_q = nbrs.kneighbors(X_query)
    # drop self only if queries are a subset of train; harmless to keep all k
    if ind_q.shape[1] > k: 
        ind_q = ind_q[:, 1:]
    diffs_q = X_query[:, None, :] - X_train[ind_q, :]  # (n_q, k, d)

    if aleatoric is None:
        tau_q = np.full(X_query.shape[0], ridge_scale)
    else:
        s2 = np.asarray(aleatoric, dtype=float).reshape(-1)
        if s2.size != X_query.shape[0]:
            raise ValueError("aleatoric must have length len(X_query)")
        tau_q = ridge_scale + s2

    # compute per-query mean m2 over its k neighbors (vectorized over queries)
    m2_q = np.empty(X_query.shape[0])
    for i in range(X_query.shape[0]):
        m2_q[i] = maha2_with_tau(diffs_q[i], tau=tau_q[i]).mean()

    # ---- median-heuristic scaling (≈ Chi^2_d units) ----
    m2_tilde = (d * m2_q) / median_train

    # ---- monotone map to (0,1] ----
    if mapping == "exp":
        # classical Gaussian kernel (can underflow)
        w = np.exp(-0.5 * m2_tilde)
    elif mapping == "root-exp":
        # softer than Gaussian
        w = np.exp(-np.sqrt(m2_tilde))
    elif mapping == "sigmoid":
        # slope chosen so that w=0.5 at m2_tilde=1
        w = 1.0 / (1.0 + np.exp(m2_tilde - 1.0))
    elif mapping == "cauchy":
        # heavy-tailed, tuning-free and stable
        w = 1.0 / (1.0 + m2_tilde)
    elif mapping == "log10":
        # decibel-like: clip to keep in (0,1]
        w = 1.0 / (1.0 + np.log10(1.0 + m2_tilde))
    else:
        raise ValueError(f"unknown mapping '{mapping}'")

    return np.clip(w, 1e-6, 1.0)



def epistemic_scaling(Dist, knn=10, alpha0=1.0, beta_leaf=0.5, eta=1.0, zeta=1.0):
    """
    Post-hoc OOD scaling for a NIG-like RegressionDistn.

    Inputs expected on Dist:
      • Dist.X_test   : (n_test, d)
      • Dist.X_train  : (n_train, d)
      • Dist.leaf_tree: final μ-tree (sklearn DecisionTreeRegressor or similar)

    Behavior:
      1) Get base uncertainties on X_test: aleatoric, epistemic.
      2) w_leaf from Dirichlet-smoothed train leaf densities (robustly normalized).
      3) w_maha from kNN Mahalanobis with neighbor aleatorics (train), median-calibrated.
         If Dist.aleatoric_train is present, we use it; otherwise we fall back gracefully to
         a plain distance version.
      4) Combine: w = w_maha^eta * w_leaf^zeta
      5) Return scaled uncertainties: epistemic' = w * epistemic, predictive' = ale + epistemic'.

    Returns
    -------
    dict with keys: mean, aleatoric, epistemic, predictive, w_leaf, w_maha, w
    """
    # 1) base uncertainties at test points
    base = Dist.pred_uncertainty()
    mean = base["mean"]
    ale_test = np.asarray(base["aleatoric"])
    epi_test = np.asarray(base["epistemic"])

    X_test = getattr(Dist, "X_test", None)
    X_train = getattr(Dist, "X_train", None)
    tree = getattr(Dist, "leaf_tree", None)
    ale_train = getattr(Dist, "aleatoric_train", None)  # optional

    # If we don't have the needed pieces, return base
    if (X_test is None) or (X_train is None) or (tree is None):
        return {
            "mean": mean,
            "aleatoric": ale_test,
            "epistemic": epi_test,
            "predictive": ale_test + epi_test,
        }

    # 2) Dirichlet-smoothed leaf density weight
    w_leaf = leaf_density_dirichlet(tree, X_train, X_test, alpha0=alpha0, beta=beta_leaf)

    # 3) kNN Mahalanobis (aleatoric-aware if train aleatorics available)
    w_maha = knn_mahalanobis_proximity(
        X_train=X_train,
        X_query=X_test,
        aleatoric_train=ale_train,
        k=knn,
        median_subset=min(2048, len(X_train)),
        ridge=1e-6,
        mapping="cauchy"  # or "exp", "sigmoid", etc.
    )

    # 4) Combine (weights multiply → precision divides; here we just scale epi)
    w = np.sqrt(w_maha ** eta) #+ (w_leaf ** zeta))

    epi_scaled = epi_test /w 
    pred_scaled = ale_test + epi_scaled

    return {
        "mean": mean,
        "aleatoric": ale_test,
        "epistemic": epi_scaled,
        "predictive": pred_scaled,
        #"w_leaf": w_leaf,
        #"w_maha": w_maha,
        #"w": w,
    }
