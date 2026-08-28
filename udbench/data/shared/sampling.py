"""Stratified Mahalanobis sampling utilities.

Provides `make_stratified_mahalanobis_sampler` (the sampler factory) and
`SAMPLING_CFG_PRESETS` (four canonical named configurations used by the
generation workflow). Both the offline generation workflow and the runtime
`DataSet` import from this module so sampling config stays in one place.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import jax
import jax.numpy as jnp


SAMPLING_CFG_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "method": "stratified_mahalanobis",
        "quantile_edges": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "bin_fractions": [0.2, 0.2, 0.2, 0.2, 0.2],
        "shrinkage": 0.1,
        "standardize": True,
        "prefer_center_bins": True,
    },
    "ood": {
        "method": "stratified_mahalanobis",
        "quantile_edges": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "bin_fractions": [0.4, 0.3, 0.1, 0.1, 0.1],
        "shrinkage": 0.1,
        "standardize": True,
        "prefer_center_bins": True,
    },
    "strong_ood": {
        "method": "stratified_mahalanobis",
        "quantile_edges": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "bin_fractions": [0.5, 0.3, 0.15, 0.05, 0.0],
        "shrinkage": 0.1,
        "standardize": True,
        "prefer_center_bins": True,
    },
    "multimode": {
        "method": "stratified_mahalanobis",
        "quantile_edges": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "bin_fractions": [0.4, 0.1, 0.0, 0.1, 0.4],
        "shrinkage": 0.1,
        "standardize": True,
        "prefer_center_bins": True,
    },
}
"""Named sampling configurations for `make_stratified_mahalanobis_sampler`.

Keys: ``"balanced"``, ``"ood"``, ``"strong_ood"``, ``"multimode"``.
Each value is a dict that can be unpacked directly as keyword arguments to
`make_stratified_mahalanobis_sampler` (after removing the ``"method"`` key).
"""


def make_stratified_mahalanobis_sampler(
    *,
    quantile_edges: Sequence[float],
    bin_fractions: Sequence[float],
    shrinkage: float = 0.1,
    standardize: bool = True,
    prefer_center_bins: bool = True,
    eps: float = 1e-8,
) -> Callable:
    """Return a sampling function that stratifies observations by Mahalanobis distance.

    The returned callable draws a subset of a given dataset such that the
    distribution of Mahalanobis distances across the drawn points follows the
    bin fractions specified here. This allows controlled covariate-shift
    scenarios (OOD, bimodal, balanced) without changing the underlying dataset.

    Args:
        quantile_edges: Monotone sequence ``[0.0, ..., 1.0]`` defining the bin
            boundaries as quantiles of the empirical Mahalanobis distribution.
            Must have at least two entries and start at 0.0 / end at 1.0.
        bin_fractions: Relative weight of each bin. Length must equal
            ``len(quantile_edges) - 1``. Values are normalised to sum to 1.
        shrinkage: Ledoit–Wolf-style shrinkage coefficient in ``[0, 1]`` for the
            covariance estimate. Higher values produce a more isotropic metric.
        standardize: If ``True``, standardise each feature to zero mean and unit
            variance before computing the covariance, making the metric
            scale-invariant.
        prefer_center_bins: When leftover points must be redistributed (because
            a bin has fewer points than requested), fill centre bins first if
            ``True``, or outer bins first if ``False``.
        eps: Small constant added for numerical stability.

    Returns:
        A callable ``(X, y, size, key) -> (X_sampled, y_sampled, indices)``
        that draws *size* rows without replacement.

    Raises:
        ValueError: If ``quantile_edges`` does not start at 0 and end at 1, or
            if ``bin_fractions`` has the wrong length.
    """
    q = jnp.asarray(quantile_edges, dtype=jnp.float32)
    if q.ndim != 1 or q.shape[0] < 2:
        raise ValueError("quantile_edges must be a 1D list with at least 2 entries.")
    if float(q[0]) != 0.0 or float(q[-1]) != 1.0:
        raise ValueError("quantile_edges must start with 0.0 and end with 1.0.")

    num_bins = q.shape[0] - 1
    fractions = jnp.asarray(bin_fractions, dtype=jnp.float32)
    if fractions.shape[0] != num_bins:
        raise ValueError(
            f"bin_fractions must have length {num_bins} (len(quantile_edges)-1)."
        )
    fractions = fractions / jnp.sum(fractions)

    def _compute_mahalanobis_sq(X: jnp.ndarray) -> jnp.ndarray:
        n_rows, n_features = X.shape

        if standardize:
            mu_x = jnp.mean(X, axis=0)
            std_x = jnp.std(X, axis=0) + eps
            Z = (X - mu_x) / std_x
        else:
            Z = X

        Zc = Z - jnp.mean(Z, axis=0)
        cov = (Zc.T @ Zc) / jnp.maximum(n_rows - 1, 1)

        lam = jnp.clip(shrinkage, 0.0, 1.0)
        tau = jnp.trace(cov) / jnp.maximum(n_features, 1)
        cov_shrunk = (1.0 - lam) * cov + lam * tau * jnp.eye(n_features, dtype=cov.dtype)
        cov_shrunk = cov_shrunk + eps * jnp.eye(n_features, dtype=cov.dtype)
        chol = jnp.linalg.cholesky(cov_shrunk)

        whitened = jax.scipy.linalg.solve_triangular(chol, Zc.T, lower=True)
        return jnp.sum(whitened * whitened, axis=0)

    def _allocate_counts(size: int, available_per_bin: jnp.ndarray) -> jnp.ndarray:
        raw = fractions * size
        base = jnp.floor(raw).astype(jnp.int32)
        remainder = int(size - int(jnp.sum(base)))

        fractional = raw - jnp.floor(raw)
        order = jnp.argsort(-fractional)
        base = base.at[order[:remainder]].add(1)

        capped = jnp.minimum(base, available_per_bin)
        leftover = int(size - int(jnp.sum(capped)))
        if leftover <= 0:
            return capped

        spare = available_per_bin - capped
        redistribution_order = (
            jnp.arange(num_bins) if prefer_center_bins
            else jnp.arange(num_bins - 1, -1, -1)
        )

        allocated = capped
        remaining = leftover
        for bin_idx in redistribution_order.tolist():
            if remaining <= 0:
                break
            take = int(jnp.minimum(spare[bin_idx], remaining))
            if take > 0:
                allocated = allocated.at[bin_idx].add(take)
                remaining -= take

        if remaining > 0:
            raise ValueError(
                f"Not enough points to sample {size} without replacement "
                f"(still missing {remaining} samples after redistribution)."
            )
        return allocated

    def sampling_fn(
        X: jnp.ndarray,
        y: jnp.ndarray,
        size: int,
        key: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        n_rows = X.shape[0]
        if size > n_rows:
            raise ValueError(f"Requested size={size} but only n={n_rows} points available.")

        # Stratification is irrelevant when drawing all points — just permute.
        if size == n_rows:
            perm = jax.random.permutation(key, n_rows)
            return X[perm], y[perm], perm

        distances_sq = _compute_mahalanobis_sq(X)
        thresholds = jnp.quantile(distances_sq, q)

        bin_indices = []
        available = []
        for bin_idx in range(num_bins):
            lo = thresholds[bin_idx]
            hi = thresholds[bin_idx + 1]
            if bin_idx < num_bins - 1:
                mask = (distances_sq >= lo) & (distances_sq < hi)
            else:
                mask = (distances_sq >= lo) & (distances_sq <= hi)
            idx = jnp.where(mask, size=n_rows, fill_value=-1)[0]
            idx = idx[idx >= 0]
            bin_indices.append(idx)
            available.append(idx.shape[0])

        # Float32 quantile interpolation can land exactly on a data value, leaving
        # that point outside every bin's half-open interval. Collect strays and
        # append them to the last bin as a fallback.
        total_assigned = sum(available)
        if total_assigned < n_rows:
            assigned_mask = jnp.zeros(n_rows, dtype=jnp.bool_)
            for idx in bin_indices:
                if idx.shape[0] > 0:
                    assigned_mask = assigned_mask.at[idx].set(True)
            missed = jnp.where(~assigned_mask, size=n_rows - total_assigned, fill_value=-1)[0]
            missed = missed[missed >= 0]
            if missed.shape[0] > 0:
                bin_indices[-1] = jnp.concatenate([bin_indices[-1], missed])
                available[-1] += int(missed.shape[0])

        available_per_bin = jnp.asarray(available, dtype=jnp.int32)
        counts = _allocate_counts(size, available_per_bin)

        keys = jax.random.split(key, num_bins + 1)
        picked = []
        for bin_idx in range(num_bins):
            count = int(counts[bin_idx])
            if count == 0:
                continue
            chosen = jax.random.choice(
                keys[bin_idx],
                bin_indices[bin_idx],
                shape=(count,),
                replace=False,
            )
            picked.append(chosen)

        all_idx = jnp.concatenate(picked, axis=0)
        all_idx = jax.random.permutation(keys[-1], all_idx)
        return X[all_idx], y[all_idx], all_idx

    return sampling_fn


__all__ = ["SAMPLING_CFG_PRESETS", "make_stratified_mahalanobis_sampler"]
