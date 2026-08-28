"""Dataset registry and sampling/noise utilities shared across experiment runners."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


# (preset_name, default_n_obs) - used by single-n_obs experiments (exp01/02/03/04/05/06).
TRAINING_DATASETS: list[tuple[str, int]] = [
    ("abalone",                    750),
    ("airfoil_self_noise",         750),
    ("concrete_compressive_strength", 750),
    ("cpu_activity",              1500),
    ("qsar_fish_toxicity",         750),
    ("seoul_bike_sharing_demand", 1500),
    ("kin8nm:kin8nm_matern32",    1500),
]

#  Validation presets
VALIDATION_DATASETS: list[tuple[str, int]] = [
    ("health_insurance",                          750),
    ("kings_county_house_prices",                 750),
    ("diamonds",                                 1500),
    ("physicochemical_properties_of_protein_tertiary_structure", 1500),
    ("sarcos",                                   1500),
    ("space_ga",                                  750),
    ("white_wine_quality",                       1500),
]

# Alias for the training suite.
DATASETS: list[tuple[str, int]] = TRAINING_DATASETS

DATASET_SUITES: dict[str, list[tuple[str, int]]] = {
    "train": TRAINING_DATASETS,
    "training": TRAINING_DATASETS,
    "val": VALIDATION_DATASETS,
    "validation": VALIDATION_DATASETS,
    "all": TRAINING_DATASETS + VALIDATION_DATASETS,
}

DATASET_SAMPLE_SIZES: dict[str, int] = dict(TRAINING_DATASETS + VALIDATION_DATASETS)
DEFAULT_DATASET_NAMES: list[str] = [name for name, _ in TRAINING_DATASETS]
VALIDATION_DATASET_NAMES: list[str] = [name for name, _ in VALIDATION_DATASETS]


def resolve_dataset_names(
    datasets: list[str] | None = None,
    *,
    dataset_suite: str = "training",
) -> list[str]:
    if datasets:
        return list(datasets)
    suite = str(dataset_suite or "training").strip().lower()
    if suite == "custom":
        raise ValueError("dataset_suite='custom' requires an explicit datasets list.")
    if suite not in DATASET_SUITES:
        available = ", ".join(sorted(DATASET_SUITES))
        raise ValueError(f"Unknown dataset_suite {dataset_suite!r}. Available: {available}.")
    return [name for name, _ in DATASET_SUITES[suite]]


def _probe_preset(
    preset: str,
    eval_size: int,
    seed: int,
    *,
    max_pool_size: int | None = None,
) -> tuple[int, float]:
    """Load the dataset once to get pool size and mean aleatoric std.

    Returns (pool_size, base_std) without drawing a training set.
    """
    from udbench.datasets import DataSet

    ds = DataSet.from_preset(
        preset,
        num_observations=1,
        eval_size=eval_size,
        key=seed,
        max_pool_size=max_pool_size,
    )
    pool_size = int(ds.X_eval.shape[0])
    base_std = float(np.asarray(ds.aleatoric_std_fn(ds.X_eval)).mean())
    return pool_size, base_std


def _build_noise_fn(
    preset: str,
    eval_size: int,
    seed: int,
    noise_mode: str = "heteroscedastic",
):
    """Return a spatially-varying noise function, or None for homoscedastic mode.

    Heteroscedastic: linear rank ramp along the first principal component (PC1).
      factor = 0.5 + rank_pc1  ∈ [0.5, 1.5],  mean factor = 1.0  →  mean std = base_std

    Homoscedastic: returns None; callers should omit aleatoric_std_fn override.
    """
    mode = noise_mode.strip().lower()
    if mode in {"homoscedastic", "homo"}:
        return None

    from udbench.datasets import DataSet
    from udbench.datasets.presets import PRESETS

    config = PRESETS[preset]
    base_std = float(np.asarray(config["aleatoric_std_fn"](np.zeros((1, 1)))).item())

    ds_pool = DataSet.from_preset(preset, num_observations=1, eval_size=1000,
                                  max_pool_size=2000, key=seed)
    X_pool  = np.asarray(ds_pool.X_eval, dtype=np.float64)

    feat_mean = X_pool.mean(axis=0)
    feat_std  = X_pool.std(axis=0) + 1e-8
    _, _, Vt  = np.linalg.svd((X_pool - feat_mean) / feat_std, full_matrices=False)
    pc1 = Vt[0] 

    def _pc1_hetero_std_fn(x: jnp.ndarray) -> jnp.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        n = x.shape[0]
        if n <= 1:
            return jnp.full((n,), base_std)
        scores = (x - feat_mean) / feat_std @ pc1          # scalar projection onto PC1
        ranks  = np.argsort(np.argsort(scores)).astype(np.float64) / (n - 1)
        factor = 0.5 + ranks                                # ∈ [0.5, 1.5], mean ≈ 1.0
        return jnp.asarray(base_std * factor)

    return _pc1_hetero_std_fn
