from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import logging

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# --- Local imports --- #
from udbench.datasets.datasets_utils import print_dataset_diagnostics
from udbench.gp import (
    StandardizationSpec,
    build_callable_prior,
    condition_posterior,
    wrap_prior,
)
from .presets import PRESETS, load_preset_config
from .sampler import make_stratified_mahalanobis_sampler

class DataSet:
    """
    Synthetic regression dataset generated from a Gaussian Process prior.

    A latent function is sampled from a GP prior and observed with
    input-dependent Gaussian (aleatoric) noise. Observations are selected
    according to a configurable sampling strategy.

    This dataset exposes both:
    - noisy observations (training data)
    - dense evaluation data with access to the latent ground truth

    Parameters
    ----------
    mean_fn
        Mean function of the Gaussian Process prior.
    kernel_fn
        Kernel function of the Gaussian Process prior.
    aleatoric_std_fn
        Callable mapping inputs ``X`` to aleatoric noise standard deviation.
        If ``None``, ``aleatoric_std_scale`` is used instead.
    aleatoric_std_scale
        When ``aleatoric_std_fn`` is ``None``, set noise as a constant
        ``aleatoric_std_scale * std(y_eval_clean)`` per dataset draw.
    keep_target_standardized
        If ``True`` and ``standardize=True``, do not map the target back to the
        original scale. Targets remain standardized throughout the dataset.
    input_range
        Input domain definition. Accepts:
        - Tuple ``(min, max)`` where each entry is a scalar or array-like.
        - List of per-feature ``(min, max)`` pairs.
        Scalars are broadcast to all dimensions.
    num_observations
        Number of observed (training) samples.
    sampling_fn
        Optional callable defining how training samples are selected
        from the evaluation set.
    key
        Random seed.
    eval_size
        Number of evaluation points. Defaults to ``max(1000, num_observations)``.
    """
    PRESETS = PRESETS

    def __init__(
        self,
        mean_fn: Optional[Callable] = None,
        kernel_fn: Optional[Callable] = None,
        aleatoric_std_fn: Optional[Callable] = None,
        aleatoric_std_scale: Optional[float] = None,
        standardize: bool = False,
        keep_target_standardized: bool = False,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        y_mean: Optional[float] = None,
        y_std: Optional[float] = None,
        input_range: Tuple[float, float] | list[tuple[float, float]] = (-1.0, 1.0),
        num_observations: int = 100,
        sampling_fn: Optional[Callable] = None,
        key: int = 42,
        eval_size: Optional[int] = None,
        name: Optional[str] = None,
        uci_dataset_id: Optional[int] = None,
        uci_dataset_name: Optional[str] = None,
        uci_feature_subset: Optional[list[str]] = None,
        base_dataset_name: Optional[str] = None,
        preset_variant: Optional[str] = None,
        gpjax_kernel_expr: Optional[str] = None,
        gpjax_mean_expr: Optional[str] = None,
        max_pool_size: Optional[int] = None,
    ) -> None:
        # Configuration
        self.mean_fn = mean_fn
        self.kernel_fn = kernel_fn
        self.aleatoric_std_fn = aleatoric_std_fn
        self.aleatoric_std_scale = aleatoric_std_scale
        self.standardize = bool(standardize)
        self.keep_target_standardized = bool(keep_target_standardized)
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.input_range = input_range
        self.input_range_min, self.input_range_max = self._normalize_input_range(
            input_range
        )
        self.train_size = num_observations
        self.eval_size = eval_size
        self.sampling_fn = sampling_fn
        self.key = key
        self.name = name
        self.uci_dataset_id = uci_dataset_id
        self.uci_dataset_name = uci_dataset_name
        self.uci_feature_subset = uci_feature_subset
        self.base_dataset_name = base_dataset_name
        self.preset_variant = preset_variant
        self.gpjax_kernel_expr = gpjax_kernel_expr
        self.gpjax_mean_expr = gpjax_mean_expr
        self.max_pool_size = max_pool_size
        self.feature_names: list[str] | None = None

        if self.standardize:
            if self.x_mean is None or self.x_std is None or self.y_mean is None or self.y_std is None:
                raise ValueError(
                    "standardize=True requires x_mean, x_std, y_mean, y_std to be provided."
                )

        # Generated data (populated by generate_dataset)
        self.X_eval: Optional[jnp.ndarray] = None
        self.y_eval: Optional[jnp.ndarray] = None
        self.y_eval_clean: Optional[jnp.ndarray] = None
        self.aleatoric_std_eval: Optional[jnp.ndarray] = None

        self.X_obs: Optional[jnp.ndarray] = None
        self.y_obs: Optional[jnp.ndarray] = None
        self.aleatoric_std_obs: Optional[jnp.ndarray] = None

        self.posterior: Optional[object] = None
        self.train_indices: Optional[jnp.ndarray] = None

        # Generate dataset immediately
        self.generate_dataset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def input_dim(self) -> int:
        """Dimensionality of the input space."""
        if np.ndim(self.input_range_min) == 0:
            return 1
        return int(len(self.input_range_min))

    @staticmethod
    def _normalize_input_range(
        input_range: Tuple[float, float] | list[tuple[float, float]]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if isinstance(input_range, tuple) and len(input_range) == 2:
            min_val, max_val = input_range
            return jnp.asarray(min_val), jnp.asarray(max_val)

        pairs = list(input_range)
        if not pairs:
            raise ValueError("input_range list cannot be empty.")
        if not all(len(pair) == 2 for pair in pairs):
            raise ValueError("input_range list must contain (min, max) pairs.")
        mins = [pair[0] for pair in pairs]
        maxs = [pair[1] for pair in pairs]
        return jnp.asarray(mins), jnp.asarray(maxs)

    @staticmethod
    def _canonicalize_name(name: str) -> str:
        return "".join(ch.lower() for ch in str(name) if ch.isalnum())

    @classmethod
    def _match_feature_subset(
        cls,
        *,
        available_names: list[str],
        requested_names: list[str],
    ) -> tuple[list[int], list[str]]:
        lookup = {
            cls._canonicalize_name(feature_name): (index, feature_name)
            for index, feature_name in enumerate(available_names)
        }
        indices: list[int] = []
        missing: list[str] = []
        seen: set[int] = set()
        for requested_name in requested_names:
            match = lookup.get(cls._canonicalize_name(requested_name))
            if match is None:
                missing.append(requested_name)
                continue
            index, _ = match
            if index not in seen:
                indices.append(index)
                seen.add(index)
        return indices, missing

    @property
    def train_samples(self) -> jnp.ndarray:
        """
        Training samples as a concatenated array ``[X | y]``.

        Returns
        -------
        jax.numpy.ndarray
            Array of shape ``(N, D + 1)``.
        """
        if self.X_obs is None or self.y_obs is None:
            raise ValueError("Training data has not been generated.")
        return jnp.concatenate([self.X_obs, self.y_obs[:, None]], axis=1)

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------
    def _load_uci_features(
        self,
        *,
        dataset_id: int | None = None,
        name: str | None = None,
        feature_subset: list[str] | None = None,
        max_points: int | None = None,
    ):
        if dataset_id is None:
            raise ValueError(
                "Semi-synthetic preset loading requires a source dataset_id."
            )

        try:
            from udbench.semi_synth_datasets.UCI_loader_utils import Utils
        except ImportError as exc:
            raise ImportError(
                "This preset requires the semi-synthetic UCI loader and its "
                "`ucimlrepo` dependency."
            ) from exc

        dataset = Utils().load_uci_dataset(int(dataset_id), show_summary=False)
        metadata = getattr(dataset, "metadata", None)
        if isinstance(metadata, dict):
            dataset_name = metadata.get("name")
        else:
            dataset_name = getattr(metadata, "name", None)
        if name is not None and dataset_name is not None and str(dataset_name) != str(name):
            logger.warning(
                "Preset dataset name mismatch (requested=%r, fetched=%r).", name, dataset_name
            )

        def _get(container, key, default=None):
            if container is None:
                return default
            if isinstance(container, dict):
                return container.get(key, default)
            return getattr(container, key, default)

        data = _get(dataset, "data", None)
        X_df = _get(data, "features", None)
        if X_df is None:
            raise ValueError("Fetched source dataset has no feature table.")
        X_df = X_df.copy()
        available_names = [str(column) for column in X_df.columns]
        if feature_subset:
            idx, missing = self._match_feature_subset(
                available_names=available_names,
                requested_names=feature_subset,
            )
            if missing:
                logger.warning(
                    "UCI feature subset missing (preset=%s, dataset_id=%s, name=%s): "
                    "missing=%s, available=%s",
                    self.name, dataset_id, name, missing, available_names,
                )
            if not idx:
                raise ValueError(
                    "Requested UCI feature subset could not be matched to any columns."
                )
            X_df = X_df.iloc[:, idx]
            available_names = [available_names[i] for i in idx]
        else:
            if X_df.shape[1] == 0:
                raise ValueError("Fetched source dataset has no available features.")

        encoded_columns = []
        for feature_name in available_names:
            column = X_df[feature_name]
            if np.issubdtype(column.dtype, np.number):
                encoded_columns.append(np.asarray(column.to_numpy(), dtype=float))
            else:
                codes = column.astype("category").cat.codes.to_numpy(dtype=float)
                encoded_columns.append(codes)

        X_np = np.column_stack(encoded_columns)
        mask = np.isfinite(X_np).all(axis=1)
        removed = int((~mask).sum())
        if removed:
            logger.warning("Removed %d rows with NaN/inf values.", removed)
        X = X_np[mask]
        if max_points is not None and X.shape[0] > max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], size=max_points, replace=False)
            X = X[idx]
        return X, available_names

    def _build_gpjax_prior(self):
        try:
            from udbench.semi_synth_datasets.GPJax_Utils import GPSpec
        except ImportError as exc:
            raise ImportError(
                "This preset requires GPJax support. Install `gpjax`, `optax`, and "
                "`flax` to use the semi-synthetic presets."
            ) from exc

        if self.gpjax_kernel_expr is None or self.gpjax_mean_expr is None:
            raise ValueError("GPJax prior requested without kernel/mean expressions.")

        spec = GPSpec(
            kernel_expr=self.gpjax_kernel_expr,
            mean_expr=self.gpjax_mean_expr,
        )
        return spec.build_prior()

    def _build_callable_prior(self):
        if self.mean_fn is None or self.kernel_fn is None:
            raise ValueError(
                "Dataset generation requires both mean_fn and kernel_fn, or a "
                "GPJax kernel/mean expression pair."
            )
        return build_callable_prior(self.mean_fn, self.kernel_fn)

    def _build_standardization_spec(self) -> StandardizationSpec:
        return StandardizationSpec.from_optional(
            enabled=self.standardize,
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
            unstandardize_y=not self.keep_target_standardized,
        )

    def generate_dataset(
        self,
    ) -> tuple[jnp.ndarray, object, jnp.ndarray]:
        """
        Generate synthetic data and condition the GP posterior.

        This method:
        1. Samples a latent function from the GP prior
        2. Adds input-dependent Gaussian noise
        3. Selects training observations
        4. Conditions the GP posterior on noisy observations

        Returns
        -------
        y_eval_clean : jax.numpy.ndarray
            Noise-free latent function evaluated on the evaluation grid.
        posterior : object
            Posterior GP conditioned on training data.
        train_samples : jax.numpy.ndarray
            Training samples as ``[X | y]``.
        """
        use_gpjax_expr = (
            self.gpjax_kernel_expr is not None and self.gpjax_mean_expr is not None
        )
        raw_prior = (
            self._build_gpjax_prior() if use_gpjax_expr else self._build_callable_prior()
        )
        standardization = self._build_standardization_spec()
        prior = wrap_prior(raw_prior, standardization=standardization)

        eval_size = self.eval_size or max(1000, self.train_size)

        key = jax.random.PRNGKey(self.key)
        key_x, key_fn, key_noise, key_sample, key_eval, key_pool = jax.random.split(key, 6)

        # ------------------------------------------------------------------
        # Step 1: Build pool — ALL available points, no cap.
        # For UCI datasets: load every row from the source; the actual count
        # may differ from what preset.json records.
        # For synthetic datasets: generate train_size + eval_size points so
        # there are enough for both splits.
        # ------------------------------------------------------------------
        X_pool = None
        feature_names: list[str] | None = None
        uci_name = self.uci_dataset_name
        uci_id = self.uci_dataset_id
        if uci_name or uci_id is not None:
            try:
                X, feature_names = self._load_uci_features(
                    dataset_id=uci_id,
                    name=uci_name,
                    feature_subset=self.uci_feature_subset,
                    max_points=None,   # load every available row
                )
                X_pool = jnp.asarray(X, dtype=float)
                logger.info(
                    "UCI pool loaded (preset=%s, uci_id=%s, uci_name=%s): shape=%s, expected_dim=%d",
                    self.name, uci_id, uci_name, X_pool.shape, self.input_dim,
                )
                if X_pool.shape[1] != self.input_dim:
                    raise ValueError(
                        "UCI feature mismatch "
                        f"(preset={self.name}, uci_id={uci_id}, uci_name={uci_name}): "
                        f"got_dim={X_pool.shape[1]}, expected_dim={self.input_dim}, "
                        f"cols={feature_names}"
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load UCI dataset id={uci_id!r} name={uci_name!r} "
                    f"for preset '{self.name}'."
                ) from exc

        if X_pool is None:
            # For synthetic data "all available" means a large dense pool.
            # Use 10 × eval_size so the sampler has ample room to stratify
            # regardless of train_size.
            pool_size = 10 * eval_size
            X_pool = jax.random.uniform(
                key_x,
                (pool_size, self.input_dim),
                minval=self.input_range_min,
                maxval=self.input_range_max,
            )

        pool_size = int(X_pool.shape[0])
        if self.max_pool_size is not None and pool_size > int(self.max_pool_size):
            cap = int(self.max_pool_size)
            idx = jax.random.permutation(key_pool, pool_size)[:cap]
            X_pool = X_pool[idx]
            logger.info("Pool capped from %d to %d points (max_pool_size=%d).", pool_size, cap, cap)
            pool_size = cap

        if pool_size < self.train_size:
            raise ValueError(
                f"Pool has only {pool_size} points but train_size={self.train_size}. "
                "Reduce train_size or use a larger dataset."
            )

        # ------------------------------------------------------------------
        # Step 2: Latent function + aleatoric noise at every pool point.
        # ------------------------------------------------------------------
        y_pool_clean = prior(X_pool).sample(key_fn, num_samples=1).squeeze()

        if self.aleatoric_std_fn is None:
            if self.aleatoric_std_scale is None:
                raise ValueError("aleatoric_std_fn is None and aleatoric_std_scale is not set.")
            noise_sigma = float(self.aleatoric_std_scale) * float(jnp.std(y_pool_clean))
            logger.debug("Using constant aleatoric noise std: %.6g", noise_sigma)
            def aleatoric_std_fn(x):
                return jnp.full((x.shape[0],), noise_sigma, dtype=jnp.float64)
            self.aleatoric_std_fn = aleatoric_std_fn
        else:
            aleatoric_std_fn = self.aleatoric_std_fn

        aleatoric_std_pool = aleatoric_std_fn(X_pool)
        noise = jax.random.normal(key_noise, y_pool_clean.shape)
        y_pool = y_pool_clean + noise * aleatoric_std_pool

        # ------------------------------------------------------------------
        # Step 3: Uniformly split pool → eval_pool ∪ train_pool (disjoint).
        #
        # The split happens BEFORE the sampling config so that the eval pool
        # is a plain uniform random subset of the full data (same input
        # density as the raw dataset).  The sampling config is applied only
        # to the training pool in Step 4, so it cannot deplete any region
        # of the eval pool.
        #
        # Size policy: reserve eval_size points for eval; give the rest to
        # the training pool.  If the pool is too small, reduce eval (keep
        # training fixed).
        # ------------------------------------------------------------------
        n_eval_pool = min(eval_size, pool_size - self.train_size)
        if n_eval_pool <= 0:
            raise ValueError(
                f"Pool has only {pool_size} points but train_size={self.train_size} "
                "leaves no room for an eval set. Reduce train_size or eval_size."
            )
        if n_eval_pool < eval_size:
            logger.warning(
                "Eval set reduced from %d to %d (pool_size=%d, train_size=%d).",
                eval_size, n_eval_pool, pool_size, self.train_size,
            )

        shuffled = jax.random.permutation(key_eval, pool_size)
        eval_pool_idx  = shuffled[:n_eval_pool]   # uniform draw → eval
        train_pool_idx = shuffled[n_eval_pool:]   # rest → sampling config

        # ------------------------------------------------------------------
        # Step 4: Apply sampling config to the training pool only.
        # ------------------------------------------------------------------
        if self.sampling_fn is None:
            def sampling_fn(x, y, size, key):
                idx = jax.random.choice(key, x.shape[0], shape=(size,), replace=False)
                return x[idx], y[idx], idx
        else:
            sampling_fn = self.sampling_fn

        X_train_pool = X_pool[train_pool_idx]
        y_train_pool = y_pool[train_pool_idx]
        sample_result = sampling_fn(X_train_pool, y_train_pool, self.train_size, key_sample)
        if isinstance(sample_result, tuple) and len(sample_result) == 3:
            X_obs, y_obs, local_train_idx = sample_result
        else:
            X_obs, y_obs = sample_result
            local_train_idx = jnp.array([], dtype=jnp.int32)

        X_obs = jnp.atleast_2d(X_obs)
        y_obs = jnp.atleast_1d(y_obs)
        # Map local-pool indices back to global pool indices (stored for diagnostics).
        train_indices = train_pool_idx[local_train_idx]

        # ------------------------------------------------------------------
        # Step 5: Eval set = eval_pool (balanced, no sampling config).
        # ------------------------------------------------------------------
        X_eval = X_pool[eval_pool_idx]
        y_eval_clean = y_pool_clean[eval_pool_idx]
        y_eval = y_pool[eval_pool_idx]
        aleatoric_std_eval = aleatoric_std_pool[eval_pool_idx]

        self.eval_size = n_eval_pool

        # ------------------------------------------------------------------
        # Step 6: Condition GP posterior on training observations only.
        # ------------------------------------------------------------------
        # Use pool-level noise stds (same values used to generate y_obs), not
        # locally recomputed ones. Rank-based noise functions assign different
        # ranks depending on batch size, so recomputing on X_obs alone would
        # give inconsistent noise levels vs. the actual observations.
        if local_train_idx is not None and len(local_train_idx) == self.train_size:
            aleatoric_std_obs = aleatoric_std_pool[train_pool_idx[local_train_idx]]
        else:
            aleatoric_std_obs = aleatoric_std_fn(X_obs).squeeze()
        posterior = condition_posterior(
            raw_prior,
            X_obs,
            y_obs,
            obs_stddev=aleatoric_std_obs,
            standardization=standardization,
        )

        # Store state
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.y_eval_clean = y_eval_clean
        self.aleatoric_std_eval = aleatoric_std_eval

        self.X_obs = X_obs
        self.y_obs = y_obs
        self.aleatoric_std_obs = aleatoric_std_obs
        self.posterior = posterior
        self.train_indices = train_indices
        self.feature_names = feature_names or [f"x{i + 1}" for i in range(self.input_dim)]

        # Print diagnostics each time a dataset is generated.
        print_dataset_diagnostics(self, preset=self.name or "n/a")

        return y_eval_clean, posterior, self.train_samples

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def list_presets(cls) -> list[str]:
        """List available dataset presets."""
        return sorted(cls.PRESETS.keys())

    @classmethod
    def _instantiate_from_config(cls, config: dict) -> "DataSet":
        """Instantiate a dataset from one preset-style config dictionary."""

        sampling_cfg = config.pop("sampling_cfg", None)

        if config.get("sampling_fn") is None and sampling_cfg is not None:
            method = sampling_cfg.get("method")
            if method == "stratified_mahalanobis":
                config["sampling_fn"] = make_stratified_mahalanobis_sampler(
                    quantile_edges=sampling_cfg["quantile_edges"],
                    bin_fractions=sampling_cfg["bin_fractions"],
                    shrinkage=sampling_cfg.get("shrinkage", 0.1),
                    standardize=sampling_cfg.get("standardize", True),
                    prefer_center_bins=sampling_cfg.get("prefer_center_bins", True),
                )
            else:
                raise ValueError(f"Unknown sampling method: {method}")

        return cls(**config)

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "DataSet":
        if name not in cls.PRESETS:
            raise KeyError(
                f"Unknown preset '{name}'. Available presets: {cls.list_presets()}"
            )

        config = dict(cls.PRESETS[name])
        config.update(overrides)
        config.setdefault("name", name)
        return cls._instantiate_from_config(config)

    @classmethod
    def from_bundle_path(cls, bundle_path: str | Path, **overrides) -> "DataSet":
        """Instantiate a dataset directly from one exported ``preset.json`` bundle."""

        bundle_path = Path(bundle_path)
        config = load_preset_config(bundle_path)
        config.update(overrides)
        config.setdefault("name", bundle_path.parent.name)
        return cls._instantiate_from_config(config)


    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_dataset(
        self,
        ax: Optional[plt.Axes] = None,
        *,
        use_tsne: bool = True,
        tsne_random_state: int = 0,
        tsne_perplexity: int = 30,
        show_posterior: bool = True,
        posterior_scale: float = 1.0,
    ):
        """
        Visualize the dataset.

        Parameters
        ----------
        ax
            Optional matplotlib axis.
        use_tsne
            Whether to apply t-SNE for visualization.
        tsne_random_state
            Random seed for t-SNE.
        tsne_perplexity
            Perplexity parameter for t-SNE.
        """
        from udbench.datasets.dataset_plotting_utils import plot_dataset_overview
        return plot_dataset_overview(
            self,
            ax=ax,
            use_tsne=use_tsne,
            tsne_random_state=tsne_random_state,
            tsne_perplexity=tsne_perplexity,
            show_posterior=show_posterior,
            posterior_scale=posterior_scale,
        )

    def plot_posterior_variance_calibration(
        self,
        ax: Optional[plt.Axes] = None,
        *,
        bin_width: float = 0.1,
    ):
        from udbench.datasets.dataset_plotting_utils import (
            plot_posterior_variance_calibration,
        )
        return plot_posterior_variance_calibration(
            self,
            ax=ax,
            bin_width=bin_width,
        )
