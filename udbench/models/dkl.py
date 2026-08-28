from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import gpjax as gpx
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.parameters import Parameter

from udbench.BaseUDRegressor import BaseUDRegressor
from udbench.tuning.objectives import ensure_metric, regression_metrics
from udbench.tuning.search_spaces import make_wandb_sweep_config
from udbench.tuning.sweep import prepare_train_val, run_wandb_bayes_sweep


@dataclass
class _DeepKernelFunction(AbstractKernel):
    """GPJax kernel composed with a learned neural feature map."""

    base_kernel: AbstractKernel
    network: nnx.Module
    compute_engine: AbstractKernelComputation = field(
        default_factory=lambda: DenseKernelComputation()
    )

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        x_features = self.network(x)
        y_features = self.network(y)
        return self.base_kernel(x_features, y_features)


class _FeatureNetwork(nnx.Module):
    """Small MLP used as the DKL feature extractor."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        *,
        input_dim: int,
        hidden_features: Sequence[int],
        feature_space_dim: int,
    ) -> None:
        hidden_sizes = tuple(int(width) for width in hidden_features)
        if not hidden_sizes:
            raise ValueError("hidden_features must contain at least one layer width.")
        if any(width <= 0 for width in hidden_sizes):
            raise ValueError("hidden_features must only contain positive integers.")
        if feature_space_dim <= 0:
            raise ValueError("feature_space_dim must be a positive integer.")

        hidden_layer_names: list[str] = []
        for layer_index, hidden_width in enumerate(hidden_sizes):
            layer_name = f"hidden_layer_{layer_index}"
            setattr(
                self,
                layer_name,
                nnx.Linear(
                    input_dim if layer_index == 0 else hidden_sizes[layer_index - 1],
                    hidden_width,
                    rngs=rngs,
                ),
            )
            hidden_layer_names.append(layer_name)
        self.hidden_layer_names = tuple(hidden_layer_names)
        self.output_layer = nnx.Linear(
            hidden_sizes[-1],
            int(feature_space_dim),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        is_single_input = x.ndim == 1
        if is_single_input:
            x = x.reshape(1, -1)
        else:
            x = x.reshape((x.shape[0], -1))
        for layer_name in self.hidden_layer_names:
            layer = getattr(self, layer_name)
            x = jax.nn.relu(layer(x))
        x = self.output_layer(x).reshape((x.shape[0], -1))
        if is_single_input:
            return x.reshape(-1)
        return x


@dataclass
class DKLRegressor(BaseUDRegressor):
    """Deep-kernel GP regressor built from Flax NNX and GPJax."""

    requires_wandb_project: ClassVar[bool] = True
    hidden_features: tuple[int, ...] = (32,)
    feature_space_dim: int = 3
    noise_std: float | None = None  # initialize at 0.2 * std(y_train)
    default_noise_std_scale: float = 0.2
    base_kernel_variance: float = 1.0
    base_kernel_lengthscale: float = 1.0
    num_iters: int = 800
    warmup_steps: int = 75
    peak_learning_rate: float = 1e-2
    end_learning_rate: float = 0.0
    gradient_clip: float = 1.0
    weight_decay: float = 0.0
    standardize_inputs: bool = True
    standardize_targets: bool = True
    verbose: bool = False
    rng: int = 123

    _eps: float = 1e-12

    @staticmethod
    def _normalize_hidden_features(hidden_features: Sequence[int] | int) -> tuple[int, ...]:
        if isinstance(hidden_features, (int, np.integer)):
            hidden_sizes = (int(hidden_features),)
        else:
            hidden_sizes = tuple(int(width) for width in hidden_features)
        if not hidden_sizes:
            raise ValueError("hidden_features must contain at least one layer width.")
        if any(width <= 0 for width in hidden_sizes):
            raise ValueError("hidden_features must only contain positive integers.")
        return hidden_sizes

    def _coerce_inputs(self, X: Any) -> jnp.ndarray:
        X_array = jnp.asarray(X, dtype=jnp.float64)
        if X_array.ndim == 1:
            return X_array.reshape(-1, 1)
        if X_array.ndim != 2:
            raise ValueError("X must have shape (N,) or (N, D).")
        return X_array

    def _coerce_targets(self, y: Any) -> jnp.ndarray:
        return jnp.asarray(y, dtype=jnp.float64).reshape(-1)

    def _standardize_xy(self, X: jnp.ndarray, y: jnp.ndarray) -> dict[str, jnp.ndarray | float]:
        x_mean = jnp.mean(X, axis=0)
        x_std = jnp.std(X, axis=0)
        x_std = jnp.where(x_std > self._eps, x_std, 1.0)
        y_mean = jnp.mean(y)
        y_std = jnp.std(y)
        y_std = jnp.where(y_std > self._eps, y_std, 1.0)

        X_model = (X - x_mean) / x_std if self.standardize_inputs else X
        y_model = (y - y_mean) / y_std if self.standardize_targets else y
        return {
            "X_model": X_model,
            "y_model": y_model,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": float(y_mean),
            "y_std": float(y_std),
        }

    def _build_optimizer(self, *, num_iters: int, warmup_steps: int) -> optax.GradientTransformation:
        # decay_steps is the *total* horizon; optax internally subtracts warmup_steps
        # to get the cosine phase length, so we must not subtract warmup here.
        effective_warmup = max(0, min(int(warmup_steps), max(int(num_iters) - 1, 0)))
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=float(self.peak_learning_rate),
            warmup_steps=effective_warmup,
            decay_steps=int(num_iters),
            end_value=float(self.end_learning_rate),
        )
        return optax.chain(
            optax.clip(float(self.gradient_clip)),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=float(self.weight_decay),
            ),
        )

    def _build_model(
        self,
        *,
        input_dim: int,
        num_datapoints: int,
        noise_std: float,
    ) -> Any:
        feature_dims = list(range(int(self.feature_space_dim)))
        network = _FeatureNetwork(
            nnx.Rngs(int(self.rng)),
            input_dim=int(input_dim),
            hidden_features=self.hidden_features,
            feature_space_dim=int(self.feature_space_dim),
        )
        base_kernel = gpx.kernels.Matern52(
            active_dims=feature_dims,
            lengthscale=jnp.full(
                (int(self.feature_space_dim),),
                float(self.base_kernel_lengthscale),
                dtype=jnp.float64,
            ),
            variance=float(self.base_kernel_variance),
        )
        kernel = _DeepKernelFunction(
            base_kernel=base_kernel,
            network=network,
        )
        prior = gpx.gps.Prior(
            mean_function=gpx.mean_functions.Zero(),
            kernel=kernel,
        )
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=int(num_datapoints),
            obs_stddev=float(noise_std),
        )
        return prior * likelihood

    def _call_posterior(self, posterior: Any, X_query: jnp.ndarray, train_data: gpx.Dataset) -> Any:
        try:
            return posterior(
                X_query,
                train_data=train_data,
                return_covariance_type="diagonal",
            )
        except TypeError:
            try:
                return posterior(X_query, train_data=train_data)
            except TypeError:
                return posterior(X_query, train_data)

    def train(self, X, y, tune: bool = False, **kwargs: Any):
        del tune
        kwargs = dict(kwargs)
        kwargs, save_tuned_params, _ = self._resolve_tuned_params(kwargs)
        X = self._coerce_inputs(X)
        y = self._coerce_targets(y)

        hidden_features_cfg = kwargs.pop("hidden_features", self.hidden_features)
        if "n_layers" in kwargs or "hidden_dim" in kwargs:
            default_hidden = self._normalize_hidden_features(hidden_features_cfg)
            hidden_dim = int(kwargs.pop("hidden_dim", default_hidden[0]))
            n_layers = int(kwargs.pop("n_layers", len(default_hidden)))
            hidden_features_cfg = (hidden_dim,) * n_layers

        self.hidden_features = self._normalize_hidden_features(hidden_features_cfg)
        self.feature_space_dim = int(kwargs.pop("feature_space_dim", self.feature_space_dim))
        noise_std_value = kwargs.pop("noise_std", self.noise_std)
        self.base_kernel_variance = float(
            kwargs.pop("base_kernel_variance", self.base_kernel_variance)
        )
        self.base_kernel_lengthscale = float(
            kwargs.pop("base_kernel_lengthscale", self.base_kernel_lengthscale)
        )
        self.num_iters = int(kwargs.pop("num_iters", self.num_iters))
        self.warmup_steps = int(kwargs.pop("warmup_steps", self.warmup_steps))
        self.peak_learning_rate = float(
            kwargs.pop("peak_learning_rate", self.peak_learning_rate)
        )
        self.end_learning_rate = float(kwargs.pop("end_learning_rate", self.end_learning_rate))
        self.gradient_clip = float(kwargs.pop("gradient_clip", self.gradient_clip))
        self.weight_decay = float(kwargs.pop("weight_decay", self.weight_decay))
        self.standardize_inputs = bool(
            kwargs.pop("standardize_inputs", self.standardize_inputs)
        )
        self.standardize_targets = bool(
            kwargs.pop("standardize_targets", self.standardize_targets)
        )
        self.verbose = bool(kwargs.pop("verbose", self.verbose))
        self.rng = int(kwargs.pop("rng", self.rng))

        num_iters = self.num_iters
        warmup_steps = self.warmup_steps
        verbose = self.verbose
        rng = self.rng
        if num_iters <= 0:
            raise ValueError("num_iters must be a positive integer.")

        stats = self._standardize_xy(X, y)
        if noise_std_value is None:
            noise_std_orig = float(self.default_noise_std_scale) * float(stats["y_std"])
            noise_std = (
                noise_std_orig / float(stats["y_std"])
                if self.standardize_targets
                else noise_std_orig
            )
        else:
            noise_std = float(noise_std_value)
            noise_std_orig = (
                noise_std * float(stats["y_std"])
                if self.standardize_targets
                else noise_std
            )
        if noise_std <= 0.0:
            raise ValueError("noise_std must be > 0.")
        self.noise_std = noise_std
        X_model = jnp.asarray(stats["X_model"], dtype=jnp.float64)
        y_model = jnp.asarray(stats["y_model"], dtype=jnp.float64).reshape(-1, 1)
        train_data = gpx.Dataset(X=X_model, y=y_model)

        posterior = self._build_model(
            input_dim=X_model.shape[1],
            num_datapoints=train_data.n,
            noise_std=noise_std,
        )
        optimizer = self._build_optimizer(
            num_iters=num_iters,
            warmup_steps=warmup_steps,
        )
        objective = lambda model, data: -gpx.objectives.conjugate_mll(model, data)
        trained_posterior, history = gpx.fit(
            model=posterior,
            objective=objective,
            train_data=train_data,
            optim=optimizer,
            num_iters=num_iters,
            key=jax.random.PRNGKey(rng),
            trainable=(Parameter, nnx.Param),
            safe=True,
            verbose=verbose,
        )

        self.model = {
            "posterior": trained_posterior,
            "train_data": train_data,
            "history": np.asarray(history, dtype=np.float64),
            "noise_std_init": noise_std,
            "noise_std_init_original": noise_std_orig,
            "x_mean": np.asarray(stats["x_mean"], dtype=np.float64),
            "x_std": np.asarray(stats["x_std"], dtype=np.float64),
            "y_mean": float(stats["y_mean"]),
            "y_std": float(stats["y_std"]),
            "standardize_inputs": bool(self.standardize_inputs),
            "standardize_targets": bool(self.standardize_targets),
        }
        self._store_tuned_params(
            {
                "hidden_features": tuple(self.hidden_features),
                "feature_space_dim": int(self.feature_space_dim),
                "noise_std": float(self.noise_std),
                "base_kernel_variance": float(self.base_kernel_variance),
                "base_kernel_lengthscale": float(self.base_kernel_lengthscale),
                "num_iters": int(self.num_iters),
                "warmup_steps": int(self.warmup_steps),
                "peak_learning_rate": float(self.peak_learning_rate),
                "end_learning_rate": float(self.end_learning_rate),
                "gradient_clip": float(self.gradient_clip),
                "weight_decay": float(self.weight_decay),
                "standardize_inputs": bool(self.standardize_inputs),
                "standardize_targets": bool(self.standardize_targets),
                "rng": int(self.rng),
            },
            save=save_tuned_params,
        )
        return self.model

    def predict(self, model, X, **kwargs: Any):
        del kwargs
        self._check_fitted()
        active_model = self.model if model is None else model
        X_query = self._prepare_query_inputs(X, active_model)
        latent_dist = self._call_posterior(
            active_model["posterior"],
            X_query,
            active_model["train_data"],
        )
        predictive_dist = active_model["posterior"].likelihood(latent_dist)
        mean = np.asarray(predictive_dist.mean, dtype=np.float64).reshape(-1)
        if active_model["standardize_targets"]:
            mean = mean * active_model["y_std"] + active_model["y_mean"]
        return mean

    def ud_fit_predict(self, X_train, y_train, X_eval, **kwargs: Any) -> dict[str, np.ndarray]:
        self.train(X_train, y_train, **kwargs)
        self._check_fitted()

        X_query = self._prepare_query_inputs(X_eval, self.model)
        latent_dist = self._call_posterior(
            self.model["posterior"],
            X_query,
            self.model["train_data"],
        )
        predictive_dist = self.model["posterior"].likelihood(latent_dist)

        mean = np.asarray(predictive_dist.mean, dtype=np.float64).reshape(-1)
        total_variance = np.asarray(predictive_dist.variance, dtype=np.float64).reshape(-1)
        epistemic_variance = np.asarray(latent_dist.variance, dtype=np.float64).reshape(-1)
        aleatoric_variance = np.clip(total_variance - epistemic_variance, a_min=0.0, a_max=None)

        if self.model["standardize_targets"]:
            y_std = float(self.model["y_std"])
            y_mean = float(self.model["y_mean"])
            mean = mean * y_std + y_mean
            variance_scale = y_std ** 2
            total_variance = total_variance * variance_scale
            epistemic_variance = epistemic_variance * variance_scale
            aleatoric_variance = aleatoric_variance * variance_scale

        return {
            "y_pred": mean,
            "total_uncertainty": total_variance,
            "epistemic_uncertainty": epistemic_variance,
            "aleatoric_uncertainty": aleatoric_variance,
        }

    @classmethod
    def _wandb_best_config_to_train_kwargs(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        hidden_dim = int(cfg["hidden_dim"])
        n_layers = int(cfg["n_layers"])
        return {
            "hidden_features": (hidden_dim,) * n_layers,
            "feature_space_dim": int(cfg["feature_space_dim"]),
            "noise_std": float(cfg["noise_std"]),
            "base_kernel_variance": float(cfg["base_kernel_variance"]),
            "base_kernel_lengthscale": float(cfg["base_kernel_lengthscale"]),
            "num_iters": int(cfg["num_iters"]),
            "warmup_steps": int(cfg["warmup_steps"]),
            "peak_learning_rate": float(cfg["peak_learning_rate"]),
            "end_learning_rate": float(cfg["end_learning_rate"]),
            "gradient_clip": float(cfg["gradient_clip"]),
            "weight_decay": float(cfg["weight_decay"]),
            "standardize_inputs": bool(cfg["standardize_inputs"]),
            "standardize_targets": bool(cfg["standardize_targets"]),
        }

    @classmethod
    def tune_hyperparameters_wandb(
        cls,
        X: Any,
        y: Any,
        *,
        project: str,
        count: int = 30,
        metric_name: str = "val_rmse",
        rng: int | np.random.Generator | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        **wandb_init_kwargs: Any,
    ) -> dict[str, Any]:
        X_tr, y_tr, X_v, y_v = prepare_train_val(
            X,
            y,
            val_fraction=0.2,
            rng=rng,
        )

        sweep_config = make_wandb_sweep_config("dkl", metric_name)

        def objective(cfg: dict[str, Any]) -> dict[str, float]:
            model = cls(
                hidden_features=(int(cfg["hidden_dim"]),) * int(cfg["n_layers"]),
                feature_space_dim=int(cfg["feature_space_dim"]),
                noise_std=float(cfg["noise_std"]),
                base_kernel_variance=float(cfg["base_kernel_variance"]),
                base_kernel_lengthscale=float(cfg["base_kernel_lengthscale"]),
                peak_learning_rate=float(cfg["peak_learning_rate"]),
                end_learning_rate=float(cfg["end_learning_rate"]),
                gradient_clip=float(cfg["gradient_clip"]),
                weight_decay=float(cfg["weight_decay"]),
                standardize_inputs=bool(cfg["standardize_inputs"]),
                standardize_targets=bool(cfg["standardize_targets"]),
                verbose=False,
            )
            preds = model.ud_fit_predict(
                X_tr,
                y_tr,
                X_v,
                num_iters=int(cfg["num_iters"]),
                warmup_steps=int(cfg["warmup_steps"]),
            )
            metrics = regression_metrics(y_v, preds["y_pred"], preds.get("total_uncertainty"))
            ensure_metric(metrics, metric_name)
            return metrics

        result = run_wandb_bayes_sweep(
            sweep_config=sweep_config,
            objective_fn=objective,
            project=project,
            entity=entity,
            count=count,
            tags=tags,
            **wandb_init_kwargs,
        )
        best_config = result.get("best_config")
        if best_config is None:
            return result
        normalized_result = dict(result)
        normalized_result["best_config"] = cls._wandb_best_config_to_train_kwargs(best_config)
        return normalized_result

    def _prepare_query_inputs(self, X: Any, model: dict[str, Any]) -> jnp.ndarray:
        X_array = self._coerce_inputs(X)
        if model["standardize_inputs"]:
            x_mean = jnp.asarray(model["x_mean"], dtype=jnp.float64)
            x_std = jnp.asarray(model["x_std"], dtype=jnp.float64)
            X_array = (X_array - x_mean) / x_std
        return X_array

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")
