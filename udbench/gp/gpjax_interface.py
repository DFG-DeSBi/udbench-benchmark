from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import gpjax as gpx


def _coerce_inputs(x: Any) -> jnp.ndarray:
    x_array = jnp.asarray(x, dtype=jnp.float64)
    if x_array.ndim == 0:
        return x_array.reshape(1, 1)
    if x_array.ndim == 1:
        return x_array.reshape(-1, 1)
    if x_array.ndim != 2:
        raise ValueError("GP inputs must have shape (N,) or (N, D).")
    return x_array


def _coerce_targets(y: Any) -> jnp.ndarray:
    return jnp.asarray(y, dtype=jnp.float64).reshape(-1)


def _coerce_obs_stddev(obs_stddev: Any, *, size: int) -> jnp.ndarray:
    std_array = jnp.asarray(obs_stddev, dtype=jnp.float64)
    if std_array.ndim == 0:
        return std_array.reshape(())
    std_array = std_array.reshape(-1)
    if std_array.size not in (1, size):
        raise ValueError(
            "Observation noise must be scalar or match the number of targets."
        )
    if std_array.size == 1:
        return std_array.reshape(())
    return std_array


@dataclass(frozen=True)
class StandardizationSpec:
    enabled: bool = False
    x_mean: Any | None = None
    x_std: Any | None = None
    y_mean: float = 0.0
    y_std: float = 1.0
    unstandardize_y: bool = True

    @classmethod
    def from_optional(
        cls,
        *,
        enabled: bool = False,
        x_mean: Any | None = None,
        x_std: Any | None = None,
        y_mean: Any | None = None,
        y_std: Any | None = None,
        unstandardize_y: bool = True,
    ) -> "StandardizationSpec":
        safe_y_std = 1.0 if y_std is None or float(y_std) == 0.0 else float(y_std)
        safe_y_mean = 0.0 if y_mean is None else float(y_mean)
        return cls(
            enabled=bool(enabled),
            x_mean=x_mean,
            x_std=x_std,
            y_mean=safe_y_mean,
            y_std=safe_y_std,
            unstandardize_y=bool(unstandardize_y),
        )

    def public_to_model_inputs(self, x: Any) -> jnp.ndarray:
        x_array = _coerce_inputs(x)
        if not self.enabled:
            return x_array
        mean = (
            jnp.asarray(self.x_mean, dtype=jnp.float64)
            if self.x_mean is not None
            else jnp.zeros((x_array.shape[1],), dtype=jnp.float64)
        )
        std = (
            jnp.asarray(self.x_std, dtype=jnp.float64)
            if self.x_std is not None
            else jnp.ones((x_array.shape[1],), dtype=jnp.float64)
        )
        std = jnp.where(std == 0.0, 1.0, std)
        return (x_array - mean) / std

    def public_to_model_targets(self, y: Any) -> jnp.ndarray:
        y_array = _coerce_targets(y)
        if not self.enabled or not self.unstandardize_y:
            return y_array
        return (y_array - self.y_mean) / self.y_std

    def public_to_model_obs_stddev(self, obs_stddev: Any, *, size: int) -> jnp.ndarray:
        std_array = _coerce_obs_stddev(obs_stddev, size=size)
        if not self.enabled or not self.unstandardize_y:
            return std_array
        return std_array / self.y_std

    def model_to_public_mean(self, mean: Any) -> jnp.ndarray:
        mean_array = jnp.asarray(mean, dtype=jnp.float64).reshape(-1)
        if not self.enabled or not self.unstandardize_y:
            return mean_array
        return mean_array * self.y_std + self.y_mean

    def model_to_public_variance(self, variance: Any) -> jnp.ndarray:
        variance_array = jnp.asarray(variance, dtype=jnp.float64).reshape(-1)
        if not self.enabled or not self.unstandardize_y:
            return variance_array
        return variance_array * (self.y_std ** 2)

    def model_to_public_samples(self, samples: Any) -> jnp.ndarray:
        samples_array = jnp.asarray(samples, dtype=jnp.float64)
        if not self.enabled or not self.unstandardize_y:
            return samples_array
        return samples_array * self.y_std + self.y_mean


class CallableMeanFunction(gpx.mean_functions.AbstractMeanFunction):
    """Wrap a plain Python mean callable behind GPJax's mean-function API."""

    def __init__(self, fn: Callable[[jnp.ndarray], Any]) -> None:
        super().__init__()
        self.fn = fn

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        values = jnp.asarray(self.fn(jnp.asarray(x, dtype=jnp.float64)), dtype=jnp.float64)
        if values.ndim == 0:
            return jnp.full((x.shape[0], 1), values, dtype=jnp.float64)
        if values.ndim == 1:
            return values.reshape(-1, 1)
        if values.ndim == 2 and values.shape[1] == 1:
            return values
        raise ValueError("Mean functions must return shape (N,) or (N, 1).")


class CallableKernel(gpx.kernels.AbstractKernel):
    """Wrap a plain Python kernel callable behind GPJax's kernel API."""

    def __init__(self, fn: Callable[[jnp.ndarray, jnp.ndarray], Any]) -> None:
        super().__init__()
        self.fn = fn

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            self.fn(
                jnp.asarray(x, dtype=jnp.float64),
                jnp.asarray(y, dtype=jnp.float64),
            ),
            dtype=jnp.float64,
        )


class GPJaxLatentDistribution:
    """Expose GPJax distributions through mu/std/sample API."""

    def __init__(
        self,
        distribution: Any,
        *,
        standardization: StandardizationSpec | None = None,
    ) -> None:
        self._distribution = distribution
        self._standardization = standardization or StandardizationSpec()

    @property
    def mean(self) -> jnp.ndarray:
        return self._standardization.model_to_public_mean(self._distribution.mean)

    @property
    def mu(self) -> jnp.ndarray:
        return self.mean

    @property
    def variance(self) -> jnp.ndarray:
        return self._standardization.model_to_public_variance(
            self._distribution.variance
        )

    @property
    def std(self) -> jnp.ndarray:
        return jnp.sqrt(jnp.clip(self.variance, a_min=0.0))

    def sample(self, key: int | jax.Array, num_samples: int = 1) -> jnp.ndarray:
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        samples = self._distribution.sample(key=key, sample_shape=(int(num_samples),))
        return self._standardization.model_to_public_samples(samples)


class GPJaxPriorModel:
    """Public prior wrapper that applies input/output conventions."""

    def __init__(
        self,
        prior: gpx.gps.Prior,
        *,
        standardization: StandardizationSpec | None = None,
    ) -> None:
        self.raw_prior = prior
        self.standardization = standardization or StandardizationSpec()

    def __call__(self, x: Any) -> GPJaxLatentDistribution:
        x_model = self.standardization.public_to_model_inputs(x)
        return GPJaxLatentDistribution(
            self.raw_prior(x_model),
            standardization=self.standardization,
        )


class GPJaxPosteriorModel:
    """Conditioned GPJax posterior with public callable interface."""

    def __init__(
        self,
        *,
        posterior: Any,
        train_data: gpx.Dataset,
        prior: gpx.gps.Prior,
        standardization: StandardizationSpec | None = None,
    ) -> None:
        self.posterior = posterior
        self.raw_posterior = posterior
        self.train_data = train_data
        self.likelihood = posterior.likelihood
        self.standardization = standardization or StandardizationSpec()
        self.prior = GPJaxPriorModel(prior, standardization=self.standardization)

    def __call__(self, x: Any) -> GPJaxLatentDistribution:
        x_model = self.standardization.public_to_model_inputs(x)
        return GPJaxLatentDistribution(
            self.posterior(x_model, self.train_data),
            standardization=self.standardization,
        )


def build_callable_prior(
    mean_fn: Callable[[jnp.ndarray], Any],
    kernel_fn: Callable[[jnp.ndarray, jnp.ndarray], Any],
    *,
    jitter: float = 1e-6,
) -> gpx.gps.Prior:
    return gpx.gps.Prior(
        mean_function=CallableMeanFunction(mean_fn),
        kernel=CallableKernel(kernel_fn),
        jitter=jitter,
    )


def wrap_prior(
    prior: gpx.gps.Prior,
    *,
    standardization: StandardizationSpec | None = None,
) -> GPJaxPriorModel:
    return GPJaxPriorModel(prior, standardization=standardization)


def condition_posterior(
    prior: gpx.gps.Prior,
    X: Any,
    y: Any,
    *,
    obs_stddev: Any,
    standardization: StandardizationSpec | None = None,
) -> GPJaxPosteriorModel:
    spec = standardization or StandardizationSpec()
    X_model = spec.public_to_model_inputs(X)
    y_model = spec.public_to_model_targets(y)
    obs_model = spec.public_to_model_obs_stddev(obs_stddev, size=y_model.shape[0])

    train_data = gpx.Dataset(
        X=X_model,
        y=y_model.reshape(-1, 1),
    )
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=train_data.n,
        obs_stddev=obs_model,
    )
    posterior = prior * likelihood
    return GPJaxPosteriorModel(
        posterior=posterior,
        train_data=train_data,
        prior=prior,
        standardization=spec,
    )


__all__ = [
    "CallableKernel",
    "CallableMeanFunction",
    "GPJaxLatentDistribution",
    "GPJaxPosteriorModel",
    "GPJaxPriorModel",
    "StandardizationSpec",
    "build_callable_prior",
    "condition_posterior",
    "wrap_prior",
]
