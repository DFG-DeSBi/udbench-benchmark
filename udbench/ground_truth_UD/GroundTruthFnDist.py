from __future__ import annotations
from abc import ABC
import inspect
from typing import Any

import jax
import jax.numpy as jnp

class GroundTruthFnDist(ABC):
    """
    Distribution over ground-truth functions conditioned on S.

    Subclasses must implement either:
      - sample(...)
      - mean(...) AND variance(...)
    """

    # ---------- TEMPLATE API ----------

    def sample(self, x, n=1, key=0):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement sample()"
        )

    def mean(self, x):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement mean()"
        )

    def variance(self, x):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement variance()"
        )

    # ---------- RUNTIME CONTRACT ----------

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if inspect.isabstract(cls):
            return

        has_sample = cls.sample is not GroundTruthFnDist.sample
        has_mean = cls.mean is not GroundTruthFnDist.mean
        has_variance = cls.variance is not GroundTruthFnDist.variance

        if not (has_sample or (has_mean and has_variance)):
            raise TypeError(
                f"{cls.__name__} must implement either:\n"
                f"  - sample(x, n, key)\n"
                f"  - mean(x) AND variance(x)"
            )


# Concrete implementation of the Groundtruth Function Distribution as a Gaussian Process
class GPPosteriorFnDist(GroundTruthFnDist):
    def __init__(self, posterior: Any, train_data: Any | None = None):
        self._posterior = posterior
        self._train_data = train_data

        if (
            not callable(posterior)
            and (
                hasattr(posterior, "posterior")
                and hasattr(posterior, "train_data")
                and getattr(posterior, "posterior") is not None
                and getattr(posterior, "train_data") is not None
            )
        ):
            self._posterior = posterior.posterior
            self._train_data = posterior.train_data

    def _distribution(self, x):
        if self._train_data is None:
            return self._posterior(x)
        try:
            return self._posterior(x, self._train_data, return_covariance_type="dense")
        except TypeError:
            return self._posterior(x, self._train_data)

    def sample(self, x, n=1, key: int | jax.Array = 0):
        key = jax.random.PRNGKey(key) if isinstance(key, int) else key
        distribution = self._distribution(x)
        try:
            return distribution.sample(key, num_samples=n)
        except TypeError:
            return distribution.sample(key=key, sample_shape=(n,))

    def mean(self, x):
        distribution = self._distribution(x)
        if hasattr(distribution, "mu"):
            return distribution.mu
        return jnp.asarray(distribution.mean).reshape(-1)

    def variance(self, x):
        distribution = self._distribution(x)
        if hasattr(distribution, "std"):
            std = distribution.std
            return std**2
        return jnp.asarray(distribution.variance).reshape(-1)
