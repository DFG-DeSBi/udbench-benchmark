"""Native GPJax interfaces used across UDBench."""

from .gpjax_interface import (
    CallableKernel,
    CallableMeanFunction,
    GPJaxLatentDistribution,
    GPJaxPosteriorModel,
    GPJaxPriorModel,
    StandardizationSpec,
    build_callable_prior,
    condition_posterior,
    wrap_prior,
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
