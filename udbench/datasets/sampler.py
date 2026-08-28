"""Backward-compatibility shim. Import from udbench.data.shared.sampling instead."""
from udbench.data.shared.sampling import make_stratified_mahalanobis_sampler  # noqa: F401

__all__ = ["make_stratified_mahalanobis_sampler"]
