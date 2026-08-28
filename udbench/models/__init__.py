from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from udbench.BaseUDRegressor import BaseUDRegressor

# Maps registry key → (module, class_name).
# Imports are deferred so heavy deps (JAX, CatBoost, …) load only on access.
_REGISTRY: dict[str, tuple[str, str]] = {
    # --- Linear ---
    "BayesianLinearRegressor":            ("udbench.models.linear",                "BayesianLinearRegressor"),
    # --- Tree boosting ---
    "NGBoostNIGRegressor":                ("udbench.models.tree.ngboost",          "NGBoostNIGRegressor"),
    "NGBoostBaggingRegressor":            ("udbench.models.tree.ngboost",          "NGBoostBaggingRegressor"),
    "CatBoostPosteriorSamplingRegressor": ("udbench.models.tree.catboost_posterior","CatBoostPosteriorSamplingRegressor"),
    "CatBoostKGBRegressor":               ("udbench.models.tree.catboost_kgb",     "CatBoostKGBRegressor"),
    # --- Deep Kernel Learning ---
    "DKLRegressor":                       ("udbench.models.dkl",                   "DKLRegressor"),
    # --- BNN (JAX) ---
    "TabularBNNBaggingRegressor":         ("udbench.models.bnn.bagging",           "TabularBNNBaggingRegressor"),
    "TabularBNNDeepRegressor":            ("udbench.models.bnn.deep_ensemble",     "TabularBNNDeepRegressor"),
    "TabularBNNDropoutRegressor":         ("udbench.models.bnn.dropout",           "TabularBNNDropoutRegressor"),
    "TabularBNNLaplaceRegressor":         ("udbench.models.bnn.laplace",           "TabularBNNLaplaceRegressor"),
    "TabularBNNFSPLaplaceRegressor":      ("udbench.models.bnn.fsp_laplace",       "TabularBNNFSPLaplaceRegressor"),
    "TabularBNNSWAGRegressor":            ("udbench.models.bnn.swag",              "TabularBNNSWAGRegressor"),
    "TabularBNNEDLRegressor":             ("udbench.models.bnn.edl",               "TabularBNNEDLRegressor"),
    "TabularBNNDEUPRegressor":            ("udbench.models.bnn.deup",              "TabularBNNDEUPRegressor"),
}

__all__ = ["MODEL_REGISTRY", "get_model_class"]


class _LazyRegistry(dict):
    """Dict subclass that imports model classes on first access."""

    def __getitem__(self, key: str) -> type:
        value = super().__getitem__(key)
        if isinstance(value, tuple):
            module_path, class_name = value
            cls = getattr(import_module(module_path), class_name)
            super().__setitem__(key, cls)
            return cls
        return value

    def __contains__(self, key: object) -> bool:
        return super().__contains__(key)


MODEL_REGISTRY: dict[str, type] = _LazyRegistry(_REGISTRY)


def get_model_class(name: str) -> type:
    """Return the model class for *name*, raising KeyError with a helpful message."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model {name!r}. Available: {available}")
    return MODEL_REGISTRY[name]
