from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "TabularBNNBaggingRegressor": "udbench.models.bnn.bagging",
    "TabularBNNDeepRegressor": "udbench.models.bnn.deep_ensemble",
    "TabularBNNDropoutRegressor": "udbench.models.bnn.dropout",
    "TabularBNNLaplaceRegressor": "udbench.models.bnn.laplace",
    "TabularBNNFSPLaplaceRegressor": "udbench.models.bnn.fsp_laplace",
    "TabularBNNSWAGRegressor": "udbench.models.bnn.swag",
    "TabularBNNEDLRegressor": "udbench.models.bnn.edl",
    "TabularBNNDEUPRegressor": "udbench.models.bnn.deup",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
