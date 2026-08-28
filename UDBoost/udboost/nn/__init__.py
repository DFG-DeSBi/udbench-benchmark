"""
Neural-network utilities for udboost.

This subpackage is optional and requires PyTorch.
"""

from __future__ import annotations

__all__ = [
    "TabularRegressorMLP",
    "TabularRegressorResNet",
    "TabularEDLRegressorMLP",
    "TabularEDLRegressorResNet",
    "bagging_ensemble_predict",
    "fit_mc_dropout_ensemble_predict",
    "fit_laplace_diagonal_ensemble_predict",
    "fit_regressor_torch",
    "DEUPRegressor",
    "deup_fit_regressor_torch",
    "deup_predict_regressor_torch",
    "fit_bagging_ensemble",
    "mc_dropout_ensemble_predict",
    "laplace_diagonal_ensemble_predict",
    "disentangle_regression_uncertainty",
    "edl_nig_uncertainty",
    "edl_nig_loss",
]


def __getattr__(name: str):
    if name in {"TabularRegressorMLP", "TabularRegressorResNet", "TabularEDLRegressorMLP", "TabularEDLRegressorResNet"}:
        from .nn_models import (
            TabularEDLRegressorMLP,
            TabularEDLRegressorResNet,
            TabularRegressorMLP,
            TabularRegressorResNet,
        )

        return {
            "TabularRegressorMLP": TabularRegressorMLP,
            "TabularRegressorResNet": TabularRegressorResNet,
            "TabularEDLRegressorMLP": TabularEDLRegressorMLP,
            "TabularEDLRegressorResNet": TabularEDLRegressorResNet,
        }[name]

    if name in {
        "bagging_ensemble_predict",
        "fit_mc_dropout_ensemble_predict",
        "fit_laplace_diagonal_ensemble_predict",
        "fit_regressor_torch",
        "DEUPRegressor",
        "deup_fit_regressor_torch",
        "deup_predict_regressor_torch",
        "fit_bagging_ensemble",
        "mc_dropout_ensemble_predict",
        "laplace_diagonal_ensemble_predict",
        "disentangle_regression_uncertainty",
        "edl_nig_uncertainty",
        "edl_nig_loss",
    }:
        from .nn_uncertainty import (
            DEUPRegressor,
            bagging_ensemble_predict,
            deup_fit_regressor_torch,
            deup_predict_regressor_torch,
            disentangle_regression_uncertainty,
            edl_nig_loss,
            edl_nig_uncertainty,
            fit_laplace_diagonal_ensemble_predict,
            fit_mc_dropout_ensemble_predict,
            fit_regressor_torch,
            fit_bagging_ensemble,
            laplace_diagonal_ensemble_predict,
            mc_dropout_ensemble_predict,
        )

        return {
            "DEUPRegressor": DEUPRegressor,
            "bagging_ensemble_predict": bagging_ensemble_predict,
            "deup_fit_regressor_torch": deup_fit_regressor_torch,
            "deup_predict_regressor_torch": deup_predict_regressor_torch,
            "fit_mc_dropout_ensemble_predict": fit_mc_dropout_ensemble_predict,
            "fit_laplace_diagonal_ensemble_predict": fit_laplace_diagonal_ensemble_predict,
            "fit_regressor_torch": fit_regressor_torch,
            "fit_bagging_ensemble": fit_bagging_ensemble,
            "mc_dropout_ensemble_predict": mc_dropout_ensemble_predict,
            "laplace_diagonal_ensemble_predict": laplace_diagonal_ensemble_predict,
            "disentangle_regression_uncertainty": disentangle_regression_uncertainty,
            "edl_nig_uncertainty": edl_nig_uncertainty,
            "edl_nig_loss": edl_nig_loss,
        }[name]

    raise AttributeError(name)
