from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def rmse(y_true: Any, y_pred: Any) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: Any, y_pred: Any) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_pred - y_true)))


def gaussian_nll(
    y_true: Any,
    y_pred: Any,
    total_uncertainty: Any,
    *,
    eps: float = 1e-12,
) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    var = np.asarray(total_uncertainty).reshape(-1)
    var = np.clip(var, eps, None)
    err = y_pred - y_true
    nll = 0.5 * (np.log(2.0 * np.pi * var) + (err**2) / var)
    return float(np.mean(nll))


def combined_nll_rmse(
    y_true: Any,
    y_pred: Any,
    total_uncertainty: Any,
    *,
    rmse_weight: float = 0.1,
    eps: float = 1e-12,
) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_scale = max(float(np.std(y_true)), float(eps))
    return float(
        gaussian_nll(y_true, y_pred, total_uncertainty, eps=eps)
        + float(rmse_weight) * rmse(y_true, y_pred) / y_scale
    )


OBJECTIVE_FUNCTIONS: Dict[str, Callable[..., float]] = {
    "val_rmse": rmse,
    "val_mae": mae,
    "val_nll": gaussian_nll,
    "val_objective": combined_nll_rmse,
}


def regression_metrics(
    y_true: Any,
    y_pred: Any,
    total_uncertainty: Any | None = None,
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_scale = max(float(np.std(y_true)), float(eps))
    val_rmse = rmse(y_true, y_pred)

    metrics = {
        "val_rmse": val_rmse,
        "val_rmse_scaled": float(val_rmse / y_scale),
        "val_mae": mae(y_true, y_pred),
    }

    if total_uncertainty is not None:
        metrics["val_nll"] = gaussian_nll(y_true, y_pred, total_uncertainty, eps=eps)
        metrics["val_objective"] = combined_nll_rmse(
            y_true,
            y_pred,
            total_uncertainty,
            eps=eps,
        )
    else:
        metrics["val_objective"] = float(metrics["val_rmse_scaled"])

    return metrics


def ensure_metric(metrics: Dict[str, float], metric_name: str) -> None:
    if metric_name not in metrics:
        raise ValueError(
            f"Metric {metric_name!r} was not computed. Available metrics: {sorted(metrics)}"
        )
