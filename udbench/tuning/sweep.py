from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _fmt_param_spec(spec: Dict[str, Any]) -> str:
    if "values" in spec:
        vals = spec["values"]
        return f"fixed {vals}" if len(vals) == 1 else f"choice {vals}"
    dist = spec.get("distribution", "?")
    lo = spec.get("min", "?")
    hi = spec.get("max", "?")
    short = dist.replace("_values", "").replace("_uniform", "_unif")
    return f"{short:<14} [{lo}, {hi}]"


def _format_search_space(parameters: Dict[str, Any]) -> str:
    if not parameters:
        return "  (no free parameters)"
    width = max(len(k) for k in parameters)
    lines = [f"  {k:<{width}}  {_fmt_param_spec(v)}" for k, v in parameters.items()]
    return "\n".join(lines)


def _format_best_params(params: Dict[str, Any]) -> str:
    if not params:
        return "  (none)"
    width = max(len(k) for k in params)
    lines = [f"  {k:<{width}} = {v}" for k, v in sorted(params.items())]
    return "\n".join(lines)


def prepare_train_val(
    X: Any,
    y: Any,
    X_val: Any | None = None,
    y_val: Any | None = None,
    *,
    val_fraction: float = 0.2,
    rng: int | np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xn = np.asarray(X)
    yn = np.asarray(y).reshape(-1)

    if X_val is not None and y_val is not None:
        return Xn, yn, np.asarray(X_val), np.asarray(y_val).reshape(-1)

    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")

    if Xn.shape[0] < 2:
        raise ValueError("Need at least 2 samples for train/val split.")

    if isinstance(rng, np.random.Generator):
        random_state = int(rng.integers(0, np.iinfo(np.int32).max))
    else:
        random_state = rng

    X_tr, X_v, y_tr, y_v = train_test_split(
        Xn,
        yn,
        test_size=val_fraction,
        random_state=random_state,
    )
    return X_tr, y_tr, X_v, y_v


def run_wandb_bayes_sweep(
    *,
    sweep_config: Dict[str, Any],
    objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    project: str,
    entity: Optional[str] = None,
    count: int = 30,
    tags: Optional[Sequence[str]] = None,
    **init_kwargs: Any,
) -> Dict[str, Any]:
    parameters = sweep_config.get("parameters", {})
    metric_name = sweep_config.get("metric", {}).get("name")
    logger.info(
        "Tuning  objective=%s  trials=%d  params=%d\n%s",
        metric_name or "?", count, len(parameters),
        _format_search_space(parameters),
    )

    _silent = os.environ.get("WANDB_SILENT", "").lower() in ("1", "true", "yes")
    if _silent:
        _old_err = os.dup(2)
        _old_out = os.dup(1)
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 2)
        os.dup2(_devnull, 1)
        os.close(_devnull)
        try:
            sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        finally:
            os.dup2(_old_err, 2)
            os.dup2(_old_out, 1)
            os.close(_old_err)
            os.close(_old_out)
    else:
        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    metric_goal = sweep_config.get("metric", {}).get("goal", "minimize")
    best_config: Optional[Dict[str, Any]] = None
    best_metric: Optional[float] = None

    desc = f"sweep [{metric_name or '?'}]"
    pbar = tqdm(total=int(count), desc=desc, unit="trial", leave=False, dynamic_ncols=True)

    def _objective():
        nonlocal best_config, best_metric
        with wandb.init(project=project, entity=entity, tags=tags,
                        settings=wandb.Settings(init_timeout=120), **init_kwargs):
            cfg = dict(wandb.config)
            metrics = objective_fn(cfg)
            if metrics:
                wandb.log(metrics)
            if metric_name and metrics and metric_name in metrics:
                value = float(metrics[metric_name])
                if best_metric is None:
                    best_metric = value
                    best_config = dict(cfg)
                else:
                    better = value < best_metric if metric_goal == "minimize" else value > best_metric
                    if better:
                        best_metric = value
                        best_config = dict(cfg)
        pbar.update(1)
        if best_metric is not None:
            pbar.set_postfix({metric_name or "best": f"{best_metric:.4f}"})

    wandb.agent(sweep_id, function=_objective, count=int(count))
    pbar.close()

    if best_metric is not None:
        logger.info(
            "Tuning done  best %s=%.4f\n%s",
            metric_name or "metric", best_metric,
            _format_best_params(best_config or {}),
        )

    return {
        "sweep_id": sweep_id,
        "best_config": best_config,
        "best_metric": best_metric,
    }
