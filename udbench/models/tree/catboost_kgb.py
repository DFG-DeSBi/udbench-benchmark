from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from udbench.BaseUDRegressor import BaseUDRegressor
from udbench.tuning.objectives import ensure_metric, regression_metrics
from udbench.tuning.search_spaces import make_wandb_sweep_config
from udbench.tuning.sweep import prepare_train_val, run_wandb_bayes_sweep


def _import_catboost():
    try:
        from catboost import sample_gaussian_process
        return sample_gaussian_process
    except ImportError as exc:
        raise ImportError(
            "catboost is required for CatBoostKGBRegressor. "
            "Install with: pip install catboost"
        ) from exc


def _standardize_targets(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    mean = float(np.mean(y))
    std = float(np.std(y)) + 1e-12
    return (y - mean) / std, mean, std


def _rescale_mean_var(
    mean: np.ndarray,
    var: np.ndarray,
    *,
    y_mean: float,
    y_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = mean * y_std + y_mean
    var = var * (y_std * y_std)
    return mean, var


@dataclass
class CatBoostKGBRegressor(BaseUDRegressor):
    """
    CatBoost Kernel Gradient Boosting (Gaussian process sampling) ensemble.

    Epistemic uncertainty: variance across the KGB posterior sample paths.
    Aleatoric uncertainty: mean squared residual on a held-out validation split
      (same strategy as DEUP). Set aleatoric_val_fraction=0 to disable.
    """

    posterior_iterations: int = 900
    prior_iterations: int = 100
    learning_rate: float = 0.1
    depth: int = 6
    sigma: float = 0.1
    delta: float = 0.0
    random_strength: float = 0.1
    random_score_type: str = "Gumbel"
    eps: float = 1e-4
    n_regressors: int = 10
    random_seed: int | None = None
    standardize_y: bool = False
    verbose: bool = False
    aleatoric_val_fraction: float = 0.1

    _y_mean: float = field(default=0.0, init=False)
    _y_std: float = field(default=1.0, init=False)
    _aleatoric_var: float = field(default=0.0, init=False)

    def train(self, X, y, tune: bool = False, **kwargs: Any):
        kwargs = dict(kwargs)
        kwargs, save_tuned_params, tuned_params = self._resolve_tuned_params(kwargs)
        sample_gaussian_process = _import_catboost()

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y, dtype=float).reshape(-1)

        def _get_param(name: str, default: Any) -> Any:
            if name in kwargs:
                return kwargs.pop(name)
            if tuned_params and name in tuned_params:
                return tuned_params[name]
            return default

        standardize_y = bool(_get_param("standardize_y", self.standardize_y))
        self.standardize_y = standardize_y
        if standardize_y:
            y, self._y_mean, self._y_std = _standardize_targets(y)
        else:
            self._y_mean, self._y_std = 0.0, 1.0

        n_regressors = int(_get_param("n_regressors", self.n_regressors))
        random_seed = _get_param("random_seed", self.random_seed)

        params: Dict[str, Any] = {
            "samples": n_regressors,
            "posterior_iterations": int(_get_param("posterior_iterations", self.posterior_iterations)),
            "prior_iterations": int(_get_param("prior_iterations", self.prior_iterations)),
            "learning_rate": float(_get_param("learning_rate", self.learning_rate)),
            "depth": int(_get_param("depth", self.depth)),
            "sigma": float(_get_param("sigma", self.sigma)),
            "delta": float(_get_param("delta", self.delta)),
            "random_strength": float(_get_param("random_strength", self.random_strength)),
            "random_score_type": _get_param("random_score_type", self.random_score_type),
            "eps": float(_get_param("eps", self.eps)),
            "random_seed": random_seed,
            "verbose": bool(kwargs.pop("verbose", self.verbose)),
        }
        params.update(kwargs)

        models: List[Any] = sample_gaussian_process(X, y, **params)

        self.model = models
        self._store_tuned_params(
            {
                "posterior_iterations": params["posterior_iterations"],
                "prior_iterations": params["prior_iterations"],
                "learning_rate": params["learning_rate"],
                "depth": params["depth"],
                "sigma": params["sigma"],
                "delta": params["delta"],
                "random_strength": params["random_strength"],
                "random_score_type": params["random_score_type"],
                "eps": params["eps"],
                "n_regressors": n_regressors,
                "standardize_y": standardize_y,
                "random_seed": random_seed,
            },
            save=save_tuned_params,
        )
        return models

    def predict(self, model, X, **kwargs: Any):
        if model is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]

        models = model if isinstance(model, (list, tuple)) else [model]
        means = [np.asarray(m.predict(X)).reshape(-1) for m in models]
        mean = np.mean(np.stack(means, axis=0), axis=0).reshape(-1)
        mean = mean * self._y_std + self._y_mean
        return mean

    def ud_fit_predict(self, X_train, y_train, X_eval, **kwargs: Any) -> Dict:
        X_train = np.asarray(X_train, dtype=float)
        if X_train.ndim == 1:
            X_train = X_train[:, None]
        y_train = np.asarray(y_train, dtype=float).reshape(-1)

        # Estimate aleatoric variance from held-out residuals (same strategy as DEUP).
        ale_frac = float(kwargs.pop("aleatoric_val_fraction", self.aleatoric_val_fraction))
        if ale_frac > 0.0 and len(y_train) >= 10:
            n_val = max(1, int(len(y_train) * ale_frac))
            seed = self.random_seed if self.random_seed is not None else 0
            idx = np.random.default_rng(seed).permutation(len(y_train))
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            inner = self.spawn_refit_worker()
            inner.train(X_train[tr_idx], y_train[tr_idx])
            mu_val = inner.predict(inner.model, X_train[val_idx])   # original-space predictions
            self._aleatoric_var = float(np.mean((y_train[val_idx] - mu_val) ** 2))
        else:
            self._aleatoric_var = 0.0

        # Full training on all data.
        self.train(X_train, y_train, **kwargs)
        self._check_fitted()

        X_eval = np.asarray(X_eval, dtype=float)
        if X_eval.ndim == 1:
            X_eval = X_eval[:, None]

        models = self.model if isinstance(self.model, (list, tuple)) else [self.model]
        if not models:
            raise RuntimeError("No trained CatBoost models available.")

        means = np.asarray([m.predict(X_eval) for m in models])
        mean = np.mean(means, axis=0).reshape(-1)
        epi = np.var(means, axis=0).reshape(-1)

        if self._y_std != 1.0 or self._y_mean != 0.0:
            mean, _ = _rescale_mean_var(mean, mean * 0.0, y_mean=self._y_mean, y_std=self._y_std)
            epi = epi * (self._y_std * self._y_std)

        # _aleatoric_var is already in original-space (computed from predict() output).
        ale = np.full_like(epi, self._aleatoric_var)
        total = ale + epi

        return {
            "y_pred": mean,
            "total_uncertainty": total,
            "epistemic_uncertainty": epi,
            "aleatoric_uncertainty": ale,
        }

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")

    @classmethod
    def tune_hyperparameters_wandb(
        cls,
        X: Any,
        y: Any,
        *,
        project: str,
        count: int = 20,
        metric_name: str = "val_rmse",
        rng: int | np.random.Generator | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        **wandb_init_kwargs: Any,
    ) -> str:
        X_tr, y_tr, X_v, y_v = prepare_train_val(
            X,
            y,
            val_fraction=0.2,
            rng=rng,
        )

        sweep_config = make_wandb_sweep_config("catboost_kgb", metric_name)

        def objective(cfg: Dict[str, Any]) -> Dict[str, float]:
            model = cls(
                posterior_iterations=int(cfg["posterior_iterations"]),
                prior_iterations=int(cfg["prior_iterations"]),
                learning_rate=float(cfg["learning_rate"]),
                depth=int(cfg["depth"]),
                sigma=float(cfg["sigma"]),
                delta=float(cfg["delta"]),
                random_strength=float(cfg["random_strength"]),
                eps=float(cfg["eps"]),
                n_regressors=int(cfg["n_regressors"]),
                standardize_y=bool(cfg["standardize_y"]),
            )
            preds = model.ud_fit_predict(X_tr, y_tr, X_v)
            metrics = regression_metrics(y_v, preds["y_pred"], preds.get("total_uncertainty"))
            ensure_metric(metrics, metric_name)
            return metrics

        return run_wandb_bayes_sweep(
            sweep_config=sweep_config,
            objective_fn=objective,
            project=project,
            entity=entity,
            count=count,
            tags=tags,
            **wandb_init_kwargs,
        )
