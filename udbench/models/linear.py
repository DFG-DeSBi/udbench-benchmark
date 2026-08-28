from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
from udbench.BaseUDRegressor import BaseUDRegressor
from udbench.tuning.objectives import ensure_metric, regression_metrics
from udbench.tuning.search_spaces import make_wandb_sweep_config
from udbench.tuning.sweep import prepare_train_val, run_wandb_bayes_sweep

@dataclass
class BayesianLinearRegressor(BaseUDRegressor):
    """
    Conjugate Bayesian linear regression with Gaussian prior on weights and known Gaussian noise.

    Model:
      w ~ N(0, alpha^{-1} I)
      y | X, w ~ N(X w, sigma^2 I)  where sigma^2

    Predictive for x:
      mean:  x^T m_N
      var:   sigma^2 + x^T S_N x

    Uncertainty decomposition:
      aleatoric(x) = sigma^2
      epistemic(x) = x^T S_N x
      total(x)     = aleatoric + epistemic
    """

    alpha: float = 0.1
    sigma: float = 1.0
    fit_intercept: bool = True
    jitter: float = 1e-10

    # ---------- public API ----------
    def train(self, X, y, tune: bool = False, **kwargs: Any):
        kwargs = dict(kwargs)
        kwargs, save_tuned_params, tuned_params = self._resolve_tuned_params(kwargs)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        alpha = float(kwargs.get("alpha", tuned_params.get("alpha", self.alpha) if tuned_params else self.alpha))
        sigma = float(kwargs.get("sigma", tuned_params.get("sigma", self.sigma) if tuned_params else self.sigma))
        fit_intercept = bool(
            kwargs.get("fit_intercept", tuned_params.get("fit_intercept", self.fit_intercept) if tuned_params else self.fit_intercept)
        )
        self.fit_intercept = fit_intercept
        Xd = self._design(X)
        n, d = Xd.shape
        if sigma <= 0:
            raise ValueError("sigma must be > 0 (noise standard deviation).")

        beta = 1.0 / (sigma * sigma)  # noise precision

        # Posterior covariance: S_N = (alpha I + beta X^T X)^{-1}
        precision = alpha * np.eye(d) + beta * (Xd.T @ Xd)
        precision = precision + self.jitter * np.eye(d)

        S_N = np.linalg.inv(precision)
        m_N = beta * (S_N @ (Xd.T @ y))

        self.model = {
            "alpha": alpha,
            "sigma": sigma,
            "beta": beta,
            "sigma2": sigma * sigma,
            "posterior_mean": m_N,
            "posterior_cov": S_N,
            "fit_intercept": self.fit_intercept,
        }
        self._store_tuned_params(
            {
                "alpha": alpha,
                "sigma": sigma,
                "fit_intercept": self.fit_intercept,
            },
            save=save_tuned_params,
        )
        return self.model

    def predict(self, model, X, **kwargs: Any):
        if model is None:
            raise RuntimeError("No fitted model provided to predict().")
        fit_intercept = bool(model.get("fit_intercept", self.fit_intercept))
        X = np.asarray(X, dtype=float)
        Xd = self._design(X) if fit_intercept == self.fit_intercept else self._design_with(X, fit_intercept)
        m_N = model["posterior_mean"]
        return Xd @ m_N

    def _design_with(self, X: np.ndarray, fit_intercept: bool) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X.shape}.")
        if not fit_intercept:
            return X
        return np.concatenate([np.ones((X.shape[0], 1), dtype=X.dtype), X], axis=1)

    def ud_fit_predict(self, X_train, y_train, X_eval, **kwargs: Any) -> Dict:
        """
        Fits on (X_train, y_train), then returns predictions + uncertainties on X_eval.
        Pass sigma either in the constructor or here via kwargs: ud_fit_predict(..., sigma=...)
        """
        self.train(X_train, y_train, **kwargs)
        self._check_fitted()

        X_eval = np.asarray(X_eval, dtype=float)
        Xd = self._design(X_eval)

        m_N = self.model["posterior_mean"]
        S_N = self.model["posterior_cov"]
        sigma2 = float(self.model["sigma2"])

        y_pred = Xd @ m_N

        epistemic = np.sum((Xd @ S_N) * Xd, axis=1)
        aleatoric = np.full_like(epistemic, sigma2, float)
        total = epistemic + aleatoric

        return {
            "y_pred": y_pred,
            "total_uncertainty": total,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
        }

    @classmethod
    def tune_hyperparameters_wandb(
        cls,
        X: Any,
        y: Any,
        *,
        project: str,
        count: int = 30,
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

        sweep_config = make_wandb_sweep_config("linear", metric_name)

        def objective(cfg: Dict[str, Any]) -> Dict[str, float]:
            model = cls(
                alpha=float(cfg["alpha"]),
                sigma=float(cfg["sigma"]),
                fit_intercept=bool(cfg["fit_intercept"]),
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

    # ---------- helpers ----------
    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")

    def _design(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features). Got shape {X.shape}.")
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate([ones, X], axis=1)
