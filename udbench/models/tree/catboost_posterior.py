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
        from catboost import CatBoostRegressor
        return CatBoostRegressor
    except ImportError as exc:
        raise ImportError(
            "catboost is required for CatBoostPosteriorSamplingRegressor. "
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
class CatBoostPosteriorSamplingRegressor(BaseUDRegressor):
    """
    CatBoost ensemble with posterior sampling (RMSEWithUncertainty).
    """

    iterations: int = 200
    learning_rate: float = 0.1
    depth: int = 6
    l2_leaf_reg: float = 3.0
    random_strength: float = 1.0
    bagging_temperature: float = 1.0
    bootstrap_type: str | None = "Bayesian"
    posterior_sampling: bool = True
    langevin: bool = False
    diffusion_temperature: float | None = None
    model_shrink_rate: float | None = None
    model_shrink_mode: str | None = None
    n_regressors: int = 10
    bagging_frac: float = 1.0
    random_seed: int | None = None
    standardize_y: bool = False
    verbose: bool = False
    allow_writing_files: bool = False

    _y_mean: float = field(default=0.0, init=False)
    _y_std: float = field(default=1.0, init=False)

    def train(self, X, y, tune: bool = False, **kwargs: Any):
        kwargs = dict(kwargs)
        kwargs, save_tuned_params, tuned_params = self._resolve_tuned_params(kwargs)
        CatBoostRegressor = _import_catboost()

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
        bagging_frac = float(_get_param("bagging_frac", self.bagging_frac))
        if not (0.0 < bagging_frac <= 1.0):
            raise ValueError("bagging_frac must be in (0, 1].")

        random_seed = _get_param("random_seed", self.random_seed)

        loss_function = _get_param("loss_function", "RMSEWithUncertainty")
        posterior_sampling = bool(_get_param("posterior_sampling", self.posterior_sampling))
        langevin = bool(_get_param("langevin", self.langevin))
        diffusion_temperature = _get_param("diffusion_temperature", self.diffusion_temperature)
        model_shrink_rate = _get_param("model_shrink_rate", self.model_shrink_rate)
        model_shrink_mode = _get_param("model_shrink_mode", self.model_shrink_mode)
        bootstrap_type = _get_param("bootstrap_type", self.bootstrap_type)

        base_params: Dict[str, Any] = {
            "loss_function": loss_function,
            "iterations": int(_get_param("iterations", self.iterations)),
            "learning_rate": float(_get_param("learning_rate", self.learning_rate)),
            "depth": int(_get_param("depth", self.depth)),
            "l2_leaf_reg": float(_get_param("l2_leaf_reg", self.l2_leaf_reg)),
            "random_strength": float(_get_param("random_strength", self.random_strength)),
            "bagging_temperature": float(_get_param("bagging_temperature", self.bagging_temperature)),
            "verbose": bool(kwargs.pop("verbose", self.verbose)),
            "allow_writing_files": bool(_get_param("allow_writing_files", self.allow_writing_files)),
        }
        if bootstrap_type is not None:
            base_params["bootstrap_type"] = bootstrap_type
        if posterior_sampling:
            base_params["posterior_sampling"] = True
        if langevin:
            base_params["langevin"] = True
        if diffusion_temperature is not None:
            base_params["diffusion_temperature"] = float(diffusion_temperature)
        if model_shrink_rate is not None:
            base_params["model_shrink_rate"] = float(model_shrink_rate)
        if model_shrink_mode is not None:
            base_params["model_shrink_mode"] = model_shrink_mode

        base_params.update(kwargs)

        rng = np.random.default_rng(random_seed)
        n_train = X.shape[0]
        models: List[Any] = []
        for _ in range(n_regressors):
            seed = int(rng.integers(0, 2**31 - 1)) if rng is not None else None
            X_fit = X
            y_fit = y
            if bagging_frac < 1.0:
                n_samples = max(1, int(n_train * bagging_frac))
                idx = rng.integers(0, n_train, size=n_samples)
                X_fit = X[idx]
                y_fit = y[idx]
            model = CatBoostRegressor(random_seed=seed, **base_params)
            model.fit(X_fit, y_fit)
            models.append(model)

        self.model = models
        self._store_tuned_params(
            {
                "iterations": base_params["iterations"],
                "learning_rate": base_params["learning_rate"],
                "depth": base_params["depth"],
                "l2_leaf_reg": base_params["l2_leaf_reg"],
                "random_strength": base_params["random_strength"],
                "bagging_temperature": base_params["bagging_temperature"],
                "bootstrap_type": bootstrap_type,
                "posterior_sampling": posterior_sampling,
                "langevin": langevin,
                "diffusion_temperature": diffusion_temperature,
                "model_shrink_rate": model_shrink_rate,
                "model_shrink_mode": model_shrink_mode,
                "n_regressors": n_regressors,
                "bagging_frac": bagging_frac,
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
        means = []
        for estimator in models:
            params = np.asarray(
                estimator.predict(X, prediction_type="RMSEWithUncertainty")
            )
            if params.ndim == 1:
                mean = params
            else:
                mean = params[:, 0]
            means.append(mean)
        mean = np.mean(np.stack(means, axis=0), axis=0).reshape(-1)
        mean = mean * self._y_std + self._y_mean
        return mean

    def ud_fit_predict(self, X_train, y_train, X_eval, **kwargs: Any) -> Dict:
        self.train(X_train, y_train, **kwargs)
        self._check_fitted()

        X_eval = np.asarray(X_eval, dtype=float)
        if X_eval.ndim == 1:
            X_eval = X_eval[:, None]

        models = self.model if isinstance(self.model, (list, tuple)) else [self.model]
        if not models:
            raise RuntimeError("No trained CatBoost models available.")

        params = np.asarray(
            [model.predict(X_eval, prediction_type="RMSEWithUncertainty") for model in models]
        )
        means = params[:, :, 0]
        ale = params[:, :, 1]

        mean = np.mean(means, axis=0)
        ale = np.mean(ale, axis=0)
        epi = np.var(means, axis=0)
        total = ale + epi

        mean = mean.reshape(-1)
        ale = ale.reshape(-1)
        epi = epi.reshape(-1)
        total = total.reshape(-1)

        if self._y_std != 1.0 or self._y_mean != 0.0:
            mean, ale = _rescale_mean_var(mean, ale, y_mean=self._y_mean, y_std=self._y_std)
            epi = epi * (self._y_std * self._y_std)
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

        sweep_config = make_wandb_sweep_config("catboost_posterior", metric_name)

        def objective(cfg: Dict[str, Any]) -> Dict[str, float]:
            model = cls(
                iterations=int(cfg["iterations"]),
                learning_rate=float(cfg["learning_rate"]),
                depth=int(cfg["depth"]),
                l2_leaf_reg=float(cfg["l2_leaf_reg"]),
                random_strength=float(cfg["random_strength"]),
                bagging_temperature=float(cfg["bagging_temperature"]),
                n_regressors=int(cfg["n_regressors"]),
                bagging_frac=float(cfg["bagging_frac"]),
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
