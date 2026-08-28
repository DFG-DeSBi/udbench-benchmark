from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import sys

import numpy as np

from udbench.BaseUDRegressor import BaseUDRegressor
from udbench.tuning.objectives import ensure_metric, regression_metrics
from udbench.tuning.search_spaces import make_wandb_sweep_config
from udbench.tuning.sweep import prepare_train_val, run_wandb_bayes_sweep


def _import_ngboost():
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal, NormalInverseGamma, NIGLogScore, NIGLogScoreSVGD
        return NGBRegressor, Normal, NormalInverseGamma, NIGLogScore, NIGLogScoreSVGD
    except ImportError as exc:
        repo_root = Path(__file__).resolve().parents[3]
        candidates = [
            repo_root / "UDBoost",                  # sibling to repo root (expected default)
            repo_root / "UDBench" / "UDBoost",      # nested inside the repo (current layout)
        ]
        for local_ngboost in candidates:
            if local_ngboost.exists():
                sys.path.insert(0, str(local_ngboost))
                from ngboost import NGBRegressor
                from ngboost.distns import Normal, NormalInverseGamma, NIGLogScore, NIGLogScoreSVGD
                return NGBRegressor, Normal, NormalInverseGamma, NIGLogScore, NIGLogScoreSVGD
        raise ImportError(
            "ngboost is required for NGBoost wrappers. Install ngboost or ensure "
            "the local UDBoost/ngboost package is available on PYTHONPATH."
        ) from exc


def _standardize_targets(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
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
) -> Tuple[np.ndarray, np.ndarray]:
    mean = mean * y_std + y_mean
    var = var * (y_std * y_std)
    return mean, var


@dataclass
class NGBoostNIGRegressor(BaseUDRegressor):
    """
    NGBoost with Normal-Inverse-Gamma (NIG) evidential regression.
    """

    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 4
    minibatch_frac: float = 1.0
    col_sample: float = 1.0
    natural_gradient: bool = True
    verbose: bool = False
    random_state: int | None = None
    use_svgd: bool = False
    epistemic_scaling: bool | None = None
    standardize_y: bool = False

    _y_mean: float = field(default=0.0, init=False)
    _y_std: float = field(default=1.0, init=False)

    def train(self, X, y, tune: bool = False, **kwargs: Any):
        kwargs = dict(kwargs)
        kwargs, save_tuned_params, tuned_params = self._resolve_tuned_params(kwargs)
        NGBRegressor, _, NormalInverseGamma, NIGLogScore, NIGLogScoreSVGD = _import_ngboost()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        def _get_param(name: str, default: Any) -> Any:
            if name in kwargs:
                return kwargs.pop(name)
            if tuned_params and name in tuned_params:
                return tuned_params[name]
            return default

        use_svgd = bool(_get_param("use_svgd", self.use_svgd))
        standardize_y = bool(_get_param("standardize_y", self.standardize_y))
        self.standardize_y = standardize_y

        Score = NIGLogScoreSVGD if use_svgd else NIGLogScore
        evid_strength = _get_param("evid_strength", None)
        kl_strength = _get_param("kl_strength", None)
        if evid_strength is not None or kl_strength is not None:
            Score.set_params(evid_strength=evid_strength, kl_strength=kl_strength)

        length_scale = _get_param("length_scale", None)
        warmup = _get_param("warmup", None)
        if use_svgd:
            if length_scale is not None:
                Score.length_scale = float(length_scale)
            if warmup is not None:
                Score.warmup = int(warmup)

        if standardize_y:
            y, self._y_mean, self._y_std = _standardize_targets(y)
        else:
            self._y_mean, self._y_std = 0.0, 1.0

        params = dict(
            Dist=NormalInverseGamma,
            Score=Score,
            n_estimators=int(_get_param("n_estimators", self.n_estimators)),
            learning_rate=float(_get_param("learning_rate", self.learning_rate)),
            max_depth=int(_get_param("max_depth", self.max_depth)),
            minibatch_frac=float(_get_param("minibatch_frac", self.minibatch_frac)),
            col_sample=float(_get_param("col_sample", self.col_sample)),
            natural_gradient=bool(_get_param("natural_gradient", self.natural_gradient)),
            verbose=bool(kwargs.pop("verbose", self.verbose)),
            random_state=_get_param("random_state", self.random_state),
            metadistribution_method="evidential_regression",
            epistemic_scaling=_get_param("epistemic_scaling", self.epistemic_scaling),
        )
        params.update(kwargs)

        model = NGBRegressor(**params)
        model.fit(X, y)
        self.model = model
        self._store_tuned_params(
            {
                "n_estimators": params["n_estimators"],
                "learning_rate": params["learning_rate"],
                "max_depth": params["max_depth"],
                "minibatch_frac": params["minibatch_frac"],
                "col_sample": params["col_sample"],
                "natural_gradient": params["natural_gradient"],
                "use_svgd": use_svgd,
                "evid_strength": evid_strength,
                "kl_strength": kl_strength,
                "length_scale": length_scale if use_svgd else None,
                "warmup": warmup if use_svgd else None,
                "standardize_y": standardize_y,
                "random_state": params["random_state"],
                "epistemic_scaling": params["epistemic_scaling"],
            },
            save=save_tuned_params,
        )
        return model

    def predict(self, model, X, **kwargs: Any):
        if model is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")
        X = np.asarray(X, dtype=float)
        mean = np.asarray(model.predict(X)).reshape(-1)
        mean = mean * self._y_std + self._y_mean
        return mean

    def ud_fit_predict(self, X_train, y_train, X_eval, **kwargs: Any) -> Dict:
        self.train(X_train, y_train, **kwargs)
        self._check_fitted()

        X_eval = np.asarray(X_eval, dtype=float)
        unc = self.model.pred_uncertainty(X_eval)

        mean = np.asarray(unc["mean"]).reshape(-1)
        ale  = np.asarray(unc["aleatoric"]).reshape(-1)
        epi  = np.asarray(unc["epistemic"]).reshape(-1)

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

        sweep_config = make_wandb_sweep_config("ngboost_nig", metric_name)

        def objective(cfg: Dict[str, Any]) -> Dict[str, float]:
            model = cls(
                n_estimators=int(cfg["n_estimators"]),
                learning_rate=float(cfg["learning_rate"]),
                max_depth=int(cfg["max_depth"]),
                minibatch_frac=float(cfg["minibatch_frac"]),
                col_sample=float(cfg["col_sample"]),
                natural_gradient=bool(cfg["natural_gradient"]),
                use_svgd=bool(cfg["use_svgd"]),
            )
            fit_kwargs: Dict[str, Any] = {
                "evid_strength": float(cfg["evid_strength"]),
                "kl_strength": float(cfg["kl_strength"]),
                "standardize_y": bool(cfg["standardize_y"]),
            }
            if bool(cfg["use_svgd"]):
                fit_kwargs["length_scale"] = float(cfg["length_scale"])
                fit_kwargs["warmup"] = int(cfg["warmup"])
            preds = model.ud_fit_predict(X_tr, y_tr, X_v, **fit_kwargs)
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

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")


@dataclass
class NGBoostBaggingRegressor(BaseUDRegressor):
    """
    NGBoost bagging ensemble with Gaussian distribution.
    """

    n_estimators: int = 500
    learning_rate: float = 0.01
    max_depth: int = 4
    minibatch_frac: float = 1.0
    col_sample: float = 1.0
    natural_gradient: bool = True
    verbose: bool = False
    random_state: int | None = None
    n_regressors: int = 10
    sample_fraction: float = 0.8
    replace: bool = True
    standardize_y: bool = False

    _y_mean: float = field(default=0.0, init=False)
    _y_std: float = field(default=1.0, init=False)

    def train(self, X, y, tune: bool = False, **kwargs: Any):
        kwargs = dict(kwargs)
        kwargs, save_tuned_params, tuned_params = self._resolve_tuned_params(kwargs)
        NGBRegressor, Normal, _, _, _ = _import_ngboost()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        def _get_param(name: str, default: Any) -> Any:
            if name in kwargs:
                return kwargs.pop(name)
            if tuned_params and name in tuned_params:
                return tuned_params[name]
            return default

        standardize_y = bool(_get_param("standardize_y", self.standardize_y))
        sample_fraction = float(_get_param("sample_fraction", self.sample_fraction))
        replace = bool(_get_param("replace", self.replace))
        self.standardize_y = standardize_y
        self.sample_fraction = sample_fraction
        self.replace = replace
        if standardize_y:
            y, self._y_mean, self._y_std = _standardize_targets(y)
        else:
            self._y_mean, self._y_std = 0.0, 1.0

        params = dict(
            Dist=Normal,
            n_estimators=int(_get_param("n_estimators", self.n_estimators)),
            learning_rate=float(_get_param("learning_rate", self.learning_rate)),
            max_depth=int(_get_param("max_depth", self.max_depth)),
            minibatch_frac=float(_get_param("minibatch_frac", self.minibatch_frac)),
            col_sample=float(_get_param("col_sample", self.col_sample)),
            natural_gradient=bool(_get_param("natural_gradient", self.natural_gradient)),
            verbose=bool(kwargs.pop("verbose", self.verbose)),
            random_state=_get_param("random_state", self.random_state),
            n_regressors=int(_get_param("n_regressors", self.n_regressors)),
            metadistribution_method="bagging",
            bagging_frac=sample_fraction,
            replace=replace,
        )
        kwargs.pop("boostrap", None)
        kwargs.pop("bootstrap", None)
        params.update(kwargs)

        model = NGBRegressor(**params)
        model.fit(X, y)
        self.model = model
        self._store_tuned_params(
            {
                "n_estimators": params["n_estimators"],
                "learning_rate": params["learning_rate"],
                "max_depth": params["max_depth"],
                "minibatch_frac": params["minibatch_frac"],
                "col_sample": params["col_sample"],
                "natural_gradient": params["natural_gradient"],
                "n_regressors": params["n_regressors"],
                "sample_fraction": sample_fraction,
                "replace": replace,
                "standardize_y": standardize_y,
                "random_state": params["random_state"],
            },
            save=save_tuned_params,
        )
        return model

    def predict(self, model, X, **kwargs: Any):
        if model is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")
        if not hasattr(model, "ensemble_models") or not model.ensemble_models:
            raise RuntimeError("Bagging ensemble is missing member models.")
        X = np.asarray(X, dtype=float)
        params = [m.pred_dist(X).params for m in model.ensemble_models]
        locs = np.asarray([p["loc"] for p in params], dtype=float)
        mean = np.mean(locs, axis=0).reshape(-1)
        mean = mean * self._y_std + self._y_mean
        return mean

    def ud_fit_predict(self, X_train, y_train, X_eval, **kwargs: Any) -> Dict:
        self.train(X_train, y_train, **kwargs)
        self._check_fitted()

        X_eval = np.asarray(X_eval, dtype=float)
        params = [model.pred_dist(X_eval).params for model in self.model.ensemble_models]
        locs   = np.asarray([p["loc"]   for p in params], dtype=float)
        scales = np.asarray([p["scale"] for p in params], dtype=float)

        mean = np.mean(locs, axis=0).reshape(-1)
        ale  = np.mean(scales * scales, axis=0).reshape(-1)
        epi  = np.var(locs, axis=0).reshape(-1)

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

        sweep_config = make_wandb_sweep_config("ngboost_bagging", metric_name)

        def objective(cfg: Dict[str, Any]) -> Dict[str, float]:
            model = cls(
                n_estimators=int(cfg["n_estimators"]),
                learning_rate=float(cfg["learning_rate"]),
                max_depth=int(cfg["max_depth"]),
                minibatch_frac=float(cfg["minibatch_frac"]),
                col_sample=float(cfg["col_sample"]),
                natural_gradient=bool(cfg["natural_gradient"]),
                n_regressors=int(cfg["n_regressors"]),
                sample_fraction=float(cfg["sample_fraction"]),
                replace=bool(cfg["replace"]),
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

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")
