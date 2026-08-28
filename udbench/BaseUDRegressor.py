from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np

@dataclass
class BaseUDRegressor(ABC):
    """
    Template for a regression model with uncertainty decomposition.

    Required public API:
      - train(X, y): fits and stores the model
      - predict(X): returns y_pred
      - ud_fit_predict(X, y): returns a dict with:
          'y_pred', 'total_uncertainty', 'epistemic_uncertainty', 'aleatoric_uncertainty'
      - spawn_refit_worker(): returns a fresh estimator with the same init config
    """
    model: Any = field(default=None, init=False)
    tuned_params: Optional[Dict[str, Any]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """No-op hook so subclasses can safely call super().__post_init__()."""

    @abstractmethod
    def train(self, X, y, tune: bool = False, **kwargs: Any):
        """
        Fit the model on training data and store fitted state in self.model.
        Must set self.is_fitted = True when done.

        If tune=True, hyperparameter tuning may be triggered depending on the model.
        """

    @abstractmethod
    def predict(self, model, X, **kwargs: Any):
        """
        Predict y for features X.
        Returns shape (n_samples,) or (n_samples, 1).
        """

    @abstractmethod
    def ud_fit_predict(self, X_train, y_train, X_eval,  **kwargs: Any) -> Dict:
        """
        Compute predictions and uncertainty components on provided data.


        Returns a dictionary with keys:
          - 'y_pred'
          - 'total_uncertainty'
          - 'epistemic_uncertainty'
          - 'aleatoric_uncertainty'
        """

    def tune(
        self,
        X,
        y,
        *,
        project: str | None = None,
        count: int = 30,
        metric_name: str = "val_rmse",
        rng: Any = None,
        entity: str | None = None,
        tags: Any = None,
    ) -> Optional[Dict[str, Any]]:
        if not hasattr(self.__class__, "tune_hyperparameters_wandb"):
            raise AttributeError(f"{self.__class__.__name__} does not support wandb tuning.")

        requires_project = bool(getattr(self.__class__, "requires_wandb_project", True))
        if project is None and requires_project:
            raise ValueError("tune requires tune_project to be set for wandb sweeps.")

        tune_kwargs = {
            "project": project,
            "count": count,
            "metric_name": metric_name,
            "rng": rng,
            "entity": entity,
            "tags": tags,
        }
        tune_signature = inspect.signature(self.__class__.tune_hyperparameters_wandb)
        if "fixed_model_kwargs" in tune_signature.parameters:
            init_params = set(inspect.signature(self.__class__).parameters.keys())
            fixed_model_kwargs = {
                key: getattr(self, key)
                for key in init_params
                if hasattr(self, key)
            }
            tune_kwargs["fixed_model_kwargs"] = fixed_model_kwargs
        if "fixed_sweep_params" in tune_signature.parameters:
            fixed_sweep_params = getattr(self, "fixed_sweep_params", None)
            if fixed_sweep_params:
                tune_kwargs["fixed_sweep_params"] = tuple(fixed_sweep_params)

        result = self.__class__.tune_hyperparameters_wandb(
            X,
            y,
            **tune_kwargs,
        )
        tuned_params: Optional[Dict[str, Any]] = None
        sweep_id: Optional[str] = None
        if isinstance(result, dict):
            tuned_params = (
                result.get("best_config")
                or result.get("tuned_params")
                or result.get("config")
            )
            sweep_id = result.get("sweep_id")
        else:
            sweep_id = str(result)
        self._tuned = True
        self._last_sweep_id = sweep_id
        if tuned_params:
            self._store_tuned_params(tuned_params, save=True, overwrite=True)
        return tuned_params

    def _resolve_tuned_params(
        self,
        kwargs: Dict[str, Any],
        *,
        default_reuse: bool = True,
        default_save: bool = True,
    ) -> Tuple[Dict[str, Any], bool, Optional[Dict[str, Any]]]:
        kwargs = dict(kwargs)
        reuse_tuned_params = bool(kwargs.pop("reuse_tuned_params", default_reuse))
        save_tuned_params = bool(kwargs.pop("save_tuned_params", default_save))
        tuned_params = self.tuned_params if reuse_tuned_params else None
        if tuned_params:
            for key, value in tuned_params.items():
                kwargs[key] = value
        return kwargs, save_tuned_params, tuned_params

    def _store_tuned_params(
        self,
        params: Dict[str, Any],
        *,
        save: bool = True,
        overwrite: bool = False,
    ) -> None:
        if not save:
            return
        if self.tuned_params is not None and not overwrite:
            return
        self.tuned_params = dict(params)

    def spawn_refit_worker(self):
        """Return a fresh estimator carrying the current init-time configuration."""

        init_kwargs: Dict[str, Any] = {}
        if is_dataclass(self):
            for dataclass_field in fields(self):
                if not dataclass_field.init:
                    continue
                value = getattr(self, dataclass_field.name)
                try:
                    init_kwargs[dataclass_field.name] = deepcopy(value)
                except Exception:
                    init_kwargs[dataclass_field.name] = value
        else:
            signature = inspect.signature(self.__class__)
            for name, parameter in signature.parameters.items():
                if name == "self":
                    continue
                if parameter.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if hasattr(self, name):
                    value = getattr(self, name)
                    try:
                        init_kwargs[name] = deepcopy(value)
                    except Exception:
                        init_kwargs[name] = value

        worker = self.__class__(**init_kwargs)
        if self.tuned_params is not None:
            worker.tuned_params = dict(self.tuned_params)
        return worker
