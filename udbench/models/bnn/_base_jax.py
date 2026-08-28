import inspect
import logging
from typing import Any, Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
import jaxtyping

logger = logging.getLogger(__name__)

from udbench.tuning.objectives import ensure_metric, regression_metrics
from udbench.tuning.search_spaces import (
    get_wandb_search_space,
    jax_bnn_base_search_space,
    jax_bnn_optimizer_search_space,
)
from udbench.tuning.sweep import prepare_train_val, run_wandb_bayes_sweep

_WARNED_VARIANCE_HEAD_KEYS: set[str] = set()
DEFAULT_JAX_BNN_SWEEP_METRIC = "val_rmse"


def _jax_bnn_cast_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _jax_bnn_hidden_features_from_cfg(cfg: Dict[str, Any]) -> tuple[int, ...]:
    n_layers = int(cfg["n_layers"])
    hidden_dim = int(cfg["hidden_dim"])
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    return tuple([hidden_dim] * n_layers)


class JaxBNNTuningMixin:
    def _fit_jax_bnn_standardization(self, X: Any, y: Any) -> None:
        eps = float(getattr(self, "standardization_eps", 1e-8))
        if not np.isfinite(eps) or eps <= 0.0:
            eps = 1e-8

        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64).reshape(-1)

        if bool(getattr(self, "standardize_inputs", True)):
            x_mean = np.mean(X_np, axis=0)
            x_std = np.std(X_np, axis=0)
            x_std = np.where(np.isfinite(x_std) & (x_std > eps), x_std, 1.0)
        else:
            x_mean = np.zeros((X_np.shape[1],), dtype=np.float64)
            x_std = np.ones((X_np.shape[1],), dtype=np.float64)

        if bool(getattr(self, "standardize_targets", True)):
            y_mean = float(np.mean(y_np))
            y_std = float(np.std(y_np))
            if not np.isfinite(y_std) or y_std <= eps:
                y_std = 1.0
        else:
            y_mean = 0.0
            y_std = 1.0

        self._jax_bnn_x_mean = np.asarray(x_mean, dtype=np.float32)
        self._jax_bnn_x_std = np.asarray(x_std, dtype=np.float32)
        self._jax_bnn_y_mean = float(y_mean)
        self._jax_bnn_y_std = float(y_std)

    def _transform_jax_bnn_inputs(self, X: Any) -> jnp.ndarray:
        X_array = jnp.asarray(X, dtype=jnp.float32)
        x_mean = jnp.asarray(
            getattr(self, "_jax_bnn_x_mean", np.zeros((X_array.shape[1],), dtype=np.float32)),
            dtype=jnp.float32,
        )
        x_std = jnp.asarray(
            getattr(self, "_jax_bnn_x_std", np.ones((X_array.shape[1],), dtype=np.float32)),
            dtype=jnp.float32,
        )
        return (X_array - x_mean) / x_std

    def _transform_jax_bnn_targets(self, y: Any) -> jnp.ndarray:
        y_array = jnp.asarray(y, dtype=jnp.float32).reshape((-1, 1))
        y_mean = float(getattr(self, "_jax_bnn_y_mean", 0.0))
        y_std = float(getattr(self, "_jax_bnn_y_std", 1.0))
        return (y_array - y_mean) / y_std

    def _inverse_transform_jax_bnn_mean(self, mean: Any) -> jnp.ndarray:
        mean_array = jnp.asarray(mean, dtype=jnp.float32)
        y_mean = float(getattr(self, "_jax_bnn_y_mean", 0.0))
        y_std = float(getattr(self, "_jax_bnn_y_std", 1.0))
        return mean_array * y_std + y_mean

    def _inverse_transform_jax_bnn_variance(self, variance: Any) -> jnp.ndarray:
        variance_array = jnp.asarray(variance, dtype=jnp.float32)
        y_std = float(getattr(self, "_jax_bnn_y_std", 1.0))
        return variance_array * (y_std**2)

    def _restore_jax_bnn_ud_outputs(
        self,
        mean: Any,
        total: Any,
        epistemic: Any,
        aleatoric: Any,
    ) -> Dict[str, Any]:
        return {
            "y_pred": self._inverse_transform_jax_bnn_mean(mean),
            "total_uncertainty": self._inverse_transform_jax_bnn_variance(total),
            "epistemic_uncertainty": self._inverse_transform_jax_bnn_variance(epistemic),
            "aleatoric_uncertainty": self._inverse_transform_jax_bnn_variance(aleatoric),
        }

    @classmethod
    def _jax_bnn_shared_sweep_params(cls) -> Dict[str, Any]:
        return jax_bnn_base_search_space()

    @classmethod
    def _jax_bnn_extra_sweep_params(cls) -> Dict[str, Any]:
        return get_wandb_search_space(getattr(cls, "tuning_model_key", None))

    @classmethod
    def _jax_bnn_extra_fit_kwargs_from_cfg(cls, cfg: Dict[str, Any]) -> Dict[str, Any]:
        del cfg
        return {}

    @classmethod
    def _jax_bnn_model_kwargs_from_cfg(cls, cfg: Dict[str, Any]) -> Dict[str, Any]:
        model_kwargs: Dict[str, Any] = {
            "hidden_features": _jax_bnn_hidden_features_from_cfg(cfg),
            "backbone": "resnet",
            "resnet_width": int(cfg.get("resnet_width", 128)),
            "resnet_blocks": int(cfg.get("resnet_blocks", 2)),
        }
        if "activation" in cfg:
            model_kwargs["activation"] = cfg["activation"]
        if "dropout" in cfg:
            model_kwargs["dropout"] = float(cfg["dropout"])
        if "norm" in cfg:
            model_kwargs["norm"] = cfg["norm"]
        return model_kwargs

    @classmethod
    def _jax_bnn_instance_kwargs_from_cfg(cls, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Build model __init__ kwargs from sweep config."""
        model_kwargs = cls._jax_bnn_model_kwargs_from_cfg(cfg)
        init_params = set(inspect.signature(cls).parameters.keys())
        for key, value in cfg.items():
            if key in init_params and key not in model_kwargs:
                model_kwargs[key] = value
        return model_kwargs

    @classmethod
    def _jax_bnn_fit_kwargs_from_cfg(cls, cfg: Dict[str, Any]) -> Dict[str, Any]:
        fit_kwargs: Dict[str, Any] = {
            "n_epochs": int(cfg["n_epochs"]),
            "batch_size": int(cfg["batch_size"]),
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg["weight_decay"]),
            "optimizer": cfg.get("optimizer", "nadamw"),
            "momentum": float(cfg.get("momentum", 0.9)),
        }
        optional_casts = {
            "adam_b2": float,
            "adam_eps": float,
            "sgd_nesterov": _jax_bnn_cast_bool,
            "muon_ns_steps": int,
            "muon_adam_b2": float,
            "muon_preconditioning": str,
            "muon_nesterov": _jax_bnn_cast_bool,
            "soap_b2": float,
            "soap_precondition_frequency": int,
            "soap_precondition_1d": _jax_bnn_cast_bool,
        }
        for key, caster in optional_casts.items():
            if key in cfg:
                fit_kwargs[key] = caster(cfg[key])
        fit_kwargs.update(cls._jax_bnn_extra_fit_kwargs_from_cfg(cfg))
        return fit_kwargs

    @classmethod
    def _jax_bnn_tuning_predictions(
        cls,
        model: Any,
        X_tr: Any,
        y_tr: Any,
        X_v: Any,
        fit_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            return model.ud_fit_predict(X_tr, y_tr, X_v, **fit_kwargs)
        except (RuntimeError, AttributeError, TypeError) as exc:
            exc_msg = str(exc).lower()
            is_unfitted_runtime = "not fitted" in exc_msg or "call train" in exc_msg
            if isinstance(exc, RuntimeError) and not is_unfitted_runtime:
                raise
            model.train(X_tr, y_tr, **fit_kwargs)
            return model.ud_fit_predict(X_tr, y_tr, X_v, **fit_kwargs)

    @classmethod
    def _jax_bnn_sweep_config(
        cls,
        metric_name: str,
        fixed_model_kwargs: Dict[str, Any] | None = None,
        fixed_sweep_params: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        params = cls._jax_bnn_shared_sweep_params()
        params.update(cls._jax_bnn_extra_sweep_params())
        if fixed_model_kwargs:
            params.update(jax_bnn_optimizer_search_space(fixed_model_kwargs.get("optimizer")))
        for key in fixed_sweep_params or ():
            params.pop(str(key), None)
        return {
            "method": "bayes",
            "metric": {"name": metric_name, "goal": "minimize"},
            "parameters": params,
        }

    @classmethod
    def _jax_bnn_merge_fixed_sweep_cfg(
        cls,
        cfg: Dict[str, Any],
        fixed_model_kwargs: Dict[str, Any] | None = None,
        fixed_sweep_params: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(fixed_model_kwargs or {})
        hidden_features = tuple(merged.get("hidden_features", ()))
        if hidden_features:
            merged.setdefault("n_layers", len(hidden_features))
            merged.setdefault("hidden_dim", int(hidden_features[0]))

        fixed = {str(key) for key in fixed_sweep_params or ()}
        for key, value in cfg.items():
            if key not in fixed:
                merged[key] = value
        return merged

    @classmethod
    def _jax_bnn_objective_metrics(
        cls,
        cfg: Dict[str, Any],
        X_tr: Any,
        y_tr: Any,
        X_v: Any,
        y_v: Any,
        metric_name: str,
        fixed_model_kwargs: Dict[str, Any] | None = None,
        fixed_sweep_params: Sequence[str] | None = None,
    ) -> Dict[str, float]:
        merged_cfg = cls._jax_bnn_merge_fixed_sweep_cfg(
            cfg,
            fixed_model_kwargs=fixed_model_kwargs,
            fixed_sweep_params=fixed_sweep_params,
        )
        instance_kwargs = cls._jax_bnn_instance_kwargs_from_cfg(merged_cfg)
        model = cls(**instance_kwargs)
        fit_kwargs = cls._jax_bnn_fit_kwargs_from_cfg(merged_cfg)
        preds = cls._jax_bnn_tuning_predictions(model, X_tr, y_tr, X_v, fit_kwargs)
        raw_pred = _jax_bnn_raw_validation_prediction(model, X_v)
        if preds.get("total_uncertainty") is None and raw_pred is not None:
            total_uncertainty = _jax_bnn_total_uncertainty_from_raw(model, raw_pred)
            if total_uncertainty is not None:
                preds["total_uncertainty"] = total_uncertainty
        metrics = regression_metrics(y_v, preds["y_pred"], preds.get("total_uncertainty"))
        val_np_loss = _jax_bnn_np_validation_loss(model, raw_pred, y_v)
        if val_np_loss is not None:
            metrics["val_np_loss"] = val_np_loss
        ensure_metric(metrics, metric_name)
        return metrics

    @classmethod
    def tune_hyperparameters_wandb(
        cls,
        X: Any,
        y: Any,
        *,
        project: str,
        count: int = 30,
        metric_name: str = DEFAULT_JAX_BNN_SWEEP_METRIC,
        rng: int | np.random.Generator | None = None,
        entity: str | None = None,
        tags: Sequence[str] | None = None,
        fixed_model_kwargs: Dict[str, Any] | None = None,
        fixed_sweep_params: Sequence[str] | None = None,
        **wandb_init_kwargs: Any,
    ) -> Dict[str, Any]:
        X_tr, y_tr, X_v, y_v = prepare_train_val(
            X,
            y,
            val_fraction=0.2,
            rng=rng,
        )
        sweep_config = cls._jax_bnn_sweep_config(
            metric_name,
            fixed_model_kwargs=fixed_model_kwargs,
            fixed_sweep_params=fixed_sweep_params,
        )

        def objective(cfg: Dict[str, Any]) -> Dict[str, float]:
            return cls._jax_bnn_objective_metrics(
                cfg,
                X_tr,
                y_tr,
                X_v,
                y_v,
                metric_name,
                fixed_model_kwargs=fixed_model_kwargs,
                fixed_sweep_params=fixed_sweep_params,
            )

        return run_wandb_bayes_sweep(
            sweep_config=sweep_config,
            objective_fn=objective,
            project=project,
            entity=entity,
            count=count,
            tags=tags,
            **wandb_init_kwargs,
        )


def apply_jax_bnn_runtime_overrides(model: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    resolved_kwargs, _, _ = model._resolve_tuned_params(kwargs)
    resolved_kwargs = dict(resolved_kwargs)

    current_cfg: Dict[str, Any] = {}
    init_params = set(inspect.signature(model.__class__).parameters.keys())
    for key in init_params:
        if hasattr(model, key):
            current_cfg[key] = getattr(model, key)

    hidden_features = tuple(getattr(model, "hidden_features", ()))
    if hidden_features:
        current_cfg.setdefault("n_layers", len(hidden_features))
        current_cfg.setdefault("hidden_dim", int(hidden_features[0]))
    current_cfg.update(resolved_kwargs)

    instance_kwargs = model.__class__._jax_bnn_instance_kwargs_from_cfg(current_cfg)
    fit_kwargs = model.__class__._jax_bnn_fit_kwargs_from_cfg(current_cfg)

    for key, value in {**instance_kwargs, **fit_kwargs}.items():
        if hasattr(model, key):
            setattr(model, key, value)
        elif key in fit_kwargs:
            setattr(model, key, value)

    remaining_kwargs = dict(resolved_kwargs)
    remaining_kwargs.pop("n_layers", None)
    remaining_kwargs.pop("hidden_dim", None)
    for key in instance_kwargs:
        remaining_kwargs.pop(key, None)
    for key in fit_kwargs:
        remaining_kwargs.pop(key, None)

    for key in list(remaining_kwargs.keys()):
        if hasattr(model, key):
            setattr(model, key, remaining_kwargs.pop(key))

    return remaining_kwargs


def _jax_bnn_raw_validation_prediction(model: Any, X: Any) -> jnp.ndarray | None:
    model_bundle = getattr(model, "model_bundle", None)
    if model_bundle is None or not isinstance(model_bundle, tuple):
        return None
    if len(model_bundle) == 4:
        module, state, batch_stats, has_batchnorm = model_bundle
    elif len(model_bundle) == 3:
        module, state, batch_stats = model_bundle
        has_batchnorm = getattr(model, "norm", None) == "batchnorm"
    else:
        return None

    if hasattr(model, "_transform_jax_bnn_inputs"):
        Xj = model._transform_jax_bnn_inputs(X)
    else:
        Xj = jnp.asarray(X, dtype=jnp.float32)
    vars_in = {"params": state.params}
    if has_batchnorm and batch_stats is not None:
        vars_in["batch_stats"] = batch_stats

    params_leaves = jax.tree_util.tree_leaves(state.params)
    n_members = int(getattr(model, "n_members", 0) or 0)
    member_batched = (
        n_members > 0
        and bool(params_leaves)
        and getattr(params_leaves[0], "ndim", 0) > 0
        and int(params_leaves[0].shape[0]) == n_members
    )
    if member_batched:
        Xj = jnp.broadcast_to(Xj[None, :, :], (n_members,) + Xj.shape)

    return module.apply(vars_in, Xj, True, True)


def _jax_bnn_total_uncertainty_from_raw(model: Any, raw_pred: jnp.ndarray) -> np.ndarray | None:
    if raw_pred is None or raw_pred.shape[-1] != 2:
        return None
    eps = float(getattr(model, "np_eps", 1e-6))
    if bool(getattr(model, "NPLoss", False)):
        _, _, mu, std = natural_gaussian_stats_from_raw(
            raw_pred,
            link=getattr(model, "np_link_fn", "softplus"),
            eps=eps,
        )
        ale = std**2
    else:
        mu = raw_pred[..., 0]
        scale = jax.nn.softplus(raw_pred[..., 1]) + eps
        ale = scale**2

    if raw_pred.ndim == 3:
        total = jnp.mean(ale, axis=0) + jnp.var(mu, axis=0)
    else:
        total = ale
    if hasattr(model, "_inverse_transform_jax_bnn_variance"):
        total = model._inverse_transform_jax_bnn_variance(total)
    return np.asarray(total).reshape(-1)


def _jax_bnn_np_validation_loss(model: Any, raw_pred: jnp.ndarray, y_true: Any) -> float | None:
    if raw_pred is None or raw_pred.shape[-1] != 2:
        return None
    if not bool(getattr(model, "NPLoss", False)):
        return None
    if hasattr(model, "_transform_jax_bnn_targets"):
        y = model._transform_jax_bnn_targets(y_true).reshape(-1)
    else:
        y = jnp.asarray(y_true, dtype=jnp.float32).reshape(-1)
    if raw_pred.ndim == 3:
        y = jnp.broadcast_to(y[None, :], raw_pred.shape[:-1])
    loss = natural_gaussian_nll(
        raw_pred,
        y,
        link=getattr(model, "np_link_fn", "softplus"),
        eps=float(getattr(model, "np_eps", 1e-6)),
    )
    y_std = float(getattr(model, "_jax_bnn_y_std", 1.0))
    return float(loss + np.log(max(y_std, 1e-12)))


def make_jax_optimizer(
    optimizer: str,
    *,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    sgd_nesterov: bool = False,
    muon_ns_steps: int = 5,
    muon_adam_b2: float = 0.999,
    muon_preconditioning: str = "frobenius",
    muon_nesterov: bool = True,
    soap_b2: float = 0.95,
    soap_precondition_frequency: int = 5,
    soap_precondition_1d: bool = False,
) -> optax.GradientTransformation:
    opt_name = str(optimizer).lower()
    if opt_name == "adam":
        if not np.isfinite(momentum) or not (0.0 <= float(momentum) < 1.0):
            raise ValueError("momentum must be finite and in [0, 1) for adam.")
        if not np.isfinite(adam_b2) or not (0.0 <= float(adam_b2) < 1.0):
            raise ValueError("adam_b2 must be finite and in [0, 1).")
        if not np.isfinite(adam_eps) or float(adam_eps) <= 0.0:
            raise ValueError("adam_eps must be positive and finite.")
        adam_tx = optax.adam(
            learning_rate=learning_rate,
            b1=float(momentum),
            b2=float(adam_b2),
            eps=float(adam_eps),
        )
        if float(weight_decay) == 0.0:
            return adam_tx
        return optax.chain(optax.add_decayed_weights(weight_decay), adam_tx)
    if opt_name == "adamw":
        return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    if opt_name == "nadamw":
        if not np.isfinite(momentum) or not (0.0 <= float(momentum) < 1.0):
            raise ValueError("momentum must be finite and in [0, 1) for nadamw.")
        return optax.nadamw(
            learning_rate=learning_rate,
            b1=float(momentum),
            weight_decay=weight_decay,
        )
    if opt_name == "sgd":
        if not np.isfinite(momentum) or float(momentum) < 0.0:
            raise ValueError("momentum must be finite and >= 0 for sgd.")
        return optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(
                learning_rate=learning_rate,
                momentum=float(momentum),
                nesterov=bool(sgd_nesterov),
            ),
        )
    if opt_name == "muon":
        if not hasattr(optax, "contrib") or not hasattr(optax.contrib, "muon"):
            raise ImportError("optimizer='muon' requires optax.contrib.muon, available in recent Optax versions.")
        if not np.isfinite(momentum) or not (0.0 <= float(momentum) < 1.0):
            raise ValueError("momentum must be finite and in [0, 1) for muon.")
        return optax.contrib.muon(
            learning_rate=learning_rate,
            beta=float(momentum),
            ns_steps=int(muon_ns_steps),
            weight_decay=weight_decay,
            adam_b1=float(momentum),
            adam_b2=float(muon_adam_b2),
            adam_weight_decay=weight_decay,
            nesterov=bool(muon_nesterov),
            preconditioning=str(muon_preconditioning),
        )
    if opt_name == "soap":
        try:
            from soap_jax import soap
        except ImportError as exc:
            raise ImportError(
                "optimizer='soap' requires the optional SOAP_JAX package. "
                "Install it with `uv pip install git+https://github.com/haydn-jones/SOAP_JAX`."
            ) from exc
        if not np.isfinite(momentum) or not (0.0 <= float(momentum) < 1.0):
            raise ValueError("momentum must be finite and in [0, 1) for soap.")
        return soap(
            learning_rate=learning_rate,
            b1=float(momentum),
            b2=float(soap_b2),
            weight_decay=weight_decay,
            # SOAP_JAX bias correction upcasts updates to float64 when other
            # imported modules enable JAX x64, which breaks its lax.cond branch
            # typing against the float32 first-step updates.
            correct_bias=False,
            precondition_frequency=int(soap_precondition_frequency),
            precondition_1d=bool(soap_precondition_1d),
        )
    raise ValueError(f"Unknown optimizer: {optimizer!r}")


def jax_optimizer_extra_kwargs(model: Any) -> Dict[str, Any]:
    keys = (
        "adam_b2",
        "adam_eps",
        "sgd_nesterov",
        "muon_ns_steps",
        "muon_adam_b2",
        "muon_preconditioning",
        "muon_nesterov",
        "soap_b2",
        "soap_precondition_frequency",
        "soap_precondition_1d",
    )
    return {key: getattr(model, key) for key in keys if hasattr(model, key)}


def apply_model_with_batch_stats(
    apply_fn: Any,
    *,
    params: Any,
    batch_stats: Any,
    has_batchnorm: bool,
    x: jaxtyping.Array,
    dropout_deterministic: bool,
    use_running_average: bool,
    rngs: Dict[str, Any] | None = None,
    update_batch_stats: bool = False,
) -> tuple[jaxtyping.Array, Any]:
    vars_in = {"params": params}
    if has_batchnorm:
        if batch_stats is None:
            raise ValueError("batch_stats must be initialized when norm='batchnorm'.")
        vars_in["batch_stats"] = batch_stats

    if update_batch_stats:
        if rngs is None:
            out, updates = apply_fn(
                vars_in,
                x,
                dropout_deterministic,
                use_running_average,
                mutable=["batch_stats"],
            )
        else:
            out, updates = apply_fn(
                vars_in,
                x,
                dropout_deterministic,
                use_running_average,
                rngs=rngs,
                mutable=["batch_stats"],
            )
        return out, updates.get("batch_stats", batch_stats)

    if rngs is None:
        out = apply_fn(vars_in, x, dropout_deterministic, use_running_average)
    else:
        out = apply_fn(vars_in, x, dropout_deterministic, use_running_average, rngs=rngs)
    return out, batch_stats


def _natural_positive_link(raw: jaxtyping.Array, link: str, eps: float) -> jaxtyping.Array:
    if link == "softplus":
        pos = jax.nn.softplus(raw)
    elif link == "exp":
        pos = jnp.exp(raw)
    else:
        raise ValueError(f"Unsupported natural link: {link!r}. Use 'softplus' or 'exp'.")
    return pos + eps


def natural_gaussian_stats_from_raw(
    pred: jaxtyping.Array,
    *,
    link: str = "softplus",
    eps: float = 1e-6,
) -> tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array, jaxtyping.Array]:
    eta1 = pred[..., 0]
    eta2 = -_natural_positive_link(pred[..., 1], link=link, eps=eps)
    var = jnp.maximum(-0.5 / eta2, eps)
    mean = -eta1 / (2.0 * eta2)
    std = jnp.sqrt(var)
    return eta1, eta2, mean, std


def natural_gaussian_nll(
    pred: jaxtyping.Array,
    target: jaxtyping.Array,
    *,
    link: str = "softplus",
    eps: float = 1e-6,
) -> jaxtyping.Array:
    y = target.squeeze(-1) if target.ndim == pred.ndim else target
    eta1, eta2, _, _ = natural_gaussian_stats_from_raw(pred, link=link, eps=eps)
    nll = -(eta1 * y + eta2 * (y**2) + (eta1**2) / (4.0 * eta2) + 0.5 * jnp.log(-2.0 * eta2))
    return jnp.mean(nll)


def variance_head_regularization(
    pred: jaxtyping.Array,
    *,
    target_variance: float,
    np_loss: bool,
    eps: float = 1e-6,
    clip_multiplier: float = 20.0,
    link: str = "softplus",
) -> jaxtyping.Array:
    if pred.shape[-1] < 2:
        return jnp.asarray(0.0, dtype=pred.dtype)

    tv = float(target_variance)
    if not np.isfinite(tv) or tv <= 0.0:
        tv = float(eps)
    max_var = max(float(eps), float(clip_multiplier) * tv)

    if np_loss:
        _, _, _, std = natural_gaussian_stats_from_raw(pred, link=link, eps=eps)
    else:
        std = jax.nn.softplus(pred[..., 1]) + eps

    var = jnp.nan_to_num(std**2, nan=max_var * 1e3, posinf=max_var * 1e3, neginf=eps)
    log_excess = jax.nn.relu(jnp.log(var) - jnp.log(max_var))
    return jnp.mean(log_excess**2)


def target_variance_from_targets(y: jaxtyping.Array, *, eps: float = 1e-6) -> float:
    y_np = np.asarray(y).reshape(-1)
    if y_np.size == 0:
        return float(eps)
    var = float(np.var(y_np))
    if not np.isfinite(var):
        return float(eps)
    return float(max(var, eps))


def _softplus_inverse(x: jaxtyping.Array) -> jaxtyping.Array:
    return x + jnp.log(-jnp.expm1(-x))


def clip_and_check_ensemble_variance_heads(
    pred: jaxtyping.Array,
    *,
    target_variance: float,
    np_loss: bool,
    eps: float = 1e-6,
    clip_multiplier: float = 50.0,
    identical_rtol: float = 1e-5,
    identical_atol: float = 1e-8,
    warning_prefix: str = "JAX ensemble",
    warning_key: str | None = None,
) -> jaxtyping.Array:
    if pred.ndim < 3 or pred.shape[0] <= 1 or pred.shape[-1] < 2:
        return pred

    tv = float(target_variance)
    if not np.isfinite(tv) or tv <= 0.0:
        tv = float(eps)
    clip_max_var = max(float(eps), float(clip_multiplier) * tv)

    if np_loss:
        std = jnp.maximum(pred[..., 1], eps)
    else:
        std = jax.nn.softplus(pred[..., 1]) + eps

    var = jnp.clip(std**2, a_min=eps, a_max=clip_max_var)
    std_clipped = jnp.sqrt(var)

    if np_loss:
        clipped = pred.at[..., 1].set(std_clipped)
    else:
        target = jnp.maximum(std_clipped - eps, 1e-12)
        raw_clipped = _softplus_inverse(target)
        clipped = pred.at[..., 1].set(raw_clipped)

    var_np = np.asarray(var).reshape((int(var.shape[0]), -1))
    if var_np.shape[1] > 0:
        pairwise_close = np.all(
            np.isclose(
                var_np[:, None, :],
                var_np[None, :, :],
                rtol=float(identical_rtol),
                atol=float(identical_atol),
            ),
            axis=-1,
        )
        duplicate_pairs = int(np.triu(pairwise_close, k=1).sum())
        if duplicate_pairs > 0:
            should_warn = warning_key is None or warning_key not in _WARNED_VARIANCE_HEAD_KEYS
            if should_warn:
                logger.warning(
                    "%s: %d near-identical variance-head pair(s) after clipping to max "
                    "variance %.6g (= %.1f * target variance %.6g). Variance heads may be unstable.",
                    warning_prefix, duplicate_pairs, clip_max_var, clip_multiplier, tv,
                )
                if warning_key is not None:
                    _WARNED_VARIANCE_HEAD_KEYS.add(warning_key)

    return clipped


def _activation_fn(name: str):
    act = str(name).lower()
    if act == "relu":
        return nn.relu
    if act == "gelu":
        return nn.gelu
    if act == "silu":
        return nn.silu
    if act == "tanh":
        return jnp.tanh
    raise ValueError(f"Unsupported activation function {name!r}")


def _apply_norm(
    x: jaxtyping.Array,
    *,
    norm: str,
    epsilon: float,
    use_running_average: bool,
) -> jaxtyping.Array:
    norm_name = str(norm).lower()
    if norm_name == "none":
        return x
    if norm_name == "layernorm":
        return nn.LayerNorm(epsilon=epsilon)(x)
    if norm_name == "batchnorm":
        return nn.BatchNorm(
            use_running_average=use_running_average,
            momentum=0.9,
            epsilon=epsilon,
        )(x)
    if norm_name == "rmsnorm":
        return nn.RMSNorm(epsilon=epsilon)(x)
    raise ValueError(f"Unsupported normalization type {norm!r}")


def _activation_aware_dense_kernel_init(*, activation: str, init_scale: float = 1.0):
    act = str(activation).lower()
    scale_mult = float(init_scale) ** 2
    if scale_mult <= 0.0 or not np.isfinite(scale_mult):
        raise ValueError("init_scale must be a positive finite scalar.")

    if act in {"relu", "silu"}:
        return nn.initializers.variance_scaling(
            scale=2.0 * scale_mult,
            mode="fan_in",
            distribution="truncated_normal",
        )
    if act in {"tanh", "gelu"}:
        return nn.initializers.variance_scaling(
            scale=1.0 * scale_mult,
            mode="fan_avg",
            distribution="truncated_normal",
        )
    raise ValueError(f"Unsupported activation function {activation!r} for initializer selection.")


class TabMLPBackbone(nn.Module):
    hidden_features: tuple[int, ...]
    norm: str = "none"
    activation: str = "relu"
    dropout: float = 0.0
    init_scale: float = 1.0
    epsilon: float = 1e-5

    @nn.compact
    def __call__(
        self,
        x: jaxtyping.Array,
        dropout_deterministic: bool,
        use_running_average: bool,
    ) -> jaxtyping.Array:
        h = x
        activation_fn = _activation_fn(self.activation)
        kernel_init = _activation_aware_dense_kernel_init(
            activation=self.activation,
            init_scale=self.init_scale,
        )
        for width in self.hidden_features:
            width = int(width)
            if width <= 0:
                raise ValueError("hidden_features entries must be positive.")
            h = nn.Dense(width, kernel_init=kernel_init)(h)
            h = _apply_norm(
                h,
                norm=self.norm,
                epsilon=self.epsilon,
                use_running_average=use_running_average,
            )
            h = activation_fn(h)
            if self.dropout > 0:
                h = nn.Dropout(rate=self.dropout)(h, deterministic=dropout_deterministic)
        return h


class TabResBlock(nn.Module):
    hidden_dim: int
    norm: str = "none"
    activation: str = "relu"
    dropout: float = 0.0
    init_scale: float = 1.0
    epsilon: float = 1e-5

    @nn.compact
    def __call__(
        self,
        x: jaxtyping.Array,
        dropout_deterministic: bool,
        use_running_average: bool,
    ) -> jaxtyping.Array:
        if x.shape[-1] != int(self.hidden_dim):
            raise ValueError(
                f"ResNet block expected width {self.hidden_dim}, got {x.shape[-1]}."
            )

        activation_fn = _activation_fn(self.activation)
        kernel_init = _activation_aware_dense_kernel_init(
            activation=self.activation,
            init_scale=self.init_scale,
        )

        h = nn.Dense(self.hidden_dim, kernel_init=kernel_init)(x)
        h = _apply_norm(
            h,
            norm=self.norm,
            epsilon=self.epsilon,
            use_running_average=use_running_average,
        )
        h = activation_fn(h)

        if self.dropout > 0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=dropout_deterministic)

        h = nn.Dense(self.hidden_dim, kernel_init=kernel_init)(h)
        h = _apply_norm(
            h,
            norm=self.norm,
            epsilon=self.epsilon,
            use_running_average=use_running_average,
        )
        return activation_fn(h + x)


class TabResNetBackbone(nn.Module):
    resnet_width: int
    resnet_blocks: int
    norm: str = "none"
    activation: str = "relu"
    dropout: float = 0.0
    init_scale: float = 1.0
    epsilon: float = 1e-5

    @nn.compact
    def __call__(
        self,
        x: jaxtyping.Array,
        dropout_deterministic: bool,
        use_running_average: bool,
    ) -> jaxtyping.Array:
        width = int(self.resnet_width)
        blocks = int(self.resnet_blocks)
        if width <= 0:
            raise ValueError("resnet_width must be positive")
        if blocks < 0:
            raise ValueError("resnet_blocks must be >= 0")

        activation_fn = _activation_fn(self.activation)
        kernel_init = _activation_aware_dense_kernel_init(
            activation=self.activation,
            init_scale=self.init_scale,
        )

        h = nn.Dense(width, kernel_init=kernel_init)(x)
        h = _apply_norm(
            h,
            norm=self.norm,
            epsilon=self.epsilon,
            use_running_average=use_running_average,
        )
        h = activation_fn(h)
        if self.dropout > 0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=dropout_deterministic)

        for _ in range(blocks):
            h = TabResBlock(
                hidden_dim=width,
                norm=self.norm,
                activation=self.activation,
                dropout=self.dropout,
                init_scale=self.init_scale,
                epsilon=self.epsilon,
            )(
                h,
                dropout_deterministic=dropout_deterministic,
                use_running_average=use_running_average,
            )
        return h


class TabularBNNBaseJax(nn.Module):
    input_dim: int
    output_dim: int
    hidden_features: tuple[int, ...] = (256, 256)
    backbone: str = "resnet"
    resnet_width: int = 128
    resnet_blocks: int = 2
    norm: str = "none"
    activation: str = "relu"
    dropout: float = 0.0
    init_scale: float = 1.0
    normalize_features: bool = False

    @nn.compact
    def __call__(
        self,
        x: jaxtyping.Array,
        dropout_deterministic: bool,
        use_running_average: bool,
    ) -> jaxtyping.Array:
        kernel_init = _activation_aware_dense_kernel_init(
            activation=self.activation,
            init_scale=self.init_scale,
        )
        backbone_name = str(self.backbone).lower()
        if backbone_name == "mlp":
            h = TabMLPBackbone(
                hidden_features=tuple(self.hidden_features),
                norm=self.norm,
                activation=self.activation,
                dropout=self.dropout,
                init_scale=self.init_scale,
            )(
                x,
                dropout_deterministic=dropout_deterministic,
                use_running_average=use_running_average,
            )
        elif backbone_name == "resnet":
            h = TabResNetBackbone(
                resnet_width=self.resnet_width,
                resnet_blocks=self.resnet_blocks,
                norm=self.norm,
                activation=self.activation,
                dropout=self.dropout,
                init_scale=self.init_scale,
            )(
                x,
                dropout_deterministic=dropout_deterministic,
                use_running_average=use_running_average,
            )
        else:
            raise ValueError(f"Unknown backbone: {self.backbone!r}")
        if self.normalize_features:
            h = nn.LayerNorm(use_scale=False, use_bias=False)(h)
        return nn.Dense(self.output_dim, kernel_init=kernel_init)(h)
