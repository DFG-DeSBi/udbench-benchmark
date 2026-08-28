from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Sequence, Tuple

import logging

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from udbench.BaseUDRegressor import BaseUDRegressor
from udbench.tuning.search_spaces import make_wandb_sweep_config
from udbench.tuning.sweep import prepare_train_val, run_wandb_bayes_sweep
from ._base_jax import (
    JaxBNNTuningMixin,
    TabularBNNBaseJax,
    apply_model_with_batch_stats,
    apply_jax_bnn_runtime_overrides,
    make_jax_optimizer,
    jax_optimizer_extra_kwargs,
)

logger = logging.getLogger(__name__)


@dataclass
class TabularBNNDEUPRegressor(JaxBNNTuningMixin, BaseUDRegressor):
    tuning_model_key = "bnn_deup"

    hidden_features: Tuple[int, ...] = (256, 256)
    backbone: str = "resnet"
    resnet_width: int = 128
    resnet_blocks: int = 2
    activation: str = "relu"
    init_scale: float = 1.0
    dropout: float = 0.0
    norm: str = "none"
    n_epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "nadamw"
    momentum: float = 0.9
    grad_clip_norm: float = 1.0
    epi_clip_multiplier: float = 50.0
    n_epochs_excess: int = 100
    batch_size_excess: int = 512
    lr_excess: float = 1e-3
    weight_decay_excess: float = 0.0
    val_fraction: float = 0.2
    np_eps: float = 1e-6
    aleatoric_var_: float | None = field(default=None, init=False)
    rng: int | None = None
    predict_batch_size: int = 4096
    device: Any = None
    dtype: Any = None
    verbose: bool = False
    standardize_inputs: bool = True
    standardize_targets: bool = True
    standardization_eps: float = 1e-8

    model_bundle: Any = field(default=None, init=False)
    excess_model_bundle: Any = field(default=None, init=False)
    target_variance_: float = field(default=1e-6, init=False)

    def _fit_excess_model(
        self,
        X_excess: jnp.ndarray,
        y_excess: jnp.ndarray,
        *,
        lr_excess: float,
        weight_decay_excess: float,
        n_epochs_excess: int,
        batch_size_excess: int,
        rng: Any,
        has_batchnorm: bool,
        grad_clip_norm: float,
        x_dummy: jnp.ndarray,
    ) -> tuple[tuple, float]:
        """Train the excess (epistemic) model on pre-computed log-excess targets.

        Returns (excess_model_bundle, train_r2). train_r2 < 0.05 indicates
        the model failed to learn anything useful from the excess targets.
        """

        def mse_loss(pred, y):
            y = y.squeeze(-1) if y.ndim == 2 else y
            return jnp.mean((pred.squeeze(-1) - y) ** 2)

        @partial(jax.jit, static_argnames=("has_batchnorm",))
        def train_step_excess(state, batch_stats, rng, xb, yb, has_batchnorm: bool):
            rng, drop_rng = jax.random.split(rng)

            def loss_fn(params):
                pred, next_batch_stats = apply_model_with_batch_stats(
                    state.apply_fn,
                    params=params,
                    batch_stats=batch_stats,
                    has_batchnorm=has_batchnorm,
                    x=xb,
                    dropout_deterministic=False,
                    use_running_average=False,
                    rngs={"dropout": drop_rng},
                    update_batch_stats=has_batchnorm,
                )
                return mse_loss(pred, yb), next_batch_stats

            (loss, next_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, next_batch_stats, rng, loss

        N_excess = X_excess.shape[0]
        excess_model = TabularBNNBaseJax(
            input_dim=X_excess.shape[1],
            output_dim=1,
            hidden_features=self.hidden_features,
            backbone=self.backbone,
            resnet_width=self.resnet_width,
            resnet_blocks=self.resnet_blocks,
            norm=self.norm,
            activation=self.activation,
            init_scale=self.init_scale,
            dropout=self.dropout,
        )
        rng, init_rng, drop_rng = jax.random.split(rng, 3)
        excess_variables = excess_model.init(
            {"params": init_rng, "dropout": drop_rng}, x_dummy, True, False
        )
        excess_params = excess_variables["params"]
        excess_batch_stats = excess_variables.get("batch_stats", None)

        tx_excess = optax.chain(
            optax.clip_by_global_norm(grad_clip_norm),
            make_jax_optimizer(
                self.optimizer,
                learning_rate=lr_excess,
                weight_decay=weight_decay_excess,
                momentum=self.momentum,
                **jax_optimizer_extra_kwargs(self),
            ),
        )
        excess_state = train_state.TrainState.create(
            apply_fn=excess_model.apply, params=excess_params, tx=tx_excess
        )

        effective_batch_size = min(batch_size_excess, N_excess)
        n_batches = max(1, (N_excess + effective_batch_size - 1) // effective_batch_size)

        for epoch in range(1, n_epochs_excess + 1):
            rng, ep_rng = jax.random.split(rng)
            epoch_idx = jax.random.permutation(ep_rng, N_excess)
            epoch_loss = 0.0
            for start in range(0, N_excess, effective_batch_size):
                batch_idx = epoch_idx[start:start + effective_batch_size]
                xb, yb = X_excess[batch_idx], y_excess[batch_idx]
                excess_state, excess_batch_stats, rng, loss = train_step_excess(
                    excess_state, excess_batch_stats, rng, xb, yb, has_batchnorm
                )
                epoch_loss += float(loss)
            epoch_loss /= n_batches
            if epoch in (1, n_epochs_excess) or epoch % max(1, n_epochs_excess // 10) == 0:
                logger.debug("excess epoch %4d/%d  loss=%.6f", epoch, n_epochs_excess, epoch_loss)

        # Evaluate convergence on training data (catches non-convergence / bad hyperparams).
        vars_final = {"params": excess_state.params}
        if has_batchnorm and excess_batch_stats is not None:
            vars_final["batch_stats"] = excess_batch_stats
        pred_tr = excess_model.apply(vars_final, X_excess, True, True).reshape(-1)
        y_flat = y_excess.reshape(-1)
        tr_mse = float(jnp.mean((pred_tr - y_flat) ** 2))
        baseline_mse = float(jnp.var(y_flat) + 1e-30)
        train_r2 = 1.0 - tr_mse / baseline_mse

        return (excess_model, excess_state, excess_batch_stats, has_batchnorm), train_r2

    def train(self, X, y, tune: bool = False, **kwargs: Any):
        kwargs = apply_jax_bnn_runtime_overrides(self, kwargs)
        build_excess = bool(kwargs.pop("build_excess", True))
        self.excess_model_bundle = None

        def mse_loss(pred, y):
            y = y.squeeze(-1) if y.ndim == 2 else y
            return jnp.mean((pred.squeeze(-1) - y) ** 2)

        @partial(jax.jit, static_argnames=("has_batchnorm",))
        def train_step(state, batch_stats, rng, xb, yb, has_batchnorm: bool):
            rng, drop_rng = jax.random.split(rng)

            def loss_fn(params):
                pred, next_batch_stats = apply_model_with_batch_stats(
                    state.apply_fn,
                    params=params,
                    batch_stats=batch_stats,
                    has_batchnorm=has_batchnorm,
                    x=xb,
                    dropout_deterministic=False,
                    use_running_average=False,
                    rngs={"dropout": drop_rng},
                    update_batch_stats=has_batchnorm,
                )
                return mse_loss(pred, yb), next_batch_stats

            (loss, next_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, next_batch_stats, rng, loss

        self._fit_jax_bnn_standardization(X, y)
        X = self._transform_jax_bnn_inputs(X)
        y = self._transform_jax_bnn_targets(y)
        self.target_variance_ = float(jnp.var(y) + self.np_eps)
        N = X.shape[0]
        if N < 2:
            raise ValueError("DEUP requires at least 2 samples.")

        rng_seed = 0 if self.rng is None else int(self.rng)
        rng = jax.random.PRNGKey(rng_seed)
        rng, split_rng = jax.random.split(rng)
        perm = jax.random.permutation(split_rng, N)
        n_val = int(self.val_fraction * N)
        n_val = min(max(n_val, 1), N - 1)

        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        X_val, y_val = X[val_idx], y[val_idx]
        X, y = X[tr_idx], y[tr_idx]
        N = X.shape[0]

        has_batchnorm = (self.norm == "batchnorm")
        model = TabularBNNBaseJax(
            input_dim=X.shape[1],
            output_dim=1,
            hidden_features=self.hidden_features,
            backbone=self.backbone,
            resnet_width=self.resnet_width,
            resnet_blocks=self.resnet_blocks,
            norm=self.norm,
            activation=self.activation,
            init_scale=self.init_scale,
            dropout=self.dropout,
        )

        rng, init_rng, drop_rng = jax.random.split(rng, 3)
        self.drop_rng = drop_rng

        x_dummy = jnp.zeros((1, X.shape[1]), dtype=jnp.float32)
        variables = model.init({"params": init_rng, "dropout": drop_rng}, x_dummy, True, False)
        params = variables["params"]
        batch_stats = variables.get("batch_stats", None)

        grad_clip_norm = float(self.grad_clip_norm)
        if not np.isfinite(grad_clip_norm) or grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be a positive finite scalar.")

        tx = optax.chain(
            optax.clip_by_global_norm(grad_clip_norm),
            make_jax_optimizer(
                self.optimizer,
                learning_rate=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                **jax_optimizer_extra_kwargs(self),
            ),
        )
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        effective_batch_size = min(self.batch_size, N)
        n_batches = max(1, (N + effective_batch_size - 1) // effective_batch_size)

        for epoch in range(1, self.n_epochs + 1):
            rng, ep_rng = jax.random.split(rng)
            epoch_idx = jax.random.permutation(ep_rng, N)
            epoch_loss = 0.0
            for start in range(0, N, effective_batch_size):
                batch_idx = epoch_idx[start:start + effective_batch_size]
                xb, yb = X[batch_idx], y[batch_idx]
                state, batch_stats, rng, loss = train_step(state, batch_stats, rng, xb, yb, has_batchnorm)
                epoch_loss += float(loss)
            epoch_loss /= n_batches
            if epoch in (1, self.n_epochs) or epoch % max(1, self.n_epochs // 10) == 0:
                logger.debug("epoch %4d/%d  loss=%.6f", epoch, self.n_epochs, epoch_loss)

        self.model_bundle = (model, state, batch_stats, has_batchnorm)
        self.model = self.model_bundle

        if not build_excess:
            return self.model

        # Compute aleatoric variance and excess targets from val set.
        vars_in = {"params": state.params}
        if has_batchnorm and batch_stats is not None:
            vars_in["batch_stats"] = batch_stats
        mu_val = model.apply(vars_in, X_val, True, True).reshape(-1)
        y_val_flat = y_val.reshape(-1)
        self.aleatoric_var_ = float(jnp.mean((y_val_flat - mu_val) ** 2))

        # Paper: train error predictor on raw squared residuals (no subtraction, no log).
        # Aleatoric is subtracted at inference time only.
        y_excess = ((y_val_flat - mu_val) ** 2).reshape((-1, 1))

        excess_bundle, excess_r2 = self._fit_excess_model(
            X_val, y_excess,
            lr_excess=float(kwargs.get("lr_excess", self.lr_excess)),
            weight_decay_excess=float(kwargs.get("weight_decay_excess", self.weight_decay_excess)),
            n_epochs_excess=int(kwargs.get("n_epochs_excess", self.n_epochs_excess)),
            batch_size_excess=int(kwargs.get("batch_size_excess", self.batch_size_excess)),
            rng=rng,
            has_batchnorm=has_batchnorm,
            grad_clip_norm=grad_clip_norm,
            x_dummy=x_dummy,
        )

        if excess_r2 < 0.05:
            logger.warning(
                "excess model train-R²=%.3f — barely beats a constant predictor. "
                "Epistemic estimates will be unreliable. Consider increasing val_fraction, "
                "n_epochs_excess, or adjusting lr_excess.",
                excess_r2,
            )

        self.excess_model_bundle = excess_bundle
        return self.model

    def predict(self, model_bundle=None, X=None, **kwargs: Any):
        if X is None:
            X = model_bundle
            model_bundle = self.model_bundle
        if model_bundle is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")

        model, state, batch_stats, has_batchnorm = model_bundle
        X = self._transform_jax_bnn_inputs(X)
        N = X.shape[0]
        bs = min(self.predict_batch_size, N)

        vars_in = {"params": state.params}
        if has_batchnorm:
            vars_in["batch_stats"] = batch_stats

        out_batches = []
        for start in range(0, N, bs):
            xb = X[start:start + bs]
            out_batches.append(model.apply(vars_in, xb, True, True))

        return self._inverse_transform_jax_bnn_mean(jnp.concatenate(out_batches, axis=0).reshape(-1))

    def ud_fit_predict(self, X_train, y_train, X_eval, **kwargs: Any) -> Dict:
        self.train(X_train, y_train, **kwargs)
        model_bundle = self.model_bundle
        excess_model_bundle = self.excess_model_bundle
        if excess_model_bundle is None:
            raise RuntimeError("DEUP ud_fit_predict requires build_excess=True.")

        self.model = model_bundle

        X_eval_std = self._transform_jax_bnn_inputs(X_eval)
        mean = self.predict(model_bundle, X_eval)

        excess_model, excess_state, excess_batch_stats, excess_has_batchnorm = excess_model_bundle
        vars_in = {"params": excess_state.params}
        if excess_has_batchnorm and excess_batch_stats is not None:
            vars_in["batch_stats"] = excess_batch_stats

        N = X_eval_std.shape[0]
        bs = min(self.predict_batch_size, N)
        epi_batches = []
        for start in range(0, N, bs):
            epi_batches.append(excess_model.apply(vars_in, X_eval_std[start:start + bs], True, True))

        # Paper: error predictor outputs total risk e(x); subtract aleatoric at inference.
        # u(x) = e(x) - a(x), where a(x) = global MSE estimate (no replicates available).
        e_pred = jnp.concatenate(epi_batches, axis=0).reshape(-1)
        max_e = max(float(self.epi_clip_multiplier) * float(self.target_variance_), float(self.np_eps))
        e_pred = jnp.nan_to_num(e_pred, nan=0.0, posinf=max_e, neginf=0.0)
        e_pred = jnp.clip(e_pred, a_min=0.0, a_max=max_e)
        aleatoric_var = float(self.aleatoric_var_) if self.aleatoric_var_ is not None else 0.0
        epi = jnp.maximum(e_pred - aleatoric_var, 0.0)
        ale = jnp.full_like(epi, aleatoric_var)
        total = ale + epi
        return {
            "y_pred": jnp.asarray(mean),
            "total_uncertainty": self._inverse_transform_jax_bnn_variance(total),
            "epistemic_uncertainty": self._inverse_transform_jax_bnn_variance(epi),
            "aleatoric_uncertainty": self._inverse_transform_jax_bnn_variance(ale),
        }

    @classmethod
    def _run_excess_sweep(
        cls,
        X: Any,
        y: Any,
        *,
        fixed_model_kwargs: Dict[str, Any],
        project: str,
        count: int,
        entity: str | None = None,
        tags: Sequence[str] | None = None,
        **wandb_init_kwargs: Any,
    ) -> Dict[str, Any]:
        """Train the main model once with fixed hyperparams, then sweep excess-model hyperparams."""
        instance_kwargs = cls._jax_bnn_instance_kwargs_from_cfg(fixed_model_kwargs)
        fit_kwargs = cls._jax_bnn_fit_kwargs_from_cfg(fixed_model_kwargs)
        main_model = cls(**instance_kwargs)

        # Use the model's configured val_fraction to split main train / excess data.
        val_fraction = float(getattr(main_model, "val_fraction", 0.2))
        X_tr, y_tr, X_v, y_v = prepare_train_val(X, y, val_fraction=val_fraction)
        main_model.train(X_tr, y_tr, **{**fit_kwargs, "build_excess": False})

        X_v_std = jnp.asarray(main_model._transform_jax_bnn_inputs(X_v), dtype=jnp.float32)
        y_v_std = jnp.asarray(main_model._transform_jax_bnn_targets(y_v), dtype=jnp.float32)

        model, state, batch_stats, has_batchnorm = main_model.model_bundle
        vars_map = {"params": state.params}
        if has_batchnorm and batch_stats is not None:
            vars_map["batch_stats"] = batch_stats
        mu_val = model.apply(vars_map, X_v_std, True, True).reshape(-1)

        y_v_flat = y_v_std.reshape(-1)
        aleatoric_var = float(jnp.mean((y_v_flat - mu_val) ** 2))
        main_model.aleatoric_var_ = aleatoric_var
        main_model.target_variance_ = float(jnp.var(y_v_std) + main_model.np_eps)
        y_excess_full = ((y_v_flat - mu_val) ** 2).reshape((-1, 1))

        # 80/20 split of the excess data for excess train / held-out eval.
        N_v = X_v_std.shape[0]
        n_eval = max(1, N_v // 5)
        X_excess_tr, X_excess_eval = X_v_std[:-n_eval], X_v_std[-n_eval:]
        y_excess_tr, y_excess_eval = y_excess_full[:-n_eval], y_excess_full[-n_eval:]

        grad_clip_norm = float(fixed_model_kwargs.get("grad_clip_norm", main_model.grad_clip_norm))
        x_dummy = jnp.zeros((1, X_v_std.shape[1]), dtype=jnp.float32)

        def objective(cfg: Dict[str, Any]) -> Dict[str, float]:
            bundle, train_r2 = main_model._fit_excess_model(
                X_excess_tr, y_excess_tr,
                lr_excess=float(cfg["lr_excess"]),
                weight_decay_excess=float(cfg["weight_decay_excess"]),
                n_epochs_excess=int(cfg["n_epochs_excess"]),
                batch_size_excess=int(cfg["batch_size_excess"]),
                rng=jax.random.PRNGKey(42),
                has_batchnorm=has_batchnorm,
                grad_clip_norm=grad_clip_norm,
                x_dummy=x_dummy,
            )
            excess_model, excess_state, excess_batch_stats, _ = bundle
            vars_eval = {"params": excess_state.params}
            if has_batchnorm and excess_batch_stats is not None:
                vars_eval["batch_stats"] = excess_batch_stats
            pred_eval = excess_model.apply(vars_eval, X_excess_eval, True, True).reshape(-1)
            y_eval_flat = y_excess_eval.reshape(-1)
            eval_mse = float(jnp.mean((pred_eval - y_eval_flat) ** 2))
            baseline_mse = float(jnp.var(y_eval_flat) + 1e-30)
            eval_r2 = 1.0 - eval_mse / baseline_mse
            if eval_r2 < 0.0:
                logger.error(
                    "excess model eval-R²=%.3f — worse than a constant predictor. "
                    "lr_excess=%.2e, n_epochs_excess=%s.",
                    eval_r2, cfg["lr_excess"], cfg["n_epochs_excess"],
                )
            return {"excess_val_mse": eval_mse, "excess_val_r2": eval_r2, "excess_train_r2": train_r2}

        return run_wandb_bayes_sweep(
            sweep_config=make_wandb_sweep_config("bnn_deup_excess", "excess_val_mse"),
            objective_fn=objective,
            project=project,
            entity=entity,
            count=count,
            tags=tags,
            **wandb_init_kwargs,
        )

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
        tags: Sequence[str] | None = None,
        fixed_model_kwargs: Dict[str, Any] | None = None,
        fixed_sweep_params: Sequence[str] | None = None,
        **wandb_init_kwargs: Any,
    ) -> Dict[str, Any]:
        main_result = super().tune_hyperparameters_wandb(
            X, y,
            project=project,
            count=count,
            metric_name=metric_name,
            rng=rng,
            entity=entity,
            tags=tags,
            fixed_model_kwargs=fixed_model_kwargs,
            fixed_sweep_params=fixed_sweep_params,
            **wandb_init_kwargs,
        )

        best_main = (main_result or {}).get("best_config") or {}
        if not best_main:
            return main_result
        fixed_best_main = cls._jax_bnn_merge_fixed_sweep_cfg(
            best_main,
            fixed_model_kwargs=fixed_model_kwargs,
            fixed_sweep_params=fixed_sweep_params,
        )

        excess_result = cls._run_excess_sweep(
            X, y,
            fixed_model_kwargs=fixed_best_main,
            project=project + "_excess",
            count=count,
            entity=entity,
            tags=tags,
            **wandb_init_kwargs,
        )

        best_excess = (excess_result or {}).get("best_config") or {}
        return {
            "sweep_id": (main_result or {}).get("sweep_id"),
            "excess_sweep_id": (excess_result or {}).get("sweep_id"),
            "best_config": {**best_main, **best_excess},
            "best_metric": (main_result or {}).get("best_metric"),
            "excess_best_metric": (excess_result or {}).get("best_metric"),
        }

    @classmethod
    def _jax_bnn_extra_fit_kwargs_from_cfg(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        return {
            "n_epochs_excess": int(cfg.get("n_epochs_excess", 100)),
            "batch_size_excess": int(cfg.get("batch_size_excess", 512)),
            "lr_excess": float(cfg.get("lr_excess", 1e-3)),
            "weight_decay_excess": float(cfg.get("weight_decay_excess", 0.0)),
        }

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")
