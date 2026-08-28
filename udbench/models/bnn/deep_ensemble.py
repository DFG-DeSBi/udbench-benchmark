from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Tuple

import logging

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import distrax

from udbench.BaseUDRegressor import BaseUDRegressor
from ._base_jax import (
    JaxBNNTuningMixin,
    TabularBNNBaseJax,
    apply_model_with_batch_stats,
    apply_jax_bnn_runtime_overrides,
    clip_and_check_ensemble_variance_heads,
    make_jax_optimizer,
    jax_optimizer_extra_kwargs,
    natural_gaussian_nll,
    natural_gaussian_stats_from_raw,
    target_variance_from_targets,
    variance_head_regularization,
)

logger = logging.getLogger(__name__)


@dataclass
class TabularBNNDeepRegressor(JaxBNNTuningMixin, BaseUDRegressor):
    tuning_model_key = "bnn_deep_ensemble"

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
    variance_penalty_weight: float = 1e-3
    train_variance_clip_multiplier: float = 20.0
    n_members: int = 8
    NPLoss: bool = True
    np_link_fn: str = "softplus"
    np_eps: float = 1e-6
    rng: int | None = None
    predict_batch_size: int = 4096
    device: Any = None
    dtype: Any = None
    verbose: bool = False
    standardize_inputs: bool = True
    standardize_targets: bool = True
    standardization_eps: float = 1e-8

    model_bundle: Any = field(default=None, init=False)
    target_variance_: float = field(default=1e-6, init=False)

    def _to_interpretable_pred(self, pred: jnp.ndarray) -> jnp.ndarray:
        if not self.NPLoss:
            return pred
        _, _, mu, std = natural_gaussian_stats_from_raw(pred, link=self.np_link_fn, eps=self.np_eps)
        return jnp.stack([mu, std], axis=-1)

    def make_ensemble_module(self, n_members: int, has_batchnorm: bool):
        variable_axes = {'params': 0}
        if has_batchnorm:
            variable_axes['batch_stats'] = 0

        Ensemble = nn.vmap(
            TabularBNNBaseJax,
            variable_axes=variable_axes,                 # separate vars per member
            split_rngs={'params': True, 'dropout': True},# separate init + dropout rng per member
            in_axes=(0, None, None),                    # x is (E, B, Din), flags are shared
            out_axes=0,                                  # out is (E, B, Dout)
            axis_size=n_members,
        )
        return Ensemble
    
    def train(
            self, 
            X, 
            y, 
            tune: bool = False, 
            **kwargs: Any):
        kwargs = apply_jax_bnn_runtime_overrides(self, kwargs)
        
        def gaussian_nll(pred, y):
            # pred: (E, B, 2), y: (E, B) or (E, B, 1)
            y = y.squeeze(-1) if y.ndim == 3 else y
            mu = pred[..., 0]
            raw = pred[..., 1]
            scale = jax.nn.softplus(raw) + 1e-6
            dist = distrax.Normal(loc=mu, scale=scale)
            return -jnp.mean(dist.log_prob(y))  # mean over E and B

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
                    rngs={'dropout': drop_rng},
                    update_batch_stats=has_batchnorm,
                )
                variance_penalty = variance_head_regularization(
                    pred,
                    target_variance=self.target_variance_,
                    np_loss=self.NPLoss,
                    eps=self.np_eps,
                    clip_multiplier=self.train_variance_clip_multiplier,
                    link=self.np_link_fn,
                )
                if self.NPLoss:
                    base_loss = natural_gaussian_nll(pred, yb, link=self.np_link_fn, eps=self.np_eps)
                else:
                    base_loss = gaussian_nll(pred, yb)
                return base_loss + float(self.variance_penalty_weight) * variance_penalty, next_batch_stats

            (loss, next_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, next_batch_stats, rng, loss
        

        self._fit_jax_bnn_standardization(X, y)
        X = self._transform_jax_bnn_inputs(X)
        y = self._transform_jax_bnn_targets(y)
        self.target_variance_ = target_variance_from_targets(y, eps=self.np_eps)
        N = X.shape[0]

        has_batchnorm = (self.norm == "batchnorm")
        Ensemble = self.make_ensemble_module(self.n_members, has_batchnorm)

        model = Ensemble(
            input_dim=X.shape[1],
            output_dim=2,
            hidden_features=self.hidden_features,
            backbone=self.backbone,
            resnet_width=self.resnet_width,
            resnet_blocks=self.resnet_blocks,
            norm=self.norm,
            activation=self.activation,
            init_scale=self.init_scale,
            dropout=self.dropout,
        )

        rng_seed = 0 if self.rng is None else int(self.rng)
        rng = jax.random.PRNGKey(rng_seed)
        rng, init_rng = jax.random.split(rng, 2)

        # init requires input with leading E because in_axes=0
        x_dummy = jnp.zeros((self.n_members, 1, X.shape[1]), dtype=jnp.float32)
        variables = model.init(
            {'params': init_rng, 'dropout': init_rng},
            x_dummy,
            True,
            False,
        )
        params = variables['params']
        batch_stats = variables.get('batch_stats', None)

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
            epoch_idx = jax.random.permutation(ep_rng, N)  # (N,)

            epoch_loss = 0.0
            for start in range(0, N, effective_batch_size):
                batch_idx = epoch_idx[start:start + effective_batch_size]  # (B,)
                xb_single = X[batch_idx]  # (B, Din)
                yb_single = y[batch_idx]  # (B, 1)

                xb = jnp.broadcast_to(xb_single[None, :, :], (self.n_members,) + xb_single.shape)  # (E, B, Din)
                yb = jnp.broadcast_to(yb_single[None, :, :], (self.n_members,) + yb_single.shape)  # (E, B, 1)

                state, batch_stats, rng, loss = train_step(state, batch_stats, rng, xb, yb, has_batchnorm)
                epoch_loss += float(loss)

            epoch_loss /= n_batches
            if epoch in (1, self.n_epochs) or epoch % max(1, self.n_epochs // 10) == 0:
                logger.debug("epoch %4d/%d  loss=%.6f", epoch, self.n_epochs, epoch_loss)

        self.model_bundle = (model, state, batch_stats, has_batchnorm)
        self.model = self.model_bundle
        return self.model


    def forward(self, model_bundle, Xb: jnp.ndarray):
        """
        Xb: (B, Din)
        Returns pred: (E, B, Dout)
        """
        model, state, batch_stats, has_batchnorm = model_bundle
        xb = jnp.broadcast_to(Xb[None, :, :], (self.n_members,) + Xb.shape)
        # xb: (E, B, Din)

        vars_in = {"params": state.params}
        if has_batchnorm:
            vars_in["batch_stats"] = batch_stats

        pred = model.apply(vars_in, xb, True, True)
        pred = self._to_interpretable_pred(pred)
        pred = clip_and_check_ensemble_variance_heads(
            pred,
            target_variance=self.target_variance_,
            np_loss=self.NPLoss,
            eps=self.np_eps,
            clip_multiplier=50.0,
            warning_prefix=f"{self.__class__.__name__}.forward",
            warning_key=f"{self.__class__.__name__}:{id(self)}",
        )
        return pred


    def predict(self, model_bundle=None, X=None, **kwargs: Any):
        """
        Returns the ensemble mean prediction (mean of member means).
        For Gaussian head (Dout=2), returns mean(mu) shape (N,).
        """
        # Keep BaseUDRegressor-compatible calling: predict(X) or predict(model_bundle, X)
        if X is None:
            X = model_bundle
            model_bundle = self.model_bundle

        if model_bundle is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if isinstance(model_bundle, tuple):
            if len(model_bundle) == 4:
                model, state, batch_stats, has_batchnorm = model_bundle
            elif len(model_bundle) == 3:
                model, state, batch_stats = model_bundle
                has_batchnorm = (self.norm == "batchnorm")
            else:
                raise ValueError(
                    "Pass model as (model, state, batch_stats[, has_batchnorm]) or leave model=None."
                )
        else:
            raise ValueError(
                "Pass model as (model, state, batch_stats[, has_batchnorm]) or leave model=None."
            )
        model_bundle = (model, state, batch_stats, has_batchnorm)

        X = np.asarray(self._transform_jax_bnn_inputs(X))
        N = X.shape[0]
        bs = int(kwargs.get("batch_size", self.predict_batch_size))

        y_mean = np.empty((N,), dtype=np.float32)

        for start in range(0, N, bs):
            stop = min(N, start + bs)
            Xb = jnp.asarray(X[start:stop], dtype=jnp.float32)  # (B, Din)

            pred = self.forward(model_bundle, Xb)  # (E, B, Dout)

            if pred.shape[-1] == 2:
                mu = pred[..., 0]                    # (E, B)
                mean_b = jnp.mean(mu, axis=0)        # (B,)
            else:
                # generic regression head Dout=1 or Dout=k
                mean_b = jnp.mean(pred, axis=0)      # (B, Dout)
                mean_b = mean_b.squeeze(-1) if mean_b.ndim == 2 and mean_b.shape[-1] == 1 else mean_b

            y_mean[start:stop] = np.array(mean_b)

        return np.asarray(self._inverse_transform_jax_bnn_mean(y_mean))

    
    def ud_fit_predict(self, X_train, y_train, X_eval, **kwargs: Any) -> Dict:
        self.train(X_train, y_train, **kwargs)
        model_bundle = self.model_bundle

        self.model = model_bundle

        X_eval = np.asarray(self._transform_jax_bnn_inputs(X_eval))
        N = X_eval.shape[0]
        bs = int(kwargs.get("predict_batch_size", self.predict_batch_size))

        mean = np.empty((N,), dtype=np.float32)
        ale = np.empty((N,), dtype=np.float32)
        epi = np.empty((N,), dtype=np.float32)
        total = np.empty((N,), dtype=np.float32)

        for start in range(0, N, bs):
            stop = min(N, start + bs)
            Xb = jnp.asarray(X_eval[start:stop], dtype=jnp.float32)  # (B, Din)

            pred = self.forward(model_bundle, Xb)  # (E, B, Dout)

            if pred.shape[-1] != 2:
                raise ValueError("ud_fit_predict expects Gaussian head output_dim=2 => [mu, raw_scale].")

            mu = pred[..., 0]                           # (E, B)
            if self.NPLoss:
                scale = pred[..., 1]
            else:
                raw = pred[..., 1]
                scale = jax.nn.softplus(raw) + self.np_eps

            mean_b = jnp.mean(mu, axis=0)               # (B,)
            ale_b = jnp.mean(scale**2, axis=0)          # (B,)
            epi_b = jnp.var(mu, axis=0)                 # (B,)
            total_b = ale_b + epi_b

            mean[start:stop] = np.array(mean_b)
            ale[start:stop] = np.array(ale_b)
            epi[start:stop] = np.array(epi_b)
            total[start:stop] = np.array(total_b)

        return self._restore_jax_bnn_ud_outputs(mean, total, epi, ale)

    @classmethod
    def _jax_bnn_extra_fit_kwargs_from_cfg(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        return {"n_members": int(cfg["n_members"])}

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")

# Alias for backward compatibility with non-JAX naming
TabularBNNDeepEnsembleRegressor = TabularBNNDeepRegressor
