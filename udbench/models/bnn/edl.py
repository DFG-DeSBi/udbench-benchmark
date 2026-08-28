from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Tuple

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
    make_jax_optimizer,
    jax_optimizer_extra_kwargs,
)

logger = logging.getLogger(__name__)


@dataclass
class TabularBNNEDLRegressor(JaxBNNTuningMixin, BaseUDRegressor):
    tuning_model_key = "bnn_edl"

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
    evi_reg: float = 0.1
    optimizer: str = "nadamw"
    momentum: float = 0.9
    n_members: int = 8
    rng: int | None = None
    predict_batch_size: int = 4096
    device: Any = None
    dtype: Any = None
    verbose: bool = False
    standardize_inputs: bool = True
    standardize_targets: bool = True
    standardization_eps: float = 1e-8

    model_bundle: Any = field(default=None, init=False)

    def _nig_params(self, preds, eps=1e-6):
        # preds: (E, N, 4)
        mu = preds[..., 0]
        v = jax.nn.softplus(preds[..., 1]) + eps
        alpha = jax.nn.softplus(preds[..., 2]) + 1.0 + eps
        beta = jax.nn.softplus(preds[..., 3]) + eps
        return mu, v, alpha, beta
    
    def train(
            self, 
            X, 
            y, 
            tune: bool = False, 
            **kwargs: Any):
        kwargs = apply_jax_bnn_runtime_overrides(self, kwargs)
        
        def nig_nll(pred, y, *, eps: float = 1e-6):
            """
            NIG negative log marginal likelihood for regression (EDL).
            pred: (..., 4) = [mu, v_raw, alpha_raw, beta_raw]
                  works for (B,4) or (E,B,4)
            y:    (...,) or (...,1)
            Returns: scalar mean NLL over all leading dims.
            """
            y = y.squeeze(-1) if y.ndim == pred.ndim else y  # handle (...,1)

            mu = pred[..., 0]
            v = jax.nn.softplus(pred[..., 1]) + eps
            alpha = jax.nn.softplus(pred[..., 2]) + 1.0 + eps   # ensure alpha > 1
            beta = jax.nn.softplus(pred[..., 3]) + eps

            # error term
            err2 = (y - mu) ** 2

            # NIG marginal NLL (see Amini et al. "Deep Evidential Regression")
            # L = 0.5*log(pi/v) - alpha*log(2*beta*(1+v)) + (alpha+0.5)*log(v*err2 + 2*beta*(1+v))
            #     + lgamma(alpha) - lgamma(alpha+0.5)
            nll = (
                0.5 * (jnp.log(jnp.pi) - jnp.log(v))
                - alpha * jnp.log(2.0 * beta * (1.0 + v))
                + (alpha + 0.5) * jnp.log(v * err2 + 2.0 * beta * (1.0 + v))
                + jax.lax.lgamma(alpha)
                - jax.lax.lgamma(alpha + 0.5)
            )

            return jnp.mean(nll)
        
        def nig_evidence_reg(pred, y, *, eps: float = 1e-6):
            y = y.squeeze(-1) if y.ndim == pred.ndim else y
            mu = pred[..., 0]
            v = jax.nn.softplus(pred[..., 1]) + eps
            alpha = jax.nn.softplus(pred[..., 2]) + 1.0 + eps
            err = jnp.abs(y - mu)
            reg = err * (2.0 * v + alpha)
            return jnp.mean(reg)
        
        def edl_nig_loss(pred, y, reg_weight: float = 1e-2):
            return nig_nll(pred, y) + reg_weight * nig_evidence_reg(pred, y)

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
                return edl_nig_loss(pred, yb, self.evi_reg), next_batch_stats

            (loss, next_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, next_batch_stats, rng, loss
        

        self._fit_jax_bnn_standardization(X, y)
        X = self._transform_jax_bnn_inputs(X)
        y = self._transform_jax_bnn_targets(y)
        N = X.shape[0]

        has_batchnorm = (self.norm == "batchnorm")
        model = TabularBNNBaseJax(
            input_dim=X.shape[1],
            output_dim=4,
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
        rng, init_rng, drop_rng = jax.random.split(rng, 3)
        self.drop_rng = drop_rng  

        x_dummy = jnp.zeros((1, X.shape[1]), dtype=jnp.float32)
        variables = model.init(
            {'params': init_rng, 'dropout': drop_rng},
            x_dummy,
            True,
            False,
        )
        params = variables['params']
        batch_stats = variables.get('batch_stats', None)

        tx = make_jax_optimizer(
            self.optimizer,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            **jax_optimizer_extra_kwargs(self),
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
                xb, yb = X[batch_idx], y[batch_idx]
                state, batch_stats, rng, loss = train_step(state, batch_stats, rng, xb, yb, has_batchnorm)
                epoch_loss += float(loss)

            epoch_loss /= n_batches
            if epoch in (1, self.n_epochs) or epoch % max(1, self.n_epochs // 10) == 0:
                logger.debug("epoch %4d/%d  loss=%.6f", epoch, self.n_epochs, epoch_loss)

        self.model_bundle = (model, state, batch_stats, has_batchnorm)
        self.model = self.model_bundle

        return self.model

    def forward(self, model_bundle, X):
        """Single-model forward pass, batched over X."""
        model, state, batch_stats, has_batchnorm = model_bundle
        X = jnp.asarray(X, dtype=jnp.float32)  # (N, Din)
        N = X.shape[0]
        bs = min(self.predict_batch_size, N)

        vars_in = {"params": state.params}
        if has_batchnorm:
            vars_in["batch_stats"] = batch_stats

        out_batches = []
        for start in range(0, N, bs):
            xb = X[start:start + bs]  # (B, Din)
            pred_b = model.apply(vars_in, xb, True, True)  # (B, 4)
            out_batches.append(pred_b)

        preds = jnp.concatenate(out_batches, axis=0)  # (N, 4)
        return preds

    def predict(self, model_bundle=None, X=None, **kwargs: Any):
        if X is None:
            X = model_bundle
            model_bundle = getattr(self, "model_bundle", None)
        if model_bundle is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")

        X = self._transform_jax_bnn_inputs(X)
        preds = self.forward(model_bundle, X)   # (N, 4)
        mu, _, _, _ = self._nig_params(preds)

        return self._inverse_transform_jax_bnn_mean(mu)

    
    def ud_fit_predict(self, X_train, y_train, X_eval, **kwargs: Any) -> Dict:
        self.train(X_train, y_train, **kwargs)
        model_bundle = self.model_bundle

        self.model = model_bundle

        X_eval = self._transform_jax_bnn_inputs(X_eval)
        preds = self.forward(model_bundle, X_eval)  # (N, 4)
        mu, v, alpha, beta = self._nig_params(preds)

        mean = mu                                     # (N,)
        ale = beta / (alpha - 1.0)                   # (N,)
        epi = beta / (v * (alpha - 1.0))             # (N,)
        total = ale + epi

        return self._restore_jax_bnn_ud_outputs(mean, total, epi, ale)

    @classmethod
    def _jax_bnn_extra_fit_kwargs_from_cfg(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        return {
            "evi_reg": float(cfg["evi_reg"]),
        }

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")
