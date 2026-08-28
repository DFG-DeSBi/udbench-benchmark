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


def swag_init(params):
    return {
        "n": jnp.array(1, dtype=jnp.int32),
        "mean": params,
        "sq_mean": jax.tree_util.tree_map(lambda p: p * p, params),
    }

def swag_update(swag, params):
    # online update of E[p] and E[p^2]
    n = swag["n"] + 1
    mean = jax.tree_util.tree_map(lambda m, p: m + (p - m) / n, swag["mean"], params)
    sq_mean = jax.tree_util.tree_map(lambda s, p: s + (p * p - s) / n, swag["sq_mean"], params)
    return {"n": n, "mean": mean, "sq_mean": sq_mean}

def swag_var(swag, eps=1e-30):
    # Var[p] = E[p^2] - (E[p])^2  (clipped)
    return jax.tree_util.tree_map(lambda s, m: jnp.maximum(s - m * m, eps), swag["sq_mean"], swag["mean"])

def split_like_tree(key, tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(treedef, keys)

def sample_swag_diag(key, mean, var, scale=1.0):
    key_tree = split_like_tree(key, mean)
    noise = jax.tree_util.tree_map(lambda k, p: jax.random.normal(k, p.shape, p.dtype), key_tree, mean)
    return jax.tree_util.tree_map(lambda mu, va, z: mu + scale * jnp.sqrt(va) * z, mean, var, noise)

@dataclass
class TabularBNNSWAGRegressor(JaxBNNTuningMixin, BaseUDRegressor):
    tuning_model_key = "bnn_swag"

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
    n_checkpoints: int = 20
    epoch_per_checkpoint: int = 10
    swag_lr: float = 0.01
    swag_momentum: float = 0.9
    swag_batch_size: int | None = None
    calibration_fraction: float = 0.1
    calibration_var_scale: float = 1.0
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
    swag: Any = field(default=None, init=False)
    swag_sample_rng: Any = field(default=None, init=False)
    target_variance_: float = field(default=1e-6, init=False)

    def _to_interpretable_pred(self, pred: jnp.ndarray) -> jnp.ndarray:
        if not self.NPLoss:
            return pred
        _, _, mu, std = natural_gaussian_stats_from_raw(pred, link=self.np_link_fn, eps=self.np_eps)
        return jnp.stack([mu, std], axis=-1)

    def _resolve_model_bundle(self, model_bundle: Any = None):
        if model_bundle is None:
            model_bundle = self.model_bundle
        if model_bundle is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")

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
            return model, state, batch_stats, has_batchnorm

        raise ValueError(
            "Pass model as (model, state, batch_stats[, has_batchnorm]) or leave model=None."
        )
    
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

        rng_seed = 0 if self.rng is None else int(self.rng)
        train_call_idx = int(getattr(self, "_train_call_idx", 0))
        self._train_call_idx = train_call_idx + 1
        rng = jax.random.PRNGKey(rng_seed + 9973 * train_call_idx)

        calibration_fraction = float(kwargs.get("calibration_fraction", self.calibration_fraction))
        if not (0.0 <= calibration_fraction <= 1.0):
            raise ValueError("calibration_fraction must be in [0, 1].")
        cal_n = int(calibration_fraction * N)
        if cal_n >= N:
            cal_n = max(0, N - 1)
        if cal_n > 0:
            rng, split_rng = jax.random.split(rng)
            perm = jax.random.permutation(split_rng, N)
            cal_idx = perm[:cal_n]
            tr_idx = perm[cal_n:]
            X_cal, y_cal = X[cal_idx], y[cal_idx]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
        else:
            X_tr, y_tr = X, y
            X_cal, y_cal = None, None

        X, y = X_tr, y_tr
        N = X.shape[0]
        if N <= 0:
            raise ValueError("No training points left after calibration split.")

        has_batchnorm = (self.norm == "batchnorm")
        model = TabularBNNBaseJax(
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

        swag_batch_size = kwargs.get("swag_batch_size", self.swag_batch_size)
        if swag_batch_size is None:
            swag_batch_size = self.batch_size
        swag_batch_size = int(swag_batch_size)
        if swag_batch_size <= 0:
            raise ValueError("swag_batch_size must be positive.")
        swag_effective_batch_size = min(swag_batch_size, N)

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

        base_state = state
        self.model_bundle = (model, base_state, batch_stats, has_batchnorm)
        self.model = self.model_bundle

        # SWAG collection phase: switch to SGD with a constant LR and sample along
        # a continuous trajectory (never restart). Adam is only used to reach the basin.
        swag_lr = float(kwargs.get("swag_lr", self.swag_lr))
        swag_momentum = float(kwargs.get("swag_momentum", self.swag_momentum))
        swag_tx = optax.chain(
            optax.clip_by_global_norm(grad_clip_norm),
            make_jax_optimizer(
                "sgd",
                learning_rate=swag_lr,
                momentum=swag_momentum,
                weight_decay=0.0,
            ),
        )
        swag_state = train_state.TrainState.create(
            apply_fn=model.apply, params=base_state.params, tx=swag_tx
        )
        swag_batch_stats = batch_stats

        swag = swag_init(base_state.params)  # include Adam-converged params as first SWAG point
        logger.info(
            "Training complete. Started SWAG collection phase (SGD lr=%s, momentum=%s).",
            swag_lr, swag_momentum,
        )

        for ckpt_id in range(1, self.n_checkpoints + 1):
            ckpt_loss = 0.0
            ckpt_batches = 0

            for _ in range(self.epoch_per_checkpoint):
                rng, ep_rng = jax.random.split(rng)
                epoch_idx = jax.random.permutation(ep_rng, N)  # (N,)

                for start in range(0, N, swag_effective_batch_size):
                    batch_idx = epoch_idx[start:start + swag_effective_batch_size]  # (B,)
                    xb, yb = X[batch_idx], y[batch_idx]
                    swag_state, swag_batch_stats, rng, loss = train_step(
                        swag_state,
                        swag_batch_stats,
                        rng,
                        xb,
                        yb,
                        has_batchnorm,
                    )
                    ckpt_loss += float(loss)
                    ckpt_batches += 1

            swag = swag_update(swag, swag_state.params)
            logger.debug("SWAG checkpoint %3d/%d  loss=%.6f", ckpt_id, self.n_checkpoints, ckpt_loss / max(1, ckpt_batches))

        self.swag = swag
        if self.rng is None:
            sample_seed = int(np.random.SeedSequence().generate_state(1)[0])
        else:
            sample_seed = int(self.rng) + 999 + 9973 * train_call_idx
        self.swag_sample_rng = jax.random.PRNGKey(sample_seed % (2**32))

        # Calibrate predictive variance scale on a held-out split.
        self.calibration_var_scale = 1.0
        if X_cal is not None and y_cal is not None and X_cal.shape[0] > 0:
            cal_preds = self.forward(self.model_bundle, X_cal, scale=1.0)
            cal_mu_members = cal_preds[..., 0]                        # (E, Ncal)
            if self.NPLoss:
                cal_obs_scale = cal_preds[..., 1]
            else:
                cal_raw = cal_preds[..., 1]
                cal_obs_scale = jax.nn.softplus(cal_raw) + self.np_eps
            cal_ale = jnp.mean(cal_obs_scale**2, axis=0)              # (Ncal,)
            cal_epi = jnp.var(cal_mu_members, axis=0)                 # (Ncal,)
            cal_total = cal_ale + cal_epi
            cal_mean = jnp.mean(cal_mu_members, axis=0)               # (Ncal,)
            cal_target = y_cal.squeeze(-1)                            # (Ncal,)

            mse = jnp.mean((cal_mean - cal_target) ** 2)
            avg_total = jnp.mean(cal_total)
            raw_scale = mse / (avg_total + 1e-12)
            raw_scale = jnp.clip(raw_scale, 1e-3, 1e3)
            if bool(jnp.isfinite(raw_scale)):
                self.calibration_var_scale = float(raw_scale)

        return self.model_bundle

    def forward(self, model_bundle, X, *, scale: float = 1.0):
        """
        Returns preds: (E, N, 2) where E=self.n_members
        """
        model, state, batch_stats, has_batchnorm = self._resolve_model_bundle(model_bundle)
        X = jnp.asarray(X, dtype=jnp.float32)  # (N, Din)
    
        if not hasattr(self, "swag") or self.swag is None:
            raise RuntimeError("SWAG stats not available. Call train() first.")
    
        mean = self.swag["mean"]
        var = swag_var(self.swag)
    
        if self.swag_sample_rng is None:
            raise RuntimeError("SWAG sample RNG not initialized. Call train() first.")
        member_keys = jax.random.split(self.swag_sample_rng, self.n_members)
    
        # sample params for each member -> stack into pytree with leading axis E
        samples = [sample_swag_diag(k, mean, var, scale=scale) for k in member_keys]
        params_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *samples)
    
        # Reuse one deterministic BN statistics tree for all sampled members.
        vars_common = {}
        if has_batchnorm and batch_stats is not None:
            vars_common["batch_stats"] = batch_stats
    
        # vmapped apply over params axis
        def apply_one(p):
            vars_in = {"params": p, **vars_common}
            return model.apply(vars_in, X, True, True)  # deterministic=True
    
        preds = jax.vmap(apply_one)(params_stack)  # (E, N, 2)
        preds = self._to_interpretable_pred(preds)
        preds = clip_and_check_ensemble_variance_heads(
            preds,
            target_variance=self.target_variance_,
            np_loss=self.NPLoss,
            eps=self.np_eps,
            clip_multiplier=50.0,
            warning_prefix=f"{self.__class__.__name__}.forward",
            warning_key=f"{self.__class__.__name__}:{id(self)}",
        )
        return preds
        


    def predict(self, model_bundle=None, X=None, **kwargs: Any):
        """
        Returns the ensemble mean prediction (mean of member means).
        For Gaussian head (Dout=2), returns mean(mu) shape (N,).
        """
        # Keep BaseUDRegressor-compatible calling: predict(X) or predict(model_bundle, X)
        if X is None:
            X = model_bundle
            model_bundle = self.model_bundle
        elif model_bundle is None:
            model_bundle = self.model_bundle

        model_bundle = self._resolve_model_bundle(model_bundle)
        X = self._transform_jax_bnn_inputs(X)
        member_means = self.forward(model_bundle, X)[..., 0]  # (E, N)
        y_mean = jnp.mean(member_means, axis=0)  # (N, D)

        return self._inverse_transform_jax_bnn_mean(y_mean)

    
    def ud_fit_predict(self, X_train, y_train, X_eval, **kwargs):
        self.train(X_train, y_train, **kwargs)
        self._check_fitted()

        X_eval = self._transform_jax_bnn_inputs(X_eval)
        preds = self.forward(self.model_bundle, X_eval, scale=float(kwargs.get("scale", 1.0)))  # (E,N,2)
    
        mu = preds[..., 0]                             # (E,N)
        if self.NPLoss:
            obs_scale = preds[..., 1]
        else:
            raw = preds[..., 1]
            obs_scale = jax.nn.softplus(raw) + self.np_eps
    
        mean = jnp.mean(mu, axis=0)                    # (N,)
        ale = jnp.mean(obs_scale**2, axis=0)           # (N,)
        epi = jnp.var(mu, axis=0)                      # (N,)
        var_scale = float(kwargs.get("calibration_var_scale", self.calibration_var_scale))
        if np.isfinite(var_scale) and var_scale > 0:
            ale = ale * var_scale
            epi = epi * var_scale
        total = ale + epi
    
        return self._restore_jax_bnn_ud_outputs(mean, total, epi, ale)

    @classmethod
    def _jax_bnn_extra_fit_kwargs_from_cfg(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        return {
            "n_members": int(cfg["n_members"]),
            "n_checkpoints": int(cfg["n_checkpoints"]),
            "epoch_per_checkpoint": int(cfg["epoch_per_checkpoint"]),
            "swag_lr": float(cfg["swag_lr"]),
            "swag_momentum": float(cfg["swag_momentum"]),
            "calibration_fraction": float(cfg["calibration_fraction"]),
            "grad_clip_norm": float(cfg["grad_clip_norm"]),
        }

    def _check_fitted(self) -> None:
        if self.model_bundle is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")
