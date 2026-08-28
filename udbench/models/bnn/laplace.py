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
from laplax import laplace, calibration
from laplax.enums import (
    CurvApprox,
    Pushforward,
    Predictive,
    CalibrationObjective,
    CalibrationMethod,
)
from laplax.eval.pushforward import get_dist_state

from udbench.BaseUDRegressor import BaseUDRegressor
from ._base_jax import (
    JaxBNNTuningMixin,
    TabMLPBackbone,
    TabularBNNBaseJax,
    TabResNetBackbone,
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
class TabularBNNLaplaceRegressor(JaxBNNTuningMixin, BaseUDRegressor):
    tuning_model_key = "bnn_laplace"

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
    n_members: int = 30
    val_fraction: float = 0.2
    NPLoss: bool = True
    np_link_fn: str = "softplus"
    np_eps: float = 1e-6
    laplace_loss_fn: str = "mse"
    curv_type: str = CurvApprox.DIAGONAL
    num_curv_samples: int | None = None
    num_total_samples: int | None = None
    vmap_over_data: bool = True
    predictive_type: Predictive = Predictive.NONE
    pushforward_type: Pushforward = Pushforward.NONLINEAR
    calib_num_samples: int = 128
    calibration_objective: CalibrationObjective = CalibrationObjective.NLL
    calibration_method: CalibrationMethod = CalibrationMethod.GRID_SEARCH
    log_prior_prec_min: float = -1.0
    log_prior_prec_max: float = 1.0
    grid_size: int = 100
    posterior_temperature: float = 0.5
    laplace_posthoc_search: bool = True
    laplace_search_val_fractions: Tuple[float, ...] = (0.2, 0.25)
    laplace_search_temperatures: Tuple[float, ...] = (0.3, 0.5, 0.7)
    rng: int | None = None
    predict_batch_size: int = 4096
    device: Any = None
    dtype: Any = None
    verbose: bool = False
    standardize_inputs: bool = True
    standardize_targets: bool = True
    standardization_eps: float = 1e-8

    model_bundle: Any = field(default=None, init=False)
    laplax_state: Any = field(default=None, init=False)
    laplax_rng: Any = field(default=None, init=False)
    target_variance_: float = field(default=1e-6, init=False)

    def _to_interpretable_pred(self, pred: jnp.ndarray) -> jnp.ndarray:
        if not self.NPLoss:
            return pred
        _, _, mu, std = natural_gaussian_stats_from_raw(pred, link=self.np_link_fn, eps=self.np_eps)
        return jnp.stack([mu, std], axis=-1)

    def _make_last_layer_functions(
        self,
        model_bundle: tuple[Any, Any, Any, bool],
    ) -> tuple[Any, Any, Any]:
        model, state, batch_stats, has_batchnorm = model_bundle
        params = state.params
        if "Dense_0" not in params:
            raise RuntimeError("Expected final Dense_0 head in TabularBNNBaseJax params.")

        backbone_keys = [key for key in params.keys() if key != "Dense_0"]
        if len(backbone_keys) != 1:
            raise RuntimeError(f"Expected exactly one backbone subtree, got {backbone_keys!r}.")
        backbone_key = backbone_keys[0]
        backbone_params = params[backbone_key]
        backbone_batch_stats = None
        if has_batchnorm and batch_stats is not None:
            backbone_batch_stats = batch_stats.get(backbone_key)

        if str(self.backbone).lower() == "resnet":
            backbone_module = TabResNetBackbone(
                resnet_width=self.resnet_width,
                resnet_blocks=self.resnet_blocks,
                norm=self.norm,
                activation=self.activation,
                dropout=self.dropout,
                init_scale=self.init_scale,
            )
        elif str(self.backbone).lower() == "mlp":
            backbone_module = TabMLPBackbone(
                hidden_features=tuple(self.hidden_features),
                norm=self.norm,
                activation=self.activation,
                dropout=self.dropout,
                init_scale=self.init_scale,
            )
        else:
            raise ValueError(f"Unsupported backbone {self.backbone!r} for Laplace.")

        def feature_fn(input: jnp.ndarray) -> jnp.ndarray:
            vars_in = {"params": backbone_params}
            if has_batchnorm and backbone_batch_stats is not None:
                vars_in["batch_stats"] = backbone_batch_stats
            return backbone_module.apply(vars_in, input, True, True)

        full_head = params["Dense_0"]
        mean_head_params = {
            "kernel": full_head["kernel"][:, 0],
            "bias": full_head["bias"][0],
        }
        var_head_kernel = full_head["kernel"][:, 1]
        var_head_bias = full_head["bias"][1]

        def model_fn_mu(input: jnp.ndarray, params: Any) -> jnp.ndarray:
            features = feature_fn(input)
            eta1 = jnp.dot(features, params["kernel"]) + params["bias"]
            if self.NPLoss:
                raw2 = jnp.dot(features, var_head_kernel) + var_head_bias
                pred = jnp.stack([eta1, raw2], axis=-1)
                _, _, mu, _ = natural_gaussian_stats_from_raw(pred, link=self.np_link_fn, eps=self.np_eps)
                return mu
            return eta1

        def map_aleatoric_var_fn(input: jnp.ndarray) -> jnp.ndarray:
            vars_in = {"params": state.params}
            if has_batchnorm and batch_stats is not None:
                vars_in["batch_stats"] = batch_stats
            out_map = model.apply(vars_in, input, True, True)
            if self.NPLoss:
                _, _, _, std_map = natural_gaussian_stats_from_raw(out_map, link=self.np_link_fn, eps=self.np_eps)
                return jnp.clip(std_map**2, a_min=float(self.np_eps))
            scale = jax.nn.softplus(out_map[..., 1]) + self.np_eps
            return jnp.clip(scale**2, a_min=float(self.np_eps))

        return model_fn_mu, mean_head_params, map_aleatoric_var_fn

    def _laplace_val_score(
        self,
        candidate: dict[str, Any],
        *,
        temperature: float,
        sample_key: Any,
    ) -> float:
        posterior = candidate["posterior_fn"](candidate["prior_args"], float(temperature))
        score_num_samples = max(32, min(int(self.n_members), 64))
        dist_state = get_dist_state(
            mean_params=candidate["mean_head_params"],
            model_fn=candidate["model_fn_mu"],
            posterior_state=posterior,
            linearized=False,
            num_samples=score_num_samples,
            key=sample_key,
        )
        get_w = dist_state["get_weight_samples"]
        params_samples = jax.vmap(get_w)(jnp.arange(score_num_samples))
        mu_samples = jax.vmap(lambda p: candidate["model_fn_mu"](candidate["X_val"], p))(params_samples)
        mean = jnp.mean(mu_samples, axis=0).reshape(-1)
        epi = jnp.var(mu_samples, axis=0).reshape(-1)
        ale = candidate["map_aleatoric_var_fn"](candidate["X_val"]).reshape(-1)
        total = jnp.clip(ale + epi, a_min=float(self.np_eps))
        target = candidate["y_val"].reshape(-1)
        nll = 0.5 * (
            jnp.log(2.0 * jnp.pi * total)
            + ((target - mean) ** 2) / total
        )
        return float(jnp.mean(nll))

    def _build_last_layer_laplax_candidate(
        self,
        model_bundle: tuple[Any, Any, Any, bool],
        X: jnp.ndarray,
        y: jnp.ndarray,
        perm: jnp.ndarray,
        *,
        val_fraction: float,
    ) -> dict[str, Any]:
        n_total = int(X.shape[0])
        n_val = min(max(1, int(float(val_fraction) * n_total)), n_total - 1)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model_fn_mu, mean_head_params, map_aleatoric_var_fn = self._make_last_layer_functions(model_bundle)
        train_data = {"input": X_tr, "target": y_tr.reshape(-1)}
        val_data = {"input": X_val, "target": y_val.reshape(-1)}

        num_curv_samples = X_tr.shape[0] if self.num_curv_samples is None else int(self.num_curv_samples)
        num_total_samples = X_tr.shape[0] if self.num_total_samples is None else int(self.num_total_samples)

        raw_curv_type = str(self.curv_type).lower()
        legacy_curv_aliases = {
            "ggn": CurvApprox.DIAGONAL.value,
            "full-ggn": CurvApprox.FULL.value,
            "full_ggn": CurvApprox.FULL.value,
            "diag-ggn": CurvApprox.DIAGONAL.value,
            "diag_ggn": CurvApprox.DIAGONAL.value,
        }
        curv_type = legacy_curv_aliases.get(raw_curv_type, raw_curv_type)
        valid_curv_types = {item.value for item in CurvApprox}
        if curv_type not in valid_curv_types:
            valid_curv_types_str = ", ".join(sorted(valid_curv_types))
            raise ValueError(
                f"Invalid curv_type={self.curv_type!r}. "
                f"Use one of: {valid_curv_types_str}."
            )

        posterior_fn, curv_est = laplace(
            model_fn_mu,
            mean_head_params,
            train_data,
            loss_fn=self.laplace_loss_fn,
            curv_type=curv_type,
            num_curv_samples=num_curv_samples,
            num_total_samples=num_total_samples,
            vmap_over_data=self.vmap_over_data,
        )

        calib_key = jax.random.key(0) if self.rng is None else jax.random.key(int(self.rng) + 123)
        prior_args, _ = calibration(
            posterior_fn,
            model_fn_mu,
            mean_head_params,
            val_data,
            loss_fn=self.laplace_loss_fn,
            curv_estimate=curv_est,
            curv_type=curv_type,
            predictive_type=self.predictive_type,
            pushforward_type=self.pushforward_type,
            num_samples=int(self.calib_num_samples),
            calibration_objective=self.calibration_objective,
            calibration_method=self.calibration_method,
            sample_key=calib_key,
            log_prior_prec_min=self.log_prior_prec_min,
            log_prior_prec_max=self.log_prior_prec_max,
            grid_size=int(self.grid_size),
        )

        return {
            "model_fn_mu": model_fn_mu,
            "posterior_fn": posterior_fn,
            "prior_args": prior_args,
            "mean_head_params": mean_head_params,
            "map_aleatoric_var_fn": map_aleatoric_var_fn,
            "X_val": X_val,
            "y_val": y_val,
            "selected_val_fraction": float(val_fraction),
        }

    
    
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

        rng_seed = 0 if self.rng is None else int(self.rng)
        init_seed = getattr(self, "init_seed", None)
        train_seed = getattr(self, "train_seed", None)
        laplace_seed = getattr(self, "laplace_seed", None)
        predict_seed = getattr(self, "predict_seed", None)
        has_explicit_seed_split = any(
            seed is not None for seed in (init_seed, train_seed, laplace_seed, predict_seed)
        )
        if has_explicit_seed_split:
            init_seed_value = rng_seed if init_seed is None else int(init_seed)
            train_seed_value = rng_seed if train_seed is None else int(train_seed)
            init_rng = jax.random.PRNGKey(init_seed_value)
            drop_rng = jax.random.PRNGKey(init_seed_value + 1)
            rng = jax.random.PRNGKey(train_seed_value)
        else:
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
                xb, yb = X[batch_idx], y[batch_idx]
                state, batch_stats, rng, loss = train_step(state, batch_stats, rng, xb, yb, has_batchnorm)
                epoch_loss += float(loss)

            epoch_loss /= n_batches
            if epoch in (1, self.n_epochs) or epoch % max(1, self.n_epochs // 10) == 0:
                logger.debug("epoch %4d/%d  loss=%.6f", epoch, self.n_epochs, epoch_loss)

        self.model_bundle = (model, state, batch_stats, has_batchnorm)
        self.model = self.model_bundle

        build_laplace = bool(kwargs.get("build_laplace", True))
        if not build_laplace:
            self.laplax_state = None
            self.laplax_rng = None
            return self.model

        laplace_rng = jax.random.PRNGKey(int(laplace_seed)) if laplace_seed is not None else rng
        laplace_rng, split_rng = jax.random.split(laplace_rng)
        perm = jax.random.permutation(split_rng, X.shape[0])
        candidate_val_fractions = (
            tuple(float(v) for v in self.laplace_search_val_fractions)
            if self.laplace_posthoc_search
            else (float(self.val_fraction),)
        )
        candidate_temperatures = (
            tuple(float(v) for v in self.laplace_search_temperatures)
            if self.laplace_posthoc_search
            else (float(self.posterior_temperature),)
        )

        best_state: dict[str, Any] | None = None
        best_temperature = float(self.posterior_temperature)
        best_score = float("inf")
        for val_fraction in candidate_val_fractions:
            candidate = self._build_last_layer_laplax_candidate(
                self.model_bundle,
                X,
                y,
                perm,
                val_fraction=val_fraction,
            )
            for temperature in candidate_temperatures:
                laplace_rng, score_key = jax.random.split(laplace_rng)
                score = self._laplace_val_score(
                    candidate,
                    temperature=float(temperature),
                    sample_key=score_key,
                )
                if score < best_score:
                    best_score = score
                    best_state = dict(candidate)
                    best_temperature = float(temperature)

        if best_state is None:
            raise RuntimeError("Failed to build a valid last-layer Laplace state.")

        self.val_fraction = float(best_state["selected_val_fraction"])
        self.posterior_temperature = float(best_temperature)
        self.laplax_state = {
            "model_fn_mu": best_state["model_fn_mu"],
            "posterior_fn": best_state["posterior_fn"],
            "prior_args": best_state["prior_args"],
            "mean_head_params": best_state["mean_head_params"],
            "selected_val_fraction": float(self.val_fraction),
            "selected_temperature": float(self.posterior_temperature),
            "selection_score_nll": float(best_score),
        }
        sample_seed = (
            (999 if self.rng is None else int(self.rng) + 999)
            if predict_seed is None
            else int(predict_seed)
        )
        self.laplax_rng = jax.random.key(sample_seed)
        return self.model

    def _predict_backbone_mean(self, X):
        if self.model_bundle is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")
        model, state, batch_stats, has_batchnorm = self.model_bundle
        X = self._transform_jax_bnn_inputs(X)
        vars_in = {"params": state.params}
        if has_batchnorm and batch_stats is not None:
            vars_in["batch_stats"] = batch_stats
        out = model.apply(vars_in, X, True, True)
        if self.NPLoss:
            _, _, mu, _ = natural_gaussian_stats_from_raw(out, link=self.np_link_fn, eps=self.np_eps)
            return self._inverse_transform_jax_bnn_mean(mu.reshape(-1))
        return self._inverse_transform_jax_bnn_mean(out[..., 0].reshape(-1))

    def forward(self, model_bundle, X):
        """
        Laplace pushforward: returns (E, N, D) = (n_members, N, 2)
        where [:, :, 0] are mu samples and [:, :, 1] is the MAP raw_scale replicated.
        """
        if self.laplax_state is None:
            raise RuntimeError("Laplace state not built. Call train() first.")
    
        model, state, batch_stats, has_batchnorm = model_bundle
        X = jnp.asarray(X, dtype=jnp.float32)  # (N, Din)
        E = int(self.n_members)
    
        model_fn_mu = self.laplax_state["model_fn_mu"]
        posterior_fn = self.laplax_state["posterior_fn"]
        prior_args = self.laplax_state["prior_args"]
    
        # build posterior object (contains state + scale/cov mv callables)
        posterior = posterior_fn(prior_args, float(self.posterior_temperature))
    
        if self.laplax_rng is None:
            raise RuntimeError("Laplace RNG not initialized. Call train() first.")
        key = self.laplax_rng
        dist_state = get_dist_state(
            mean_params=self.laplax_state["mean_head_params"],
            model_fn=model_fn_mu,
            posterior_state=posterior,
            linearized=False,
            num_samples=E,
            key=key,
        )
        get_w = dist_state["get_weight_samples"]
        params_samples = jax.vmap(get_w)(jnp.arange(E))  # pytree with leading axis E
        mu_samples = jax.vmap(lambda p: model_fn_mu(X, p))(params_samples)  # (E, N)
    
        # MAP aleatoric head, replicated across E
        vars_in = {"params": state.params}
        if has_batchnorm and batch_stats is not None:
            vars_in["batch_stats"] = batch_stats
        out_map = model.apply(vars_in, X, True, True)          # (N, 2)
        if self.NPLoss:
            _, _, _, std_map = natural_gaussian_stats_from_raw(out_map, link=self.np_link_fn, eps=self.np_eps)
            rep = jnp.broadcast_to(std_map[None, :], (E, X.shape[0]))  # (E, N)
        else:
            raw_map = out_map[..., 1]                        # (N,)
            rep = jnp.broadcast_to(raw_map[None, :], (E, X.shape[0]))  # (E, N)

        preds = jnp.stack([mu_samples, rep], axis=-1)  # (E, N, 2)
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

    def predict(self, model, X, **kwargs: Any):
        X = self._transform_jax_bnn_inputs(X)
        preds = self.forward(model, X)          # (E, N, 2)
        mu = preds[..., 0]                      # (E, N)
        return self._inverse_transform_jax_bnn_mean(jnp.mean(mu, axis=0))  # (N,)

    
    def ud_fit_predict(self, X_train, y_train, X_eval, **kwargs: Any) -> Dict:
        self.train(X_train, y_train, **kwargs)
    
        X_eval = self._transform_jax_bnn_inputs(X_eval)
        preds = self.forward(self.model_bundle, X_eval)  # (E, N, 2)
        mu = preds[..., 0]                               # (E, N)
        aux = preds[..., 1]                              # (E, N) but identical across E
    
        mean = jnp.mean(mu, axis=0)                      # (N,)
        epi = jnp.var(mu, axis=0)                        # (N,)
    
        if self.NPLoss:
            ale = aux[0] ** 2
        else:
            scale = jax.nn.softplus(aux[0]) + self.np_eps
            ale = scale**2
    
        total = ale + epi
    
        return self._restore_jax_bnn_ud_outputs(mean, total, epi, ale)

    @classmethod
    def _jax_bnn_extra_fit_kwargs_from_cfg(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        return {
            "n_members": int(cfg["n_members"]),
        }

    @classmethod
    def _jax_bnn_tuning_predictions(
        cls,
        model: Any,
        X_tr: Any,
        y_tr: Any,
        X_v: Any,
        fit_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        model.train(X_tr, y_tr, **fit_kwargs)

        X_v_std = model._transform_jax_bnn_inputs(X_v)
        preds = model.forward(model.model_bundle, X_v_std)
        mu = preds[..., 0]
        aux = preds[..., 1]

        mean = jnp.mean(mu, axis=0)
        epi = jnp.var(mu, axis=0)
        if model.NPLoss:
            ale = aux[0] ** 2
        else:
            scale = jax.nn.softplus(aux[0]) + model.np_eps
            ale = scale**2
        total = ale + epi

        return model._restore_jax_bnn_ud_outputs(mean, total, epi, ale)

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call train(X, y) first.")
