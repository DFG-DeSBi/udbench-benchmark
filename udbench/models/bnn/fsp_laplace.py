from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import math
from typing import Any, Callable, Tuple

import logging

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
from flax import linen as nn
from flax.training import train_state
import distrax

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


def _first_primes(n: int) -> list[int]:
    if n <= 0:
        return []
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        limit = int(candidate ** 0.5)
        for p in primes:
            if p > limit:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def _van_der_corput(n: int, base: int, start_index: int = 1) -> np.ndarray:
    seq = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        idx = start_index + i
        denom = 1.0
        value = 0.0
        while idx > 0:
            idx, rem = divmod(idx, base)
            denom *= float(base)
            value += rem / denom
        seq[i] = value
    return seq


def _halton_sequence(n: int, d: int, start_index: int = 1) -> np.ndarray:
    bases = _first_primes(d)
    cols = [_van_der_corput(n, base, start_index=start_index) for base in bases]
    return np.stack(cols, axis=1)


@dataclass
class TabularBNNFSPLaplaceRegressor(JaxBNNTuningMixin, BaseUDRegressor):
    tuning_model_key = "bnn_fsp_laplace"

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
    fsp_reg_weight: float = 1.0
    kernel_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None
    prior_mean_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    kernel_lengthscale: float = 1.0
    kernel_variance: float = 1.0
    kernel_jitter: float = 1e-5
    n_context_train: int = 64
    n_context_laplace: int = 256
    context_train_strategy: str = "train_data"
    context_laplace_strategy: str = "halton_train_bounds"
    context_train_points: Any = None
    context_laplace_points: Any = None
    context_bounds_padding: float = 0.0
    context_resample_each_step: bool = True
    context_halton_start_index: int = 1
    laplace_rank: int = 128
    laplace_data_size: int | None = None
    laplace_min_eig: float = 1e-8
    prior_var_tolerance: float = 1e-6
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

    def _default_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        ls = jnp.asarray(self.kernel_lengthscale, dtype=x1.dtype)
        if ls.ndim == 0:
            ls = jnp.ones((x1.shape[1],), dtype=x1.dtype) * ls
        ls = jnp.clip(ls, a_min=self.np_eps)
        var = jnp.asarray(self.kernel_variance, dtype=x1.dtype)
        diff = (x1[:, None, :] - x2[None, :, :]) / ls[None, None, :]
        r = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + self.np_eps)
        scaled = jnp.sqrt(jnp.asarray(3.0, dtype=x1.dtype)) * r

        # Use the same ARD scaling for the additive linear term.
        x1_scaled = x1 / ls[None, :]
        x2_scaled = x2 / ls[None, :]
        linear = jnp.matmul(x1_scaled, x2_scaled.T)

        matern32 = (1.0 + scaled) * jnp.exp(-scaled)
        return (var**2) * (matern32 + linear)

    def _kernel_matrix(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        if self.kernel_fn is None:
            return self._default_kernel(x1, x2)
        return self.kernel_fn(x1, x2)

    def _prior_mean(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.prior_mean_fn is None:
            return jnp.zeros((x.shape[0],), dtype=x.dtype)
        out = self.prior_mean_fn(x)
        out = jnp.asarray(out, dtype=x.dtype).reshape((-1,))
        if out.shape[0] != x.shape[0]:
            raise ValueError(
                f"prior_mean_fn must return shape ({x.shape[0]},), got {tuple(out.shape)}."
            )
        return out

    def _context_bounds(self, x_ref: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        lo = jnp.min(x_ref, axis=0)
        hi = jnp.max(x_ref, axis=0)
        span = hi - lo
        pad = float(self.context_bounds_padding)
        lo = lo - pad * span
        hi = hi + pad * span
        return lo, hi

    def _sample_context_points(
        self,
        key: Any,
        x_ref: jnp.ndarray,
        *,
        n_context: int,
        strategy: str,
        fixed_points: Any,
        step_idx: int,
    ) -> jnp.ndarray:
        x_ref = jnp.asarray(x_ref, dtype=jnp.float32)
        n_context = int(n_context)
        if n_context <= 0:
            raise ValueError("n_context must be > 0.")

        if fixed_points is not None:
            ctx = jnp.asarray(fixed_points, dtype=jnp.float32)
            if ctx.ndim != 2:
                raise ValueError("context points must have shape (n_context, input_dim).")
            if ctx.shape[1] != x_ref.shape[1]:
                raise ValueError(
                    f"context point dim mismatch: expected {x_ref.shape[1]}, got {ctx.shape[1]}."
                )
            if ctx.shape[0] == n_context:
                return ctx
            idx = jax.random.randint(key, (n_context,), 0, ctx.shape[0])
            return ctx[idx]

        strat = str(strategy).lower()
        if strat in {"train_data", "data", "dataset"}:
            idx = jax.random.randint(key, (n_context,), 0, x_ref.shape[0])
            return x_ref[idx]

        if strat in {"uniform_train_bounds", "uniform_bounds", "uniform"}:
            lo, hi = self._context_bounds(x_ref)
            u = jax.random.uniform(key, (n_context, x_ref.shape[1]), minval=0.0, maxval=1.0)
            return lo[None, :] + u * (hi - lo)[None, :]

        if strat in {"halton_train_bounds", "halton_bounds", "halton"}:
            lo, hi = self._context_bounds(x_ref)
            start = int(self.context_halton_start_index) + int(step_idx) * n_context
            seq = jnp.asarray(
                _halton_sequence(n_context, x_ref.shape[1], start_index=start),
                dtype=jnp.float32,
            )
            return lo[None, :] + seq * (hi - lo)[None, :]

        raise ValueError(
            f"Unknown context sampling strategy={strategy!r}. "
            f"Use one of: train_data, uniform_train_bounds, halton_train_bounds."
        )

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
            raise ValueError(f"Unsupported backbone {self.backbone!r} for FSP-Laplace.")

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
            scale = jax.nn.softplus(out_map[..., 1]) + float(self.np_eps)
            return jnp.clip(scale**2, a_min=float(self.np_eps))

        return model_fn_mu, mean_head_params, map_aleatoric_var_fn

    def _fsp_laplace_val_score(
        self,
        candidate: dict[str, Any],
        *,
        temperature: float,
        sample_key: Any,
    ) -> float:
        flat_mean = candidate["flat_mean"]
        posterior_sqrt = candidate["posterior_sqrt"]
        unravel_fn = candidate["unravel_fn"]
        model_fn_mu = candidate["model_fn_mu"]
        X_val = candidate["X_val"]
        y_val = candidate["y_val"].reshape(-1)

        score_num_samples = max(32, min(int(self.n_members), 64))
        rank = int(posterior_sqrt.shape[1])
        if rank > 0:
            z = jax.random.normal(sample_key, (score_num_samples, rank), dtype=flat_mean.dtype)
            scale = jnp.sqrt(jnp.asarray(temperature, dtype=flat_mean.dtype))
            flat_samples = flat_mean[None, :] + scale * (z @ posterior_sqrt.T)
        else:
            flat_samples = jnp.broadcast_to(flat_mean[None, :], (score_num_samples, flat_mean.shape[0]))

        params_samples = jax.vmap(unravel_fn)(flat_samples)
        mu_samples = jax.vmap(lambda p: model_fn_mu(X_val, p))(params_samples)
        mean = jnp.mean(mu_samples, axis=0).reshape(-1)
        epi = jnp.var(mu_samples, axis=0).reshape(-1)
        ale = candidate["map_aleatoric_var_fn"](X_val).reshape(-1)
        total = jnp.clip(ale + epi, a_min=float(self.np_eps))
        nll = 0.5 * (
            jnp.log(2.0 * jnp.pi * total)
            + ((y_val - mean) ** 2) / total
        )
        return float(jnp.mean(nll))

    def _build_laplax_candidate(
        self,
        model_bundle: tuple[Any, Any, Any, bool],
        x_all: jnp.ndarray,
        y_all: jnp.ndarray,
        rng: Any,
        perm: jnp.ndarray,
        *,
        val_fraction: float,
    ) -> dict[str, Any]:
        n_total_all = int(x_all.shape[0])
        n_val = min(max(1, int(float(val_fraction) * n_total_all)), n_total_all - 1)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        x_train = x_all[tr_idx]
        x_val = x_all[val_idx]
        y_val = y_all[val_idx]

        model_fn_mu, mean_head_params, map_aleatoric_var_fn = self._make_last_layer_functions(model_bundle)

        rng, cov_ctx_key = jax.random.split(rng)
        cov_ctx = self._sample_context_points(
            cov_ctx_key,
            x_train,
            n_context=self.n_context_laplace,
            strategy=self.context_laplace_strategy,
            fixed_points=self.context_laplace_points,
            step_idx=0,
        )
        k_cc = self._kernel_matrix(cov_ctx, cov_ctx)
        k_cc = k_cc + float(self.kernel_jitter) * jnp.eye(k_cc.shape[0], dtype=k_cc.dtype)

        eigvals_k, eigvecs_k = jnp.linalg.eigh(k_cc)
        order = jnp.argsort(eigvals_k)[::-1]
        eigvals_k = eigvals_k[order]
        eigvecs_k = eigvecs_k[:, order]

        valid = eigvals_k > float(self.laplace_min_eig)
        n_valid = int(jnp.sum(valid))
        if n_valid <= 0:
            raise RuntimeError("Kernel Gram matrix has no positive eigenvalues above laplace_min_eig.")
        rank = min(int(self.laplace_rank), n_valid)
        eigvals_k = jnp.clip(eigvals_k[:rank], a_min=float(self.laplace_min_eig))
        eigvecs_k = eigvecs_k[:, :rank]
        l_mat = eigvecs_k / jnp.sqrt(eigvals_k[None, :])

        flat_params, unravel_fn = ravel_pytree(mean_head_params)
        if not bool(jnp.all(jnp.isfinite(flat_params))):
            logger.warning(
                "FSP-Laplace: last-layer parameters contain non-finite values (training "
                "diverged); Jacobian contributions will be zeroed — posterior will be prior-only."
            )

        def mu_context_flat(flat_w: jnp.ndarray) -> jnp.ndarray:
            params = unravel_fn(flat_w)
            return model_fn_mu(cov_ctx, params)

        j_ctx = jax.jacrev(mu_context_flat)(flat_params)
        # Diverged model weights produce NaN Jacobians; zero them so the SVD below
        # stays finite (zero m_mat → zero singular values → prior-only a_mat).
        j_ctx = jnp.nan_to_num(j_ctx, nan=0.0, posinf=0.0, neginf=0.0)
        j_ctx_rms = jnp.sqrt(jnp.mean(j_ctx ** 2) + 1e-12)
        j_ctx = j_ctx / jnp.maximum(j_ctx_rms, 1.0)
        m_mat = j_ctx.T @ l_mat

        u_m, s_m, _ = jnp.linalg.svd(m_mat, full_matrices=False)
        a_mat = jnp.diag(s_m**2)

        n_total = x_train.shape[0]
        if self.laplace_data_size is None or int(self.laplace_data_size) >= n_total:
            x_hess = x_train
        else:
            rng, hess_key = jax.random.split(rng)
            idx_h = jax.random.randint(hess_key, (int(self.laplace_data_size),), 0, n_total)
            x_hess = x_train[idx_h]

        var_h = map_aleatoric_var_fn(x_hess)
        # NaN var_h (from NaN model weights) would make h_nll = NaN and poison the
        # outer product; replace with zero so the Hessian contribution is suppressed.
        h_nll = jnp.nan_to_num(
            1.0 / var_h, nan=0.0, posinf=float(1.0 / self.np_eps), neginf=0.0
        )
        h_nll = h_nll * (float(n_total) / float(x_hess.shape[0]))

        def mu_hess_flat(flat_w: jnp.ndarray) -> jnp.ndarray:
            params = unravel_fn(flat_w)
            return model_fn_mu(x_hess, params)

        _, lin_mu_hess = jax.linearize(mu_hess_flat, flat_params)
        j_proj = jax.vmap(lin_mu_hess, in_axes=1, out_axes=1)(u_m)

        # Compute j_proj.T @ diag(h_nll) @ j_proj without float32 overflow.
        # h_nll = 1/var_h is bounded by 1/np_eps ≤ 1e6 (eps floor in _natural_positive_link).
        # To keep h_nll * j_proj[i]^2 * n_hess within float32 range we need |j_proj| ≤ ~1e9.
        # For well-trained networks this holds easily; diverged models (extreme features) are
        # clipped — their Hessian contribution dominates a_mat regardless, so the truncation
        # has negligible effect on the posterior.
        # Safe threshold: float32_max^(1/4) ≈ 4.3e9 guarantees the outer-product sum stays
        # below float32_max even with n_hess=5000 and h_nll_max=1e6.
        # nan_to_num must run before clip: jnp.clip propagates NaN unchanged (IEEE 754),
        # so a diverged model with NaN weights would still produce NaN in the outer product.
        j_clip = float(jnp.finfo(j_proj.dtype).max) ** 0.25
        j_proj_safe = jnp.clip(
            jnp.nan_to_num(j_proj, nan=0.0, posinf=j_clip, neginf=-j_clip),
            a_min=-j_clip, a_max=j_clip,
        )
        a_mat_with_hess = a_mat + j_proj_safe.T @ (h_nll[:, None] * j_proj_safe)
        a_mat_with_hess = 0.5 * (a_mat_with_hess + a_mat_with_hess.T)
        if bool(jnp.any(~jnp.isfinite(a_mat_with_hess))):
            logger.warning(
                "FSP-Laplace: Hessian contribution overflowed even after clipping; "
                "falling back to prior-only curvature approximation."
            )
            a_mat_fallback = 0.5 * (a_mat + a_mat.T)
            if not bool(jnp.all(jnp.isfinite(a_mat_fallback))):
                # a_mat itself is non-finite (e.g. SVD of degenerate kernel returned NaN).
                # Use a scaled identity so eigdecomp below is always well-posed.
                logger.warning(
                    "FSP-Laplace: prior curvature matrix is also non-finite; "
                    "falling back to scaled identity."
                )
                a_mat = float(self.laplace_min_eig) * jnp.eye(
                    a_mat.shape[0], dtype=a_mat.dtype
                )
            else:
                a_mat = a_mat_fallback
        else:
            a_mat = a_mat_with_hess

        eigvals_a, u_a = jnp.linalg.eigh(a_mat)
        eigvals_a = jnp.clip(eigvals_a, a_min=float(self.laplace_min_eig))
        s_full = u_m @ u_a @ jnp.diag(1.0 / jnp.sqrt(eigvals_a))

        prior_diag = jnp.clip(jnp.diag(k_cc), a_min=float(self.laplace_min_eig))
        k_idx = 0
        for cur_k in range(s_full.shape[1] + 1):
            s_tail = s_full[:, cur_k:]
            if s_tail.shape[1] == 0:
                k_idx = cur_k
                break
            proj_tail = j_ctx @ s_tail
            post_diag = jnp.sum(proj_tail * proj_tail, axis=1)
            cond = bool(jnp.all(post_diag <= prior_diag + float(self.prior_var_tolerance)))
            if cond:
                k_idx = cur_k
                break
            k_idx = cur_k

        posterior_sqrt = s_full[:, k_idx:]
        return {
            "model_fn_mu": model_fn_mu,
            "flat_mean": flat_params,
            "unravel_fn": unravel_fn,
            "posterior_sqrt": posterior_sqrt,
            "map_aleatoric_var_fn": map_aleatoric_var_fn,
            "X_val": x_val,
            "y_val": y_val,
            "selected_val_fraction": float(val_fraction),
            "cov_context_points": cov_ctx,
            "cov_prior_diag": prior_diag,
            "laplace_rank": int(rank),
            "laplace_rank_truncated": int(posterior_sqrt.shape[1]),
        }

    def train(
        self,
        X,
        y,
        tune: bool = False,
        **kwargs: Any,
    ):
        del tune
        kwargs = apply_jax_bnn_runtime_overrides(self, kwargs)

        def gaussian_nll(pred, yb):
            yb = yb.squeeze(-1) if yb.ndim == 3 else yb
            mu = pred[..., 0]
            raw = pred[..., 1]
            scale = jax.nn.softplus(raw) + 1e-6
            dist = distrax.Normal(loc=mu, scale=scale)
            return -jnp.mean(dist.log_prob(yb))

        @partial(jax.jit, static_argnames=("has_batchnorm",))
        def train_step(state, batch_stats, rng, xb, yb, cb, n_total, has_batchnorm: bool):
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
                variance_penalty = variance_head_regularization(
                    pred,
                    target_variance=self.target_variance_,
                    np_loss=self.NPLoss,
                    eps=self.np_eps,
                    clip_multiplier=self.train_variance_clip_multiplier,
                    link=self.np_link_fn,
                )
                if self.NPLoss:
                    nll_batch = natural_gaussian_nll(pred, yb, link=self.np_link_fn, eps=self.np_eps)
                else:
                    nll_batch = gaussian_nll(pred, yb)
                nll_term = n_total * (nll_batch + float(self.variance_penalty_weight) * variance_penalty)

                pred_ctx, _ = apply_model_with_batch_stats(
                    state.apply_fn,
                    params=params,
                    batch_stats=next_batch_stats,
                    has_batchnorm=has_batchnorm,
                    x=cb,
                    dropout_deterministic=True,
                    use_running_average=True,
                    update_batch_stats=False,
                )
                if self.NPLoss:
                    _, _, mu_ctx, _ = natural_gaussian_stats_from_raw(
                        pred_ctx, link=self.np_link_fn, eps=self.np_eps
                    )
                else:
                    mu_ctx = pred_ctx[..., 0]
                centered = mu_ctx - self._prior_mean(cb)

                k_cc = self._kernel_matrix(cb, cb)
                # Eigendecomposition instead of linalg.solve: the linear kernel term makes
                # k_cc rank-deficient when n_context > n_features (rank ≤ n_features), giving
                # condition numbers above float32's 1/eps ≈ 8e6. jnp.linalg.solve silently
                # returns NaN for such ill-conditioned matrices, corrupting weights permanently.
                # k_cc has no dependence on model params, so stop_gradient on the decomposition
                # is mathematically exact and avoids the unstable eigh backward pass.
                _eigvals_cc, _eigvecs_cc = jnp.linalg.eigh(k_cc)
                _eps_f32 = float(jnp.finfo(k_cc.dtype).eps)
                _eig_floor = jnp.maximum(
                    jnp.max(_eigvals_cc) * 10.0 * _eps_f32,
                    float(self.kernel_jitter),
                )
                _eigvals_cc_s = jax.lax.stop_gradient(jnp.clip(_eigvals_cc, a_min=_eig_floor))
                _eigvecs_cc_s = jax.lax.stop_gradient(_eigvecs_cc)
                _proj = _eigvecs_cc_s.T @ centered
                rkhs_term = 0.5 * jnp.sum(_proj**2 / _eigvals_cc_s)

                return nll_term + float(self.fsp_reg_weight) * rkhs_term, next_batch_stats

            (loss, next_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, next_batch_stats, rng, loss

        self._fit_jax_bnn_standardization(X, y)
        X = self._transform_jax_bnn_inputs(X)
        y = self._transform_jax_bnn_targets(y)
        self.target_variance_ = target_variance_from_targets(y, eps=self.np_eps)
        n_total = int(X.shape[0])

        has_batchnorm = self.norm == "batchnorm"
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
            {"params": init_rng, "dropout": drop_rng},
            x_dummy,
            True,
            False,
        )
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

        effective_batch_size = min(int(self.batch_size), n_total)
        n_batches = max(1, (n_total + effective_batch_size - 1) // effective_batch_size)

        step_idx = 0
        train_context_fixed = None
        if not bool(self.context_resample_each_step):
            rng, ctx_key = jax.random.split(rng)
            train_context_fixed = self._sample_context_points(
                ctx_key,
                X,
                n_context=self.n_context_train,
                strategy=self.context_train_strategy,
                fixed_points=self.context_train_points,
                step_idx=0,
            )

        for epoch in range(1, self.n_epochs + 1):
            rng, ep_rng = jax.random.split(rng)
            epoch_idx = jax.random.permutation(ep_rng, n_total)

            epoch_loss = 0.0
            for start in range(0, n_total, effective_batch_size):
                batch_idx = epoch_idx[start : start + effective_batch_size]
                xb, yb = X[batch_idx], y[batch_idx]

                if train_context_fixed is None:
                    rng, ctx_key = jax.random.split(rng)
                    cb = self._sample_context_points(
                        ctx_key,
                        X,
                        n_context=self.n_context_train,
                        strategy=self.context_train_strategy,
                        fixed_points=self.context_train_points,
                        step_idx=step_idx,
                    )
                else:
                    cb = train_context_fixed

                state, batch_stats, rng, loss = train_step(
                    state,
                    batch_stats,
                    rng,
                    xb,
                    yb,
                    cb,
                    float(n_total),
                    has_batchnorm,
                )
                epoch_loss += float(loss)
                step_idx += 1

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
        fallback_state: dict[str, Any] | None = None
        fallback_temperature = float(self.posterior_temperature)
        for val_fraction in candidate_val_fractions:
            laplace_rng, cand_key = jax.random.split(laplace_rng)
            candidate = self._build_laplax_candidate(
                self.model_bundle,
                X,
                y,
                cand_key,
                perm,
                val_fraction=val_fraction,
            )
            if fallback_state is None:
                fallback_state = dict(candidate)
                fallback_temperature = float(candidate_temperatures[0])
            for temperature in candidate_temperatures:
                laplace_rng, score_key = jax.random.split(laplace_rng)
                score = self._fsp_laplace_val_score(
                    candidate,
                    temperature=float(temperature),
                    sample_key=score_key,
                )
                if math.isfinite(score) and score < best_score:
                    best_score = score
                    best_state = dict(candidate)
                    best_temperature = float(temperature)

        if best_state is None:
            if fallback_state is None:
                raise RuntimeError("Failed to build a valid FSP-Laplace state.")
            logger.warning(
                "FSP-Laplace: all candidate scores were non-finite; falling back to first candidate."
            )
            best_state = fallback_state
            best_temperature = fallback_temperature

        self.val_fraction = float(best_state["selected_val_fraction"])
        self.posterior_temperature = float(best_temperature)
        self.laplax_state = {
            "model_fn_mu": best_state["model_fn_mu"],
            "flat_mean": best_state["flat_mean"],
            "unravel_fn": best_state["unravel_fn"],
            "posterior_sqrt": best_state["posterior_sqrt"],
            "selected_val_fraction": float(self.val_fraction),
            "selected_temperature": float(self.posterior_temperature),
            "selection_score_nll": float(best_score),
            "cov_context_points": best_state["cov_context_points"],
            "cov_prior_diag": best_state["cov_prior_diag"],
            "laplace_rank": best_state["laplace_rank"],
            "laplace_rank_truncated": best_state["laplace_rank_truncated"],
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
        FSP-LAPLACE pushforward: returns (E, N, D) = (n_members, N, 2)
        where [:, :, 0] are mu samples and [:, :, 1] is the MAP raw_scale replicated.
        """
        if self.laplax_state is None:
            raise RuntimeError("Laplace state not built. Call train() first.")

        model, state, batch_stats, has_batchnorm = model_bundle
        X = jnp.asarray(X, dtype=jnp.float32)
        E = int(self.n_members)

        model_fn_mu = self.laplax_state["model_fn_mu"]
        flat_mean = self.laplax_state["flat_mean"]
        unravel_fn = self.laplax_state["unravel_fn"]
        posterior_sqrt = self.laplax_state["posterior_sqrt"]

        if self.laplax_rng is None:
            raise RuntimeError("Laplace RNG not initialized. Call train() first.")
        key = self.laplax_rng
        rank = int(posterior_sqrt.shape[1])
        if rank > 0:
            z = jax.random.normal(key, (E, rank), dtype=flat_mean.dtype)
            scale = jnp.sqrt(jnp.asarray(self.posterior_temperature, dtype=flat_mean.dtype))
            flat_samples = flat_mean[None, :] + scale * (z @ posterior_sqrt.T)
        else:
            flat_samples = jnp.broadcast_to(flat_mean[None, :], (E, flat_mean.shape[0]))

        params_samples = jax.vmap(unravel_fn)(flat_samples)
        mu_samples = jax.vmap(lambda p: model_fn_mu(X, p))(params_samples)

        vars_in = {"params": state.params}
        if has_batchnorm and batch_stats is not None:
            vars_in["batch_stats"] = batch_stats
        out_map = model.apply(vars_in, X, True, True)
        if self.NPLoss:
            _, _, _, std_map = natural_gaussian_stats_from_raw(out_map, link=self.np_link_fn, eps=self.np_eps)
            rep = jnp.broadcast_to(std_map[None, :], (E, X.shape[0]))
        else:
            raw_map = out_map[..., 1]
            rep = jnp.broadcast_to(raw_map[None, :], (E, X.shape[0]))

        preds = jnp.stack([mu_samples, rep], axis=-1)
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
        del kwargs
        X = self._transform_jax_bnn_inputs(X)
        preds = self.forward(model, X)
        mu = preds[..., 0]
        return self._inverse_transform_jax_bnn_mean(jnp.mean(mu, axis=0))

    def ud_fit_predict(self, X_train, y_train, X_eval, **kwargs: Any) -> Dict:
        self.train(X_train, y_train, **kwargs)

        X_eval = self._transform_jax_bnn_inputs(X_eval)
        preds = self.forward(self.model_bundle, X_eval)
        mu = preds[..., 0]
        aux = preds[..., 1]

        mean = jnp.mean(mu, axis=0)
        epi = jnp.var(mu, axis=0)

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
