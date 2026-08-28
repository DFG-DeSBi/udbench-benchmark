from __future__ import annotations

import math
from typing import Callable
import warnings
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from scipy.special import ndtr as _norm_cdf

# Local imports
from .GroundTruthFnDist import GroundTruthFnDist, GPPosteriorFnDist
from udbench.datasets import DataSet
from udbench.BaseUDRegressor import BaseUDRegressor


def _gaussian_scores(
    mu: np.ndarray,
    sigma2: np.ndarray,
    y: np.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Point-wise NLL and CRPS for N(μ, σ²) evaluated at y.

    Returns (nll, crps), both shape (n,).
    """
    sigma2 = np.maximum(sigma2, 1e-12)
    sigma  = np.sqrt(sigma2)
    z      = (y - mu) / sigma
    nll    = 0.5 * (np.log(2 * np.pi * sigma2) + z ** 2)
    phi    = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    crps   = sigma * (z * (2 * _norm_cdf(z) - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
    return jnp.asarray(nll), jnp.asarray(crps)

class GTDisentangle:
    def __init__(
            self,
            dataset: DataSet = None,
            X_eval: jnp.ndarray = None,
            y_eval_preds: jnp.ndarray = None,
            GroundTruthFnDist: GroundTruthFnDist = None,
            aleatoric_noise_fn: Callable = None,
            mc_samples: int = 100,
            random_key = 42
    ):
        self.dataset = dataset
        self.X_eval = X_eval
        self.y_eval_preds = y_eval_preds
        self.GroundTruthFnDist = GroundTruthFnDist
        self.aleatoric_noise_fn = aleatoric_noise_fn
        self.mc_samples = mc_samples
        self.random_key = random_key

        if dataset is not None:
            if self.X_eval is None:
                self.X_eval = dataset.X_eval
            if self.GroundTruthFnDist is None:
                if dataset.posterior is None:
                    raise ValueError("Dataset posterior is not initialized.")
                self.GroundTruthFnDist = GPPosteriorFnDist(dataset.posterior)
            if self.aleatoric_noise_fn is None:
                self.aleatoric_noise_fn = dataset.aleatoric_std_fn

    def compute_gt_uncertainty(self):
        """
        Compute ground-truth uncertainty targets at eval points.

        Each target is available in three scoring-rule variants (MSE, NLL, CRPS)
        and three uncertainty scopes (total, epistemic, aleatoric):

          {total,epistemic,aleatoric}_mse
          {total,epistemic,aleatoric}_nll
          {total,epistemic,aleatoric}_crps

        Distributions and observations per scope:
          total      — N(μ, σ²_f + σ²_noise)  scored at y_eval
          epistemic  — N(μ, σ²_f)              scored at y_eval_clean (noiseless f_true)
          aleatoric  — N(f_true, σ²_noise)      scored at y_eval

        MSE-specific decomposition terms (no NLL/CRPS analogues):
          posterior_variance — σ²_f
          squared_bias       — (y_pred - μ)²
          realized_error     — (y_pred - f_true)²
        """
        # Posterior moments 
        if hasattr(self.GroundTruthFnDist, "mean") and hasattr(self.GroundTruthFnDist, "variance"):
            posterior_mean = self.GroundTruthFnDist.mean(self.X_eval)
            posterior_var  = self.GroundTruthFnDist.variance(self.X_eval)
        else:
            f_samples      = self.GroundTruthFnDist.sample(self.X_eval, self.mc_samples, self.random_key)
            posterior_mean = jnp.mean(f_samples, axis=0)
            posterior_var  = jnp.var(f_samples, axis=0)

        # Store
        self._posterior_mean = posterior_mean
        self._posterior_var  = posterior_var

        aleatoric_var = self.aleatoric_noise_fn(self.X_eval) ** 2

        # Arrays
        mu  = np.asarray(posterior_mean, dtype=np.float64).ravel()
        s2e = np.asarray(posterior_var,  dtype=np.float64).ravel()
        s2a = np.asarray(aleatoric_var,  dtype=np.float64).ravel()

        ds      = self.dataset
        y_eval  = np.asarray(ds.y_eval,       dtype=np.float64).ravel() if ds is not None and getattr(ds, "y_eval",       None) is not None else None
        y_clean = np.asarray(ds.y_eval_clean, dtype=np.float64).ravel() if ds is not None and getattr(ds, "y_eval_clean", None) is not None else None

        # MSE targets
        total_mse     = (self.y_eval_preds - posterior_mean) ** 2 + posterior_var + aleatoric_var
        epistemic_mse = (self.y_eval_preds - posterior_mean) ** 2 + posterior_var
        aleatoric_mse = aleatoric_var

        posterior_variance = posterior_var
        squared_bias       = (self.y_eval_preds - posterior_mean) ** 2
        realized_error     = (self.y_eval_preds - jnp.asarray(y_clean)) ** 2 if y_clean is not None else None

        # NLL and CRPS targets 
        total_nll     = total_crps     = None
        epistemic_nll = epistemic_crps = None
        aleatoric_nll = aleatoric_crps = None

        if y_eval is not None:
            total_nll,     total_crps     = _gaussian_scores(mu,      s2e + s2a, y_eval)
        if y_clean is not None:
            epistemic_nll, epistemic_crps = _gaussian_scores(mu,      s2e,       y_clean)
        if y_eval is not None and y_clean is not None:
            aleatoric_nll, aleatoric_crps = _gaussian_scores(y_clean, s2a,       y_eval)

        return {
            # MSE targets
            'total_mse':          total_mse,
            'epistemic_mse':      epistemic_mse,
            'aleatoric_mse':      aleatoric_mse,
            'posterior_variance': posterior_variance,
            'squared_bias':       squared_bias,
            'realized_error':     realized_error,
            # NLL targets
            'total_nll':          total_nll,
            'epistemic_nll':      epistemic_nll,
            'aleatoric_nll':      aleatoric_nll,
            # CRPS targets
            'total_crps':         total_crps,
            'epistemic_crps':     epistemic_crps,
            'aleatoric_crps':     aleatoric_crps,
        }
    
    def approx_excess_risk_decomp(
            self,
            model: BaseUDRegressor,
            mc_samples: int = 100,
            plot: bool = False):
        
        n_eval = self.X_eval.shape[0]

        possible_gt_functions = self.GroundTruthFnDist.sample(
            self.X_eval, mc_samples, self.random_key
        )  # (mc_samples, n_eval)

        estimation_error_samples = []
        approximation_error_samples = []

        total_iters = possible_gt_functions.shape[0]
        progress = 0

        def _print_progress():
            bar_width = 30
            filled = int(bar_width * progress / total_iters)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(f"\rExcess risk decomp: [{bar}] {progress}/{total_iters}", end="")

        best_in_class_function_dist = []

        # Pre-compute per-feature GT PDP grids once
        gt_pdp_cache: dict | None = None
        if plot:
            gt_pdp_cache = self._build_gt_pdp_cache(num_grid_points=31)

        original_model_state = getattr(model, "model", None)

        x_plot = np.asarray(self.X_eval).reshape(-1)
        order = np.argsort(x_plot)
        x_plot = x_plot[order]
        for i, gt_function in enumerate(possible_gt_functions):
            key = jax.random.PRNGKey(self.random_key + i * mc_samples)

            noise = (
                jax.random.normal(key, shape=self.dataset.y_eval.shape)
                * self.aleatoric_noise_fn(self.X_eval)
            )


            gt_samples = gt_function + noise
            print('Training the best in class model for GT function sample', i + 1)
            print('Sample size for training the best in class model:', self.X_eval.shape[0])
            best_in_class_predictor = model.spawn_refit_worker()
            best_in_class_model = best_in_class_predictor.train(self.X_eval, gt_samples)
            best_in_class_function = best_in_class_predictor.predict(best_in_class_model, self.X_eval)
            print()
            best_in_class_performance = jnp.mean((best_in_class_function - gt_function) ** 2)
            model_performance = jnp.mean((self.y_eval_preds - gt_function) ** 2)
            print(f"Best-in-class MSE: {best_in_class_performance:.6f}, Model MSE: {model_performance:.6f}")
            print(f"Best-in-class pearson corr: {jnp.corrcoef(best_in_class_function, gt_function)[0, 1]:.4f}, Model pearson corr: {jnp.corrcoef(self.y_eval_preds, gt_function)[0, 1]:.4f}")
            best_in_class_function_dist.append(best_in_class_function)
            approx_error = (best_in_class_function - gt_function) ** 2
            estim_error = (
                (self.y_eval_preds - gt_function) ** 2
                - (best_in_class_function - gt_function) ** 2 
            )

            approximation_error_samples.append(approx_error)
            estimation_error_samples.append(estim_error)
            progress += 1
            _print_progress()

            if plot:
                self._plot_excess_risk_pdp(
                    iteration_idx=i,
                    gt_samples=gt_samples,
                    best_in_class_model=best_in_class_model,
                    best_in_class_predictor=best_in_class_predictor,
                    model=model,
                    gt_pdp_cache=gt_pdp_cache,
                    original_model_state=original_model_state,
                )


        approximation_error = jnp.mean(jnp.stack(approximation_error_samples), axis=0)
        estimation_error = jnp.mean(jnp.stack(estimation_error_samples), axis=0)
        if best_in_class_function_dist:
            mean_best_in_class_function = jnp.mean(
                jnp.stack(best_in_class_function_dist), axis=0
            )
        else:
            mean_best_in_class_function = None
        if hasattr(model, "model"):
            model.model = original_model_state
        print()
        return estimation_error, approximation_error, mean_best_in_class_function, jnp.mean(possible_gt_functions, axis=0)

    def _build_gt_pdp_cache(self, num_grid_points: int = 31) -> dict:
        """Pre-compute per-feature GT posterior-mean PDP grids (once, outside the iteration loop)."""
        X_eval_np = np.asarray(self.X_eval, dtype=np.float64)
        if X_eval_np.ndim == 1:
            X_eval_np = X_eval_np.reshape(-1, 1)
        n_features = X_eval_np.shape[1]

        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        X_mod = X_eval_np.copy()
        for d in range(n_features):
            feat_vals = X_eval_np[:, d]
            x_grid = np.unique(
                np.quantile(feat_vals, np.linspace(0.0, 1.0, num_grid_points))
            )
            pdp_gt = np.empty(x_grid.shape[0], dtype=np.float64)
            for g_idx, g_val in enumerate(x_grid):
                X_mod[:, d] = g_val
                gt_vals = np.asarray(
                    self.GroundTruthFnDist.mean(jnp.asarray(X_mod)), dtype=np.float64
                ).reshape(-1)
                pdp_gt[g_idx] = float(np.mean(gt_vals))
            X_mod[:, d] = X_eval_np[:, d]  # restore column
            cache[d] = (x_grid, pdp_gt)
        return cache

    def _plot_excess_risk_pdp(
            self,
            iteration_idx: int,
            gt_samples: jnp.ndarray,
            best_in_class_model,
            best_in_class_predictor: BaseUDRegressor,
            model: BaseUDRegressor,
            gt_pdp_cache: dict,
            original_model_state=None,
            num_grid_points: int = 31,
    ) -> None:
        """PDP plots for one excess-risk-decomp iteration.

        For each input feature one subplot shows:
          - Data scatter      : (X_eval[:, d], gt_samples) — the noisy observations
          - GT function PDP   : posterior-mean PDP (pre-computed, same for every iteration)
          - Model PDP         : marginalized predictions of the original model
          - Best-in-class PDP : marginalized predictions of the best-in-class model
        """
        X_eval_np = np.asarray(self.X_eval, dtype=np.float64)
        if X_eval_np.ndim == 1:
            X_eval_np = X_eval_np.reshape(-1, 1)
        n_features = X_eval_np.shape[1]

        gt_samples_np = np.asarray(gt_samples, dtype=np.float64).reshape(-1)

        feature_names = list(
            getattr(self.dataset, "feature_names", [f"x{i + 1}" for i in range(n_features)])
        )

        n_cols = min(3, n_features)
        n_rows = math.ceil(n_features / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5.5 * n_cols, 4.5 * n_rows),
            squeeze=False,
        )
        axes_flat = axes.reshape(-1)

        has_original_model = original_model_state is not None

        X_mod = X_eval_np.copy()
        for d in range(n_features):
            ax = axes_flat[d]
            feat_vals = X_eval_np[:, d]

            # --- data scatter ---
            ax.scatter(
                feat_vals,
                gt_samples_np,
                s=14, alpha=0.30, color="#7d8597", edgecolors="none",
                label="Data (gt + noise)" if d == 0 else None,
                zorder=1,
            )

            # --- GT posterior-mean PDP (pre-computed) ---
            x_grid_gt, pdp_gt = gt_pdp_cache[d]
            ax.plot(
                x_grid_gt, pdp_gt,
                color="black", lw=2.0,
                label="GT function (PDP)" if d == 0 else None,
                zorder=3,
            )

            # --- model & BIC PDP ---
            x_grid = np.unique(
                np.quantile(feat_vals, np.linspace(0.0, 1.0, num_grid_points))
            )
            pdp_model = np.empty(x_grid.shape[0], dtype=np.float64)
            pdp_bic = np.empty(x_grid.shape[0], dtype=np.float64)

            for g_idx, g_val in enumerate(x_grid):
                X_mod[:, d] = g_val
                X_mod_jnp = jnp.asarray(X_mod)

                if has_original_model:
                    pdp_model[g_idx] = float(
                        np.mean(np.asarray(model.predict(original_model_state, X_mod_jnp)).reshape(-1))
                    )
                pdp_bic[g_idx] = float(
                    np.mean(
                        np.asarray(
                            best_in_class_predictor.predict(best_in_class_model, X_mod_jnp)
                        ).reshape(-1)
                    )
                )
            X_mod[:, d] = X_eval_np[:, d]  # restore column

            if has_original_model:
                ax.plot(
                    x_grid, pdp_model,
                    color="#e63946", lw=2.0, linestyle="--",
                    label="Model PDP" if d == 0 else None,
                    zorder=4,
                )
            ax.plot(
                x_grid, pdp_bic,
                color="#2196f3", lw=2.0, linestyle="-.",
                label="Best-in-class PDP" if d == 0 else None,
                zorder=4,
            )

            ax.set_xlabel(feature_names[d])
            ax.set_title(f"PDP: {feature_names[d]}", fontsize=11)
            ax.grid(alpha=0.20, linestyle=":")
            if d == 0:
                ax.legend(fontsize=8, frameon=False)

        for unused_ax in axes_flat[n_features:]:
            unused_ax.set_visible(False)

        fig.suptitle(
            f"Excess Risk Decomp PDP — iteration {iteration_idx + 1} / {self.mc_samples}",
            fontsize=13,
        )
        fig.tight_layout()
        plt.show()
