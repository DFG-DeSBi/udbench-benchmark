from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as npd
import optax
from flax import nnx
from gpjax.fit import DEFAULT_BIJECTION, transform
from scipy.stats import pearsonr
from tqdm.auto import trange


class GPSpecParseError(ValueError):
    """Raised when a kernel or mean expression violates the strict DSL."""


_KERNEL_CONSTRUCTORS: dict[str, type[gpx.kernels.AbstractKernel]] = {
    "Matern12": gpx.kernels.Matern12,
    "Matern32": gpx.kernels.Matern32,
    "Matern52": gpx.kernels.Matern52,
    "RBF": gpx.kernels.RBF,
    "RationalQuadratic": gpx.kernels.RationalQuadratic,
    "PoweredExponential": gpx.kernels.PoweredExponential,
    "Polynomial": gpx.kernels.Polynomial,
    "Linear": gpx.kernels.Linear,
    "ArcCosine": gpx.kernels.ArcCosine,
    "Periodic": gpx.kernels.Periodic,
    "White": gpx.kernels.White,
}

_MEAN_CONSTRUCTORS: dict[str, type[gpx.mean_functions.AbstractMeanFunction]] = {
    "Zero": gpx.mean_functions.Zero,
    "Constant": gpx.mean_functions.Constant,
}


@dataclass(frozen=True)
class PriorSliceData:
    """One one-dimensional prior effect curve used for ALE-style plotting."""

    feature_index: int
    x_values: np.ndarray
    fixed_values: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    samples: np.ndarray

    @property
    def prior_mean(self) -> np.ndarray:
        """Return the prior mean for compatibility with earlier callers."""

        return self.mean

    @property
    def prior_std(self) -> np.ndarray:
        """Return the prior standard deviation for compatibility with earlier callers."""

        return self.std

    @property
    def prior_samples(self) -> np.ndarray:
        """Return prior samples for compatibility with earlier callers."""

        return self.samples


@dataclass(frozen=True)
class PosteriorSliceData:
    """One one-dimensional latent-posterior effect curve used for ALE-style plotting."""

    feature_index: int
    x_values: np.ndarray
    fixed_values: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    samples: np.ndarray

    @property
    def posterior_mean(self) -> np.ndarray:
        """Return the posterior mean for compatibility with plotting helpers."""

        return self.mean

    @property
    def posterior_std(self) -> np.ndarray:
        """Return the posterior standard deviation for compatibility with plotting helpers."""

        return self.std

    @property
    def posterior_samples(self) -> np.ndarray:
        """Return posterior samples for compatibility with plotting helpers."""

        return self.samples


@dataclass(frozen=True)
class ParameterPriorConfig:
    """Data-informed priors attached directly to the GPJax hyperparameter tree."""

    feature_scales: np.ndarray
    feature_spans: np.ndarray
    target_mean: float
    target_std: float
    target_variance: float
    target_obs_stddev: float
    log_spread: float
    strength: float

    @property
    def scale_factor(self) -> float:
        """Return the multiplicative spread corresponding to ``log_spread``."""

        return float(np.exp(self.log_spread))

    @property
    def positive_log_scale(self) -> float:
        """Return the LogNormal scale implied by the configured prior strength."""

        return float(self.log_spread / np.sqrt(self.strength))

    @property
    def real_scale(self) -> float:
        """Return the Normal scale used for unconstrained real-valued parameters."""

        return float(max(self.target_std, 1e-3) * self.scale_factor / np.sqrt(self.strength))


@dataclass
class GPSpec:
    """Central GP hub holding a strict prior specification and fitted GP state."""

    kernel_expr: str
    mean_expr: str
    train_data: gpx.Dataset | None = field(default=None, init=False, repr=False)
    prior: gpx.gps.Prior | None = field(default=None, init=False, repr=False)
    likelihood: gpx.likelihoods.Gaussian | None = field(default=None, init=False, repr=False)
    posterior: Any | None = field(default=None, init=False, repr=False)
    is_fitted: bool = field(default=False, init=False)
    last_optimization_summary: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    parameter_prior_strength: float = field(default=1.0, init=False, repr=False)
    parameter_prior_scale_factor: float = field(default=5.0, init=False, repr=False)

    @staticmethod
    def resolve_kernel_expr(expr: str) -> gpx.kernels.AbstractKernel:
        """Resolve a strict kernel expression into a concrete GPJax kernel."""

        resolver = _StrictExpressionResolver(
            expr_kind="kernel",
            constructors=_KERNEL_CONSTRUCTORS,
        )
        return resolver.resolve(expr)

    @staticmethod
    def resolve_mean_expr(expr: str) -> gpx.mean_functions.AbstractMeanFunction:
        """Resolve a strict mean expression into a concrete GPJax mean function."""

        resolver = _StrictExpressionResolver(
            expr_kind="mean",
            constructors=_MEAN_CONSTRUCTORS,
        )
        return resolver.resolve(expr)

    def build_kernel(self) -> gpx.kernels.AbstractKernel:
        """Build the kernel described by ``kernel_expr``."""

        return self.resolve_kernel_expr(self.kernel_expr)

    def build_mean_function(self) -> gpx.mean_functions.AbstractMeanFunction:
        """Build the mean function described by ``mean_expr``."""

        mean_function = self.resolve_mean_expr(self.mean_expr)
        _parameterize_mean_function_constants(mean_function)
        return mean_function

    def build_prior(self) -> gpx.gps.Prior:
        """Build the GP prior from the stored kernel and mean expressions."""

        return gpx.gps.Prior(
            mean_function=self.build_mean_function(),
            kernel=self.build_kernel(),
        )

    def build_posterior(
        self,
        train_data: gpx.Dataset,
        *,
        obs_stddev: float = 1.0,
    ) -> Any:
        """Build the conjugate posterior implied by the stored prior specification."""

        prior = self.build_prior()
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=train_data.n,
            obs_stddev=obs_stddev,
        )
        return prior * likelihood

    def fit(
        self,
        X_or_dataset: Any,
        y: Any | None = None,
        *,
        obs_stddev: float = 1.0,
    ) -> GPSpec:
        """Condition the current prior on a dataset using a Gaussian likelihood.
        """

        train_data = _coerce_training_dataset(X_or_dataset, y)
        posterior = self.build_posterior(train_data, obs_stddev=obs_stddev)
        parameter_prior = _build_parameter_prior_config(
            train_data=train_data,
            strength=self.parameter_prior_strength,
            scale_factor=self.parameter_prior_scale_factor,
        )
        _attach_parameter_priors_to_model(
            posterior,
            parameter_prior=parameter_prior,
        )
        self._store_fitted_state(
            train_data=train_data,
            posterior=posterior,
            optimization_summary=None,
        )
        return self

    def optimize_hyperparameters(
        self,
        X_or_dataset: Any | None = None,
        y: Any | None = None,
        *,
        obs_stddev: float | None = None,
        adam_learning_rate: float = 0.05,
        adam_num_iters: int = 250,
        lbfgs_num_iters: int = 100,
        use_lbfgs_refinement: bool = True,
        gradient_clip_norm: float = 1.0,
        parameter_prior_strength: float | None = None,
        parameter_prior_scale_factor: float | None = None,
        lengthscale_prior_strength: float | None = None,
        lengthscale_prior_scale_factor: float | None = None,
        verbose: bool = False,
    ) -> GPSpec:
        """Optimize GP hyperparameters with Adam warm start and L-BFGS refinement.

        The optimization target is the negative conjugate marginal log-likelihood
        plus the negative log density induced by GPJax parameter-attached priors.
        Priors are centered on data-informed scales and attached to every trainable
        GPJax hyperparameter.
        """

        train_data = _coerce_optional_training_dataset(self, X_or_dataset, y)
        initial_obs_stddev = _resolve_initial_obs_stddev(self, obs_stddev)
        _validate_optimization_config(
            adam_num_iters=adam_num_iters,
            lbfgs_num_iters=lbfgs_num_iters,
            adam_learning_rate=adam_learning_rate,
            gradient_clip_norm=gradient_clip_norm,
            use_lbfgs_refinement=use_lbfgs_refinement,
        )
        resolved_prior_strength, resolved_prior_scale_factor = _resolve_parameter_prior_settings(
            current_strength=self.parameter_prior_strength,
            current_scale_factor=self.parameter_prior_scale_factor,
            parameter_prior_strength=parameter_prior_strength,
            parameter_prior_scale_factor=parameter_prior_scale_factor,
            lengthscale_prior_strength=lengthscale_prior_strength,
            lengthscale_prior_scale_factor=lengthscale_prior_scale_factor,
        )
        parameter_prior = _build_parameter_prior_config(
            train_data=train_data,
            strength=resolved_prior_strength,
            scale_factor=resolved_prior_scale_factor,
        )
        self.parameter_prior_strength = float(resolved_prior_strength)
        self.parameter_prior_scale_factor = float(resolved_prior_scale_factor)

        best_fallback: dict[str, Any] | None = None
        attempt_errors: list[str] = []

        for attempt_index, initialization in enumerate(
            _build_retry_initializations(train_data, initial_obs_stddev),
            start=1,
        ):
            posterior = self.build_posterior(train_data, obs_stddev=initial_obs_stddev)
            _apply_initialization_to_posterior(
                posterior=posterior,
                train_data=train_data,
                initialization=initialization,
            )
            _attach_parameter_priors_to_model(
                posterior,
                parameter_prior=parameter_prior,
            )

            try:
                posterior, adam_history = _run_adam_optimization(
                    posterior=posterior,
                    train_data=train_data,
                    learning_rate=adam_learning_rate,
                    num_iters=adam_num_iters,
                    gradient_clip_norm=gradient_clip_norm,
                    parameter_prior=parameter_prior,
                    verbose=verbose,
                )
                adam_map_objective = _evaluate_negative_conjugate_map_objective(
                    posterior,
                    train_data,
                    parameter_prior=parameter_prior,
                )
                _ensure_model_parameters_are_finite(posterior)
            except Exception as exc:
                attempt_errors.append(
                    _format_attempt_error(
                        attempt_index=attempt_index,
                        attempt_name=initialization["name"],
                        stage="adam",
                        exc=exc,
                    )
                )
                continue

            if not use_lbfgs_refinement:
                optimization_summary = _build_optimization_summary(
                    attempt_index=attempt_index,
                    attempt_name=initialization["name"],
                    strategy="adam",
                    initial_obs_stddev=initial_obs_stddev,
                    adam_history=adam_history,
                    lbfgs_history=None,
                    adam_final_map_objective=adam_map_objective,
                    final_map_objective=adam_map_objective,
                    parameter_prior_strength=self.parameter_prior_strength,
                    parameter_prior_scale_factor=self.parameter_prior_scale_factor,
                    adam_model=posterior,
                    final_model=posterior,
                    train_data=train_data,
                )
                self._store_fitted_state(
                    train_data=train_data,
                    posterior=posterior,
                    optimization_summary=optimization_summary,
                )
                return self

            fallback_summary = _build_optimization_summary(
                attempt_index=attempt_index,
                attempt_name=initialization["name"],
                strategy="adam",
                initial_obs_stddev=initial_obs_stddev,
                adam_history=adam_history,
                lbfgs_history=None,
                adam_final_map_objective=adam_map_objective,
                final_map_objective=adam_map_objective,
                parameter_prior_strength=self.parameter_prior_strength,
                parameter_prior_scale_factor=self.parameter_prior_scale_factor,
                adam_model=posterior,
                final_model=posterior,
                train_data=train_data,
            )
            if best_fallback is None or adam_map_objective < best_fallback["objective"]:
                best_fallback = {
                    "objective": adam_map_objective,
                    "posterior": posterior,
                    "summary": fallback_summary,
                }

            try:
                refined_posterior, lbfgs_history = _run_lbfgs_optimization(
                    posterior=posterior,
                    train_data=train_data,
                    num_iters=lbfgs_num_iters,
                    parameter_prior=parameter_prior,
                    verbose=verbose,
                )
                final_map_objective = _evaluate_negative_conjugate_map_objective(
                    refined_posterior,
                    train_data,
                    parameter_prior=parameter_prior,
                )
                _ensure_model_parameters_are_finite(refined_posterior)
            except Exception as exc:
                attempt_errors.append(
                    _format_attempt_error(
                        attempt_index=attempt_index,
                        attempt_name=initialization["name"],
                        stage="lbfgs",
                        exc=exc,
                    )
                )
                continue

            optimization_summary = _build_optimization_summary(
                attempt_index=attempt_index,
                attempt_name=initialization["name"],
                strategy="adam+lbfgs",
                initial_obs_stddev=initial_obs_stddev,
                adam_history=adam_history,
                lbfgs_history=lbfgs_history,
                adam_final_map_objective=adam_map_objective,
                final_map_objective=final_map_objective,
                parameter_prior_strength=self.parameter_prior_strength,
                parameter_prior_scale_factor=self.parameter_prior_scale_factor,
                adam_model=posterior,
                final_model=refined_posterior,
                train_data=train_data,
            )
            self._store_fitted_state(
                train_data=train_data,
                posterior=refined_posterior,
                optimization_summary=optimization_summary,
            )
            return self

        if best_fallback is not None:
            best_fallback["summary"]["status"] = "adam_only_fallback"
            best_fallback["summary"]["attempt_errors"] = tuple(attempt_errors)
            self._store_fitted_state(
                train_data=train_data,
                posterior=best_fallback["posterior"],
                optimization_summary=best_fallback["summary"],
            )
            return self

        error_message = "Hyperparameter optimization failed for all retry initializations."
        if attempt_errors:
            error_message = f"{error_message} Attempts: {' | '.join(attempt_errors)}"
        raise RuntimeError(error_message)

    def _store_fitted_state(
        self,
        *,
        train_data: gpx.Dataset,
        posterior: Any,
        optimization_summary: dict[str, Any] | None,
    ) -> None:
        """Store the current conditioned GP state on the spec instance."""

        self.train_data = train_data
        self.prior = posterior.prior
        self.likelihood = posterior.likelihood
        self.posterior = posterior
        self.is_fitted = True
        self.last_optimization_summary = optimization_summary

    def predict(self, X_query: Any) -> tuple[np.ndarray, np.ndarray]:
        """Predict means and standard deviations for query inputs with the fitted spec."""

        return _predict_with_spec(self, X_query)

    def compute_fit_diagnostics(self, X_test: Any, y_test: Any) -> dict[str, float]:
        """Compute held-out diagnostics for the current fitted GP specification."""

        return _compute_fit_diagnostics(self, X_test, y_test)

    def collect_hyperparameter_lines(self) -> list[str]:
        """Collect formatted hyperparameter lines from the fitted kernel and likelihood."""

        return _collect_gp_hyperparameters(self)

    def build_fitted_kernel_expr(self) -> str:
        """Serialize the fitted kernel back into the strict string DSL."""

        return _build_fitted_kernel_expr(self)

    def build_fitted_mean_expr(self) -> str:
        """Serialize the fitted mean function back into the strict string DSL."""

        return _build_fitted_mean_expr(self)

    def build_fitted_spec_payload(self) -> dict[str, Any]:
        """Build a serializable GP preset payload from the current fitted state."""

        return _build_fitted_spec_payload(self)

    def build_fitted_prior(self) -> gpx.gps.Prior:
        """Build a new prior object from the currently fitted kernel and mean function."""

        return _build_prior_from_fitted_spec(self)

    def build_prior_slice_data(
        self,
        X_reference: Any,
        feature_index: int,
        *,
        num_points: int = 250,
        num_samples: int = 5,
        random_seed: int = 7,
    ) -> PriorSliceData:
        """Build one prior ALE-style effect curve along a selected feature."""

        prior = self.build_fitted_prior()
        return _build_prior_slice_data(
            prior=prior,
            X_reference=X_reference,
            feature_index=feature_index,
            num_points=num_points,
            num_samples=num_samples,
            random_seed=random_seed,
        )

    def build_prior_slice_grid_data(
        self,
        X_reference: Any,
        *,
        num_points: int = 250,
        num_samples: int = 5,
        random_seed: int = 7,
    ) -> list[PriorSliceData]:
        """Build one-dimensional prior ALE-style effect curves for every feature."""

        X_reference_np = np.asarray(X_reference, dtype=np.float64)
        if X_reference_np.ndim == 1:
            X_reference_np = X_reference_np.reshape(-1, 1)
        elif X_reference_np.ndim != 2:
            raise ValueError("Reference inputs for prior ALE curves must have shape (N,) or (N, D).")

        prior = self.build_fitted_prior()
        return [
            _build_prior_slice_data(
                prior=prior,
                X_reference=X_reference_np,
                feature_index=feature_index,
                num_points=num_points,
                num_samples=num_samples,
                random_seed=random_seed + feature_index,
            )
            for feature_index in range(X_reference_np.shape[1])
        ]

    def build_posterior_slice_data(
        self,
        X_reference: Any,
        feature_index: int,
        *,
        num_points: int = 250,
        num_samples: int = 5,
        random_seed: int = 7,
    ) -> PosteriorSliceData:
        """Build one latent-posterior ALE-style effect curve along a selected feature."""

        if not self.is_fitted or self.posterior is None or self.train_data is None:
            raise ValueError("GPSpec must be fitted before posterior ALE curves can be built.")

        return _build_posterior_slice_data(
            posterior=self.posterior,
            train_data=self.train_data,
            X_reference=X_reference,
            feature_index=feature_index,
            num_points=num_points,
            num_samples=num_samples,
            random_seed=random_seed,
        )

    def build_posterior_slice_grid_data(
        self,
        X_reference: Any,
        *,
        num_points: int = 250,
        num_samples: int = 5,
        random_seed: int = 7,
    ) -> list[PosteriorSliceData]:
        """Build one latent-posterior ALE-style effect curve for every feature."""

        if not self.is_fitted or self.posterior is None or self.train_data is None:
            raise ValueError("GPSpec must be fitted before posterior ALE curves can be built.")

        X_reference_np = np.asarray(X_reference, dtype=np.float64)
        if X_reference_np.ndim == 1:
            X_reference_np = X_reference_np.reshape(-1, 1)
        elif X_reference_np.ndim != 2:
            raise ValueError("Reference inputs for posterior ALE curves must have shape (N,) or (N, D).")

        return [
            _build_posterior_slice_data(
                posterior=self.posterior,
                train_data=self.train_data,
                X_reference=X_reference_np,
                feature_index=feature_index,
                num_points=num_points,
                num_samples=num_samples,
                random_seed=random_seed + feature_index,
            )
            for feature_index in range(X_reference_np.shape[1])
        ]

    def build_nearby_refit_specs(
        self,
        *,
        perturbation_scales: tuple[float, ...] = (0.85, 1.15),
        adam_learning_rate: float = 0.02,
        adam_num_iters: int = 60,
        lbfgs_num_iters: int = 30,
        use_lbfgs_refinement: bool = True,
        gradient_clip_norm: float = 1.0,
        parameter_prior_strength: float | None = None,
        parameter_prior_scale_factor: float | None = None,
        lengthscale_prior_strength: float | None = None,
        lengthscale_prior_scale_factor: float | None = None,
        verbose: bool = False,
    ) -> list[GPSpec]:
        """Build nearby local refits from small perturbations of the fitted hyperparameters."""

        if not self.is_fitted or self.posterior is None or self.train_data is None or self.likelihood is None:
            raise ValueError("GPSpec must be fitted before nearby refits can be built.")

        _validate_optimization_config(
            adam_num_iters=adam_num_iters,
            lbfgs_num_iters=lbfgs_num_iters,
            adam_learning_rate=adam_learning_rate,
            gradient_clip_norm=gradient_clip_norm,
            use_lbfgs_refinement=use_lbfgs_refinement,
        )

        successful_refits: list[GPSpec] = []
        base_obs_stddev = float(np.asarray(self.likelihood.obs_stddev[...], dtype=np.float64))
        positive_base_obs_stddev = max(base_obs_stddev, 1e-6)
        resolved_prior_strength, resolved_prior_scale_factor = _resolve_parameter_prior_settings(
            current_strength=self.parameter_prior_strength,
            current_scale_factor=self.parameter_prior_scale_factor,
            parameter_prior_strength=parameter_prior_strength,
            parameter_prior_scale_factor=parameter_prior_scale_factor,
            lengthscale_prior_strength=lengthscale_prior_strength,
            lengthscale_prior_scale_factor=lengthscale_prior_scale_factor,
        )
        parameter_prior = _build_parameter_prior_config(
            train_data=self.train_data,
            strength=resolved_prior_strength,
            scale_factor=resolved_prior_scale_factor,
        )

        for refit_index, scale in enumerate(perturbation_scales, start=1):
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("All perturbation scales must be positive finite scalars.")

            candidate_posterior = self.build_posterior(
                self.train_data,
                obs_stddev=positive_base_obs_stddev,
            )
            _copy_model_hyperparameters(self.posterior, candidate_posterior)
            _perturb_model_hyperparameters(candidate_posterior, scale_factor=float(scale))
            _attach_parameter_priors_to_model(
                candidate_posterior,
                parameter_prior=parameter_prior,
            )

            try:
                optimized_posterior, optimization_summary = _optimize_local_refit(
                    posterior=candidate_posterior,
                    train_data=self.train_data,
                    attempt_name=f"nearby_scale_{scale:.3g}",
                    attempt_index=refit_index,
                    initial_obs_stddev=positive_base_obs_stddev,
                    adam_learning_rate=adam_learning_rate,
                    adam_num_iters=adam_num_iters,
                    lbfgs_num_iters=lbfgs_num_iters,
                    use_lbfgs_refinement=use_lbfgs_refinement,
                    gradient_clip_norm=gradient_clip_norm,
                    parameter_prior=parameter_prior,
                    parameter_prior_strength=resolved_prior_strength,
                    parameter_prior_scale_factor=resolved_prior_scale_factor,
                    verbose=verbose,
                )
            except Exception:
                try:
                    perturbation_map_objective = _evaluate_negative_conjugate_map_objective(
                        candidate_posterior,
                        self.train_data,
                        parameter_prior=parameter_prior,
                    )
                    _ensure_model_parameters_are_finite(candidate_posterior)
                except Exception:
                    continue
                optimized_posterior = candidate_posterior
                optimization_summary = _build_optimization_summary(
                    attempt_index=refit_index,
                    attempt_name=f"nearby_scale_{scale:.3g}",
                    strategy="perturbed",
                    initial_obs_stddev=positive_base_obs_stddev,
                    adam_history=np.asarray([], dtype=np.float64),
                    lbfgs_history=None,
                    adam_final_map_objective=perturbation_map_objective,
                    final_map_objective=perturbation_map_objective,
                    parameter_prior_strength=resolved_prior_strength,
                    parameter_prior_scale_factor=resolved_prior_scale_factor,
                    adam_model=candidate_posterior,
                    final_model=candidate_posterior,
                    train_data=self.train_data,
                )
                optimization_summary["status"] = "perturbed_only_fallback"

            refit_spec = GPSpec(kernel_expr=self.kernel_expr, mean_expr=self.mean_expr)
            refit_spec.parameter_prior_strength = resolved_prior_strength
            refit_spec.parameter_prior_scale_factor = resolved_prior_scale_factor
            refit_spec._store_fitted_state(
                train_data=self.train_data,
                posterior=optimized_posterior,
                optimization_summary=optimization_summary,
            )
            successful_refits.append(refit_spec)

        return successful_refits

    @property
    def lengthscale_prior_strength(self) -> float:
        """Backward-compatible alias for the parameter-prior strength."""

        return self.parameter_prior_strength

    @lengthscale_prior_strength.setter
    def lengthscale_prior_strength(self, value: float) -> None:
        self.parameter_prior_strength = float(value)

    @property
    def lengthscale_prior_scale_factor(self) -> float:
        """Backward-compatible alias for the parameter-prior scale factor."""

        return self.parameter_prior_scale_factor

    @lengthscale_prior_scale_factor.setter
    def lengthscale_prior_scale_factor(self, value: float) -> None:
        self.parameter_prior_scale_factor = float(value)


class GPJaxUtils:
    """Compatibility wrapper that delegates public utilities to ``GPSpec``."""

    @staticmethod
    def resolve_kernel_expr(expr: str) -> gpx.kernels.AbstractKernel:
        """Resolve a strict kernel expression via ``GPSpec``."""

        return GPSpec.resolve_kernel_expr(expr)

    @staticmethod
    def resolve_mean_expr(expr: str) -> gpx.mean_functions.AbstractMeanFunction:
        """Resolve a strict mean expression via ``GPSpec``."""

        return GPSpec.resolve_mean_expr(expr)

    @staticmethod
    def build_prior(kernel_expr: str, mean_expr: str) -> gpx.gps.Prior:
        """Build a prior from raw expressions via ``GPSpec``."""

        return GPSpec(kernel_expr=kernel_expr, mean_expr=mean_expr).build_prior()

    @staticmethod
    def optimize_hyperparameters(
        kernel_expr: str,
        mean_expr: str,
        X_or_dataset: Any,
        y: Any | None = None,
        **kwargs: Any,
    ) -> GPSpec:
        """Optimize a GP spec by delegating to ``GPSpec.optimize_hyperparameters``."""

        spec = GPSpec(kernel_expr=kernel_expr, mean_expr=mean_expr)
        return spec.optimize_hyperparameters(X_or_dataset, y, **kwargs)


class _StrictExpressionResolver:
    """Strict AST walker for the kernel/mean expression DSL."""

    def __init__(self, *, expr_kind: str, constructors: Mapping[str, type[Any]]) -> None:
        """Initialize a resolver for either kernel or mean expressions."""

        self.expr_kind = expr_kind
        self.constructors = constructors

    def resolve(self, expr: str) -> Any:
        """Parse and resolve a strict DSL expression into a GPJax object."""

        if not isinstance(expr, str):
            raise GPSpecParseError(f"{self.expr_kind.title()} expression must be a string.")

        raw_expr = expr
        expr = expr.strip()
        if not expr:
            raise GPSpecParseError(f"{self.expr_kind.title()} expression must not be empty.")

        try:
            root = ast.parse(expr, mode="eval").body
        except SyntaxError as exc:
            raise GPSpecParseError(
                f"Invalid {self.expr_kind} expression '{raw_expr}': {exc.msg}."
            ) from exc

        return self._resolve_node(root, raw_expr=raw_expr)

    def _resolve_node(self, node: ast.AST, *, raw_expr: str) -> Any:
        """Resolve one AST node from the strict DSL."""

        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            left = self._resolve_node(node.left, raw_expr=raw_expr)
            right = self._resolve_node(node.right, raw_expr=raw_expr)
            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                return left * right
            except Exception as exc:
                raise GPSpecParseError(
                    f"Failed to combine {self.expr_kind} expression '{raw_expr}': {exc}"
                ) from exc

        if isinstance(node, ast.Call):
            return self._resolve_call(node, raw_expr=raw_expr)

        if isinstance(node, ast.Name):
            raise GPSpecParseError(
                f"Bare names are not allowed in {self.expr_kind} expressions. "
                f"Use constructor calls like '{node.id}()'."
            )

        if isinstance(node, ast.Attribute):
            raise GPSpecParseError(
                f"Attribute access is not allowed in {self.expr_kind} expressions. "
                "Use strict constructor names only, e.g. 'Matern52()'."
            )

        raise GPSpecParseError(
            f"Unsupported syntax in {self.expr_kind} expression '{raw_expr}'. "
            "Only constructor calls, '+', '*', and parentheses are allowed."
        )

    def _resolve_call(self, node: ast.Call, *, raw_expr: str) -> Any:
        """Resolve a whitelisted constructor call with keyword-only literals."""

        if not isinstance(node.func, ast.Name):
            if isinstance(node.func, ast.Attribute):
                raise GPSpecParseError(
                    f"Attribute access is not allowed in {self.expr_kind} expressions. "
                    "Use strict constructor names only, e.g. 'Matern52()'."
                )
            raise GPSpecParseError(
                f"Unsupported callable in {self.expr_kind} expression '{raw_expr}'."
            )

        constructor_name = node.func.id
        constructor = self.constructors.get(constructor_name)
        if constructor is None:
            allowed = ", ".join(sorted(self.constructors))
            raise GPSpecParseError(
                f"Unknown {self.expr_kind} constructor '{constructor_name}'. "
                f"Allowed {self.expr_kind} constructors: {allowed}."
            )

        if node.args:
            raise GPSpecParseError(
                f"Positional arguments are not allowed for {self.expr_kind} constructor "
                f"'{constructor_name}'. Use keyword arguments only."
            )

        allowed_kwargs = _get_allowed_keyword_arguments(constructor)
        kwargs: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise GPSpecParseError(
                    f"Expanded keyword arguments are not allowed for {self.expr_kind} "
                    f"constructor '{constructor_name}'."
                )
            if keyword.arg in kwargs:
                raise GPSpecParseError(
                    f"Duplicate keyword argument '{keyword.arg}' for {self.expr_kind} "
                    f"constructor '{constructor_name}'."
                )
            if keyword.arg not in allowed_kwargs:
                allowed = ", ".join(sorted(allowed_kwargs))
                raise GPSpecParseError(
                    f"Unsupported keyword '{keyword.arg}' for {self.expr_kind} "
                    f"constructor '{constructor_name}'. Allowed keywords: {allowed}."
                )
            kwargs[keyword.arg] = self._parse_literal_value(keyword.value)

        try:
            return constructor(**kwargs)
        except Exception as exc:
            raise GPSpecParseError(
                f"Failed to construct {self.expr_kind} '{constructor_name}' from "
                f"expression '{raw_expr}': {exc}"
            ) from exc

    def _parse_literal_value(self, node: ast.AST) -> Any:
        """Parse a literal keyword value allowed by the strict DSL."""

        if isinstance(node, ast.Constant):
            value = node.value
            if value is None or isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            raise GPSpecParseError(
                f"Unsupported literal {value!r} in {self.expr_kind} expression. "
                "Only int, float, bool, None, list, and tuple values are allowed."
            )

        # Handle unary +/- on a numeric literal (e.g. -0.5, +1.0, -2.4e-16).
        # Python's AST represents `-x` as UnaryOp(USub, Constant(x)), not as a
        # Constant with a negative value, so this case must be handled explicitly.
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = self._parse_literal_value(node.operand)
            if isinstance(operand, (int, float)) and not isinstance(operand, bool):
                return -operand if isinstance(node.op, ast.USub) else operand
            raise GPSpecParseError(
                f"Unary +/- is only supported on numeric literals in {self.expr_kind} "
                "expressions."
            )

        if isinstance(node, ast.List):
            return [self._parse_literal_value(element) for element in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(self._parse_literal_value(element) for element in node.elts)

        raise GPSpecParseError(
            f"Unsupported keyword value in {self.expr_kind} expression. "
            "Only int, float, bool, None, list, and tuple literals are allowed."
        )


def _coerce_training_dataset(X_or_dataset: Any, y: Any | None = None) -> gpx.Dataset:
    """Coerce either a GPJax dataset or raw ``X, y`` arrays into ``gpx.Dataset``."""

    if isinstance(X_or_dataset, gpx.Dataset):
        if y is not None:
            raise ValueError("When passing a gpx.Dataset, `y` must be None.")
        if X_or_dataset.X is None or X_or_dataset.y is None:
            raise ValueError("Provided gpx.Dataset must contain both X and y.")
        return X_or_dataset

    if y is None:
        raise ValueError("Raw training inputs require both `X` and `y`.")

    X_np = np.asarray(X_or_dataset, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)

    if X_np.ndim == 1:
        X_np = X_np.reshape(-1, 1)
    elif X_np.ndim != 2:
        raise ValueError("Raw `X` must have shape (N,) or (N, D).")

    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)
    elif y_np.ndim == 2 and y_np.shape[1] == 1:
        pass
    else:
        raise ValueError("Raw `y` must have shape (N,) or (N, 1).")

    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError("Raw `X` and `y` must contain the same number of samples.")

    return gpx.Dataset(X=X_np, y=y_np)


def _call_prior_distribution(
    prior: Any,
    X_query: Any,
    *,
    covariance_type: str | None = "dense",
) -> Any:
    """Call a GPJax prior across the old and new call signatures."""

    X_query_np = np.asarray(X_query, dtype=np.float64)
    try:
        if covariance_type is None:
            return prior(X_query_np)
        return prior(X_query_np, return_covariance_type=covariance_type)
    except TypeError:
        return prior(X_query_np)


def _call_posterior_distribution(
    posterior: Any,
    X_query: Any,
    train_data: gpx.Dataset,
    *,
    covariance_type: str | None = "dense",
) -> Any:
    """Call a GPJax posterior across the old and new call signatures."""

    X_query_np = np.asarray(X_query, dtype=np.float64)
    try:
        if covariance_type is None:
            return posterior(X_query_np, train_data)
        return posterior(
            X_query_np,
            train_data,
            return_covariance_type=covariance_type,
        )
    except TypeError:
        return posterior(X_query_np, train_data)


def _predict_with_spec(gp_spec: GPSpec, X_query: Any) -> tuple[np.ndarray, np.ndarray]:
    """Predict means and standard deviations for query inputs with a fitted spec."""

    if not gp_spec.is_fitted or gp_spec.posterior is None or gp_spec.train_data is None:
        raise ValueError("GPSpec must be fitted before predictions can be made.")
    if gp_spec.likelihood is None:
        raise ValueError("GPSpec likelihood is missing. Fit the spec again.")

    latent_dist = _call_posterior_distribution(
        gp_spec.posterior,
        X_query,
        gp_spec.train_data,
        covariance_type="diagonal",
    )
    predictive_dist = gp_spec.likelihood(latent_dist)

    mean = np.asarray(predictive_dist.mean, dtype=np.float64).reshape(-1)
    variance = np.asarray(predictive_dist.variance, dtype=np.float64).reshape(-1)
    std = np.sqrt(np.clip(variance, a_min=0.0, a_max=None))
    return mean, std


def _compute_pearson_correlation(y_true: Any, y_pred: Any) -> float:
    """Compute Pearson correlation and return ``nan`` for degenerate inputs."""

    y_true_np = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_np = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if (
        y_true_np.size < 2
        or np.allclose(y_true_np, y_true_np[0])
        or np.allclose(y_pred_np, y_pred_np[0])
    ):
        return float("nan")

    return float(pearsonr(y_true_np, y_pred_np).statistic)


def _compute_fit_diagnostics(
    gp_spec: GPSpec,
    X_test: Any,
    y_test: Any,
) -> dict[str, float]:
    """Compute held-out diagnostics for the current fitted GP specification."""

    if gp_spec.train_data is None or gp_spec.posterior is None:
        raise ValueError("GPSpec must be fitted before diagnostics can be computed.")

    y_test_np = np.asarray(y_test, dtype=np.float64).reshape(-1)
    test_mean, test_std = _predict_with_spec(gp_spec, X_test)
    train_mll = float(gpx.objectives.conjugate_mll(gp_spec.posterior, gp_spec.train_data))
    train_log_prior_density = float(
        np.asarray(_log_parameter_prior_density(gp_spec.posterior), dtype=np.float64)
    )
    train_log_posterior_density = train_mll + train_log_prior_density
    test_rmse = float(np.sqrt(np.mean((y_test_np - test_mean) ** 2)))
    test_pearson_r = _compute_pearson_correlation(y_test_np, test_mean)
    return {
        "train_mll": train_mll,
        "train_log_prior_density": train_log_prior_density,
        "train_log_posterior_density": train_log_posterior_density,
        "train_map_objective": -train_log_posterior_density,
        "test_rmse": test_rmse,
        "test_pearson_r": test_pearson_r,
        "test_std_mean": float(np.mean(test_std)),
    }


def _format_numeric_value(value: Any) -> str:
    """Format scalar and array hyperparameter values for terminal output."""

    values = np.asarray(value, dtype=np.float64)
    if values.shape == ():
        return f"{float(values):.6g}"
    return np.array2string(values, precision=6, separator=", ", suppress_small=False)


def _format_parameter_value_with_prior_center(parameter: gpx.parameters.Parameter) -> str:
    """Format one parameter value and its attached prior center when available."""

    formatted_value = _format_numeric_value(parameter[...])
    prior_center = getattr(parameter, "prior_center", None)
    if prior_center is None:
        return formatted_value
    return f"{formatted_value} (prior center: {_format_numeric_value(prior_center)})"


def _is_numeric_hyperparameter(value: Any) -> bool:
    """Return whether an attribute value should be printed as a hyperparameter."""

    if isinstance(value, (bool, str, bytes, slice)):
        return False

    try:
        values = np.asarray(value)
    except Exception:
        return False

    return values.size > 0 and np.issubdtype(values.dtype, np.number)


def _collect_component_hyperparameters(component: Any, prefix: str) -> list[str]:
    """Collect printable hyperparameter lines from a GPJax component tree."""

    if component is None:
        return []

    lines: list[str] = []
    ignored_attributes = {
        "_pytree__state",
        "_pytree__nodes",
        "active_dims",
        "n_dims",
        "compute_engine",
        "operator",
        "integrator",
        "num_datapoints",
        "name",
    }

    for attribute_name, attribute_value in vars(component).items():
        if attribute_name in ignored_attributes or attribute_name.startswith("_"):
            continue
        if attribute_name in {"kernels", "means"}:
            for index, child in enumerate(attribute_value):
                lines.extend(
                    _collect_component_hyperparameters(
                        child,
                        f"{prefix}.{attribute_name}[{index}:{type(child).__name__}]",
                    )
                )
            continue
        if isinstance(attribute_value, gpx.parameters.Parameter):
            lines.append(
                f"{prefix}.{attribute_name}: "
                f"{_format_parameter_value_with_prior_center(attribute_value)}"
            )
            continue
        if _is_numeric_hyperparameter(attribute_value):
            lines.append(f"{prefix}.{attribute_name}: {_format_numeric_value(attribute_value)}")

    return lines


def _collect_gp_hyperparameters(gp_spec: GPSpec) -> list[str]:
    """Collect kernel, mean, and likelihood hyperparameters from a fitted spec."""

    if gp_spec.prior is None or gp_spec.likelihood is None:
        return ["n/a"]

    hyperparameters: list[str] = []
    hyperparameters.extend(_collect_component_hyperparameters(gp_spec.prior.kernel, "kernel"))
    hyperparameters.extend(
        _collect_component_hyperparameters(gp_spec.prior.mean_function, "mean_function")
    )
    hyperparameters.extend(_collect_component_hyperparameters(gp_spec.likelihood, "likelihood"))
    return hyperparameters or ["n/a"]


def _build_fitted_kernel_expr(gp_spec: GPSpec) -> str:
    """Serialize the fitted kernel from a fitted ``GPSpec`` into the strict DSL."""

    if gp_spec.prior is None:
        raise ValueError("GPSpec must be fitted before the fitted kernel expression can be built.")
    return _serialize_kernel_expression(gp_spec.prior.kernel)


def _build_fitted_mean_expr(gp_spec: GPSpec) -> str:
    """Serialize the fitted mean function from a fitted ``GPSpec`` into the strict DSL."""

    if gp_spec.prior is None:
        raise ValueError("GPSpec must be fitted before the fitted mean expression can be built.")
    return _serialize_mean_expression(gp_spec.prior.mean_function)


def _build_fitted_spec_payload(gp_spec: GPSpec) -> dict[str, Any]:
    """Build a serializable preset payload from the current fitted GP state."""

    if not gp_spec.is_fitted or gp_spec.prior is None or gp_spec.likelihood is None:
        raise ValueError("GPSpec must be fitted before a preset payload can be built.")

    optimization_summary = None
    if gp_spec.last_optimization_summary is not None:
        optimization_summary = _to_python_value(gp_spec.last_optimization_summary)

    return {
        "original_kernel_expr": gp_spec.kernel_expr,
        "original_mean_expr": gp_spec.mean_expr,
        "fitted_kernel_expr": _build_fitted_kernel_expr(gp_spec),
        "fitted_mean_expr": _build_fitted_mean_expr(gp_spec),
        "likelihood_obs_stddev": float(np.asarray(gp_spec.likelihood.obs_stddev[...], dtype=np.float64)),
        "hyperparameter_lines": list(gp_spec.collect_hyperparameter_lines()),
        "optimization_summary": optimization_summary,
    }


def _serialize_kernel_expression(kernel: Any) -> str:
    """Serialize a fitted GPJax kernel tree into the strict string DSL."""

    return _serialize_component_expression(
        component=kernel,
        constructor_registry=_KERNEL_CONSTRUCTORS,
        child_attribute="kernels",
    )


def _serialize_mean_expression(mean_function: Any) -> str:
    """Serialize a fitted GPJax mean-function tree into the strict string DSL."""

    return _serialize_component_expression(
        component=mean_function,
        constructor_registry=_MEAN_CONSTRUCTORS,
        child_attribute="means",
    )


def _serialize_component_expression(
    *,
    component: Any,
    constructor_registry: Mapping[str, type[Any]],
    child_attribute: str,
) -> str:
    """Serialize one GPJax component tree into the strict constructor DSL."""

    if hasattr(component, child_attribute):
        children = getattr(component, child_attribute)
        if not children:
            raise ValueError("Combination components must contain at least one child.")
        operator_symbol = _resolve_operator_symbol(getattr(component, "operator", None))
        rendered_children = [
            _serialize_component_expression(
                component=child,
                constructor_registry=constructor_registry,
                child_attribute=child_attribute,
            )
            for child in children
        ]
        return f"({f' {operator_symbol} '.join(rendered_children)})"

    constructor_name = type(component).__name__
    if constructor_name == "Zero":
        return "Zero()"
    if constructor_name not in constructor_registry:
        raise ValueError(f"Unsupported fitted component type '{constructor_name}' for DSL serialization.")

    keyword_parts: list[str] = []
    for keyword_name, keyword_value in _iter_serializable_constructor_kwargs(component):
        keyword_parts.append(f"{keyword_name}={_format_dsl_literal(keyword_value)}")

    rendered_kwargs = ", ".join(keyword_parts)
    return f"{constructor_name}({rendered_kwargs})" if rendered_kwargs else f"{constructor_name}()"


def _iter_serializable_constructor_kwargs(component: Any) -> list[tuple[str, Any]]:
    """Collect constructor keyword arguments that can be losslessly serialized."""

    ignored_attributes = {
        "active_dims",
        "n_dims",
        "compute_engine",
        "operator",
        "integrator",
        "num_datapoints",
        "name",
    }

    signature = inspect.signature(type(component).__init__)
    constructor_kwargs: list[tuple[str, Any]] = []
    for parameter_name, parameter in signature.parameters.items():
        if parameter_name == "self":
            continue
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if parameter_name in ignored_attributes or not hasattr(component, parameter_name):
            continue

        raw_value = getattr(component, parameter_name)
        python_value = _to_python_value(raw_value)
        if python_value is None or isinstance(python_value, slice):
            continue
        constructor_kwargs.append((parameter_name, python_value))

    return constructor_kwargs


def _resolve_operator_symbol(operator: Any) -> str:
    """Resolve a GPJax combination operator to the DSL symbol ``+`` or ``*``."""

    operator_name = getattr(operator, "__name__", None)
    if operator_name is None and hasattr(operator, "func"):
        operator_name = getattr(operator.func, "__name__", None)
    operator_text = (operator_name or repr(operator)).lower()

    if "sum" in operator_text or "add" in operator_text:
        return "+"
    if "prod" in operator_text or "mul" in operator_text or "multiply" in operator_text:
        return "*"
    raise ValueError(f"Unsupported GPJax combination operator '{operator}'.")


def _format_dsl_literal(value: Any) -> str:
    """Format a Python value as a strict literal accepted by the GP DSL parser."""

    value = _to_python_value(value)
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, list):
        return "[" + ", ".join(_format_dsl_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        if len(value) == 1:
            return "(" + _format_dsl_literal(value[0]) + ",)"
        return "(" + ", ".join(_format_dsl_literal(item) for item in value) + ")"
    raise TypeError(f"Unsupported DSL literal type: {type(value).__name__}.")


def _to_python_value(value: Any) -> Any:
    """Convert NumPy/JAX/GPJax values into plain Python containers and scalars."""

    if isinstance(value, gpx.parameters.Parameter):
        return _to_python_value(value[...])
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return tuple(_to_python_value(item) for item in value)
    if isinstance(value, list):
        return [_to_python_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_python_value(item) for key, item in value.items()}
    if isinstance(value, slice):
        return value

    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return [_to_python_value(item) for item in array.tolist()]


def _parameterize_mean_function_constants(mean_function: Any) -> None:
    """Promote constant means to GPJax Parameters so priors can be attached."""

    if hasattr(mean_function, "means"):
        for child_mean in mean_function.means:
            _parameterize_mean_function_constants(child_mean)
        return

    if type(mean_function).__name__ != "Constant":
        return

    constant_value = getattr(mean_function, "constant", None)
    if isinstance(constant_value, gpx.parameters.Parameter):
        return

    mean_function.constant = gpx.parameters.Real(
        np.asarray(constant_value, dtype=np.float64),
    )


def _build_prior_from_fitted_spec(gp_spec: GPSpec) -> gpx.gps.Prior:
    """Build a new prior object from the currently fitted kernel and mean function."""

    if gp_spec.prior is None:
        raise ValueError("GPSpec must be fitted before a prior can be rebuilt.")

    return gpx.gps.Prior(
        mean_function=gp_spec.prior.mean_function,
        kernel=gp_spec.prior.kernel,
    )


def _copy_component_hyperparameters(source_component: Any, target_component: Any) -> None:
    """Recursively copy numeric hyperparameters from one GPJax component tree to another."""

    ignored_attributes = {
        "_pytree__state",
        "_pytree__nodes",
        "active_dims",
        "n_dims",
        "compute_engine",
        "operator",
        "integrator",
        "num_datapoints",
        "name",
    }

    for attribute_name, source_value in vars(source_component).items():
        if attribute_name in ignored_attributes or attribute_name.startswith("_"):
            continue
        if not hasattr(target_component, attribute_name):
            continue

        target_value = getattr(target_component, attribute_name)
        if attribute_name in {"kernels", "means"}:
            for source_child, target_child in zip(source_value, target_value):
                _copy_component_hyperparameters(source_child, target_child)
            continue

        if isinstance(source_value, gpx.parameters.Parameter) and isinstance(
            target_value,
            gpx.parameters.Parameter,
        ):
            _assign_parameter_value(target_value, source_value[...])
            continue

        if attribute_name == "constant" and _is_numeric_hyperparameter(source_value):
            setattr(
                target_component,
                attribute_name,
                jax.numpy.asarray(np.asarray(source_value, dtype=np.float64)),
            )


def _copy_model_hyperparameters(source_model: Any, target_model: Any) -> None:
    """Copy kernel, mean, and likelihood hyperparameters between compatible GP models."""

    _copy_component_hyperparameters(source_model.prior.kernel, target_model.prior.kernel)
    _copy_component_hyperparameters(
        source_model.prior.mean_function,
        target_model.prior.mean_function,
    )
    _copy_component_hyperparameters(source_model.likelihood, target_model.likelihood)


def _perturb_numeric_hyperparameter(
    *,
    attribute_name: str,
    values: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Apply a small multiplicative perturbation while keeping positive scales valid."""

    perturbed = values * scale_factor
    positive_names = {
        "lengthscale",
        "variance",
        "period",
        "obs_stddev",
        "alpha",
        "scale",
    }

    if attribute_name in positive_names or np.all(values >= 0.0):
        perturbed = np.maximum(perturbed, 1e-6)
    return perturbed


def _perturb_component_hyperparameters(component: Any, *, scale_factor: float) -> None:
    """Recursively perturb numeric GPJax hyperparameters in a component tree."""

    ignored_attributes = {
        "_pytree__state",
        "_pytree__nodes",
        "active_dims",
        "n_dims",
        "compute_engine",
        "operator",
        "integrator",
        "num_datapoints",
        "name",
    }

    for attribute_name, attribute_value in vars(component).items():
        if attribute_name in ignored_attributes or attribute_name.startswith("_"):
            continue
        if attribute_name in {"kernels", "means"}:
            for child in attribute_value:
                _perturb_component_hyperparameters(child, scale_factor=scale_factor)
            continue

        if isinstance(attribute_value, gpx.parameters.Parameter):
            current_value = np.asarray(attribute_value[...], dtype=np.float64)
            _assign_parameter_value(
                attribute_value,
                _perturb_numeric_hyperparameter(
                    attribute_name=attribute_name,
                    values=current_value,
                    scale_factor=scale_factor,
                ),
            )
            continue

        if attribute_name == "constant" and _is_numeric_hyperparameter(attribute_value):
            current_value = np.asarray(attribute_value, dtype=np.float64)
            setattr(
                component,
                attribute_name,
                jax.numpy.asarray(
                    _perturb_numeric_hyperparameter(
                        attribute_name=attribute_name,
                        values=current_value,
                        scale_factor=scale_factor,
                    )
                ),
            )


def _perturb_model_hyperparameters(model: Any, *, scale_factor: float) -> None:
    """Perturb a full GP posterior model in place around its current hyperparameters."""

    _perturb_component_hyperparameters(model.prior.kernel, scale_factor=scale_factor)
    _perturb_component_hyperparameters(model.prior.mean_function, scale_factor=scale_factor)
    _perturb_component_hyperparameters(model.likelihood, scale_factor=scale_factor)


def _coerce_reference_inputs(X_reference: Any, *, context: str) -> np.ndarray:
    """Coerce reference inputs into a two-dimensional float array."""

    X_reference_np = np.asarray(X_reference, dtype=np.float64)
    if X_reference_np.ndim == 1:
        X_reference_np = X_reference_np.reshape(-1, 1)
    elif X_reference_np.ndim != 2:
        raise ValueError(f"Reference inputs for {context} must have shape (N,) or (N, D).")
    return X_reference_np


def _predict_prior_mean_in_batches(
    prior: Any,
    X_query: Any,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    """Predict prior means in batches to keep dense-covariance calls manageable."""

    X_query_np = _coerce_reference_inputs(X_query, context="prior ALE curves")
    means = np.empty(X_query_np.shape[0], dtype=np.float64)
    for start_index in range(0, X_query_np.shape[0], batch_size):
        end_index = min(start_index + batch_size, X_query_np.shape[0])
        distribution = _call_prior_distribution(
            prior,
            X_query_np[start_index:end_index],
            covariance_type="diagonal",
        )
        means[start_index:end_index] = np.asarray(distribution.mean, dtype=np.float64).reshape(-1)
    return means


def _predict_posterior_latent_mean_in_batches(
    posterior: Any,
    train_data: gpx.Dataset,
    X_query: Any,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    """Predict latent posterior means in batches."""

    X_query_np = _coerce_reference_inputs(X_query, context="posterior ALE curves")
    means = np.empty(X_query_np.shape[0], dtype=np.float64)
    for start_index in range(0, X_query_np.shape[0], batch_size):
        end_index = min(start_index + batch_size, X_query_np.shape[0])
        distribution = _call_posterior_distribution(
            posterior,
            X_query_np[start_index:end_index],
            train_data,
            covariance_type="diagonal",
        )
        means[start_index:end_index] = np.asarray(distribution.mean, dtype=np.float64).reshape(-1)
    return means


def _build_ale_curve(
    *,
    X_reference: Any,
    feature_index: int,
    num_points: int,
    predict_mean_fn: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a first-order ALE curve on the original feature scale."""

    X_reference_np = _coerce_reference_inputs(X_reference, context="ALE curves")
    if feature_index < 0 or feature_index >= X_reference_np.shape[1]:
        raise IndexError("Requested ALE feature index is out of bounds.")
    if num_points <= 1:
        raise ValueError("`num_points` must be greater than one for ALE curves.")

    fixed_values = np.median(X_reference_np, axis=0)
    feature_values = X_reference_np[:, feature_index]
    baseline_prediction = float(np.mean(predict_mean_fn(X_reference_np)))
    bin_edges = np.unique(
        np.quantile(
            feature_values,
            np.linspace(0.0, 1.0, num_points + 1, dtype=np.float64),
        )
    )

    if bin_edges.size < 2:
        constant_value = float(feature_values[0]) if feature_values.size else 0.0
        x_values = np.array([constant_value, constant_value], dtype=np.float64)
        mean = np.full_like(x_values, baseline_prediction, dtype=np.float64)
        std = np.zeros_like(x_values, dtype=np.float64)
        return x_values, mean, std, fixed_values

    lower_edges = bin_edges[:-1]
    upper_edges = bin_edges[1:]
    x_values = 0.5 * (lower_edges + upper_edges)
    num_bins = x_values.shape[0]

    bin_indices = np.searchsorted(bin_edges, feature_values, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    lower_inputs = np.asarray(X_reference_np, dtype=np.float64).copy()
    upper_inputs = np.asarray(X_reference_np, dtype=np.float64).copy()
    lower_inputs[:, feature_index] = lower_edges[bin_indices]
    upper_inputs[:, feature_index] = upper_edges[bin_indices]

    local_effects = predict_mean_fn(upper_inputs) - predict_mean_fn(lower_inputs)
    counts = np.bincount(bin_indices, minlength=num_bins).astype(np.float64)
    mean_deltas = np.zeros(num_bins, dtype=np.float64)
    delta_standard_errors = np.zeros(num_bins, dtype=np.float64)

    for bin_index in range(num_bins):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        bin_effects = local_effects[mask]
        mean_deltas[bin_index] = float(np.mean(bin_effects))
        if bin_effects.size > 1:
            delta_standard_errors[bin_index] = float(
                np.std(bin_effects, ddof=1) / np.sqrt(bin_effects.size)
            )

    cumulative_prefix = np.concatenate(([0.0], np.cumsum(mean_deltas[:-1])))
    raw_effect = cumulative_prefix + 0.5 * mean_deltas
    centered_effect = raw_effect - np.average(raw_effect, weights=counts)
    mean = baseline_prediction + centered_effect

    prefix_variance = np.concatenate(([0.0], np.cumsum(delta_standard_errors[:-1] ** 2)))
    std = np.sqrt(
        np.clip(prefix_variance + 0.25 * delta_standard_errors**2, a_min=0.0, a_max=None)
    )
    return x_values, mean, std, fixed_values


def _build_prior_slice_data(
    *,
    prior: gpx.gps.Prior,
    X_reference: Any,
    feature_index: int,
    num_points: int,
    num_samples: int,
    random_seed: int,
) -> PriorSliceData:
    """Build one prior ALE-style effect curve for one selected feature."""

    if num_samples <= 0:
        raise ValueError("`num_samples` must be a positive integer.")

    x_values, prior_mean, prior_std, fixed_values = _build_ale_curve(
        X_reference=X_reference,
        feature_index=feature_index,
        num_points=num_points,
        predict_mean_fn=lambda X_query: _predict_prior_mean_in_batches(prior, X_query),
    )
    prior_samples = np.repeat(prior_mean[None, :], repeats=num_samples, axis=0)
    return PriorSliceData(
        feature_index=feature_index,
        x_values=x_values,
        fixed_values=fixed_values,
        mean=prior_mean,
        std=prior_std,
        samples=prior_samples,
    )


def _build_posterior_slice_data(
    *,
    posterior: Any,
    train_data: gpx.Dataset,
    X_reference: Any,
    feature_index: int,
    num_points: int,
    num_samples: int,
    random_seed: int,
) -> PosteriorSliceData:
    """Build one latent-posterior ALE-style effect curve for one selected feature."""

    if num_samples <= 0:
        raise ValueError("`num_samples` must be a positive integer.")

    x_values, posterior_mean, posterior_std, fixed_values = _build_ale_curve(
        X_reference=X_reference,
        feature_index=feature_index,
        num_points=num_points,
        predict_mean_fn=lambda X_query: _predict_posterior_latent_mean_in_batches(
            posterior,
            train_data,
            X_query,
        ),
    )
    posterior_samples = np.repeat(posterior_mean[None, :], repeats=num_samples, axis=0)
    return PosteriorSliceData(
        feature_index=feature_index,
        x_values=x_values,
        fixed_values=fixed_values,
        mean=posterior_mean,
        std=posterior_std,
        samples=posterior_samples,
    )


def _coerce_optional_training_dataset(
    gp_spec: GPSpec,
    X_or_dataset: Any | None,
    y: Any | None = None,
) -> gpx.Dataset:
    """Resolve optimization inputs from explicit data or the stored fitted dataset."""

    if X_or_dataset is None:
        if y is not None:
            raise ValueError("`y` must be None when reusing the stored training dataset.")
        if gp_spec.train_data is None:
            raise ValueError(
                "No training data available. Provide `X_or_dataset` to optimize hyperparameters."
            )
        return gp_spec.train_data

    return _coerce_training_dataset(X_or_dataset, y)


def _resolve_initial_obs_stddev(gp_spec: GPSpec, obs_stddev: float | None) -> float:
    """Resolve the observation noise initialization for optimization."""

    if obs_stddev is None:
        if gp_spec.likelihood is not None:
            resolved = float(np.asarray(gp_spec.likelihood.obs_stddev[...]))
        else:
            resolved = 1.0
    else:
        resolved = float(obs_stddev)

    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("`obs_stddev` must be a positive finite scalar.")
    return resolved


def _validate_optimization_config(
    *,
    adam_num_iters: int,
    lbfgs_num_iters: int,
    adam_learning_rate: float,
    gradient_clip_norm: float,
    use_lbfgs_refinement: bool,
) -> None:
    """Validate the numerical configuration used by hyperparameter optimization."""

    if adam_num_iters <= 0:
        raise ValueError("`adam_num_iters` must be a positive integer.")
    if use_lbfgs_refinement and lbfgs_num_iters <= 0:
        raise ValueError("`lbfgs_num_iters` must be positive when L-BFGS refinement is enabled.")
    if not np.isfinite(adam_learning_rate) or adam_learning_rate <= 0.0:
        raise ValueError("`adam_learning_rate` must be a positive finite scalar.")
    if not np.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0.0:
        raise ValueError("`gradient_clip_norm` must be a positive finite scalar.")


def _compute_feature_scales(train_data: gpx.Dataset) -> np.ndarray:
    """Compute per-feature empirical scales from the training covariates."""

    X_np = np.asarray(train_data.X, dtype=np.float64)
    feature_scales = np.std(X_np, axis=0)
    return np.where(
        np.isfinite(feature_scales) & (feature_scales > 1e-6),
        feature_scales,
        1.0,
    )


def _compute_feature_spans(train_data: gpx.Dataset) -> np.ndarray:
    """Compute per-feature empirical spans from the training covariates."""

    X_np = np.asarray(train_data.X, dtype=np.float64)
    feature_spans = np.max(X_np, axis=0) - np.min(X_np, axis=0)
    feature_scales = _compute_feature_scales(train_data)
    return np.where(
        np.isfinite(feature_spans) & (feature_spans > 1e-6),
        feature_spans,
        feature_scales,
    )


def _resolve_parameter_prior_settings(
    *,
    current_strength: float,
    current_scale_factor: float,
    parameter_prior_strength: float | None,
    parameter_prior_scale_factor: float | None,
    lengthscale_prior_strength: float | None,
    lengthscale_prior_scale_factor: float | None,
) -> tuple[float, float]:
    """Resolve general parameter-prior settings with legacy keyword aliases."""

    if parameter_prior_strength is not None and lengthscale_prior_strength is not None:
        if float(parameter_prior_strength) != float(lengthscale_prior_strength):
            raise ValueError(
                "Received both `parameter_prior_strength` and the deprecated "
                "`lengthscale_prior_strength` with different values."
            )
    if parameter_prior_scale_factor is not None and lengthscale_prior_scale_factor is not None:
        if float(parameter_prior_scale_factor) != float(lengthscale_prior_scale_factor):
            raise ValueError(
                "Received both `parameter_prior_scale_factor` and the deprecated "
                "`lengthscale_prior_scale_factor` with different values."
            )

    resolved_strength = (
        current_strength
        if parameter_prior_strength is None and lengthscale_prior_strength is None
        else float(
            parameter_prior_strength
            if parameter_prior_strength is not None
            else lengthscale_prior_strength
        )
    )
    resolved_scale_factor = (
        current_scale_factor
        if parameter_prior_scale_factor is None and lengthscale_prior_scale_factor is None
        else float(
            parameter_prior_scale_factor
            if parameter_prior_scale_factor is not None
            else lengthscale_prior_scale_factor
        )
    )
    return resolved_strength, resolved_scale_factor


def _build_parameter_prior_config(
    *,
    train_data: gpx.Dataset,
    strength: float,
    scale_factor: float,
) -> ParameterPriorConfig | None:
    """Construct data-informed priors for all trainable GPJax hyperparameters."""

    strength = float(strength)
    scale_factor = float(scale_factor)
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("`parameter_prior_strength` must be a non-negative finite scalar.")
    if strength == 0.0:
        return None
    if not np.isfinite(scale_factor) or scale_factor <= 1.0:
        raise ValueError(
            "`parameter_prior_scale_factor` must be a finite scalar greater than 1."
        )

    y_np = np.asarray(train_data.y, dtype=np.float64).reshape(-1)
    target_mean = float(np.mean(y_np))
    target_std = max(float(np.std(y_np)), 1e-3)
    target_variance = max(target_std**2, 1e-3)
    target_obs_stddev = max(0.2 * target_std, 1e-3)

    return ParameterPriorConfig(
        feature_scales=_compute_feature_scales(train_data),
        feature_spans=_compute_feature_spans(train_data),
        target_mean=target_mean,
        target_std=target_std,
        target_variance=target_variance,
        target_obs_stddev=target_obs_stddev,
        log_spread=float(np.log(scale_factor)),
        strength=strength,
    )


def _build_retry_initializations(
    train_data: gpx.Dataset,
    initial_obs_stddev: float,
) -> list[dict[str, Any]]:
    """Build a small set of safer retry initializations based on training-data scale."""

    X_np = np.asarray(train_data.X, dtype=np.float64)
    y_np = np.asarray(train_data.y, dtype=np.float64).reshape(-1)

    feature_scales = _compute_feature_scales(train_data)
    mean_feature_scale = float(np.median(feature_scales))

    target_mean = float(np.mean(y_np))
    target_std = float(np.std(y_np))
    target_std = max(target_std, 1e-3)
    target_variance = max(target_std**2, 1e-3)

    return [
        {
            "name": "spec_defaults",
            "lengthscale_scale": None,
            "variance": None,
            "obs_stddev": initial_obs_stddev,
            "mean_constant": None,
        },
        {
            "name": "data_scaled",
            "lengthscale_scale": feature_scales,
            "variance": target_variance,
            "obs_stddev": max(initial_obs_stddev, 0.1 * target_std),
            "mean_constant": target_mean,
        },
        {
            "name": "conservative",
            "lengthscale_scale": np.full_like(feature_scales, 2.0 * mean_feature_scale),
            "variance": target_variance,
            "obs_stddev": max(initial_obs_stddev, 0.25 * target_std),
            "mean_constant": target_mean,
        },
    ]


def _apply_initialization_to_posterior(
    *,
    posterior: Any,
    train_data: gpx.Dataset,
    initialization: Mapping[str, Any],
) -> None:
    """Apply one retry initialization recipe to a freshly built posterior."""

    _apply_kernel_initialization(
        kernel=posterior.prior.kernel,
        feature_scales=np.asarray(initialization["lengthscale_scale"])
        if initialization["lengthscale_scale"] is not None
        else None,
        variance=initialization["variance"],
    )
    _apply_mean_initialization(
        mean_function=posterior.prior.mean_function,
        mean_constant=initialization["mean_constant"],
    )
    posterior.likelihood.obs_stddev[...] = np.asarray(
        initialization["obs_stddev"],
        dtype=np.float64,
    )
    posterior.likelihood.num_datapoints = train_data.n


def _apply_kernel_initialization(
    *,
    kernel: Any,
    feature_scales: np.ndarray | None,
    variance: float | None,
) -> None:
    """Recursively apply safe numeric initializations to kernel hyperparameters."""

    if hasattr(kernel, "kernels"):
        for sub_kernel in kernel.kernels:
            _apply_kernel_initialization(
                kernel=sub_kernel,
                feature_scales=feature_scales,
                variance=variance,
            )
        return

    if feature_scales is not None and hasattr(kernel, "lengthscale"):
        _assign_parameter_value(kernel.lengthscale, feature_scales)
    if variance is not None and hasattr(kernel, "variance"):
        _assign_parameter_value(kernel.variance, variance)


def _apply_mean_initialization(*, mean_function: Any, mean_constant: float | None) -> None:
    """Recursively initialize train-time constant mean offsets when available."""

    if mean_constant is None:
        return
    if hasattr(mean_function, "means"):
        for sub_mean in mean_function.means:
            _apply_mean_initialization(mean_function=sub_mean, mean_constant=mean_constant)
        return

    if type(mean_function).__name__ == "Constant":
        if isinstance(mean_function.constant, gpx.parameters.Parameter):
            _assign_parameter_value(mean_function.constant, mean_constant)
        else:
            mean_function.constant = jax.numpy.asarray(mean_constant, dtype=jax.numpy.float64)


def _assign_parameter_value(parameter: Any, value: Any) -> None:
    """Assign a scalar or vector value to a GPJax parameter with shape matching."""

    current_value = np.asarray(parameter[...], dtype=np.float64)
    target_value = np.asarray(value, dtype=np.float64)

    if current_value.shape == ():
        if target_value.shape == ():
            parameter[...] = target_value
        else:
            parameter[...] = np.asarray(np.median(target_value), dtype=np.float64)
        return

    if target_value.shape == ():
        parameter[...] = np.full_like(current_value, float(target_value), dtype=np.float64)
        return

    if target_value.shape == current_value.shape:
        parameter[...] = target_value
        return

    flattened = np.ravel(target_value)
    if flattened.size == current_value.size:
        parameter[...] = flattened.reshape(current_value.shape)
        return

    parameter[...] = np.full_like(
        current_value,
        float(np.median(flattened)),
        dtype=np.float64,
    )


def _negative_conjugate_mll(model: Any, train_data: gpx.Dataset) -> Any:
    """Compute the negative conjugate marginal log-likelihood for optimization."""

    return -gpx.objectives.conjugate_mll(model, train_data)


def _match_parameter_prior_target_shape(
    *,
    parameter_shape: tuple[int, ...],
    target_values: np.ndarray,
) -> np.ndarray:
    """Match one data-informed target array to the shape of a parameter."""

    target_np = np.asarray(target_values, dtype=np.float64)

    if parameter_shape == ():
        return np.asarray(np.median(target_np), dtype=np.float64)
    if target_np.shape == ():
        return np.full(parameter_shape, float(target_np), dtype=np.float64)
    if parameter_shape == target_np.shape:
        return target_np

    flattened = np.ravel(target_np)
    if flattened.size == int(np.prod(parameter_shape)):
        return flattened.reshape(parameter_shape)

    return np.full(
        parameter_shape,
        float(np.median(flattened)),
        dtype=np.float64,
    )


def _build_positive_parameter_prior(
    *,
    center: np.ndarray,
    prior: ParameterPriorConfig,
) -> npd.LogNormal:
    """Build a LogNormal prior around one positive or non-negative target."""

    center_np = np.maximum(np.asarray(center, dtype=np.float64), 1e-6)
    return npd.LogNormal(
        loc=jnp.log(jnp.asarray(center_np, dtype=jnp.float64)),
        scale=jnp.asarray(
            _match_parameter_prior_target_shape(
                parameter_shape=center_np.shape,
                target_values=np.full_like(center_np, prior.positive_log_scale, dtype=np.float64),
            ),
            dtype=jnp.float64,
        ),
    )


def _build_real_parameter_prior(
    *,
    center: np.ndarray,
    prior: ParameterPriorConfig,
) -> npd.Normal:
    """Build a Normal prior for unconstrained real-valued parameters."""

    center_np = np.asarray(center, dtype=np.float64)
    scale_np = _match_parameter_prior_target_shape(
        parameter_shape=center_np.shape,
        target_values=np.full_like(center_np, prior.real_scale, dtype=np.float64),
    )
    return npd.Normal(
        loc=jnp.asarray(center_np, dtype=jnp.float64),
        scale=jnp.asarray(np.maximum(scale_np, 1e-6), dtype=jnp.float64),
    )


def _build_parameter_prior_distribution(
    *,
    attribute_name: str,
    parameter: gpx.parameters.Parameter,
    prior: ParameterPriorConfig,
) -> tuple[Any, np.ndarray]:
    """Build one GPJax-native prior distribution for a concrete parameter leaf."""

    values = np.asarray(parameter[...], dtype=np.float64)
    parameter_shape = tuple(values.shape)

    if attribute_name == "lengthscale":
        center = _match_parameter_prior_target_shape(
            parameter_shape=parameter_shape,
            target_values=prior.feature_scales,
        )
        return _build_positive_parameter_prior(center=center, prior=prior), np.asarray(
            center,
            dtype=np.float64,
        )

    if attribute_name == "period":
        center = _match_parameter_prior_target_shape(
            parameter_shape=parameter_shape,
            target_values=np.maximum(prior.feature_spans, prior.feature_scales),
        )
        return _build_positive_parameter_prior(center=center, prior=prior), np.asarray(
            center,
            dtype=np.float64,
        )

    if attribute_name == "obs_stddev":
        center = _match_parameter_prior_target_shape(
            parameter_shape=parameter_shape,
            target_values=np.asarray(prior.target_obs_stddev, dtype=np.float64),
        )
        return _build_positive_parameter_prior(center=center, prior=prior), np.asarray(
            center,
            dtype=np.float64,
        )

    if "variance" in attribute_name:
        center = _match_parameter_prior_target_shape(
            parameter_shape=parameter_shape,
            target_values=np.asarray(prior.target_variance, dtype=np.float64),
        )
        return _build_positive_parameter_prior(center=center, prior=prior), np.asarray(
            center,
            dtype=np.float64,
        )

    if attribute_name == "scale":
        center = _match_parameter_prior_target_shape(
            parameter_shape=parameter_shape,
            target_values=np.asarray(prior.target_std, dtype=np.float64),
        )
        return _build_positive_parameter_prior(center=center, prior=prior), np.asarray(
            center,
            dtype=np.float64,
        )

    if attribute_name == "alpha":
        center = _match_parameter_prior_target_shape(
            parameter_shape=parameter_shape,
            target_values=np.asarray(1.0, dtype=np.float64),
        )
        return _build_positive_parameter_prior(center=center, prior=prior), np.asarray(
            center,
            dtype=np.float64,
        )

    if attribute_name == "constant":
        center = _match_parameter_prior_target_shape(
            parameter_shape=parameter_shape,
            target_values=np.asarray(prior.target_mean, dtype=np.float64),
        )
        return _build_real_parameter_prior(center=center, prior=prior), np.asarray(
            center,
            dtype=np.float64,
        )

    if parameter.tag in {"positive", "non_negative"}:
        center = np.where(
            np.isfinite(values) & (np.abs(values) > 1e-6),
            np.abs(values),
            prior.target_std,
        )
        return _build_positive_parameter_prior(center=center, prior=prior), np.asarray(
            center,
            dtype=np.float64,
        )

    center = np.where(np.isfinite(values), values, prior.target_mean)
    return _build_real_parameter_prior(center=center, prior=prior), np.asarray(
        center,
        dtype=np.float64,
    )


def _attach_parameter_priors_to_component(
    component: Any,
    *,
    prior: ParameterPriorConfig | None,
) -> None:
    """Attach GPJax-native priors to every trainable parameter in one component tree."""

    if component is None or prior is None:
        return

    ignored_attributes = {
        "_pytree__state",
        "_pytree__nodes",
        "active_dims",
        "n_dims",
        "compute_engine",
        "operator",
        "integrator",
        "num_datapoints",
        "name",
    }

    for attribute_name, attribute_value in vars(component).items():
        if attribute_name in ignored_attributes or attribute_name.startswith("_"):
            continue
        if attribute_name in {"kernels", "means"}:
            for child in attribute_value:
                _attach_parameter_priors_to_component(child, prior=prior)
            continue
        if isinstance(attribute_value, gpx.parameters.Parameter):
            distribution, prior_center = _build_parameter_prior_distribution(
                attribute_name=attribute_name,
                parameter=attribute_value,
                prior=prior,
            )
            setattr(
                component,
                attribute_name,
                attribute_value.replace(
                    value=attribute_value.value,
                    prior=distribution,
                    prior_center=prior_center,
                ),
            )


def _attach_parameter_priors_to_model(
    posterior: Any,
    *,
    parameter_prior: ParameterPriorConfig | None,
) -> None:
    """Attach GPJax-native priors to the full posterior hyperparameter tree."""

    if parameter_prior is None:
        return

    _parameterize_mean_function_constants(posterior.prior.mean_function)
    _attach_parameter_priors_to_component(posterior.prior.kernel, prior=parameter_prior)
    _attach_parameter_priors_to_component(posterior.prior.mean_function, prior=parameter_prior)
    _attach_parameter_priors_to_component(posterior.likelihood, prior=parameter_prior)


def _log_parameter_prior_density_from_component(component: Any) -> Any:
    """Accumulate the log density from attached GPJax parameter priors."""

    if component is None:
        return jnp.asarray(0.0, dtype=jnp.float64)

    ignored_attributes = {
        "_pytree__state",
        "_pytree__nodes",
        "active_dims",
        "n_dims",
        "compute_engine",
        "operator",
        "integrator",
        "num_datapoints",
        "name",
    }
    log_density = jnp.asarray(0.0, dtype=jnp.float64)

    for attribute_name, attribute_value in vars(component).items():
        if attribute_name in ignored_attributes or attribute_name.startswith("_"):
            continue
        if attribute_name in {"kernels", "means"}:
            for child in attribute_value:
                log_density = log_density + _log_parameter_prior_density_from_component(child)
            continue
        if isinstance(attribute_value, gpx.parameters.Parameter):
            parameter_prior = attribute_value.get_metadata().get("prior")
            if parameter_prior is None:
                continue
            log_density = log_density + jnp.sum(
                jnp.asarray(
                    parameter_prior.log_prob(
                        jnp.asarray(attribute_value[...], dtype=jnp.float64)
                    ),
                    dtype=jnp.float64,
                )
            )

    return log_density


def _log_parameter_prior_density(model: Any) -> Any:
    """Evaluate the total log density of all attached GPJax parameter priors."""

    return (
        _log_parameter_prior_density_from_component(model.prior.kernel)
        + _log_parameter_prior_density_from_component(model.prior.mean_function)
        + _log_parameter_prior_density_from_component(model.likelihood)
    )


def _negative_log_parameter_prior_density(model: Any) -> Any:
    """Return the negative log density of all attached GPJax parameter priors."""

    return -_log_parameter_prior_density(model)


def _negative_conjugate_map_objective(
    model: Any,
    train_data: gpx.Dataset,
    *,
    parameter_prior: ParameterPriorConfig | None,
) -> Any:
    """Compute a MAP objective from the MLL and attached GPJax parameter priors."""

    objective = _negative_conjugate_mll(model, train_data)
    if parameter_prior is None:
        return objective
    return objective + _negative_log_parameter_prior_density(model)


def _run_adam_optimization(
    *,
    posterior: Any,
    train_data: gpx.Dataset,
    learning_rate: float,
    num_iters: int,
    gradient_clip_norm: float,
    parameter_prior: ParameterPriorConfig | None,
    verbose: bool,
) -> tuple[Any, np.ndarray]:
    """Run clipped Adam on the MAP objective with attached GPJax priors."""

    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm),
        optax.adam(learning_rate),
    )
    objective = lambda model, data: _negative_conjugate_map_objective(
        model,
        data,
        parameter_prior=parameter_prior,
    )
    optimized_posterior, history = gpx.fit(
        model=posterior,
        objective=objective,
        train_data=train_data,
        optim=optimizer,
        num_iters=num_iters,
        verbose=verbose,
        safe=True,
    )
    history_np = np.asarray(history, dtype=np.float64)
    if history_np.size == 0 or not np.all(np.isfinite(history_np)):
        raise FloatingPointError("Adam warm start produced a non-finite objective history.")
    return optimized_posterior, history_np


def _run_lbfgs_optimization(
    *,
    posterior: Any,
    train_data: gpx.Dataset,
    num_iters: int,
    parameter_prior: ParameterPriorConfig | None,
    verbose: bool,
) -> tuple[Any, np.ndarray]:
    """Run Optax L-BFGS refinement on the MAP objective with attached priors."""

    graphdef, params, *static_state = nnx.split(posterior, gpx.parameters.Parameter, ...)
    params = transform(params, DEFAULT_BIJECTION, inverse=True)

    def loss(unconstrained_params: nnx.State) -> Any:
        constrained_params = transform(unconstrained_params, DEFAULT_BIJECTION)
        candidate_model = nnx.merge(graphdef, constrained_params, *static_state)
        return _negative_conjugate_map_objective(
            candidate_model,
            train_data,
            parameter_prior=parameter_prior,
        )

    optimizer = optax.lbfgs()
    opt_state = optimizer.init(params)
    value_and_grad = optax.value_and_grad_from_state(loss)
    history: list[float] = []
    best_params = params
    best_value = np.inf

    progress_bar = (
        trange(num_iters, desc="Running", dynamic_ncols=True)
        if verbose
        else range(num_iters)
    )
    for _ in progress_bar:
        value, grad = value_and_grad(params, state=opt_state)
        value_float = float(np.asarray(value))
        history.append(value_float)
        if verbose:
            progress_bar.set_postfix({"Value": f"{value_float: .2f}"}, refresh=False)
        if not np.isfinite(value_float):
            if verbose:
                progress_bar.close()
            raise FloatingPointError("L-BFGS refinement produced a non-finite objective value.")
        if not _tree_all_finite(grad):
            if verbose:
                progress_bar.close()
            raise FloatingPointError("L-BFGS refinement produced non-finite gradients.")

        updates, opt_state = optimizer.update(
            grad,
            opt_state,
            params,
            value=value,
            grad=grad,
            value_fn=loss,
        )
        params = optax.apply_updates(params, updates)
        if not _tree_all_finite(params):
            if verbose:
                progress_bar.close()
            raise FloatingPointError("L-BFGS refinement produced non-finite parameters.")

        candidate_value = _safe_objective_value(loss, params)
        if candidate_value < best_value:
            best_value = candidate_value
            best_params = params

    if verbose:
        progress_bar.close()

    constrained_params = transform(best_params, DEFAULT_BIJECTION)
    refined_posterior = nnx.merge(graphdef, constrained_params, *static_state)
    return refined_posterior, np.asarray(history, dtype=np.float64)


def _evaluate_negative_conjugate_map_objective(
    model: Any,
    train_data: gpx.Dataset,
    *,
    parameter_prior: ParameterPriorConfig | None = None,
) -> float:
    """Evaluate the scalar MAP objective and require a finite result."""

    value = float(
        np.asarray(
            _negative_conjugate_map_objective(
                model,
                train_data,
                parameter_prior=parameter_prior,
            )
        )
    )
    if not np.isfinite(value):
        raise FloatingPointError("Optimization produced a non-finite objective value.")
    return value


def _ensure_model_parameters_are_finite(model: Any) -> None:
    """Validate that all trainable GPJax parameters in a model remain finite."""

    _, params, *_ = nnx.split(model, gpx.parameters.Parameter, ...)
    if not _tree_all_finite(params):
        raise FloatingPointError("Optimization produced non-finite GP hyperparameters.")


def _tree_all_finite(tree: Any) -> bool:
    """Return whether every numeric leaf in a pytree is finite."""

    leaves = jax.tree_util.tree_leaves(tree)
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


def _safe_objective_value(objective: Any, params: nnx.State) -> float:
    """Evaluate an objective and collapse failures to positive infinity."""

    try:
        value = float(np.asarray(objective(params)))
    except Exception:
        return float("inf")
    if not np.isfinite(value):
        return float("inf")
    return value


def _collect_fit_objective_terms(model: Any, train_data: gpx.Dataset) -> dict[str, float]:
    """Collect explicit MLL, log-prior, and MAP objective terms for one model."""

    train_mll = float(np.asarray(gpx.objectives.conjugate_mll(model, train_data), dtype=np.float64))
    train_log_prior_density = float(np.asarray(_log_parameter_prior_density(model), dtype=np.float64))
    train_log_posterior_density = train_mll + train_log_prior_density
    return {
        "train_mll": train_mll,
        "train_log_prior_density": train_log_prior_density,
        "train_log_posterior_density": train_log_posterior_density,
        "train_map_objective": -train_log_posterior_density,
    }


def _build_optimization_summary(
    *,
    attempt_index: int,
    attempt_name: str,
    strategy: str,
    initial_obs_stddev: float,
    adam_history: np.ndarray,
    lbfgs_history: np.ndarray | None,
    adam_final_map_objective: float,
    final_map_objective: float,
    parameter_prior_strength: float,
    parameter_prior_scale_factor: float,
    adam_model: Any,
    final_model: Any,
    train_data: gpx.Dataset,
) -> dict[str, Any]:
    """Build a compact optimization summary stored on ``GPSpec``."""

    adam_terms = _collect_fit_objective_terms(adam_model, train_data)
    final_terms = _collect_fit_objective_terms(final_model, train_data)
    return {
        "status": "success",
        "attempt_index": attempt_index,
        "attempt_name": attempt_name,
        "strategy": strategy,
        "initial_obs_stddev": float(initial_obs_stddev),
        "parameter_prior_strength": float(parameter_prior_strength),
        "parameter_prior_scale_factor": float(parameter_prior_scale_factor),
        "adam_final_mll": float(adam_terms["train_mll"]),
        "adam_final_log_prior_density": float(adam_terms["train_log_prior_density"]),
        "adam_final_log_posterior_density": float(adam_terms["train_log_posterior_density"]),
        "adam_final_map_objective": float(adam_final_map_objective),
        "final_mll": float(final_terms["train_mll"]),
        "final_log_prior_density": float(final_terms["train_log_prior_density"]),
        "final_log_posterior_density": float(final_terms["train_log_posterior_density"]),
        "final_map_objective": float(final_map_objective),
        "adam_history": adam_history.copy(),
        "lbfgs_history": None if lbfgs_history is None else lbfgs_history.copy(),
        "attempt_errors": (),
    }


def _optimize_local_refit(
    *,
    posterior: Any,
    train_data: gpx.Dataset,
    attempt_name: str,
    attempt_index: int,
    initial_obs_stddev: float,
    adam_learning_rate: float,
    adam_num_iters: int,
    lbfgs_num_iters: int,
    use_lbfgs_refinement: bool,
    gradient_clip_norm: float,
    parameter_prior: ParameterPriorConfig | None,
    parameter_prior_strength: float,
    parameter_prior_scale_factor: float,
    verbose: bool,
) -> tuple[Any, dict[str, Any]]:
    """Optimize one nearby refit starting from an explicit perturbed hyperparameter state."""

    posterior, adam_history = _run_adam_optimization(
        posterior=posterior,
        train_data=train_data,
        learning_rate=adam_learning_rate,
        num_iters=adam_num_iters,
        gradient_clip_norm=gradient_clip_norm,
        parameter_prior=parameter_prior,
        verbose=verbose,
    )
    adam_map_objective = _evaluate_negative_conjugate_map_objective(
        posterior,
        train_data,
        parameter_prior=parameter_prior,
    )
    _ensure_model_parameters_are_finite(posterior)

    if not use_lbfgs_refinement:
        summary = _build_optimization_summary(
            attempt_index=attempt_index,
            attempt_name=attempt_name,
            strategy="adam",
            initial_obs_stddev=initial_obs_stddev,
            adam_history=adam_history,
            lbfgs_history=None,
            adam_final_map_objective=adam_map_objective,
            final_map_objective=adam_map_objective,
            parameter_prior_strength=parameter_prior_strength,
            parameter_prior_scale_factor=parameter_prior_scale_factor,
            adam_model=posterior,
            final_model=posterior,
            train_data=train_data,
        )
        return posterior, summary

    try:
        refined_posterior, lbfgs_history = _run_lbfgs_optimization(
            posterior=posterior,
            train_data=train_data,
            num_iters=lbfgs_num_iters,
            parameter_prior=parameter_prior,
            verbose=verbose,
        )
        final_map_objective = _evaluate_negative_conjugate_map_objective(
            refined_posterior,
            train_data,
            parameter_prior=parameter_prior,
        )
        _ensure_model_parameters_are_finite(refined_posterior)
    except Exception:
        fallback_summary = _build_optimization_summary(
            attempt_index=attempt_index,
            attempt_name=attempt_name,
            strategy="adam",
            initial_obs_stddev=initial_obs_stddev,
            adam_history=adam_history,
            lbfgs_history=None,
            adam_final_map_objective=adam_map_objective,
            final_map_objective=adam_map_objective,
            parameter_prior_strength=parameter_prior_strength,
            parameter_prior_scale_factor=parameter_prior_scale_factor,
            adam_model=posterior,
            final_model=posterior,
            train_data=train_data,
        )
        fallback_summary["status"] = "adam_only_fallback"
        return posterior, fallback_summary

    summary = _build_optimization_summary(
        attempt_index=attempt_index,
        attempt_name=attempt_name,
        strategy="adam+lbfgs",
        initial_obs_stddev=initial_obs_stddev,
        adam_history=adam_history,
        lbfgs_history=lbfgs_history,
        adam_final_map_objective=adam_map_objective,
        final_map_objective=final_map_objective,
        parameter_prior_strength=parameter_prior_strength,
        parameter_prior_scale_factor=parameter_prior_scale_factor,
        adam_model=posterior,
        final_model=refined_posterior,
        train_data=train_data,
    )
    return refined_posterior, summary


def _format_attempt_error(
    *,
    attempt_index: int,
    attempt_name: str,
    stage: str,
    exc: Exception,
) -> str:
    """Format one optimization-attempt failure for the final error summary."""

    return f"attempt {attempt_index} ({attempt_name}, {stage}): {type(exc).__name__}: {exc}"


def _get_allowed_keyword_arguments(constructor: type[Any]) -> set[str]:
    """Return the keyword arguments accepted by a GPJax constructor."""

    signature = inspect.signature(constructor.__init__)
    allowed: set[str] = set()
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            allowed.add(name)
    return allowed


__all__ = [
    "PriorSliceData",
    "PosteriorSliceData",
    "GPSpec",
    "GPSpecParseError",
    "GPJaxUtils",
]
