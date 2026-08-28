"""Kernel building blocks for GP-based dataset generation.

Provides `KernelSpec`, `ParamSpec`, `make_kernel_spec`, and
`combine_kernel_specs`. These types are consumed by both the generation
workflow (fitting) and the runtime `DataSet` constructor (prior building).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import jax.scipy.special as jsp


@dataclass(frozen=True)
class ParamSpec:
    """Specification for a single named kernel hyperparameter.

    Args:
        name: Parameter name used as a key in the params dict.
        size: Number of scalar values (1 for scalars, d for ARD lengthscales).
        lower: Optional lower bound for optimisation / sampling.
        upper: Optional upper bound for optimisation / sampling.
    """

    name: str
    size: int
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class KernelSpec:
    """Complete specification of a kernel function.

    Bundles the parameter schema, the initialiser, and the kernel callable
    into a single frozen object that can be passed around and composed.

    Args:
        name: Human-readable kernel name.
        params: Ordered list of `ParamSpec` entries describing all hyperparameters.
        kernel_fn: Callable ``(x, y, params) -> scalar`` implementing k(x, y).
        init_fn: Callable ``(dim) -> dict`` that returns default parameter values.
        description: Optional human-readable description.
    """

    name: str
    params: list[ParamSpec]
    kernel_fn: Callable[[jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]
    init_fn: Callable[[int], dict]
    description: str = ""


def _matern_kernel_from_r(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    nu: float,
    eps: float,
) -> jnp.ndarray:
    if abs(nu - 0.5) < 1e-8:
        return theta**2 * jnp.exp(-r)
    if abs(nu - 1.5) < 1e-8:
        scaled = jnp.sqrt(3.0) * r
        return theta**2 * (1.0 + scaled) * jnp.exp(-scaled)
    if abs(nu - 2.5) < 1e-8:
        scaled = jnp.sqrt(5.0) * r
        return theta**2 * (1.0 + scaled + (5.0 / 3.0) * r * r) * jnp.exp(-scaled)
    scaled = jnp.sqrt(2.0 * nu) * r
    scaled = jnp.maximum(scaled, eps)
    coef = (2.0 ** (1.0 - nu)) / jsp.gamma(nu)
    return theta**2 * coef * (scaled**nu) * jsp.kv(nu, scaled)


def _matern_ard_kernel(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    ell: jnp.ndarray,
    theta: jnp.ndarray,
    nu: float,
    eps: float = 1e-12,
) -> jnp.ndarray:
    dx = (x - y) / ell
    r = jnp.sqrt(jnp.sum(dx * dx, axis=-1) + eps)
    r = jnp.maximum(r, eps)
    return _matern_kernel_from_r(r, theta, nu, eps)


def _rq_ard_kernel(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    ell: jnp.ndarray,
    alpha: jnp.ndarray,
    theta: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    dx = (x - y) / ell
    r2 = jnp.sum(dx * dx, axis=-1)
    alpha = jnp.maximum(alpha, eps)
    return theta**2 * (1.0 + 0.5 * r2 / alpha) ** (-alpha)


def _periodic_ard_kernel(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    ell: jnp.ndarray,
    period: jnp.ndarray,
    theta: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    dx = x - y
    period = jnp.maximum(period, eps)
    ell = jnp.maximum(ell, eps)
    sin_term = jnp.sin(jnp.pi * dx / period)
    r2 = jnp.sum((sin_term / ell) ** 2, axis=-1)
    return theta**2 * jnp.exp(-2.0 * r2)


def _arccosine_kernel(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    sigma_w: jnp.ndarray,
    sigma_b: jnp.ndarray,
    degree: int = 1,
    eps: float = 1e-12,
) -> jnp.ndarray:
    x_norm = jnp.sqrt(jnp.sum(x * x, axis=-1) + eps)
    y_norm = jnp.sqrt(jnp.sum(y * y, axis=-1) + eps)
    dot = jnp.sum(x * y, axis=-1)
    cos_theta = dot / (x_norm * y_norm + eps)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)

    if degree == 0:
        j_theta = (jnp.pi - theta) / jnp.pi
        return (sigma_w**2) * j_theta + sigma_b**2
    if degree == 1:
        j_theta = (jnp.sin(theta) + (jnp.pi - theta) * jnp.cos(theta)) / jnp.pi
        return (sigma_w**2 / (2.0 * jnp.pi)) * x_norm * y_norm * j_theta + sigma_b**2
    raise ValueError("Only degree 0 or 1 are supported for ArcCosine kernels.")


def make_kernel_spec(name: str, d: int, **kwargs) -> KernelSpec:
    """Build a `KernelSpec` by name for an input of dimension *d*.

    Args:
        name: Kernel name. Accepted values (case-insensitive): ``"matern32_ard"``,
            ``"matern52_ard"``, ``"rq_ard"``, ``"linear"``, ``"periodic_ard"``,
            ``"arccosine"``.
        d: Input dimensionality. Used to size ARD lengthscale vectors.
        **kwargs: Optional overrides forwarded to the specific kernel factory
            (e.g. ``nu`` for Matérn, ``alpha`` for RQ, ``degree`` for ArcCosine).

    Returns:
        A frozen `KernelSpec` instance.

    Raises:
        ValueError: If *name* does not match any known kernel.
    """
    name = name.lower()

    if name in {"matern32_ard", "matern3/2_ard", "matern_ard"}:
        nu = float(kwargs.get("nu", 1.5))

        def init_fn(dim: int) -> dict:
            return {
                "ell": jnp.ones((dim,), dtype=jnp.float64),
                "theta": jnp.array(1.0, dtype=jnp.float64),
            }

        params = [
            ParamSpec("ell", d, lower=1e-6),
            ParamSpec("theta", 1, lower=1e-6),
        ]

        def kernel_fn(x: jnp.ndarray, y: jnp.ndarray, params: dict) -> jnp.ndarray:
            return _matern_ard_kernel(
                x, y,
                ell=jnp.maximum(params["ell"], 1e-6),
                theta=jnp.maximum(params["theta"], 1e-6),
                nu=nu,
            )

        return KernelSpec(
            name="matern32_ard",
            params=params,
            kernel_fn=kernel_fn,
            init_fn=init_fn,
            description=f"Matérn ARD kernel (nu={nu}).",
        )

    if name in {"matern52_ard", "matern5/2_ard"}:
        nu = float(kwargs.get("nu", 2.5))

        def init_fn(dim: int) -> dict:
            return {
                "ell": jnp.ones((dim,), dtype=jnp.float64),
                "theta": jnp.array(1.0, dtype=jnp.float64),
            }

        params = [
            ParamSpec("ell", d, lower=1e-6),
            ParamSpec("theta", 1, lower=1e-6),
        ]

        def kernel_fn(x: jnp.ndarray, y: jnp.ndarray, params: dict) -> jnp.ndarray:
            return _matern_ard_kernel(
                x, y,
                ell=jnp.maximum(params["ell"], 1e-6),
                theta=jnp.maximum(params["theta"], 1e-6),
                nu=nu,
            )

        return KernelSpec(
            name="matern52_ard",
            params=params,
            kernel_fn=kernel_fn,
            init_fn=init_fn,
            description=f"Matérn ARD kernel (nu={nu}).",
        )

    if name in {"rq_ard", "rational_quadratic", "rq"}:
        alpha_init = float(kwargs.get("alpha", 1.0))

        def init_fn(dim: int) -> dict:
            return {
                "ell": jnp.ones((dim,), dtype=jnp.float64),
                "alpha": jnp.array(alpha_init, dtype=jnp.float64),
                "theta": jnp.array(1.0, dtype=jnp.float64),
            }

        params = [
            ParamSpec("ell", d, lower=1e-6),
            ParamSpec("alpha", 1, lower=1e-6),
            ParamSpec("theta", 1, lower=1e-6),
        ]

        def kernel_fn(x: jnp.ndarray, y: jnp.ndarray, params: dict) -> jnp.ndarray:
            return _rq_ard_kernel(
                x, y,
                ell=jnp.maximum(params["ell"], 1e-6),
                alpha=jnp.maximum(params["alpha"], 1e-6),
                theta=jnp.maximum(params["theta"], 1e-6),
            )

        return KernelSpec(
            name="rq_ard",
            params=params,
            kernel_fn=kernel_fn,
            init_fn=init_fn,
            description="Rational Quadratic ARD kernel.",
        )

    if name in {"linear"}:
        def init_fn(dim: int) -> dict:
            return {
                "lin_sigma": jnp.array(1.0, dtype=jnp.float64),
                "lin_bias": jnp.array(0.0, dtype=jnp.float64),
            }

        params = [
            ParamSpec("lin_sigma", 1, lower=1e-6),
            ParamSpec("lin_bias", 1, lower=0.0),
        ]

        def kernel_fn(x: jnp.ndarray, y: jnp.ndarray, params: dict) -> jnp.ndarray:
            linear_core = jnp.einsum("...d,...d->...", x, y)
            lin_sigma = jnp.maximum(params["lin_sigma"], 1e-6)
            lin_bias = jnp.maximum(params["lin_bias"], 0.0)
            return (lin_sigma**2) * (linear_core + lin_bias)

        return KernelSpec(
            name="linear",
            params=params,
            kernel_fn=kernel_fn,
            init_fn=init_fn,
            description="Linear kernel.",
        )

    if name in {"periodic", "periodic_ard"}:
        def init_fn(dim: int) -> dict:
            return {
                "per_ell": jnp.ones((dim,), dtype=jnp.float64),
                "per_period": jnp.array(1.0, dtype=jnp.float64),
                "per_theta": jnp.array(1.0, dtype=jnp.float64),
            }

        params = [
            ParamSpec("per_ell", d, lower=1e-6),
            ParamSpec("per_period", 1, lower=1e-6),
            ParamSpec("per_theta", 1, lower=1e-6),
        ]

        def kernel_fn(x: jnp.ndarray, y: jnp.ndarray, params: dict) -> jnp.ndarray:
            return _periodic_ard_kernel(
                x, y,
                ell=jnp.maximum(params["per_ell"], 1e-6),
                period=jnp.maximum(params["per_period"], 1e-6),
                theta=jnp.maximum(params["per_theta"], 1e-6),
            )

        return KernelSpec(
            name="periodic_ard",
            params=params,
            kernel_fn=kernel_fn,
            init_fn=init_fn,
            description="Periodic ARD kernel.",
        )

    if name in {"arccosine", "arc_cosine", "neural_network"}:
        degree = int(kwargs.get("degree", 1))

        def init_fn(dim: int) -> dict:
            return {
                "sigma_w": jnp.array(1.0, dtype=jnp.float64),
                "sigma_b": jnp.array(0.1, dtype=jnp.float64),
            }

        params = [
            ParamSpec("sigma_w", 1, lower=1e-6),
            ParamSpec("sigma_b", 1, lower=0.0),
        ]

        def kernel_fn(x: jnp.ndarray, y: jnp.ndarray, params: dict) -> jnp.ndarray:
            return _arccosine_kernel(
                x, y,
                sigma_w=jnp.maximum(params["sigma_w"], 1e-6),
                sigma_b=jnp.maximum(params["sigma_b"], 0.0),
                degree=degree,
            )

        return KernelSpec(
            name="arccosine",
            params=params,
            kernel_fn=kernel_fn,
            init_fn=init_fn,
            description=f"ArcCosine kernel (degree={degree}).",
        )

    raise ValueError(f"Unknown kernel spec '{name}'.")


def combine_kernel_specs(
    specs: list[KernelSpec],
    *,
    name: str | None = None,
    description: str | None = None,
) -> KernelSpec:
    """Additively combine multiple `KernelSpec` instances into one.

    All parameter names across the input specs must be unique, since they are
    merged into a single flat params dict at evaluation time.

    Args:
        specs: Non-empty list of `KernelSpec` instances to add together.
        name: Optional name for the combined kernel. Defaults to the component
            names joined by ``"+"``.
        description: Optional description. Defaults to ``"Sum of kernels."``.

    Returns:
        A new `KernelSpec` whose ``kernel_fn`` returns the sum of the components.

    Raises:
        ValueError: If *specs* is empty or if any parameter names collide.
    """
    if not specs:
        raise ValueError("At least one kernel spec is required.")

    params: list[ParamSpec] = []
    seen: set[str] = set()
    for spec in specs:
        for param in spec.params:
            if param.name in seen:
                raise ValueError(
                    f"Duplicate parameter name '{param.name}' when combining kernels."
                )
            seen.add(param.name)
            params.append(param)

    def init_fn(dim: int) -> dict:
        merged: dict = {}
        for spec in specs:
            merged.update(spec.init_fn(dim))
        return merged

    def kernel_fn(x: jnp.ndarray, y: jnp.ndarray, params_dict: dict) -> jnp.ndarray:
        return sum(spec.kernel_fn(x, y, params_dict) for spec in specs)

    return KernelSpec(
        name=name or "+".join(spec.name for spec in specs),
        params=params,
        kernel_fn=kernel_fn,
        init_fn=init_fn,
        description=description or "Sum of kernels.",
    )


__all__ = [
    "KernelSpec",
    "ParamSpec",
    "combine_kernel_specs",
    "make_kernel_spec",
]
