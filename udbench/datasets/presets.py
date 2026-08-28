"""Bundle-backed dataset presets.

This module discovers dataset presets exclusively from
``udbench/semi_synth_datasets/preset_outputs``. The old hand-written preset
registry is intentionally removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


PRESET_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1] / "semi_synth_datasets" / "preset_outputs"
)


def left_biased_sampling_fn(
    x,
    y,
    size: int,
    key=100,
    *,
    strength: float = 8.0,
):
    """Sample observations with higher density on the left side of the domain."""

    x = jnp.asarray(x)
    y = jnp.asarray(y)
    t = x[:, 0] if x.ndim == 2 else x
    t01 = (t - t.min()) / (t.max() - t.min() + 1e-12)

    weights = jnp.exp(-strength * t01)
    probs = weights / jnp.sum(weights)

    try:
        key = jax.random.PRNGKey(int(key))
    except (TypeError, ValueError):
        key = jnp.asarray(key)

    idx = jax.random.choice(
        key,
        x.shape[0],
        shape=(size,),
        replace=False,
        p=probs,
    )
    return x[idx], y[idx], idx


def _gated_locally_periodic_kernel_1d(
    x,
    y,
    *,
    sigma_s: float = 0.7,
    ell_s: float = 0.5,
    sigma_p: float = 6.0,
    ell_lp: float = 0.8,
    period: float = 0.3,
    gate_center: float = 1.0,
    gate_width: float = 0.25,
    sigma_lin: float = 1.0,
    lin_bias: float = 0.0,
):
    """One-dimensional kernel with smooth background and localized oscillations."""

    t = x[..., 0]
    u = y[..., 0]
    d = t - u

    k_smooth = (sigma_s**2) * jnp.exp(-0.5 * (d / ell_s) ** 2)

    wx = jnp.exp(-0.5 * ((t - gate_center) / gate_width) ** 2)
    wy = jnp.exp(-0.5 * ((u - gate_center) / gate_width) ** 2)
    gate = wx * wy

    k_cos = jnp.cos(2.0 * jnp.pi * d / period)
    k_rbf_local = jnp.exp(-0.5 * (d / ell_lp) ** 2)
    k_local_osc = (sigma_p**2) * gate * k_rbf_local * k_cos

    k_linear = (sigma_lin**2) * (t + lin_bias) * (u + lin_bias)
    return k_smooth + k_local_osc + k_linear


def _zero_mean_fn(x) -> jnp.ndarray:
    x = jnp.asarray(x)
    return jnp.zeros((x.shape[0],), dtype=jnp.float64)


def _constant_noise_std_fn(noise_std: float) -> Callable:
    def _fn(x):
        x = jnp.asarray(x)
        return jnp.full((x.shape[0],), float(noise_std), dtype=jnp.float64)

    return _fn


def _absolute_scaled_noise_std_fn(scale: float) -> Callable:
    def _fn(x):
        x = jnp.asarray(x)
        values = x if x.ndim == 1 else x[:, 0]
        return float(scale) * jnp.abs(values)

    return _fn


def _build_kernel_fn(spec: dict) -> Callable:
    kind = str(spec.get("kind", "")).strip()
    params = dict(spec.get("params", {}))
    if kind == "gated_locally_periodic_1d":
        return lambda x, y: _gated_locally_periodic_kernel_1d(x, y, **params)
    raise ValueError(f"Unsupported custom kernel kind: {kind!r}")


def _build_mean_fn(spec: dict) -> Callable:
    kind = str(spec.get("kind", "zero")).strip()
    if kind == "zero":
        return _zero_mean_fn
    raise ValueError(f"Unsupported custom mean kind: {kind!r}")


def _build_noise_fn(spec: dict) -> Callable:
    kind = str(spec.get("kind", "")).strip()
    if kind == "constant":
        return _constant_noise_std_fn(float(spec["std"]))
    if kind == "absolute_scaled":
        return _absolute_scaled_noise_std_fn(float(spec["scale"]))
    raise ValueError(f"Unsupported aleatoric noise kind: {kind!r}")


def _build_sampling_fn(spec: dict) -> Callable:
    method = str(spec.get("method", "")).strip()
    if method == "left_biased":
        strength = float(spec.get("strength", 8.0))
        return lambda x, y, size, key: left_biased_sampling_fn(
            x,
            y,
            size,
            key,
            strength=strength,
        )
    raise ValueError(f"Unsupported sampling method: {method!r}")


def _feature_input_range(feature_summaries: list[dict]) -> list[tuple[float, float]]:
    return [
        (float(feature["minimum"]), float(feature["maximum"]))
        for feature in feature_summaries
    ]


def _load_runtime_bundle_config(payload: dict) -> dict:
    dataset_payload = payload["dataset"]
    full_summary = dataset_payload["full_data_summary"]
    feature_summaries = full_summary["features"]
    runtime = payload["runtime"]

    config = {
        "input_range": runtime.get(
            "input_range",
            _feature_input_range(feature_summaries),
        ),
        "eval_size": int(runtime.get("eval_size", full_summary["num_samples"])),
        "num_observations": int(runtime["num_observations"]),
        "base_dataset_name": str(dataset_payload.get("name", "")),
        "preset_variant": str(payload.get("preset", {}).get("name", "")),
        "mean_fn": _build_mean_fn(runtime.get("mean", {"kind": "zero"})),
        "kernel_fn": _build_kernel_fn(runtime["kernel"]),
        "aleatoric_std_fn": _build_noise_fn(runtime["aleatoric_noise"]),
    }

    sampling = runtime.get("sampling")
    if sampling is not None:
        config["sampling_fn"] = _build_sampling_fn(sampling)

    return config


def _load_gpjax_bundle_config(payload: dict) -> dict:
    dataset_payload = payload["dataset"]
    full_summary = dataset_payload["full_data_summary"]
    feature_summaries = full_summary["features"]
    gp_payload = payload["gp_spec"]

    dataset_id = dataset_payload.get("id")
    feature_names = [str(feature["name"]) for feature in feature_summaries]

    # likelihood_obs_stddev is in the model (standardized) space when the preset
    # was fit on standardized data.  We back-transform it to the public (original)
    # space here so that condition_posterior's public_to_model_obs_stddev correctly
    # re-standardizes it (divides by y_std) before passing it to the GP.
    obs_stddev_model = float(gp_payload["likelihood_obs_stddev"])

    config = {
        "input_range": _feature_input_range(feature_summaries),
        "eval_size": int(full_summary["num_samples"]),
        "uci_dataset_id": int(dataset_id) if dataset_id is not None else None,
        "uci_dataset_name": (
            str(dataset_payload["name"]) if dataset_payload.get("name") is not None else None
        ),
        "uci_feature_subset": feature_names,
        "base_dataset_name": str(dataset_payload.get("name", "")),
        "preset_variant": str(payload.get("preset", {}).get("name", "")),
        "gpjax_kernel_expr": str(gp_payload["fitted_kernel_expr"]),
        "gpjax_mean_expr": str(gp_payload["fitted_mean_expr"]),
    }
    sampling_cfg = payload.get("sampling_cfg")
    if sampling_cfg is not None:
        config["sampling_cfg"] = sampling_cfg

    standardization = payload.get("standardization")
    if standardization is not None:
        y_std = float(standardization["y_std"])
        config["standardize"] = True
        config["x_mean"] = np.asarray(standardization["x_mean"], dtype=np.float64)
        config["x_std"] = np.asarray(standardization["x_std"], dtype=np.float64)
        config["y_mean"] = float(standardization["y_mean"])
        config["y_std"] = y_std
        # Back-transform obs_stddev from standardized → original scale so that
        # aleatoric_std_fn returns public-space values.  condition_posterior will
        # re-divide by y_std when converting to model space.
        obs_stddev_model = obs_stddev_model * y_std
        # Use the original (pre-standardization) feature ranges so that DataSet
        # generates X_eval in the original space and standardizes internally.
        config["input_range"] = [
            (float(lo), float(hi))
            for lo, hi in standardization["original_input_range"]
        ]

    config["aleatoric_std_fn"] = _constant_noise_std_fn(obs_stddev_model)
    return config


def _load_bundle_config(bundle_path: Path) -> dict:
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if "runtime" in payload:
        return _load_runtime_bundle_config(payload)
    if "gp_spec" in payload:
        return _load_gpjax_bundle_config(payload)
    raise ValueError(
        f"Preset bundle {bundle_path} is missing both 'runtime' and 'gp_spec' sections."
    )


def load_preset_config(bundle_path: str | Path) -> dict[str, Any]:
    """Load one preset bundle config from a ``preset.json`` file."""

    return _load_bundle_config(Path(bundle_path))


def _iter_bundle_files() -> list[Path]:
    if not PRESET_OUTPUT_ROOT.exists():
        return []
    return sorted(PRESET_OUTPUT_ROOT.glob("*/*/preset.json"))


def _bundle_key(bundle_path: Path, variant_count: int) -> str:
    dataset_slug = bundle_path.parents[1].name
    preset_slug = bundle_path.parent.name
    if variant_count == 1:
        return dataset_slug
    return f"{dataset_slug}:{preset_slug}"


def _load_presets() -> dict[str, dict]:
    if not PRESET_OUTPUT_ROOT.exists():
        return {}

    bundle_files = _iter_bundle_files()
    variant_counts = {
        dataset_dir.name: len(list(dataset_dir.glob("*/preset.json")))
        for dataset_dir in PRESET_OUTPUT_ROOT.iterdir()
        if dataset_dir.is_dir()
    }

    presets: dict[str, dict] = {}
    for bundle_path in bundle_files:
        key = _bundle_key(bundle_path, variant_counts[bundle_path.parents[1].name])
        presets[key] = _load_bundle_config(bundle_path)
    return presets


PRESETS = _load_presets()


__all__ = ["PRESETS", "left_biased_sampling_fn", "load_preset_config"]
