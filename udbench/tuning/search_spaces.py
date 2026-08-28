from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

WandbSearchSpace = Dict[str, Dict[str, Any]]


WANDB_SEARCH_SPACES: Dict[str, WandbSearchSpace] = {
    "linear": {
        "alpha": {"distribution": "log_uniform_values", "min": 1e-4, "max": 10.0},
        "sigma": {"distribution": "log_uniform_values", "min": 0.01, "max": 2.0},
        "fit_intercept": {"values": [True, False]},
    },
    "dkl": {
        "n_layers": {"distribution": "int_uniform", "min": 1, "max": 3},
        "hidden_dim": {"values": [16, 32, 64, 128]},
        "feature_space_dim": {"distribution": "int_uniform", "min": 2, "max": 8},
        "noise_std": {"distribution": "log_uniform_values", "min": 0.05, "max": 1.0},
        "base_kernel_variance": {"distribution": "log_uniform_values", "min": 0.1, "max": 5.0},
        "base_kernel_lengthscale": {"distribution": "log_uniform_values", "min": 0.1, "max": 5.0},
        "num_iters": {"distribution": "int_uniform", "min": 200, "max": 1000},
        "warmup_steps": {"distribution": "int_uniform", "min": 0, "max": 150},
        "peak_learning_rate": {"distribution": "log_uniform_values", "min": 5e-4, "max": 5e-2},
        "end_learning_rate": {"values": [0.0]},
        "gradient_clip": {"distribution": "uniform", "min": 0.5, "max": 2.0},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        "standardize_inputs": {"values": [True]},
        "standardize_targets": {"values": [True]},
    },
    "ngboost_nig": {
        "n_estimators": {"distribution": "int_uniform", "min": 200, "max": 800},
        "learning_rate": {"distribution": "log_uniform_values", "min": 0.01, "max": 0.2},
        "max_depth": {"distribution": "int_uniform", "min": 2, "max": 6},
        "minibatch_frac": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "col_sample": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "natural_gradient": {"values": [True]},
        "use_svgd": {"values": [False]},
        "evid_strength": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1.0},
        "kl_strength": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1.0},
        "length_scale": {"distribution": "log_uniform_values", "min": 0.1, "max": 5.0},
        "standardize_y": {"values": [True]},
    },
    "ngboost_bagging": {
        "n_estimators": {"distribution": "int_uniform", "min": 200, "max": 800},
        "learning_rate": {"distribution": "log_uniform_values", "min": 0.005, "max": 0.1},
        "max_depth": {"distribution": "int_uniform", "min": 2, "max": 6},
        "minibatch_frac": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "col_sample": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "natural_gradient": {"values": [True, False]},
        "n_regressors": {"distribution": "int_uniform", "min": 5, "max": 25},
        "sample_fraction": {"distribution": "uniform", "min": 0.6, "max": 0.95},
        "replace": {"values": [True, False]},
        "standardize_y": {"values": [True, False]},
    },
    "catboost_posterior": {
        "iterations": {"distribution": "int_uniform", "min": 100, "max": 600},
        "learning_rate": {"distribution": "log_uniform_values", "min": 0.01, "max": 0.3},
        "depth": {"distribution": "int_uniform", "min": 3, "max": 8},
        "l2_leaf_reg": {"distribution": "log_uniform_values", "min": 0.5, "max": 10.0},
        "random_strength": {"distribution": "uniform", "min": 0.0, "max": 2.0},
        "bagging_temperature": {"distribution": "uniform", "min": 0.0, "max": 2.0},
        "n_regressors": {"distribution": "int_uniform", "min": 5, "max": 20},
        "bagging_frac": {"distribution": "uniform", "min": 0.6, "max": 1.0},
        "standardize_y": {"values": [True, False]},
    },
    "catboost_kgb": {
        "posterior_iterations": {"distribution": "int_uniform", "min": 200, "max": 1000},
        "prior_iterations": {"distribution": "int_uniform", "min": 50, "max": 300},
        "learning_rate": {"distribution": "log_uniform_values", "min": 0.03, "max": 0.3},
        "depth": {"distribution": "int_uniform", "min": 4, "max": 8},
        "sigma": {"distribution": "log_uniform_values", "min": 0.05, "max": 0.5},
        "delta": {"distribution": "log_uniform_values", "min": 0.01, "max": 0.5},
        "random_strength": {"distribution": "log_uniform_values", "min": 0.05, "max": 1.0},
        "eps": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "n_regressors": {"distribution": "int_uniform", "min": 5, "max": 20},
        "standardize_y": {"values": [True, False]},
    },
    "bnn_bagging": {
        "n_members": {"values": [10]},
        "sample_fraction": {"distribution": "uniform", "min": 0.6, "max": 0.95},
        "replace": {"values": [True, False]},
    },
    "bnn_deep_ensemble": {
        "n_members": {"values": [10]},
    },
    "bnn_dropout": {
        "n_members": {"values": [10]},
        "dropout": {"distribution": "uniform", "min": 0.05, "max": 0.35},
    },
    "bnn_laplace": {
        "n_members": {"values": [30]},
    },
    "bnn_fsp_laplace": {
        "n_members": {"values": [30]},
    },
    "bnn_swag": {
        "n_members": {"values": [30]},
        "n_checkpoints": {"distribution": "int_uniform", "min": 10, "max": 40},
        "epoch_per_checkpoint": {"distribution": "int_uniform", "min": 2, "max": 12},
        "swag_lr": {"distribution": "log_uniform_values", "min": 1e-3, "max": 0.1},
        "swag_momentum": {"values": [0.0, 0.9, 0.95]},
        "calibration_fraction": {"distribution": "uniform", "min": 0.05, "max": 0.25},
        "grad_clip_norm": {"distribution": "uniform", "min": 0.5, "max": 2.0},
    },
    "bnn_edl": {
        "evi_reg": {"distribution": "log_uniform_values", "min": 1e-3, "max": 3e-1},
    },
    "bnn_deup": {},
    "bnn_deup_excess": {
        "lr_excess": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
        "weight_decay_excess": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        "n_epochs_excess": {"distribution": "int_uniform", "min": 30, "max": 200},
        "batch_size_excess": {"values": [32, 64, 128, 256]},
    },
}

JAX_BNN_BASE_SEARCH_SPACE: WandbSearchSpace = {
    "n_layers": {"values": [1, 2, 3]},
    "hidden_dim": {"values": [32, 64, 128, 256, 512]},
    "resnet_width": {"values": [32, 64, 128, 256, 512]},
    "resnet_blocks": {"values": [1, 2, 3, 4]},
    "n_epochs": {"distribution": "int_uniform", "min": 50, "max": 200},
    "batch_size": {"values": [64, 128, 256, 512]},
    "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 5e-3},
    "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
    "momentum": {"distribution": "uniform", "min": 0.7, "max": 0.95},
}

JAX_BNN_OPTIMIZER_SEARCH_SPACES: Dict[str, WandbSearchSpace] = {
    "adam": {
        "adam_b2": {"distribution": "uniform", "min": 0.95, "max": 0.999},
        "adam_eps": {"distribution": "log_uniform_values", "min": 1e-9, "max": 1e-6},
    },
    "sgd": {
        "sgd_nesterov": {"values": [False, True]},
    },
    "muon": {
        "muon_ns_steps": {"values": [3, 5, 7]},
        "muon_adam_b2": {"distribution": "uniform", "min": 0.95, "max": 0.999},
        "muon_preconditioning": {"values": ["frobenius", "spectral"]},
        "muon_nesterov": {"values": [True, False]},
    },
    "soap": {
        "soap_b2": {"distribution": "uniform", "min": 0.9, "max": 0.99},
        "soap_precondition_frequency": {"values": [1, 5, 10, 20]},
        "soap_precondition_1d": {"values": [False, True]},
    },
}


def get_wandb_search_space(model_key: str | None) -> WandbSearchSpace:
    if model_key is None:
        return {}
    if model_key not in WANDB_SEARCH_SPACES:
        raise KeyError(f"Unknown W&B search space {model_key!r}.")
    return deepcopy(WANDB_SEARCH_SPACES[model_key])


def make_wandb_sweep_config(
    model_key: str,
    metric_name: str,
    *,
    method: str = "bayes",
    goal: str = "minimize",
) -> Dict[str, Any]:
    return {
        "method": method,
        "metric": {"name": metric_name, "goal": goal},
        "parameters": get_wandb_search_space(model_key),
    }


def jax_bnn_base_search_space() -> WandbSearchSpace:
    return deepcopy(JAX_BNN_BASE_SEARCH_SPACE)


def jax_bnn_optimizer_search_space(optimizer: str | None) -> WandbSearchSpace:
    opt_name = str(optimizer or "").lower()
    return deepcopy(JAX_BNN_OPTIMIZER_SEARCH_SPACES.get(opt_name, {}))


def jax_bnn_search_space(
    model_key: str | None = None,
    *,
    optimizer: str | None = None,
) -> WandbSearchSpace:
    params = jax_bnn_base_search_space()
    params.update(get_wandb_search_space(model_key))
    params.update(jax_bnn_optimizer_search_space(optimizer))
    return params
