#!/usr/bin/env python3
"""Exp06 — Validation benchmark on the curated dataset suite.

Runs all models over the seven validation presets. Each dataset uses its
canonical sample size from ``udbench.experiments.datasets_registry``.

Usage:
    python run.py
    python run.py --no-tune
    python run.py --datasets sarcos,space_ga
    python run.py --models BayesianLinearRegressor,DKLRegressor
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from udbench._logging import add_logging_args, apply_logging_args
from udbench.experiments.config import ExperimentConfig
from udbench.experiments.runner import run_experiment
from udbench.tuning.search_spaces import JAX_BNN_BASE_SEARCH_SPACE

SEED = 42
# Width values searched during BNN tuning; depth is left to the search space.
BNN_WIDTH_VALUES = [16, 32, 64]

BNN_MODELS = {
    "TabularBNNEDLRegressor",
    "TabularBNNBaggingRegressor",
    "TabularBNNDeepRegressor",
    "TabularBNNDropoutRegressor",
    "TabularBNNLaplaceRegressor",
    "TabularBNNFSPLaplaceRegressor",
    "TabularBNNSWAGRegressor",
    "TabularBNNDEUPRegressor",
}


def _model_kwargs(seed: int) -> dict[str, dict[str, Any]]:
    bnn_kwargs = {"activation": "tanh", "optimizer": "adam", "rng": int(seed)}
    return {
        "NGBoostNIGRegressor": {"random_state": int(seed)},
        "NGBoostBaggingRegressor": {"random_state": int(seed)},
        "CatBoostPosteriorSamplingRegressor": {"random_seed": int(seed)},
        "CatBoostKGBRegressor": {"random_seed": int(seed)},
        "DKLRegressor": {"rng": int(seed)},
        **{name: dict(bnn_kwargs) for name in BNN_MODELS},
    }


cfg = ExperimentConfig(
    name="exp06_validation_benchmark",
    datasets=[],
    dataset_suite="validation",
    models=[
        "BayesianLinearRegressor",
        "NGBoostNIGRegressor",
        "NGBoostBaggingRegressor",
        "CatBoostPosteriorSamplingRegressor",
        "CatBoostKGBRegressor",
        "DKLRegressor",
        "TabularBNNEDLRegressor",
        "TabularBNNBaggingRegressor",
        "TabularBNNDeepRegressor",
        "TabularBNNDropoutRegressor",
        "TabularBNNLaplaceRegressor",
        "TabularBNNFSPLaplaceRegressor",
        "TabularBNNSWAGRegressor",
        "TabularBNNDEUPRegressor",
    ],
    model_kwargs=_model_kwargs(SEED),
    sample_sizes=[],
    noise_mode="heteroscedastic",
    seed=SEED,
    eval_size=1500,
    tune=True,
    tune_objective="val_nll",
    results_dir=Path(__file__).resolve().parent / "results",
    max_pool_size=5000,
    arrays_root=Path(__file__).resolve().parent / "results" / "arrays",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models",        type=str, default="", help="Comma-separated model names (default: all)")
    p.add_argument("--datasets",      type=str, default="", help="Comma-separated validation preset names (default: all 7)")
    p.add_argument("--noise-mode",    type=str, default=cfg.noise_mode)
    p.add_argument("--seed",          type=int, default=cfg.seed)
    p.add_argument("--eval-size",     type=int, default=cfg.eval_size)
    p.add_argument("--sample-sizes",  type=str, default="", help="Comma-separated n_obs values (default: per-dataset canonical sizes)")
    p.add_argument("--tune",          dest="tune", action="store_true", help="Enable tuning (default)")
    p.add_argument("--no-tune",       dest="tune", action="store_false", help="Disable tuning")
    p.add_argument("--tune-project",  type=str, default="exp06_validation")
    p.add_argument("--tune-count",    type=int, default=cfg.tune_count)
    p.add_argument("--tune-objective", type=str, default=cfg.tune_objective)
    p.add_argument("--max-pool-size", type=int, default=cfg.max_pool_size, help="Cap UCI pool before GP sampling (0 = no cap)")
    p.add_argument("--results-dir",   type=str, default=str(cfg.results_dir))
    p.set_defaults(tune=cfg.tune)
    add_logging_args(p)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    apply_logging_args(args)

    # Restrict BNN width search to the three values used in the paper.
    JAX_BNN_BASE_SEARCH_SPACE["hidden_dim"] = {"values": list(BNN_WIDTH_VALUES)}
    JAX_BNN_BASE_SEARCH_SPACE["resnet_width"] = {"values": list(BNN_WIDTH_VALUES)}

    overrides: dict = {
        "noise_mode": args.noise_mode,
        "seed": args.seed,
        "eval_size": args.eval_size,
        "tune": args.tune,
        "tune_project": args.tune_project or None,
        "tune_count": args.tune_count,
        "tune_objective": args.tune_objective,
        "results_dir": Path(args.results_dir),
        "max_pool_size": int(args.max_pool_size) if args.max_pool_size else None,
        "arrays_root": Path(args.results_dir) / "arrays",
        "model_kwargs": _model_kwargs(args.seed),
    }
    if args.datasets:
        overrides["datasets"] = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.sample_sizes:
        overrides["sample_sizes"] = [int(s.strip()) for s in args.sample_sizes.split(",") if s.strip()]
    if args.models:
        requested = {m.strip() for m in args.models.split(",") if m.strip()}
        overrides["models"] = [m for m in cfg.models if m in requested]

    raise SystemExit(run_experiment(replace(cfg, **overrides)))
