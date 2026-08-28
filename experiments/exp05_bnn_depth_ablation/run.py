#!/usr/bin/env python3
"""Exp05 - BNN depth ablation on the training dataset suite.

Runs only BNN models. For each training dataset and depth, the architecture is
fixed to width=128, ReLU, and the requested number of hidden layers. W&B tuning
optimizes the remaining BNN hyperparameters using val_rmse.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from udbench._logging import add_logging_args, apply_logging_args
from udbench.BaseUDRegressor import BaseUDRegressor
from udbench.benchmarking.Benchmark import UDBench
from udbench.datasets import DataSet
from udbench.experiments.datasets_registry import (
    DATASET_SAMPLE_SIZES,
    _build_noise_fn,
    resolve_dataset_names,
)
from udbench.benchmarking.results_io import save_run
from udbench.experiments.io import _append_row
from udbench.models import MODEL_REGISTRY

logger = logging.getLogger("udbench.experiments.exp05")


SEED = 42
DEFAULT_WIDTH = 128
DEFAULT_ACTIVATION = "relu"
DEFAULT_DEPTHS = [1, 2, 3, 4, 5, 6]
DEFAULT_TUNE_OBJECTIVE = "val_rmse"
DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parent / "results" / "exp05_bnn_depth_ablation_results.csv"
)
DEFAULT_ARRAYS_ROOT = Path(__file__).resolve().parent / "results" / "arrays"

BNN_MODELS = [
    "TabularBNNEDLRegressor",
    "TabularBNNBaggingRegressor",
    "TabularBNNDeepRegressor",
    "TabularBNNDropoutRegressor",
    "TabularBNNLaplaceRegressor",
    "TabularBNNFSPLaplaceRegressor",
    "TabularBNNSWAGRegressor",
    "TabularBNNDEUPRegressor",
]

MODEL_ALIASES = {
    "edl": "TabularBNNEDLRegressor",
    "bagging": "TabularBNNBaggingRegressor",
    "deep": "TabularBNNDeepRegressor",
    "deep_ensemble": "TabularBNNDeepRegressor",
    "dropout": "TabularBNNDropoutRegressor",
    "laplace": "TabularBNNLaplaceRegressor",
    "fsp_laplace": "TabularBNNFSPLaplaceRegressor",
    "fsp": "TabularBNNFSPLaplaceRegressor",
    "swag": "TabularBNNSWAGRegressor",
    "deup": "TabularBNNDEUPRegressor",
}

# These are the architecture choices fixed by this ablation. The BNN sweep must
# not tune them; it should tune the remaining training and model-specific knobs.
FIXED_SWEEP_PARAMS = (
    "n_layers",
    "hidden_dim",
    "resnet_width",
    "resnet_blocks",
    "activation",
)

RESULT_FIELDNAMES = [
    "experiment",
    "preset",
    "dataset_suite",
    "noise_mode",
    "n_obs",
    "eval_size",
    "model_group",
    "model_name",
    "width",
    "depth",
    "activation",
    "tune_objective",
    "tune_count",
    "tuned_params_json",
    "status",
    "tune_seconds",
    "fit_predict_seconds",
    "mse_y_pred",
    "rho_y_pred",
    "pearson_r_y_pred",
    "mse_total_uncertainty",
    "rho_total_uncertainty",
    "pearson_r_total_uncertainty",
    "mse_epistemic",
    "rho_epistemic",
    "pearson_r_epistemic",
    "mse_aleatoric",
    "rho_aleatoric",
    "pearson_r_aleatoric",
    "error",
]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_depths(value: str) -> list[int]:
    return list(dict.fromkeys(int(part) for part in _parse_csv(value) or DEFAULT_DEPTHS))


def _resolve_models(value: str) -> list[str]:
    requested = _parse_csv(value)
    if not requested:
        return list(BNN_MODELS)
    return list(dict.fromkeys(
        MODEL_ALIASES.get(name.strip().lower(), name.strip()) for name in requested
    ))


def _json_dumps(value: object) -> str:
    return json.dumps(value or {}, sort_keys=True, default=str)


def _metric(metrics: dict[str, object], key: str) -> float:
    value = metrics.get(key)
    return float("nan") if value is None else float(value)


def _model_kwargs(*, width: int, depth: int, activation: str, seed: int) -> dict[str, Any]:
    return {
        "hidden_features": tuple([int(width)] * int(depth)),
        "backbone": "resnet",
        "resnet_width": int(width),
        "resnet_blocks": int(depth),
        "activation": str(activation),
        "rng": int(seed),
    }


def _base_row(
    *,
    preset: str,
    noise_mode: str,
    n_obs: int,
    eval_size: int,
    model_name: str,
    width: int,
    depth: int,
    activation: str,
    tune_objective: str,
    tune_count: int,
) -> dict[str, object]:
    return {
        "experiment": "exp05_bnn_depth_ablation",
        "preset": preset,
        "dataset_suite": "training",
        "noise_mode": noise_mode,
        "n_obs": int(n_obs),
        "eval_size": int(eval_size),
        "model_group": "BNN",
        "model_name": model_name,
        "width": int(width),
        "depth": int(depth),
        "activation": activation,
        "tune_objective": tune_objective,
        "tune_count": int(tune_count),
        "tuned_params_json": "{}",
    }


def _run_condition(
    *,
    dataset: DataSet,
    preset: str,
    noise_mode: str,
    model_name: str,
    model_cls: type,
    width: int,
    depth: int,
    activation: str,
    tune: bool,
    tune_project: str | None,
    tune_count: int,
    tune_objective: str,
    seed: int,
    arrays_root: Path | None = None,
) -> dict[str, object]:
    row = _base_row(
        preset=preset,
        noise_mode=noise_mode,
        n_obs=int(dataset.train_size),
        eval_size=int(dataset.eval_size or dataset.X_eval.shape[0]),
        model_name=model_name,
        width=width,
        depth=depth,
        activation=activation,
        tune_objective=tune_objective,
        tune_count=tune_count,
    )
    kwargs = _model_kwargs(width=width, depth=depth, activation=activation, seed=seed)

    logger.info(
        "preset=%s | model=%s | width=%d | depth=%d | activation=%s",
        preset, model_name, width, depth, activation,
    )
    start = time.time()
    tune_seconds = 0.0
    try:
        model: BaseUDRegressor = model_cls(**kwargs)
        model.fixed_sweep_params = FIXED_SWEEP_PARAMS

        if tune:
            tune_start = time.time()
            tuned_params = BaseUDRegressor.tune(
                model,
                dataset.X_obs,
                dataset.y_obs,
                project=tune_project,
                count=tune_count,
                metric_name=tune_objective,
                rng=seed,
                tags=[
                    "exp10",
                    "depth_ablation",
                    preset,
                    model_name,
                    f"width={width}",
                    f"depth={depth}",
                    f"activation={activation}",
                ],
            )
            tune_seconds = time.time() - tune_start
            row["tuned_params_json"] = _json_dumps(tuned_params)
        else:
            row["tuned_params_json"] = "{}"

        fit_start = time.time()
        preds = model.ud_fit_predict(dataset.X_obs, dataset.y_obs, dataset.X_eval)
        fit_seconds = time.time() - fit_start
        bench = UDBench(
            dataset=dataset,
            y_pred=preds.get("y_pred"),
            pred_total_uncertainty=preds.get("total_uncertainty"),
            pred_aleatoric=preds.get("aleatoric_uncertainty"),
            pred_epistemic=preds.get("epistemic_uncertainty"),
        )
        metrics = bench.evaluate_ud()
        row.update(
            {
                "status": "ok",
                "tune_seconds": tune_seconds,
                "fit_predict_seconds": fit_seconds,
                "mse_y_pred": _metric(metrics, "mse_y_pred"),
                "rho_y_pred": _metric(metrics, "spearman_rho_y_pred"),
                "pearson_r_y_pred": _metric(metrics, "pearson_r_y_pred"),
                "mse_total_uncertainty": _metric(metrics, "mse_total_uncertainty"),
                "rho_total_uncertainty": _metric(metrics, "spearman_rho_total_uncertainty"),
                "pearson_r_total_uncertainty": _metric(metrics, "pearson_r_total_uncertainty"),
                "mse_epistemic": _metric(metrics, "mse_epistemic"),
                "rho_epistemic": _metric(metrics, "spearman_rho_epistemic"),
                "pearson_r_epistemic": _metric(metrics, "pearson_r_epistemic"),
                "mse_aleatoric": _metric(metrics, "mse_aleatoric"),
                "rho_aleatoric": _metric(metrics, "spearman_rho_aleatoric"),
                "pearson_r_aleatoric": _metric(metrics, "pearson_r_aleatoric"),
            }
        )
        if arrays_root is not None:
            try:
                save_run(
                    bench, metrics,
                    results_root=arrays_root,
                    dataset=preset,
                    model=model_name,
                    run_id=f"seed{seed}_n{int(dataset.train_size)}_{noise_mode}_d{depth}",
                    metadata={
                        "seed": seed,
                        "n_obs": int(dataset.train_size),
                        "n_eval": int(dataset.X_eval.shape[0]),
                        "n_features": int(dataset.X_eval.shape[1]),
                        "noise_mode": noise_mode,
                        "experiment": "exp05_bnn_depth_ablation",
                        "tune_objective": tune_objective,
                        "status": "ok",
                        "error": None,
                        "tune_seconds": row["tune_seconds"],
                        "fit_predict_seconds": row["fit_predict_seconds"],
                        "architecture": {"width": width, "depth": depth, "activation": activation},
                        "tuned_params": json.loads(row.get("tuned_params_json", "{}")),
                    },
                )
            except Exception as save_exc:
                logger.warning("save_run failed — %r", save_exc)
        logger.info(
            "  done  mse=%.4f  rho=%.4f  tune=%.1fs  fit=%.1fs",
            row["mse_y_pred"], row["rho_y_pred"], tune_seconds, fit_seconds,
        )
    except Exception as exc:
        logger.error("FAIL  %s depth=%d — %r", model_name, depth, exc, exc_info=True)
        row.update(
            {
                "status": "error",
                "tune_seconds": tune_seconds,
                "fit_predict_seconds": time.time() - start - tune_seconds,
                "tuned_params_json": row.get("tuned_params_json", "{}"),
                "error": repr(exc),
            }
        )
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated training preset names")
    parser.add_argument("--models", type=str, default="", help="Comma-separated BNN model names or aliases")
    parser.add_argument("--depths", type=str, default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION)
    parser.add_argument("--noise-mode", type=str, default="heteroscedastic")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--sample-sizes", type=str, default="", help="Optional comma-separated n_obs values")
    parser.add_argument("--tune-project", type=str, default="exp05_bnn_depth_ablation")
    parser.add_argument("--tune-count", type=int, default=10)
    parser.add_argument("--tune-objective", type=str, default=DEFAULT_TUNE_OBJECTIVE)
    parser.add_argument("--no-tune", dest="tune", action="store_false")
    parser.add_argument("--max-pool-size", type=int, default=5000, help="Cap UCI pool before GP sampling (0 = no cap)")
    parser.add_argument("--results-path", type=str, default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--arrays-root", type=str, default=str(DEFAULT_ARRAYS_ROOT))
    parser.set_defaults(tune=True)
    add_logging_args(parser)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    apply_logging_args(args)

    if args.width <= 0:
        raise ValueError(f"Width must be positive, got {args.width}.")

    datasets = _parse_csv(args.datasets) if args.datasets else resolve_dataset_names([], dataset_suite="training")
    models = _resolve_models(args.models)
    depths = _parse_depths(args.depths)
    requested_sizes = [int(value) for value in _parse_csv(args.sample_sizes)]
    results_path = Path(args.results_path)
    arrays_root = Path(args.arrays_root) if args.arrays_root else None

    logger.info(
        "exp05_bnn_depth_ablation  datasets=%d  models=%d  depths=%s",
        len(datasets), len(models), depths,
    )
    logger.info(
        "  width=%d  activation=%s  tune=%s  objective=%s",
        args.width, args.activation, args.tune, args.tune_objective,
    )

    for preset in datasets:
        sizes = requested_sizes or [DATASET_SAMPLE_SIZES[preset]]

        for n_obs in sizes:
            logger.info("── Dataset: %s  n_obs=%d ──", preset, n_obs)
            try:
                noise_fn = _build_noise_fn(
                    preset,
                    int(args.eval_size),
                    int(args.seed),
                    args.noise_mode,
                )
                noise_kwargs = {"aleatoric_std_fn": noise_fn} if noise_fn is not None else {}
                dataset = DataSet.from_preset(
                    preset,
                    num_observations=int(n_obs),
                    eval_size=int(args.eval_size),
                    key=int(args.seed),
                    max_pool_size=int(args.max_pool_size) if args.max_pool_size else None,
                    **noise_kwargs,
                )
                logger.info("  train=%d  eval=%d", int(dataset.train_size), int(dataset.X_eval.shape[0]))
            except Exception as exc:
                logger.error("Dataset build failed for %s n_obs=%d — %r", preset, n_obs, exc, exc_info=True)
                for model_name in models:
                    for depth in depths:
                        row = _base_row(
                            preset=preset,
                            noise_mode=args.noise_mode,
                            n_obs=int(n_obs),
                            eval_size=int(args.eval_size),
                            model_name=model_name,
                            width=int(args.width),
                            depth=depth,
                            activation=args.activation,
                            tune_objective=args.tune_objective,
                            tune_count=int(args.tune_count),
                        )
                        row.update({"status": "dataset_error", "error": repr(exc)})
                        _append_row(results_path, row, fieldnames=RESULT_FIELDNAMES)
                continue

            for model_name in models:
                try:
                    model_cls = MODEL_REGISTRY[model_name]
                except Exception as exc:
                    logger.error("Model import failed for %s — %r", model_name, exc, exc_info=True)
                    for depth in depths:
                        row = _base_row(
                            preset=preset,
                            noise_mode=args.noise_mode,
                            n_obs=int(n_obs),
                            eval_size=int(dataset.eval_size or args.eval_size),
                            model_name=model_name,
                            width=int(args.width),
                            depth=depth,
                            activation=args.activation,
                            tune_objective=args.tune_objective,
                            tune_count=int(args.tune_count),
                        )
                        row.update({"status": "import_error", "error": repr(exc)})
                        _append_row(results_path, row, fieldnames=RESULT_FIELDNAMES)
                    continue

                for depth in depths:
                    row = _run_condition(
                        dataset=dataset,
                        preset=preset,
                        noise_mode=args.noise_mode,
                        model_name=model_name,
                        model_cls=model_cls,
                        width=int(args.width),
                        depth=int(depth),
                        activation=args.activation,
                        tune=bool(args.tune),
                        tune_project=args.tune_project or None,
                        tune_count=int(args.tune_count),
                        tune_objective=args.tune_objective,
                        seed=int(args.seed),
                        arrays_root=arrays_root,
                    )
                    _append_row(results_path, row, fieldnames=RESULT_FIELDNAMES)

    logger.info("Done. Results: %s", results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
