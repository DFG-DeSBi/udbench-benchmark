#!/usr/bin/env python3
"""Exp05 - Tuning objective comparison on the training dataset suite.

Runs the selected models over the training presets and repeats the full
tune→fit→evaluate cycle once per tuning objective so that val_rmse-tuned and
val_nll-tuned models are evaluated under identical conditions. By default the
experiment keeps the historical BNN sweep, but the CLI can also target the
non-GP baseline family used in the central results.

Available objectives: val_rmse, val_nll, val_mae, val_objective
Default comparison:   val_rmse  vs  val_nll
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
from udbench.tuning.objectives import OBJECTIVE_FUNCTIONS

logger = logging.getLogger("udbench.experiments.exp02")


SEED = 42
DEFAULT_WIDTH = 128
DEFAULT_DEPTH = 2
DEFAULT_ACTIVATION = "relu"
DEFAULT_TUNE_OBJECTIVES = ["val_rmse", "val_nll"]
DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parent / "results" / "exp02_tuning_objective_results.csv"
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
BASELINE_MODELS = [
    "BayesianLinearRegressor",
    "DKLRegressor",
    "NGBoostNIGRegressor",
    "NGBoostBaggingRegressor",
    "CatBoostPosteriorSamplingRegressor",
    "CatBoostKGBRegressor",
]

MODEL_GROUPS = {
    "BayesianLinearRegressor": "Linear",
    "DKLRegressor": "DKL",
    "NGBoostNIGRegressor": "Tree",
    "NGBoostBaggingRegressor": "Tree",
    "CatBoostPosteriorSamplingRegressor": "Tree",
    "CatBoostKGBRegressor": "Tree",
    **{name: "BNN" for name in BNN_MODELS},
}

MODEL_ALIASES = {
    "bnn": BNN_MODELS,
    "bnn_only": BNN_MODELS,
    "baseline": BASELINE_MODELS,
    "baselines": BASELINE_MODELS,
    "non_bnn": BASELINE_MODELS,
    "non_gp": BASELINE_MODELS,
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
    "linear": "BayesianLinearRegressor",
    "dkl": "DKLRegressor",
    "ngboost_nig": "NGBoostNIGRegressor",
    "ngboost_bagging": "NGBoostBaggingRegressor",
    "ngboost_bag": "NGBoostBaggingRegressor",
    "catboost": "CatBoostPosteriorSamplingRegressor",
    "catboost_ps": "CatBoostPosteriorSamplingRegressor",
    "catboost_kgb": "CatBoostKGBRegressor",
}

VALID_OBJECTIVES = sorted(OBJECTIVE_FUNCTIONS)

# Architecture is fully fixed for this ablation; the sweep only tunes
# training hyperparameters and model-specific knobs.
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


def _parse_objectives(value: str) -> list[str]:
    parsed = _parse_csv(value) or list(DEFAULT_TUNE_OBJECTIVES)
    unknown = [o for o in parsed if o not in VALID_OBJECTIVES]
    if unknown:
        raise ValueError(
            f"Unknown tuning objective(s): {unknown}. "
            f"Valid: {VALID_OBJECTIVES}"
        )
    return list(dict.fromkeys(parsed))


def _resolve_models(value: str) -> list[str]:
    requested = _parse_csv(value)
    if not requested:
        return list(BNN_MODELS)
    resolved: list[str] = []
    for name in requested:
        item = MODEL_ALIASES.get(name.strip().lower(), name.strip())
        if isinstance(item, list):
            resolved.extend(item)
        else:
            resolved.append(item)
    return list(dict.fromkeys(resolved))


def _json_dumps(value: object) -> str:
    return json.dumps(value or {}, sort_keys=True, default=str)


def _metric(metrics: dict[str, object], key: str) -> float:
    value = metrics.get(key)
    return float("nan") if value is None else float(value)


def _model_group(model_name: str) -> str:
    return MODEL_GROUPS.get(model_name, "Unknown")


def _model_kwargs(
    model_name: str,
    *,
    width: int,
    depth: int,
    activation: str,
    seed: int,
) -> dict[str, Any]:
    if model_name in BNN_MODELS:
        return {
            "hidden_features": tuple([int(width)] * int(depth)),
            "backbone": "resnet",
            "resnet_width": int(width),
            "resnet_blocks": int(depth),
            "activation": str(activation),
            "rng": int(seed),
        }
    if model_name == "DKLRegressor":
        return {
            "hidden_features": tuple([int(width)] * int(depth)),
            "rng": int(seed),
        }
    if model_name in {"NGBoostNIGRegressor", "NGBoostBaggingRegressor"}:
        return {"random_state": int(seed)}
    if model_name in {"CatBoostPosteriorSamplingRegressor", "CatBoostKGBRegressor"}:
        return {"random_seed": int(seed)}
    if model_name == "BayesianLinearRegressor":
        return {}
    raise KeyError(f"No Exp05 model kwargs mapping defined for {model_name}.")


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
    group = _model_group(model_name)
    show_arch = group in {"BNN", "DKL"}
    return {
        "experiment": "exp02_tuning_objective",
        "preset": preset,
        "dataset_suite": "training",
        "noise_mode": noise_mode,
        "n_obs": int(n_obs),
        "eval_size": int(eval_size),
        "model_group": group,
        "model_name": model_name,
        "width": int(width) if show_arch else None,
        "depth": int(depth) if show_arch else None,
        "activation": activation if show_arch else None,
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
    tune_objective: str,
    tune: bool,
    tune_project: str | None,
    tune_count: int,
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
    group = _model_group(model_name)
    kwargs = _model_kwargs(
        model_name,
        width=width,
        depth=depth,
        activation=activation,
        seed=seed,
    )

    logger.info(
        "preset=%s | model=%s | objective=%s | width=%d | depth=%d",
        preset, model_name, tune_objective, width, depth,
    )
    start = time.time()
    tune_seconds = 0.0
    try:
        model: BaseUDRegressor = model_cls(**kwargs)
        if model_name in BNN_MODELS:
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
                    "exp02",
                    "tuning_objective",
                    preset,
                    model_name,
                    tune_objective,
                    f"width={width}",
                    f"depth={depth}",
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
                    run_id=f"seed{seed}_n{int(dataset.train_size)}_{noise_mode}_obj{tune_objective}",
                    metadata={
                        "seed": seed,
                        "n_obs": int(dataset.train_size),
                        "n_eval": int(dataset.X_eval.shape[0]),
                        "n_features": int(dataset.X_eval.shape[1]),
                        "noise_mode": noise_mode,
                        "experiment": "exp02_tuning_objective",
                        "tune_objective": tune_objective,
                        "status": "ok",
                        "error": None,
                        "tune_seconds": row["tune_seconds"],
                        "fit_predict_seconds": row["fit_predict_seconds"],
                        "architecture": (
                            {"width": width, "depth": depth, "activation": activation}
                            if group in {"BNN", "DKL"} else {}
                        ),
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
        logger.error("FAIL  %s obj=%s — %r", model_name, tune_objective, exc, exc_info=True)
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
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(
            "Comma-separated model names or aliases. "
            "Group aliases: bnn, baseline/non_bnn/non_gp."
        ),
    )
    parser.add_argument(
        "--tune-objectives", type=str,
        default=",".join(DEFAULT_TUNE_OBJECTIVES),
        help=f"Comma-separated objectives to compare. Valid: {VALID_OBJECTIVES}",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION)
    parser.add_argument("--noise-mode", type=str, default="heteroscedastic")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--sample-sizes", type=str, default="", help="Optional comma-separated n_obs values")
    parser.add_argument("--tune-project", type=str, default="exp02_tuning_objective")
    parser.add_argument("--tune-count", type=int, default=10)
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
    if args.depth <= 0:
        raise ValueError(f"Depth must be positive, got {args.depth}.")

    datasets = _parse_csv(args.datasets) if args.datasets else resolve_dataset_names([], dataset_suite="training")
    models = _resolve_models(args.models)
    tune_objectives = _parse_objectives(args.tune_objectives)
    requested_sizes = [int(v) for v in _parse_csv(args.sample_sizes)]
    results_path = Path(args.results_path)
    arrays_root = Path(args.arrays_root) if args.arrays_root else None

    logger.info(
        "exp02_tuning_objective  datasets=%d  models=%d  objectives=%s",
        len(datasets), len(models), tune_objectives,
    )
    logger.info(
        "  width=%d  depth=%d  activation=%s  tune=%s",
        args.width, args.depth, args.activation, args.tune,
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
                    for tune_obj in tune_objectives:
                        row = _base_row(
                            preset=preset,
                            noise_mode=args.noise_mode,
                            n_obs=int(n_obs),
                            eval_size=int(args.eval_size),
                            model_name=model_name,
                            width=int(args.width),
                            depth=int(args.depth),
                            activation=args.activation,
                            tune_objective=tune_obj,
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
                    for tune_obj in tune_objectives:
                        row = _base_row(
                            preset=preset,
                            noise_mode=args.noise_mode,
                            n_obs=int(n_obs),
                            eval_size=int(dataset.eval_size or args.eval_size),
                            model_name=model_name,
                            width=int(args.width),
                            depth=int(args.depth),
                            activation=args.activation,
                            tune_objective=tune_obj,
                            tune_count=int(args.tune_count),
                        )
                        row.update({"status": "import_error", "error": repr(exc)})
                        _append_row(results_path, row, fieldnames=RESULT_FIELDNAMES)
                    continue

                for tune_obj in tune_objectives:
                    row = _run_condition(
                        dataset=dataset,
                        preset=preset,
                        noise_mode=args.noise_mode,
                        model_name=model_name,
                        model_cls=model_cls,
                        width=int(args.width),
                        depth=int(args.depth),
                        activation=args.activation,
                        tune_objective=tune_obj,
                        tune=bool(args.tune),
                        tune_project=args.tune_project or None,
                        tune_count=int(args.tune_count),
                        seed=int(args.seed),
                        arrays_root=arrays_root,
                    )
                    _append_row(results_path, row, fieldnames=RESULT_FIELDNAMES)

    logger.info("Done. Results: %s", results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
