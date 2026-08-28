"""Core evaluation loop for the sample-size-sweep experiment pattern (exp01 and compatible)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from udbench.BaseUDRegressor import BaseUDRegressor
from udbench.benchmarking.Benchmark import UDBench
from udbench.benchmarking.results_io import save_run
from udbench.datasets import DataSet
from udbench.experiments.config import ExperimentConfig
from udbench.experiments.datasets_registry import (
    DATASET_SAMPLE_SIZES,
    _build_noise_fn,
    _probe_preset,
    resolve_dataset_names,
)
from udbench.experiments.io import _append_row
from udbench.models import MODEL_REGISTRY

logger = logging.getLogger(__name__)


# Maps registry key to display group (used in CSV output for grouping plots).
_MODEL_GROUP: dict[str, str] = {
    "BayesianLinearRegressor":            "Linear",
    "NGBoostNIGRegressor":                "TreeBoosting",
    "NGBoostBaggingRegressor":            "TreeBoosting",
    "CatBoostPosteriorSamplingRegressor": "TreeBoosting",
    "CatBoostKGBRegressor":               "TreeBoosting",
    "DKLRegressor":                       "DKL",
    "TabularBNNBaggingRegressor":         "BNN",
    "TabularBNNDeepRegressor":            "BNN",
    "TabularBNNDropoutRegressor":         "BNN",
    "TabularBNNLaplaceRegressor":         "BNN",
    "TabularBNNFSPLaplaceRegressor":      "BNN",
    "TabularBNNSWAGRegressor":            "BNN",
    "TabularBNNEDLRegressor":             "BNN",
    "TabularBNNDEUPRegressor":            "BNN",
}


def _pick_metric(metrics: dict[str, object], key: str) -> float:
    value = metrics.get(key)
    return float("nan") if value is None else float(value)


def _json_dumps(value: object) -> str:
    return json.dumps(value or {}, sort_keys=True, default=str)


def _architecture_metadata(
    model_init_kwargs: dict,
    tuned_params: dict[str, object] | None,
) -> dict[str, object]:
    arch_keys = (
        "hidden_features",
        "hidden_dim",
        "n_layers",
        "backbone",
        "resnet_width",
        "resnet_blocks",
        "activation",
        "optimizer",
    )
    architecture = {key: model_init_kwargs[key] for key in arch_keys if key in model_init_kwargs}
    if tuned_params:
        architecture.update({key: tuned_params[key] for key in arch_keys if key in tuned_params})
    return architecture


def _run_single(
    *,
    preset: str,
    noise_mode: str,
    dataset: DataSet,
    model_name: str,
    model_cls: type,
    model_init_kwargs: dict,
    tune: bool,
    tune_count: int,
    tune_project: str | None,
    tune_metric: str = "val_rmse",
    dataset_suite: str = "custom",
    seed: int = 0,
    experiment: str = "",
    arrays_root: Path | None = None,
) -> dict[str, object]:
    """Optionally tune, fit, evaluate, and return one metrics row."""
    group = _MODEL_GROUP.get(model_name, "Unknown")
    logger.info(
        "preset=%s | noise=%s | n_obs=%d | model=%s (%s)",
        preset, noise_mode, dataset.train_size, model_name, group,
    )
    row_base = {
        "experiment": experiment,
        "preset": preset,
        "dataset_suite": dataset_suite,
        "noise_mode": noise_mode,
        "n_obs": int(dataset.train_size),
        "eval_size": int(dataset.eval_size or dataset.X_eval.shape[0]),
        "model_group": group,
        "model_name": model_name,
        "tune_objective": tune_metric,
        "tune_count": int(tune_count),
        "tuned_params_json": "{}",
    }
    start = time.time()
    tune_seconds = 0.0
    try:
        model: BaseUDRegressor = model_cls(**model_init_kwargs)
        tuned_params: dict[str, object] | None = None
        if tune:
            logger.info("  tuning %s (%s, count=%d)", model_name, tune_metric, tune_count)
            tune_start = time.time()
            tuned_params = BaseUDRegressor.tune(
                model, dataset.X_obs, dataset.y_obs,
                project=tune_project, count=tune_count,
                metric_name=tune_metric,
                rng=seed,
                tags=[experiment, dataset_suite, preset, model_name],
            )
            tune_seconds = time.time() - tune_start

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
        nan = float("nan")
        logger.info(
            "  done  mse=%.4f  rho=%.4f  tune=%.1fs  fit=%.1fs",
            metrics.get("mse_y_pred", nan),
            metrics.get("spearman_rho_y_pred", nan),
            tune_seconds,
            fit_seconds,
        )
        row = {
            **row_base,
            "status": "ok",
            "tune_seconds": tune_seconds,
            "fit_predict_seconds": fit_seconds,
            "tuned_params_json": _json_dumps(tuned_params),
            "mse_y_pred":          _pick_metric(metrics, "mse_y_pred"),
            "rho_y_pred":          _pick_metric(metrics, "spearman_rho_y_pred"),
            "pearson_r_y_pred":    _pick_metric(metrics, "pearson_r_y_pred"),
            "mse_total_uncertainty":       _pick_metric(metrics, "mse_total_uncertainty"),
            "rho_total_uncertainty":       _pick_metric(metrics, "spearman_rho_total_uncertainty"),
            "pearson_r_total_uncertainty": _pick_metric(metrics, "pearson_r_total_uncertainty"),
            "mse_epistemic":       _pick_metric(metrics, "mse_epistemic"),
            "rho_epistemic":       _pick_metric(metrics, "spearman_rho_epistemic"),
            "pearson_r_epistemic": _pick_metric(metrics, "pearson_r_epistemic"),
            "mse_aleatoric":       _pick_metric(metrics, "mse_aleatoric"),
            "rho_aleatoric":       _pick_metric(metrics, "spearman_rho_aleatoric"),
            "pearson_r_aleatoric": _pick_metric(metrics, "pearson_r_aleatoric"),
            "error": None,
        }
        if arrays_root is not None:
            try:
                run_prefix = f"{experiment}_" if experiment else ""
                save_run(
                    bench, metrics,
                    results_root=arrays_root,
                    dataset=preset,
                    model=model_name,
                    run_id=f"{run_prefix}seed{seed}_n{int(dataset.train_size)}_{noise_mode}",
                    metadata={
                        "seed": seed,
                        "n_obs": int(dataset.train_size),
                        "n_eval": int(dataset.X_eval.shape[0]),
                        "n_features": int(dataset.X_eval.shape[1]),
                        "noise_mode": noise_mode,
                        "experiment": experiment,
                        "tune_objective": tune_metric,
                        "status": "ok",
                        "error": None,
                        "tune_seconds": row["tune_seconds"],
                        "fit_predict_seconds": row["fit_predict_seconds"],
                        "architecture": _architecture_metadata(model_init_kwargs, tuned_params),
                        "tuned_params": tuned_params or {},
                    },
                )
            except Exception as save_exc:
                logger.warning("save_run failed — %r", save_exc)
        return row
    except NotImplementedError as exc:
        logger.warning("SKIP  %s — not implemented: %s", model_name, exc)
        return {**row_base, "status": "skipped", "error": str(exc)}
    except ImportError as exc:
        logger.warning("SKIP  %s — missing dependency: %s", model_name, exc)
        return {**row_base, "status": "skipped", "error": str(exc)}
    except Exception as exc:
        logger.error("FAIL  %s — %r", model_name, exc, exc_info=True)
        nan = float("nan")
        return {
            **row_base,
            "status": "error",
            "tune_seconds": tune_seconds,
            "fit_predict_seconds": time.time() - start - tune_seconds,
            "mse_y_pred": nan,
            "rho_y_pred": nan,
            "pearson_r_y_pred": nan,
            "mse_total_uncertainty": nan,
            "rho_total_uncertainty": nan,
            "pearson_r_total_uncertainty": nan,
            "mse_epistemic": nan,
            "rho_epistemic": nan,
            "pearson_r_epistemic": nan,
            "mse_aleatoric": nan,
            "rho_aleatoric": nan,
            "pearson_r_aleatoric": nan,
            "error": repr(exc),
        }


def run_experiment(cfg: ExperimentConfig, *, results_path: Path | None = None) -> int:
    """Run the sample-size sweep defined by cfg.

    Loop structure:
      for preset in resolved datasets:
        for n_obs in cfg.sample_sizes (filtered by pool size):
          for model_name in cfg.models:
            _run_single optionally tunes, fits/evaluates, and appends a row to CSV

    Returns 0 on success.
    """
    if results_path is None:
        results_path = cfg.results_dir / f"{cfg.name}_results.csv"

    # Resolve model classes up front so typos fail before the first training run.
    model_entries: list[tuple[str, type, dict]] = [
        (name, MODEL_REGISTRY[name], dict(cfg.model_kwargs.get(name, {})))
        for name in cfg.models
    ]
    dataset_names = resolve_dataset_names(cfg.datasets, dataset_suite=cfg.dataset_suite)
    dataset_suite = cfg.dataset_suite
    if cfg.datasets:
        try:
            suite_names = set(resolve_dataset_names([], dataset_suite=cfg.dataset_suite))
        except ValueError:
            dataset_suite = "custom"
        else:
            dataset_suite = cfg.dataset_suite if set(dataset_names).issubset(suite_names) else "custom"

    logger.info(
        "%s  datasets=%d  models=%d  seed=%d",
        cfg.name, len(dataset_names), len(model_entries), cfg.seed,
    )
    logger.info(
        "  noise=%s  suite=%s  sample_sizes=%s",
        cfg.noise_mode, dataset_suite, cfg.sample_sizes or "default",
    )

    start_time = time.time()

    for preset in dataset_names:
        pool_size, _ = _probe_preset(
            preset,
            cfg.eval_size,
            cfg.seed,
            max_pool_size=cfg.max_pool_size,
        )
        requested_sizes = (
            list(cfg.sample_sizes)
            if cfg.sample_sizes
            else [DATASET_SAMPLE_SIZES.get(preset, min(pool_size, 1000))]
        )
        sizes = [s for s in requested_sizes if s <= pool_size]
        skipped = [s for s in requested_sizes if s > pool_size]
        if skipped:
            logger.warning("Skipping n_obs %s for %r — pool_size=%d", skipped, preset, pool_size)
        logger.info("── Dataset: %s  pool=%d  sizes=%s ──", preset, pool_size, sizes)

        noise_fn = _build_noise_fn(preset, cfg.eval_size, cfg.seed, cfg.noise_mode)
        noise_kwargs = {"aleatoric_std_fn": noise_fn} if noise_fn is not None else {}

        for n_obs in sizes:
            dataset = DataSet.from_preset(
                preset,
                num_observations=n_obs,
                eval_size=cfg.eval_size,
                key=cfg.seed,
                max_pool_size=cfg.max_pool_size,
                **noise_kwargs,
            )
            logger.info(
                "  train=%d  eval=%d", int(dataset.train_size), int(dataset.X_eval.shape[0])
            )

            for model_name, model_cls, model_kwargs in model_entries:
                row = _run_single(
                    preset=preset,
                    noise_mode=cfg.noise_mode,
                    dataset=dataset,
                    model_name=model_name,
                    model_cls=model_cls,
                    model_init_kwargs=model_kwargs,
                    tune=cfg.tune,
                    tune_count=cfg.tune_count,
                    tune_project=cfg.tune_project,
                    tune_metric=cfg.tune_objective,
                    dataset_suite=dataset_suite,
                    seed=cfg.seed,
                    experiment=cfg.name,
                    arrays_root=cfg.arrays_root,
                )
                _append_row(results_path, row)

    elapsed = time.time() - start_time
    logger.info("Done. Total elapsed: %.2fs  |  Results: %s", elapsed, results_path)
    return 0
