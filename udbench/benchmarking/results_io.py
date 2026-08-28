"""Persistence layer for UDBench evaluation runs.

Directory layout
----------------
{results_root}/
├── overview.csv                  ← one row per run, all scalar metrics from evaluate_ud()
└── {dataset}/
    └── {model}/
        ├── {run_id}.npz          ← per-point arrays, reconstructable without re-running
        └── {run_id}.json         ← metadata + full hyperparameter sidecar

Usage
-----
from udbench.benchmarking.results_io import save_run

save_run(
    bench        = bench,          # UDBench instance after evaluate_ud()
    results      = metrics,        # dict returned by bench.evaluate_ud()
    results_root = Path("results"),
    dataset      = "abalone",
    model        = "TabularBNNBaggingRegressor",
    run_id       = "seed42_n750_hetero",
    metadata     = {
        "seed": 42, "n_obs": 750, "n_eval": 1000, "n_features": 8,
        "noise_mode": "heteroscedastic", "experiment": "exp05_tuning_objective",
        "tune_objective": "val_nll", "status": "ok", "error": None,
        "tune_seconds": 12.4, "fit_predict_seconds": 3.1,
        "architecture":  {"width": 128, "depth": 2, "activation": "relu"},
        "tuned_params":  {"learning_rate": 1e-3, "n_epochs": 200},
    },
)

Reconstructing all targets from the saved arrays
-------------------------------------------------
arrays = np.load("results/abalone/TabularBNNBaggingRegressor/seed42_n750_hetero.npz")
from udbench.ground_truth_UD.GTDisentangle import _gaussian_scores

squared_bias       = (arrays["y_hat"] - arrays["posterior_mean"]) ** 2
epistemic_mse      = squared_bias + arrays["posterior_var"]
total_mse          = epistemic_mse + arrays["aleatoric_noise_var"]
realized_error     = (arrays["y_hat"] - arrays["f_true"]) ** 2
total_nll,  total_crps  = _gaussian_scores(arrays["posterior_mean"],
                              arrays["posterior_var"] + arrays["aleatoric_noise_var"],
                              arrays["y_eval"])
epistemic_nll, epistemic_crps = _gaussian_scores(arrays["posterior_mean"],
                              arrays["posterior_var"], arrays["f_true"])
aleatoric_nll, aleatoric_crps = _gaussian_scores(arrays["f_true"],
                              arrays["aleatoric_noise_var"], arrays["y_eval"])
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


# Scalar metadata keys written to both the sidecar and the overview CSV (ordered).
_META_KEYS = [
    "dataset", "model", "run_id",
    "experiment", "seed", "n_obs", "n_eval", "n_features", "noise_mode",
    "tune_objective", "status", "error", "tune_seconds", "fit_predict_seconds",
]


def save_run(
    bench,
    results: dict[str, float],
    *,
    results_root: str | Path,
    dataset: str,
    model: str,
    run_id: str,
    metadata: dict[str, Any],
) -> None:
    """Save one evaluation run: arrays (.npz) + sidecar (.json) + overview row (.csv).

    Parameters
    ----------
    bench        : UDBench instance (evaluate_ud() must have been called first).
    results      : scalar metrics dict returned by bench.evaluate_ud().
    results_root : root directory; created if absent.
    dataset      : dataset preset name  → subfolder name.
    model        : model class name     → subfolder name.
    run_id       : unique string for this run, e.g. "seed42_n750_hetero".
    metadata     : run-level metadata.  Keys "architecture" and "tuned_params"
                   are expected to be dicts; they go to the sidecar only (not CSV).
                   All other keys are written to both sidecar and overview CSV.
    """
    results_root = Path(results_root)
    run_dir = results_root / dataset / model
    run_dir.mkdir(parents=True, exist_ok=True)

    _save_arrays(bench, run_dir / f"{run_id}.npz")
    _save_sidecar(
        run_dir / f"{run_id}.json",
        dataset=dataset, model=model, run_id=run_id, metadata=metadata,
    )
    _append_overview_row(
        results_root / "overview.csv",
        dataset=dataset, model=model, run_id=run_id,
        metadata=metadata, results=results,
    )


# ── Arrays ────────────────────────────────────────────────────────────────────

def _save_arrays(bench, path: Path) -> None:
    """Write all per-point arrays to a compressed .npz file.

    Arrays
    ------
    X_eval              (N, D)  eval inputs
    y_eval              (N,)    noisy observations
    f_true              (N,)    noiseless latent function
    posterior_mean      (N,)    GP posterior mean μ
    posterior_var       (N,)    GP posterior variance σ²_f
    aleatoric_noise_var (N,)    ground-truth noise variance σ²_noise
    y_hat               (N,)    model point predictions
    pred_epistemic      (N,)    model EU estimate
    pred_aleatoric      (N,)    model AU estimate
    pred_total          (N,)    model total uncertainty
    """
    def _f32(x):
        return np.asarray(x, dtype=np.float32).ravel()

    arrays: dict[str, np.ndarray] = {}

    ds  = bench.dataset
    gtd = bench.gt_disentangle
    gt  = bench.gt_uncertainty or {}

    # ── Inputs / observations ─────────────────────────────────────────────────
    if ds is not None:
        if getattr(ds, "X_eval", None) is not None:
            arrays["X_eval"] = np.asarray(ds.X_eval, dtype=np.float32)
        if getattr(ds, "y_eval", None) is not None:
            arrays["y_eval"] = _f32(ds.y_eval)
        if getattr(ds, "y_eval_clean", None) is not None:
            arrays["f_true"] = _f32(ds.y_eval_clean)

    # ── GP posterior moments ──────────────────────────────────────────────────
    if getattr(gtd, "_posterior_mean", None) is not None:
        arrays["posterior_mean"] = _f32(gtd._posterior_mean)
    pv = gt.get("posterior_variance")
    av = gt.get("aleatoric_mse")
    if pv is not None:
        arrays["posterior_var"] = _f32(pv)
    if av is not None:
        arrays["aleatoric_noise_var"] = _f32(av)

    # ── Model outputs ─────────────────────────────────────────────────────────
    if bench.y_pred is not None:
        arrays["y_hat"] = _f32(bench.y_pred)
    if bench.pred_epistemic is not None:
        arrays["pred_epistemic"] = _f32(bench.pred_epistemic)
    if bench.pred_aleatoric is not None:
        arrays["pred_aleatoric"] = _f32(bench.pred_aleatoric)
    if bench.pred_total_uncertainty is not None:
        arrays["pred_total"] = _f32(bench.pred_total_uncertainty)

    np.savez_compressed(path, **arrays)


# ── Sidecar JSON ──────────────────────────────────────────────────────────────

def _save_sidecar(
    path: Path,
    *,
    dataset: str,
    model: str,
    run_id: str,
    metadata: dict[str, Any],
) -> None:
    """Write the metadata + hyperparameter sidecar as a JSON file."""
    doc: dict[str, Any] = {
        "dataset": dataset,
        "model":   model,
        "run_id":  run_id,
    }
    # Scalar fields in canonical order
    for key in _META_KEYS:
        if key in metadata and key not in doc:
            doc[key] = metadata[key]
    # Structured sub-dicts (architecture + tuned hyperparameters)
    for key in ("architecture", "tuned_params"):
        doc[key] = metadata.get(key) or {}
    # Any remaining caller-supplied keys
    for key, val in metadata.items():
        if key not in doc:
            doc[key] = val

    path.write_text(json.dumps(doc, indent=2, default=str))


# ── Overview CSV ──────────────────────────────────────────────────────────────

def _append_overview_row(
    path: Path,
    *,
    dataset: str,
    model: str,
    run_id: str,
    metadata: dict[str, Any],
    results: dict[str, float],
) -> None:
    """Append one row to overview.csv.

    Creates the file with a header on first call.  If a new run introduces
    columns not seen before (e.g. a new metric), the file is rewritten with
    the expanded header so no data is lost.
    """
    # Build the flat row (skip nested dicts that belong in sidecar only)
    _nested = {"architecture", "tuned_params"}
    flat_meta: dict[str, Any] = {"dataset": dataset, "model": model, "run_id": run_id}
    for key in _META_KEYS:
        if key in metadata and key not in flat_meta:
            flat_meta[key] = metadata[key]
    for key, val in metadata.items():
        if key not in _nested and key not in flat_meta:
            flat_meta[key] = val

    row = {**flat_meta, **results}

    path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing fieldnames (if any)
    existing_fieldnames: list[str] = []
    existing_rows: list[dict] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    # Union of old and new fieldnames (order-preserving)
    all_fieldnames = list(existing_fieldnames)
    new_cols = [k for k in row if k not in all_fieldnames]
    all_fieldnames.extend(new_cols)

    if new_cols:
        # New columns appeared → rewrite the whole file with the expanded header
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
    else:
        # Common case: just append
        write_header = not existing_fieldnames
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
