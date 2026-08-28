from __future__ import annotations

from typing import Callable, List, Tuple
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp

from udbench.datasets import DataSet
from udbench.ground_truth_UD.GTDisentangle import GTDisentangle
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
from udbench.BaseUDRegressor import BaseUDRegressor

DEFAULT_METRICS: Tuple[tuple[str, Callable], ...] = (
    ("mse", mean_squared_error),
    ("spearman_rho", spearmanr),
    ("pearson_r", pearsonr),
)
DEFAULT_TOP_OVERLAP_PERCENTAGES: Tuple[float, ...] = (1.0, 5.0, 10.0, 20.0)


def _format_overlap_percent_label(top_percent: float) -> str:
    top_percent = float(top_percent)
    if not 0.0 < top_percent <= 100.0:
        raise ValueError(
            f"top_percent must be in (0, 100], got {top_percent!r}."
        )
    if top_percent.is_integer():
        return f"{int(top_percent)}pct"
    return f"{str(top_percent).replace('.', '_')}pct"


def _percent_label_to_float(label: str) -> float:
    if not label.endswith("pct"):
        raise ValueError(f"Invalid overlap label: {label!r}")
    return float(label[:-3].replace("_", "."))


def _top_percent_overlap_score(
    ground_truth: jnp.ndarray | np.ndarray,
    prediction: jnp.ndarray | np.ndarray,
    top_percent: float,
) -> float:
    ground_truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            "ground_truth and prediction must have the same shape for top-percent overlap."
        )

    mask = np.isfinite(ground_truth) & np.isfinite(prediction)
    n = int(mask.sum())
    if n == 0:
        return float("nan")

    top_percent = float(top_percent)
    if not 0.0 < top_percent <= 100.0:
        raise ValueError(
            f"top_percent must be in (0, 100], got {top_percent!r}."
        )

    ground_truth = ground_truth[mask]
    prediction = prediction[mask]

    k = max(1, int(np.ceil(n * top_percent / 100.0)))
    if k >= n:
        return 1.0

    top_ground_truth = np.argsort(-ground_truth, kind="stable")[:k]
    top_prediction = np.argsort(-prediction, kind="stable")[:k]
    overlap = np.intersect1d(top_ground_truth, top_prediction).size / k
    return float(overlap)

class UDBench:
    """
    Docstring for UDBench
    """
    def __init__(
        self,
        dataset: DataSet | None = None,
        y_pred: jnp.ndarray = None,
        pred_total_uncertainty: jnp.ndarray = None,
        pred_aleatoric: jnp.ndarray = None,
        pred_epistemic: jnp.ndarray = None,
        pred_estimation_error: jnp.ndarray = None,
        pred_approximation_error: jnp.ndarray = None,
        ground_truth_fn_dist=None,
        aleatoric_noise_fn=None,
    ):
        self.dataset = dataset
        self.y_pred = y_pred
        self.pred_total_uncertainty = pred_total_uncertainty
        self.pred_aleatoric = pred_aleatoric
        self.pred_epistemic = pred_epistemic
        self.pred_estimation_error = pred_estimation_error
        self.pred_approximation_error = pred_approximation_error
        self.ground_truth_fn_dist = ground_truth_fn_dist
        self.aleatoric_noise_fn = aleatoric_noise_fn
        self.mean_best_in_class_function = None

        self.gt_disentangle = GTDisentangle(
            dataset=self.dataset,
            y_eval_preds=self.y_pred,
            GroundTruthFnDist=self.ground_truth_fn_dist,
            aleatoric_noise_fn=self.aleatoric_noise_fn,
        )
        
        self.gt_uncertainty = self.gt_disentangle.compute_gt_uncertainty()

    def evaluate_ud(
            self,
            *,
            top_overlap_percentages: Tuple[float, ...] = DEFAULT_TOP_OVERLAP_PERCENTAGES,
        ):
        metric_fns: List[tuple[str, Callable]] = list(DEFAULT_METRICS)

        results = {}

        y_true = self.dataset.y_eval if self.dataset is not None else None
        if self.y_pred is not None and y_true is not None:
            for name, fn in metric_fns:
                score = fn(y_true, self.y_pred)
                if score is None or (hasattr(score, "__len__") and len(score) == 0):
                    warnings.warn(
                        f"{name} score for y_pred could not be calculated.",
                        RuntimeWarning,
                    )
                    continue
                results[f"{name}_y_pred"] = score[0] if hasattr(score, "__len__") else score
        else:
            warnings.warn(
                "y_pred or dataset.y_eval missing; skipping prediction metrics.",
                RuntimeWarning,
            )
        

        gt  = self.gt_uncertainty or {}
        gt_total_mse     = gt.get("total_mse")
        gt_epistemic_mse = gt.get("epistemic_mse")
        gt_aleatoric_mse = gt.get("aleatoric_mse")

        if self.pred_total_uncertainty is not None and gt_total_mse is not None:
            for name, fn in metric_fns:
                score = fn(gt_total_mse, self.pred_total_uncertainty)
                if score is None or (hasattr(score, "__len__") and len(score) == 0):
                    warnings.warn(
                        f"{name} score for total uncertainty could not be calculated.",
                        RuntimeWarning,
                    )
                    continue
                results[f"{name}_total_uncertainty"] = score[0] if hasattr(score, "__len__") else score
        else:
            warnings.warn(
                "Total uncertainty predictions or ground truth missing; skipping.",
                RuntimeWarning,
            )

        if self.pred_epistemic is not None and gt_epistemic_mse is not None:
            for name, fn in metric_fns:
                score = fn(gt_epistemic_mse, self.pred_epistemic)
                if score is None or (hasattr(score, "__len__") and len(score) == 0):
                    warnings.warn(
                        f"{name} score for epistemic uncertainty could not be calculated.",
                        RuntimeWarning,
                    )
                    continue
                results[f"{name}_epistemic"] = score[0] if hasattr(score, "__len__") else score
        else:
            warnings.warn(
                "Epistemic uncertainty predictions or ground truth missing; skipping.",
                RuntimeWarning,
            )

        if self.pred_aleatoric is not None and gt_aleatoric_mse is not None:
            for name, fn in metric_fns:
                score = fn(gt_aleatoric_mse, self.pred_aleatoric)
                if score is None or (hasattr(score, "__len__") and len(score) == 0):
                    warnings.warn(
                        f"{name} score for aleatoric uncertainty could not be calculated.",
                        RuntimeWarning,
                    )
                    continue
                results[f"{name}_aleatoric"] = score[0] if hasattr(score, "__len__") else score
        else:
            warnings.warn(
                "Aleatoric uncertainty predictions or ground truth missing; skipping.",
                RuntimeWarning,
            )

        gt_posterior_variance = gt.get("posterior_variance")
        gt_squared_bias       = gt.get("squared_bias")
        gt_realized_error     = gt.get("realized_error")
        gt_total_nll          = gt.get("total_nll")
        gt_total_crps         = gt.get("total_crps")
        gt_epistemic_nll      = gt.get("epistemic_nll")
        gt_epistemic_crps     = gt.get("epistemic_crps")
        gt_aleatoric_nll      = gt.get("aleatoric_nll")
        gt_aleatoric_crps     = gt.get("aleatoric_crps")

        # additional epistemic targets → compared against pred_epistemic
        for gt_target, key in (
            (gt_posterior_variance, "posterior_variance"),
            (gt_squared_bias,       "squared_bias"),
            (gt_realized_error,     "realized_error"),
            (gt_epistemic_nll,      "epistemic_nll"),
            (gt_epistemic_crps,     "epistemic_crps"),
        ):
            if self.pred_epistemic is None or gt_target is None:
                continue
            for name, fn in metric_fns:
                score = fn(gt_target, self.pred_epistemic)
                if score is None or (hasattr(score, "__len__") and len(score) == 0):
                    continue
                results[f"{name}_{key}"] = score[0] if hasattr(score, "__len__") else score

        # additional total targets → compared against pred_total_uncertainty
        for gt_target, key in (
            (gt_total_nll,  "total_nll"),
            (gt_total_crps, "total_crps"),
        ):
            if self.pred_total_uncertainty is None or gt_target is None:
                continue
            for name, fn in metric_fns:
                score = fn(gt_target, self.pred_total_uncertainty)
                if score is None or (hasattr(score, "__len__") and len(score) == 0):
                    continue
                results[f"{name}_{key}"] = score[0] if hasattr(score, "__len__") else score

        # additional aleatoric targets → compared against pred_aleatoric
        for gt_target, key in (
            (gt_aleatoric_nll,  "aleatoric_nll"),
            (gt_aleatoric_crps, "aleatoric_crps"),
        ):
            if self.pred_aleatoric is None or gt_target is None:
                continue
            for name, fn in metric_fns:
                score = fn(gt_target, self.pred_aleatoric)
                if score is None or (hasattr(score, "__len__") and len(score) == 0):
                    continue
                results[f"{name}_{key}"] = score[0] if hasattr(score, "__len__") else score

        overlap_specs = (
            ("total_uncertainty",  gt_total_mse,          self.pred_total_uncertainty),
            ("epistemic",          gt_epistemic_mse,      self.pred_epistemic),
            ("aleatoric",          gt_aleatoric_mse,      self.pred_aleatoric),
            ("posterior_variance", gt_posterior_variance, self.pred_epistemic),
            ("squared_bias",       gt_squared_bias,       self.pred_epistemic),
            ("realized_error",     gt_realized_error,     self.pred_epistemic),
            ("total_nll",          gt_total_nll,          self.pred_total_uncertainty),
            ("total_crps",         gt_total_crps,         self.pred_total_uncertainty),
            ("epistemic_nll",      gt_epistemic_nll,      self.pred_epistemic),
            ("epistemic_crps",     gt_epistemic_crps,     self.pred_epistemic),
            ("aleatoric_nll",      gt_aleatoric_nll,      self.pred_aleatoric),
            ("aleatoric_crps",     gt_aleatoric_crps,     self.pred_aleatoric),
        )
        for component_name, ground_truth, prediction in overlap_specs:
            if ground_truth is None or prediction is None:
                continue
            for top_percent in top_overlap_percentages:
                label = _format_overlap_percent_label(top_percent)
                results[f"top_overlap_{label}_{component_name}"] = (
                    _top_percent_overlap_score(ground_truth, prediction, top_percent)
                )

        self._print_results_table("\n+++ Uncertainty Decomposition Results:", results)
        self._print_overlap_results_table("\n+++ Top-X% Overlap Results:", results)
        return results

    def approx_excess_risk_decomp(
            self,
            model: BaseUDRegressor,
            mc_samples: int = 100,
            plot_excess_risk_decomp: bool = False,
        ):
        estimation_error, approximation_error, mean_best_in_class_function, mean_gt_function = self.gt_disentangle.approx_excess_risk_decomp(
            model,
            mc_samples=mc_samples,
            plot=plot_excess_risk_decomp,
        )
        self.estimation_error = estimation_error
        self.approximation_error = approximation_error
        self.mean_best_in_class_function = mean_best_in_class_function
        return estimation_error, approximation_error, mean_best_in_class_function, mean_gt_function
        
    def plot_summary(
            self,
            *,
            ax=None,
            use_tsne: bool = True,
            tsne_random_state: int = 0,
            tsne_perplexity: int = 30,
        ):
        if self.dataset is None:
            raise ValueError("Dataset is required for plotting.")

        if self.dataset.input_dim != 1:
            raise ValueError("Summary plots support only 1D data.")

        from utils.plot_utils import PlotUtils  # noqa: PLC0415
        gt  = self.gt_uncertainty or {}
        gt_total     = gt.get("total_mse")
        gt_epistemic = gt.get("epistemic_mse")
        gt_aleatoric = gt.get("aleatoric_mse")

        fig_uq, axes_uq = PlotUtils.plot_uncertainty_comparison(
            self.dataset,
            gt_total=gt_total,
            gt_epistemic=gt_epistemic,
            gt_aleatoric=gt_aleatoric,
            pred_total=self.pred_total_uncertainty,
            pred_epistemic=self.pred_epistemic,
            pred_aleatoric=self.pred_aleatoric,
            y_pred=self.y_pred,
            ax=ax,
            use_tsne=use_tsne,
            tsne_random_state=tsne_random_state,
            tsne_perplexity=tsne_perplexity,
            title_prefix=None,
        )
        fig_err, axes_err = PlotUtils.plot_error_comparison(
            self.dataset,
            gt_estimation=getattr(self, "estimation_error", None),
            gt_approximation=getattr(self, "approximation_error", None),
            pred_estimation=getattr(self, "pred_estimation_error", None),
            pred_approximation=getattr(self, "pred_approximation_error", None),
            y_pred=self.y_pred,
            full_predictions=self.mean_best_in_class_function,
            use_tsne=use_tsne,
            tsne_random_state=tsne_random_state,
            tsne_perplexity=tsne_perplexity,
            title_prefix=None,
        )

        return fig_uq, axes_uq, fig_err, axes_err

    def evaluate_excess_uncertainty_decomp(
            self,

        ):
        metric_fns: List[tuple[str, Callable]] = list(DEFAULT_METRICS)
        results = {}

        gt_estimation = getattr(self, "estimation_error", None)
        gt_approximation = getattr(self, "approximation_error", None)

        if self.pred_estimation_error is not None and gt_estimation is not None:
            for name, fn in metric_fns:
                score = fn(gt_estimation, self.pred_estimation_error)
                if score is None or (hasattr(score, "__len__") and len(score) == 0):
                    warnings.warn(
                        f"{name} score for estimation error could not be calculated.",
                        RuntimeWarning,
                    )
                    continue
                results[f"{name}_estimation_error"] = (
                    score[0] if hasattr(score, "__len__") else score
                )
        else:
            warnings.warn(
                "Estimation error predictions or ground truth missing; skipping.",
                RuntimeWarning,
            )

        if self.pred_approximation_error is not None and gt_approximation is not None:
            for name, fn in metric_fns:
                score = fn(gt_approximation, self.pred_approximation_error)
                if score is None or (hasattr(score, "__len__") and len(score) == 0):
                    warnings.warn(
                        f"{name} score for approximation error could not be calculated.",
                        RuntimeWarning,
                    )
                    continue
                results[f"{name}_approximation_error"] = (
                    score[0] if hasattr(score, "__len__") else score
                )
        else:
            warnings.warn(
                "Approximation error predictions or ground truth missing; skipping.",
                RuntimeWarning,
            )

        self._print_results_table("\n+++ Excess Uncertainty Decomposition Results:", results)
        return results

    @staticmethod
    def _print_results_table(title: str, results: dict) -> None:
        if not results:
            print(f"{title}: no results to display.")
            return
        metrics = ["mse", "spearman_rho", "pearson_r"]
        rows = {}
        for key, value in results.items():
            for metric in metrics:
                suffix = f"{metric}_"
                if key.startswith(suffix):
                    row = key[len(suffix):]
                    rows.setdefault(row, {})[metric] = value
                    break

        if not rows:
            print(f"{title}: no results to display.")
            return

        row_names = list(rows.keys())
        row_width = max(len("Data"), max(len(str(r)) for r in row_names))
        col_widths = {
            m: max(len(m), max(len(f"{rows[r].get(m, float('nan')):.6g}") for r in row_names))
            for m in metrics
        }

        line = "+-" + "-" * row_width + "-+-" + "-+-".join(
            "-" * col_widths[m] for m in metrics
        ) + "-+"
        print(f"{title}")
        print(line)
        print(
            f"| {'Data':<{row_width}} | "
            + " | ".join(f"{m:>{col_widths[m]}}" for m in metrics)
            + " |"
        )
        print(line)
        for row in row_names:
            values = [
                f"{rows[row].get(m, float('nan')):>{col_widths[m]}.6g}"
                for m in metrics
            ]
            print(f"| {row:<{row_width}} | " + " | ".join(values) + " |")
        print(line)

    @staticmethod
    def _print_overlap_results_table(title: str, results: dict) -> None:
        overlap_pattern = re.compile(r"^top_overlap_(.+?pct)_(.+)$")
        rows: dict[str, dict[str, float]] = {}
        percent_labels: list[str] = []
        for key, value in results.items():
            match = overlap_pattern.match(key)
            if match is None:
                continue
            percent_label, row_name = match.groups()
            rows.setdefault(row_name, {})[percent_label] = value
            if percent_label not in percent_labels:
                percent_labels.append(percent_label)

        if not rows:
            return

        percent_labels = sorted(percent_labels, key=_percent_label_to_float)
        row_names = list(rows.keys())
        row_width = max(len("Data"), max(len(str(r)) for r in row_names))
        col_widths = {
            label: max(
                len(label),
                max(
                    len(f"{rows[row].get(label, float('nan')):.6g}")
                    for row in row_names
                ),
            )
            for label in percent_labels
        }

        line = "+-" + "-" * row_width + "-+-" + "-+-".join(
            "-" * col_widths[label] for label in percent_labels
        ) + "-+"
        print(title)
        print(line)
        print(
            f"| {'Data':<{row_width}} | "
            + " | ".join(f"{label:>{col_widths[label]}}" for label in percent_labels)
            + " |"
        )
        print(line)
        for row in row_names:
            values = [
                f"{rows[row].get(label, float('nan')):>{col_widths[label]}.6g}"
                for label in percent_labels
            ]
            print(f"| {row:<{row_width}} | " + " | ".join(values) + " |")
        print(line)

        

        
