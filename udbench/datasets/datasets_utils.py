#!/usr/bin/env python3
"""Shared diagnostics for the active dataset API."""

from __future__ import annotations

import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)


def _fmt_value(value: float | None, spec: str = ".6g") -> str:
    try:
        if value is None:
            return "n/a"
        val = float(value)
        if not np.isfinite(val):
            return "n/a"
        return format(val, spec)
    except Exception:
        return "n/a"


def _flag(ok: bool) -> str:
    return "OK" if ok else "WARN"


def _line(key: str, value: str, width: int = 44) -> str:
    return f"  {key:<{width}} {value}"


def print_dataset_diagnostics(dataset, *, preset: str = "n/a", noise_mode: str = "n/a") -> None:
    lines: list[str] = [
        "=" * 78,
        f"Dataset diagnostics  |  preset={preset}  |  noise={noise_mode}  |  n_obs={dataset.train_size}",
        "-" * 78,
    ]

    y_obs = np.asarray(dataset.y_obs).reshape(-1)
    y_min = np.min(y_obs) if y_obs.size else np.nan
    y_max = np.max(y_obs) if y_obs.size else np.nan
    y_std = np.std(y_obs) if y_obs.size else np.nan

    lines.append("Targets")
    lines.append(_line("range [min, max]", f"[{_fmt_value(y_min)}, {_fmt_value(y_max)}]"))
    lines.append(_line("std", _fmt_value(y_std)))
    lines.append(
        _line(
            "finite y_obs",
            f"{_flag(np.isfinite(y_obs).all())} ({int(np.isfinite(y_obs).sum())}/{y_obs.size})",
        )
    )

    lines.append("\nUncertainty decomposition")
    try:
        if dataset.posterior is None:
            raise ValueError("dataset.posterior is None")

        gp = dataset.posterior(dataset.X_eval)
        post_std = np.asarray(gp.std).reshape(-1)
        noise_std = np.asarray(dataset.aleatoric_std_eval).reshape(-1)

        avg_post_std = np.mean(post_std) if post_std.size else np.nan
        std_of_std = np.std(post_std) if post_std.size else np.nan
        avg_noise_std = np.mean(noise_std) if noise_std.size else np.nan
        snr = (avg_post_std / avg_noise_std) if (np.isfinite(avg_noise_std) and avg_noise_std > 0) else np.nan

        p10 = np.percentile(post_std, 10) if post_std.size else np.nan
        p90 = np.percentile(post_std, 90) if post_std.size else np.nan
        p10_p90_ratio = (p10 / p90) if (np.isfinite(p90) and p90 > 0) else np.nan

        lines.append(_line("avg posterior std       E[Std_f|D]", _fmt_value(avg_post_std)))
        lines.append(_line("avg noise std           E[Std_eps]", _fmt_value(avg_noise_std)))
        lines.append(_line("signal/noise            E[Std_f]/E[Std_eps]", _fmt_value(snr)))
        lines.append(_line("std of posterior std    Std(Std_f|D)", _fmt_value(std_of_std)))
        lines.append(_line("posterior std spread    p10/p90", _fmt_value(p10_p90_ratio)))

        prior = dataset.posterior.prior
        prior_gp = prior(dataset.X_eval)
        prior_std = np.asarray(prior_gp.std).reshape(-1)

        m = min(post_std.size, prior_std.size)
        ps = post_std[:m]
        prs = prior_std[:m]

        low_mask = ps < 0.2 * prs if m else np.array([], dtype=bool)
        high_mask = ps >= 0.8 * prs if m else np.array([], dtype=bool)
        low_vs_prior = bool(np.any(low_mask)) if m else False
        high_vs_prior = bool(np.any(high_mask)) if m else False
        low_pct = 100.0 * float(np.mean(low_mask)) if m else 0.0
        high_pct = 100.0 * float(np.mean(high_mask)) if m else 0.0

        lines.append(_line("posterior < 20% of prior somewhere", f"{_flag(low_vs_prior)} ({low_vs_prior})"))
        lines.append(_line("posterior >= 80% of prior somewhere", f"{_flag(high_vs_prior)} ({high_vs_prior})"))
        lines.append(_line("pct posterior < 20% of prior", f"{low_pct:.2f}%"))
        lines.append(_line("pct posterior >= 80% of prior", f"{high_pct:.2f}%"))

        finite_post = np.isfinite(post_std).all()
        finite_noise = np.isfinite(noise_std).all()
        lines.append(
            _line(
                "finite posterior stds",
                f"{_flag(finite_post)} ({int(np.isfinite(post_std).sum())}/{post_std.size})",
            )
        )
        lines.append(
            _line(
                "finite noise stds",
                f"{_flag(finite_noise)} ({int(np.isfinite(noise_std).sum())}/{noise_std.size})",
            )
        )

    except Exception as exc:
        lines.append(_line("posterior stats", f"unavailable ({type(exc).__name__}: {exc})"))

    lines.append("=" * 78)
    if logger.isEnabledFor(logging.INFO):
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()


def summarize_dataset_diagnostics(dataset) -> dict:
    y_obs = np.asarray(dataset.y_obs).reshape(-1)
    summary = {
        "n_obs": int(dataset.train_size),
        "n_eval": int(dataset.eval_size or 0),
        "y_min": float(np.min(y_obs)) if y_obs.size else float("nan"),
        "y_max": float(np.max(y_obs)) if y_obs.size else float("nan"),
        "y_std": float(np.std(y_obs)) if y_obs.size else float("nan"),
    }

    if dataset.posterior is None:
        return summary

    gp = dataset.posterior(dataset.X_eval)
    post_std = np.asarray(gp.std).reshape(-1)
    noise_std = np.asarray(dataset.aleatoric_std_eval).reshape(-1)
    prior = dataset.posterior.prior
    prior_gp = prior(dataset.X_eval)
    prior_std = np.asarray(prior_gp.std).reshape(-1)

    m = min(post_std.size, prior_std.size)
    if m > 0:
        ps = post_std[:m]
        prs = prior_std[:m]
        low_mask = ps < 0.2 * prs
        high_mask = ps >= 0.8 * prs
        summary.update(
            {
                "avg_post_std": float(np.mean(post_std)),
                "avg_noise_std": float(np.mean(noise_std)),
                "snr": float(np.mean(post_std) / np.mean(noise_std))
                if np.mean(noise_std) > 0
                else float("nan"),
                "pct_post_lt_20pct_prior": float(100.0 * np.mean(low_mask)),
                "pct_post_ge_80pct_prior": float(100.0 * np.mean(high_mask)),
            }
        )
    return summary


__all__ = ["print_dataset_diagnostics", "summarize_dataset_diagnostics"]
