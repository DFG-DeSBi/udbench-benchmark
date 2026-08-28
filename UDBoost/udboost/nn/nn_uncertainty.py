from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import math
import numpy as np

EnsembleUncertainty = Optional[Literal["bayesian", "levi"]]
DEUPTarget = Literal["epistemic", "predictive"]
DEUPAleatoricMode = Literal["from_fn", "constant", "zero"]


def _require_torch():  # pragma: no cover
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "These udboost.nn uncertainty helpers require PyTorch. Install it (e.g. `pip install torch`)."
        ) from e
    return torch


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    torch = None
    try:
        torch = _require_torch()
    except Exception:
        torch = None
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _postprocess_ensemble_predictions(
    preds: "EnsemblePredictions",
    *,
    uncertainty: EnsembleUncertainty,
    eps: float,
) -> Union["EnsemblePredictions", Dict[str, Optional[np.ndarray]]]:
    if uncertainty is None:
        return preds
    if uncertainty == "bayesian":
        return disentangle_regression_uncertainty(preds, mode="moment", eps=eps)
    if uncertainty == "levi":
        return disentangle_regression_uncertainty(preds, mode="levi_brier", eps=eps)
    raise ValueError(f"Unknown uncertainty: {uncertainty!r}")


@dataclass(frozen=True)
class ParsedModelOutput:
    mean: Any
    log_var: Optional[Any] = None  # Gaussian heteroscedastic head (optional)
    nig: Optional[Tuple[Any, Any, Any, Any]] = None  # (mu, v, alpha, beta) for evidential regression


def _parse_model_output(out: Any) -> ParsedModelOutput:
    """
    Supports:
      - Tensor -> mean
      - (mean, log_var) -> Gaussian heteroscedastic regression
      - (mu, v, alpha, beta) -> evidential regression (Normal-Inverse-Gamma, Amini et al. 2020)
      - dict with keys:
          - 'mean' (+ optionally 'log_var' or 'var')
          - or 'mu','v','alpha','beta' (optionally also 'mean' as alias for 'mu')
    """
    torch = None
    try:
        torch = _require_torch()
    except Exception:
        torch = None

    if torch is not None and isinstance(out, torch.Tensor):
        return ParsedModelOutput(mean=out)
    if all(hasattr(out, k) for k in ("mu", "v", "alpha", "beta")):
        mu = getattr(out, "mu")
        v = getattr(out, "v")
        alpha = getattr(out, "alpha")
        beta = getattr(out, "beta")
        return ParsedModelOutput(mean=mu, nig=(mu, v, alpha, beta))
    if isinstance(out, tuple) and len(out) == 2:
        return ParsedModelOutput(mean=out[0], log_var=out[1])
    if isinstance(out, tuple) and len(out) == 4:
        mu, v, alpha, beta = out
        return ParsedModelOutput(mean=mu, nig=(mu, v, alpha, beta))
    if isinstance(out, dict):
        if {"mu", "v", "alpha", "beta"}.issubset(out.keys()):
            mu = out["mu"]
            v = out["v"]
            alpha = out["alpha"]
            beta = out["beta"]
            return ParsedModelOutput(mean=mu, nig=(mu, v, alpha, beta))

        mean = out.get("mean", None)
        if mean is None and {"v", "alpha", "beta"}.issubset(out.keys()):
            mean = out.get("mu", None)
        if mean is None:
            raise ValueError("Model output dict must contain 'mean' or evidential keys {'mu','v','alpha','beta'}")

        if {"v", "alpha", "beta"}.issubset(out.keys()):
            mu = mean
            v = out["v"]
            alpha = out["alpha"]
            beta = out["beta"]
            return ParsedModelOutput(mean=mu, nig=(mu, v, alpha, beta))

        if "log_var" in out:
            return ParsedModelOutput(mean=mean, log_var=out["log_var"])
        if "var" in out:
            if torch is None:
                log_var = np.log(np.asarray(out["var"]))
            else:
                log_var = torch.log(out["var"])
            return ParsedModelOutput(mean=mean, log_var=log_var)
        return ParsedModelOutput(mean=mean)
    raise TypeError(f"Unsupported model output type: {type(out)}")


def _as_tensor(x: Any, *, device: Any, dtype: Any):
    torch = _require_torch()
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def _call_aleatoric_var_fn(aleatoric_var_fn: Callable[..., Any], X: Any, y: Any = None) -> Any:
    """
    Supports `aleatoric_var_fn(X)` and optionally `aleatoric_var_fn(X, y)` (during training).
    """
    if y is None:
        try:
            return aleatoric_var_fn(X)
        except TypeError as e:
            raise TypeError("aleatoric_var_fn must accept signature aleatoric_var_fn(X) for prediction") from e
    try:
        return aleatoric_var_fn(X, y)
    except TypeError:
        return aleatoric_var_fn(X)


def gaussian_nll_loss(mean, log_var, y, *, reduction: Literal["mean", "sum"] = "mean"):
    """
    Negative log-likelihood for y ~ Normal(mean, exp(0.5*log_var)).
    """
    torch = _require_torch()
    var = torch.exp(log_var)
    nll = 0.5 * (math.log(2.0 * math.pi) + log_var + (y - mean) ** 2 / var)
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    raise ValueError(f"Unknown reduction: {reduction!r}")


def edl_nig_aleatoric_var(mu, v, alpha, beta, *, eps: float = 1e-12):
    """
    Aleatoric variance under NIG evidential regression: beta / (alpha - 1).
    """
    torch = _require_torch()
    alpha_m1 = torch.clamp(alpha - 1.0, min=eps)
    return beta / alpha_m1


def edl_nig_epistemic_var(mu, v, alpha, beta, *, eps: float = 1e-12):
    """
    Epistemic variance proxy under NIG evidential regression: beta / (v * (alpha - 1)).
    """
    torch = _require_torch()
    alpha_m1 = torch.clamp(alpha - 1.0, min=eps)
    v_safe = torch.clamp(v, min=eps)
    return beta / (v_safe * alpha_m1)


def edl_nig_uncertainty(mu, v, alpha, beta, *, eps: float = 1e-12) -> Dict[str, Any]:
    """
    Amini et al. (2020) evidential regression uncertainty decomposition (NIG).
    """
    ale = edl_nig_aleatoric_var(mu, v, alpha, beta, eps=eps)
    epi = edl_nig_epistemic_var(mu, v, alpha, beta, eps=eps)
    return {"mean": mu, "aleatoric": ale, "epistemic": epi, "predictive": ale + epi}


def edl_nig_loss(
    mu,
    v,
    alpha,
    beta,
    y,
    *,
    reg_weight: float = 1e-2,
    reduction: Literal["mean", "sum"] = "mean",
):
    """
    Evidential regression loss (Amini et al., 2020): NIG marginal likelihood + evidence regularizer.
    """
    torch = _require_torch()
    if reg_weight < 0:
        raise ValueError("reg_weight must be >= 0")

    # Negative log marginal likelihood of the Student-t predictive implied by NIG.
    two_beta_v = 2.0 * beta * (1.0 + v)
    nll = (
        0.5 * (math.log(math.pi) - torch.log(v))
        - alpha * torch.log(two_beta_v)
        + (alpha + 0.5) * torch.log(v * (y - mu) ** 2 + two_beta_v)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    # Evidence regularizer encourages higher uncertainty where residuals are large.
    reg = torch.abs(y - mu) * (2.0 * v + alpha)

    if reduction == "mean":
        return nll.mean() + reg_weight * reg.mean()
    if reduction == "sum":
        return nll.sum() + reg_weight * reg.sum()
    raise ValueError(f"Unknown reduction: {reduction!r}")


@dataclass(frozen=True)
class EnsemblePredictions:
    member_means: np.ndarray  # (m, n, d)
    member_vars: Optional[np.ndarray]  # (m, n, d)

@dataclass(frozen=True)
class DEUPRegressor:
    """
    Direct Epistemic Uncertainty Prediction (DEUP) regressor for tabular regression.

    Trains a point predictor `predictor` and an auxiliary uncertainty model `uncertainty_model`
    on cross-fitted squared residuals. See: https://arxiv.org/pdf/2102.08501
    """

    predictor: Any
    uncertainty_model: Any
    include_mean_feature: bool = False
    aleatoric_mode: DEUPAleatoricMode = "zero"
    aleatoric_const: float = 0.0
    aleatoric_var_fn: Optional[Callable[..., Any]] = None
    eps: float = 1e-12


def _predict_model_mean(
    model: Any,
    X: Any,
    *,
    batch_size: int = 4096,
    device: Any = None,
    dtype: Any = None,
) -> np.ndarray:
    torch = _require_torch()
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")

    model = model.to(device=device)
    model.eval()
    X_t = _as_tensor(X, device=device, dtype=dtype)
    n = X_t.shape[0]

    means: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = X_t[start : start + batch_size]
            out = model(xb)
            parsed = _parse_model_output(out)
            means.append(parsed.mean.detach().cpu().numpy())
    mean_all = np.concatenate(means, axis=0)
    if mean_all.ndim == 1:
        mean_all = mean_all[:, None]
    return mean_all


def _deup_features(X: np.ndarray, mean: Optional[np.ndarray], *, include_mean_feature: bool) -> np.ndarray:
    Xn = _to_numpy(X).astype(np.float32, copy=False)
    if not include_mean_feature:
        return Xn
    if mean is None:
        raise ValueError("include_mean_feature=True requires mean predictions")
    mn = _to_numpy(mean).astype(np.float32, copy=False)
    if mn.ndim == 1:
        mn = mn[:, None]
    return np.concatenate([Xn, mn], axis=1)


def deup_fit_regressor_torch(
    predictor_fn: Callable[[], Any],
    uncertainty_model_fn: Callable[..., Any],
    X_train: Any,
    y_train: Any,
    *,
    n_folds: int = 5,
    rng: Union[int, np.random.Generator, None] = None,
    include_mean_feature: bool = False,
    target: DEUPTarget = "epistemic",
    aleatoric_mode: DEUPAleatoricMode = "zero",
    aleatoric_const: float = 0.0,
    aleatoric_var_fn: Optional[Callable[..., Any]] = None,
    fit_kwargs_predictor: Optional[Dict[str, Any]] = None,
    fit_kwargs_uncertainty: Optional[Dict[str, Any]] = None,
    batch_size_pred: int = 4096,
    device: Any = None,
    dtype: Any = None,
    eps: float = 1e-12,
) -> DEUPRegressor:
    """
    Fits DEUP for regression using K-fold cross-fitting.

    Labels for the uncertainty model are cross-fitted squared residuals:
      r_i = (y_i - f_{-fold(i)}(x_i))^2

    If `target='epistemic'`, we optionally subtract an aleatoric variance estimate:
      u_i = max(r_i - sigma_a^2(x_i), 0)

    The uncertainty model is trained on `log(u_i + eps)` with MSE loss and returns a log-variance.
    """
    if n_folds <= 1:
        raise ValueError("n_folds must be >= 2")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if aleatoric_mode == "from_fn" and aleatoric_var_fn is None:
        raise ValueError("aleatoric_mode='from_fn' requires aleatoric_var_fn")
    if aleatoric_mode == "constant" and aleatoric_const < 0:
        raise ValueError("aleatoric_const must be >= 0")

    Xn = _to_numpy(X_train).astype(np.float32, copy=False)
    yn = _to_numpy(y_train).astype(np.float32, copy=False)
    if yn.ndim == 1:
        yn = yn[:, None]

    n = Xn.shape[0]
    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)

    perm = gen.permutation(n)
    folds = np.array_split(perm, n_folds)

    fit_kwargs_predictor = {} if fit_kwargs_predictor is None else dict(fit_kwargs_predictor)
    fit_kwargs_uncertainty = {} if fit_kwargs_uncertainty is None else dict(fit_kwargs_uncertainty)
    if device is not None and "device" not in fit_kwargs_predictor:
        fit_kwargs_predictor["device"] = device
    if dtype is not None and "dtype" not in fit_kwargs_predictor:
        fit_kwargs_predictor["dtype"] = dtype
    if device is not None and "device" not in fit_kwargs_uncertainty:
        fit_kwargs_uncertainty["device"] = device
    if dtype is not None and "dtype" not in fit_kwargs_uncertainty:
        fit_kwargs_uncertainty["dtype"] = dtype
    device_eff = fit_kwargs_predictor.get("device", device)
    dtype_eff = fit_kwargs_predictor.get("dtype", dtype)

    # Cross-fitted residual targets for the uncertainty model.
    u_features: list[np.ndarray] = []
    u_targets_log: list[np.ndarray] = []

    for fold_idx in range(n_folds):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx], axis=0)

        model = predictor_fn()
        model = fit_regressor_torch(model, Xn[train_idx], yn[train_idx], **fit_kwargs_predictor)
        mean_val = _predict_model_mean(model, Xn[val_idx], batch_size=batch_size_pred, device=device_eff, dtype=dtype_eff)

        resid2 = (yn[val_idx] - mean_val) ** 2

        if target == "predictive":
            u_val = resid2
        else:
            if aleatoric_mode == "from_fn":
                ale = _to_numpy(_call_aleatoric_var_fn(aleatoric_var_fn, Xn[val_idx], yn[val_idx])).astype(
                    np.float32, copy=False
                )
                if ale.ndim == 1:
                    ale = ale[:, None]
            elif aleatoric_mode == "constant":
                ale = np.full_like(resid2, float(aleatoric_const), dtype=np.float32)
            else:  # "zero"
                ale = np.zeros_like(resid2, dtype=np.float32)
            u_val = np.maximum(resid2 - ale, 0.0)

        u_features.append(_deup_features(Xn[val_idx], mean_val, include_mean_feature=include_mean_feature))
        u_targets_log.append(np.log(u_val + float(eps)))

    X_u = np.concatenate(u_features, axis=0)
    y_u_log = np.concatenate(u_targets_log, axis=0)

    # Fit the uncertainty model on log-variance targets.
    try:
        u_model = uncertainty_model_fn(X_u.shape[1])
    except TypeError:
        u_model = uncertainty_model_fn()
    u_fit_kwargs = dict(fit_kwargs_uncertainty)
    u_fit_kwargs.setdefault("loss", "mse")
    u_model = fit_regressor_torch(u_model, X_u, y_u_log, **u_fit_kwargs)

    # Refit the predictor on full data for final mean predictions.
    predictor = predictor_fn()
    predictor = fit_regressor_torch(predictor, Xn, yn, **fit_kwargs_predictor)

    return DEUPRegressor(
        predictor=predictor,
        uncertainty_model=u_model,
        include_mean_feature=include_mean_feature,
        aleatoric_mode=aleatoric_mode,
        aleatoric_const=float(aleatoric_const),
        aleatoric_var_fn=aleatoric_var_fn,
        eps=float(eps),
    )


def deup_predict_regressor_torch(
    deup: DEUPRegressor,
    X: Any,
    *,
    batch_size: int = 4096,
    device: Any = None,
    dtype: Any = None,
    aleatoric_var_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Returns dict with keys: mean, epistemic, aleatoric, predictive.

    - epistemic is DEUP's predicted variance (exp of the uncertainty model output).
    - aleatoric is provided by `aleatoric_var_fn` / deup settings when available (otherwise None).
    """
    torch = _require_torch()
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")

    mean = _predict_model_mean(deup.predictor, X, batch_size=batch_size, device=device, dtype=dtype)
    X_feat = _deup_features(_to_numpy(X), mean, include_mean_feature=deup.include_mean_feature)

    log_u = _predict_model_mean(deup.uncertainty_model, X_feat, batch_size=batch_size, device=device, dtype=dtype)
    epistemic = np.exp(log_u)

    # Optional aleatoric component.
    if aleatoric_var_fn is None:
        aleatoric_var_fn = deup.aleatoric_var_fn

    ale = None
    if aleatoric_var_fn is not None:
        ale = _to_numpy(_call_aleatoric_var_fn(aleatoric_var_fn, _to_numpy(X), None)).astype(np.float32, copy=False)
        if ale.ndim == 1:
            ale = ale[:, None]
    elif deup.aleatoric_mode == "constant":
        ale = np.full_like(epistemic, float(deup.aleatoric_const), dtype=np.float32)
    elif deup.aleatoric_mode == "zero":
        ale = None

    pred = None if ale is None else ale + epistemic
    return {"mean": mean, "epistemic": epistemic, "aleatoric": ale, "predictive": pred}


def predict_ensemble(
    models: Sequence[Any],
    X: Any,
    *,
    batch_size: int = 4096,
    device: Any = None,
    dtype: Any = None,
    uncertainty: EnsembleUncertainty = None,
    uncertainty_eps: float = 1e-12,
) -> Union["EnsemblePredictions", Dict[str, Optional[np.ndarray]]]:
    """
    Runs a deterministic ensemble: list of models -> member predictions.

    If `uncertainty` is set, returns an uncertainty dict (see `disentangle_regression_uncertainty`).
    """
    torch = _require_torch()
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")

    X_t = _as_tensor(X, device=device, dtype=dtype)
    n = X_t.shape[0]

    member_means: list[np.ndarray] = []
    member_vars: list[np.ndarray] = []
    any_vars = False

    for model in models:
        model = model.to(device=device)
        model.eval()
        means: list[torch.Tensor] = []
        vars_: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                xb = X_t[start : start + batch_size]
                out = model(xb)
                parsed = _parse_model_output(out)
                means.append(parsed.mean)
                if parsed.log_var is not None:
                    vars_.append(torch.exp(parsed.log_var))
                    any_vars = True
                elif parsed.nig is not None:
                    mu, v, alpha, beta = parsed.nig
                    vars_.append(edl_nig_aleatoric_var(mu, v, alpha, beta))
                    any_vars = True
        mean_all = torch.cat(means, dim=0).detach().cpu().numpy()
        member_means.append(mean_all)
        if vars_:
            var_all = torch.cat(vars_, dim=0).detach().cpu().numpy()
            member_vars.append(var_all)
        else:
            member_vars.append(None)  # placeholder

    means_np = np.stack(member_means, axis=0)
    if any_vars:
        vars_np = np.stack(
            [v if v is not None else np.full_like(means_np[0], np.nan) for v in member_vars],
            axis=0,
        )
        preds = EnsemblePredictions(member_means=means_np, member_vars=vars_np)
        return _postprocess_ensemble_predictions(preds, uncertainty=uncertainty, eps=uncertainty_eps)
    preds = EnsemblePredictions(member_means=means_np, member_vars=None)
    return _postprocess_ensemble_predictions(preds, uncertainty=uncertainty, eps=uncertainty_eps)


def fit_regressor_torch(
    model: Any,
    X_train: Any,
    y_train: Any,
    *,
    n_epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Any = None,
    dtype: Any = None,
    loss: Literal["mse", "gaussian_nll", "edl_nig", "auto"] = "auto",
    edl_reg_weight: float = 1e-2,
    verbose: bool = False,
) -> Any:
    """
    Minimal PyTorch training loop for regression.
    """
    torch = _require_torch()
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")

    X_t = _as_tensor(X_train, device=device, dtype=dtype)
    y_t = _as_tensor(y_train, device=device, dtype=dtype)
    if y_t.ndim == 1:
        y_t = y_t[:, None]

    model = model.to(device=device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = X_t.shape[0]
    indices = torch.arange(n, device=device)

    for epoch in range(n_epochs):
        perm = indices[torch.randperm(n)]
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_t[idx]
            yb = y_t[idx]

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            parsed = _parse_model_output(out)
            mean_t = parsed.mean

            if loss == "auto":
                if parsed.nig is not None:
                    loss_mode = "edl_nig"
                elif parsed.log_var is not None:
                    loss_mode = "gaussian_nll"
                else:
                    loss_mode = "mse"
            else:
                loss_mode = loss

            if loss_mode == "mse":
                batch_loss = torch.nn.functional.mse_loss(mean_t, yb, reduction="mean")
            elif loss_mode == "gaussian_nll":
                if parsed.log_var is None:
                    raise ValueError("loss='gaussian_nll' requires model to output (mean, log_var)")
                batch_loss = gaussian_nll_loss(mean_t, parsed.log_var, yb, reduction="mean")
            elif loss_mode == "edl_nig":
                if parsed.nig is None:
                    raise ValueError("loss='edl_nig' requires model to output (mu, v, alpha, beta)")
                mu, v, alpha, beta = parsed.nig
                batch_loss = edl_nig_loss(mu, v, alpha, beta, yb, reg_weight=edl_reg_weight, reduction="mean")
            else:
                raise ValueError(f"Unknown loss: {loss!r}")

            batch_loss.backward()
            optimizer.step()

            total_loss += float(batch_loss.detach().cpu()) * xb.shape[0]

        if verbose and ((epoch + 1) % max(1, n_epochs // 10) == 0):
            print(f"epoch {epoch+1:4d}/{n_epochs}: loss={total_loss / n:.6f}")

    model.eval()
    return model


def fit_bagging_ensemble(
    model_fn: Callable[[], Any],
    X_train: Any,
    y_train: Any,
    *,
    n_members: int = 10,
    sample_fraction: float = 1.0,
    replace: bool = True,
    rng: Union[int, np.random.Generator, None] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> list[Any]:
    """
    Bagging ensemble via bootstrap resampling + independent training.
    """
    torch = _require_torch()
    Xn = _to_numpy(X_train)
    yn = _to_numpy(y_train)
    n = Xn.shape[0]
    if n_members <= 0:
        raise ValueError("n_members must be positive")
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError("sample_fraction must be in (0, 1]")

    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)

    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    models: list[Any] = []

    m = int(np.ceil(sample_fraction * n))
    for _ in range(n_members):
        idx = gen.choice(n, size=m, replace=replace)
        model = model_fn()
        model = fit_regressor_torch(model, Xn[idx], yn[idx], **fit_kwargs)
        models.append(model)

    return models


def bagging_ensemble_predict(
    model_fn: Callable[[], Any],
    X_train: Any,
    y_train: Any,
    X: Any,
    *,
    n_members: int = 10,
    sample_fraction: float = 1.0,
    replace: bool = True,
    rng: Union[int, np.random.Generator, None] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    predict_kwargs: Optional[Dict[str, Any]] = None,
    uncertainty: EnsembleUncertainty = None,
    uncertainty_eps: float = 1e-12,
) -> Union["EnsemblePredictions", Dict[str, Optional[np.ndarray]]]:
    """
    Convenience wrapper: train a bagging ensemble and return member predictions on X.
    """
    models = fit_bagging_ensemble(
        model_fn,
        X_train,
        y_train,
        n_members=n_members,
        sample_fraction=sample_fraction,
        replace=replace,
        rng=rng,
        fit_kwargs=fit_kwargs,
    )
    predict_kwargs = {} if predict_kwargs is None else dict(predict_kwargs)
    predict_kwargs.pop("uncertainty", None)
    predict_kwargs.pop("uncertainty_eps", None)
    preds = predict_ensemble(models, X, **predict_kwargs)
    return _postprocess_ensemble_predictions(preds, uncertainty=uncertainty, eps=uncertainty_eps)


def fit_mc_dropout_ensemble_predict(
    model_fn: Callable[[], Any],
    X_train: Any,
    y_train: Any,
    X: Any,
    *,
    n_members: int = 30,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    predict_kwargs: Optional[Dict[str, Any]] = None,
    uncertainty: EnsembleUncertainty = None,
    uncertainty_eps: float = 1e-12,
) -> Union["EnsemblePredictions", Dict[str, Optional[np.ndarray]]]:
    """
    Convenience wrapper: train one model, then run MC Dropout to obtain an ensemble.

    Notes
    -----
    - Requires the model architecture to contain dropout layers (e.g. `torch.nn.Dropout`).
    - This function is model-agnostic; it works for `TabularRegressorResNet` as long as `dropout > 0`.
    """
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    predict_kwargs = {} if predict_kwargs is None else dict(predict_kwargs)
    predict_kwargs.pop("n_members", None)
    predict_kwargs.pop("uncertainty", None)
    predict_kwargs.pop("uncertainty_eps", None)

    model = model_fn()
    model = fit_regressor_torch(model, X_train, y_train, **fit_kwargs)
    return mc_dropout_ensemble_predict(
        model,
        X,
        n_members=n_members,
        uncertainty=uncertainty,
        uncertainty_eps=uncertainty_eps,
        **predict_kwargs,
    )


def _enable_dropout_only(model: Any) -> Callable[[], None]:
    """
    Returns a restore function.
    """
    torch = _require_torch()
    was_training = model.training
    model.eval()
    dropout_types = (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train()

    def restore():
        model.train(was_training)

    return restore


def mc_dropout_ensemble_predict(
    model: Any,
    X: Any,
    *,
    n_members: int = 30,
    batch_size: int = 4096,
    device: Any = None,
    dtype: Any = None,
    uncertainty: EnsembleUncertainty = None,
    uncertainty_eps: float = 1e-12,
) -> Union["EnsemblePredictions", Dict[str, Optional[np.ndarray]]]:
    """
    MC Dropout: one trained model, multiple stochastic forward passes.

    If `uncertainty` is set, returns an uncertainty dict (see `disentangle_regression_uncertainty`).
    """
    torch = _require_torch()
    if n_members <= 0:
        raise ValueError("n_members must be positive")
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")

    model = model.to(device=device)
    X_t = _as_tensor(X, device=device, dtype=dtype)
    n = X_t.shape[0]

    restore = _enable_dropout_only(model)
    try:
        member_means: list[np.ndarray] = []
        member_vars: list[np.ndarray] = []
        any_vars = False
        with torch.no_grad():
            for _ in range(n_members):
                means: list[torch.Tensor] = []
                vars_: list[torch.Tensor] = []
                for start in range(0, n, batch_size):
                    xb = X_t[start : start + batch_size]
                    out = model(xb)
                    parsed = _parse_model_output(out)
                    means.append(parsed.mean)
                    if parsed.log_var is not None:
                        vars_.append(torch.exp(parsed.log_var))
                        any_vars = True
                    elif parsed.nig is not None:
                        mu, v, alpha, beta = parsed.nig
                        vars_.append(edl_nig_aleatoric_var(mu, v, alpha, beta))
                        any_vars = True
                member_means.append(torch.cat(means, dim=0).cpu().numpy())
                if vars_:
                    member_vars.append(torch.cat(vars_, dim=0).cpu().numpy())
                else:
                    member_vars.append(None)

        means_np = np.stack(member_means, axis=0)
        if any_vars:
            vars_np = np.stack(
                [v if v is not None else np.full_like(means_np[0], np.nan) for v in member_vars],
                axis=0,
            )
            preds = EnsemblePredictions(member_means=means_np, member_vars=vars_np)
            return _postprocess_ensemble_predictions(preds, uncertainty=uncertainty, eps=uncertainty_eps)
        preds = EnsemblePredictions(member_means=means_np, member_vars=None)
        return _postprocess_ensemble_predictions(preds, uncertainty=uncertainty, eps=uncertainty_eps)
    finally:
        restore()


def fit_laplace_diagonal_ensemble_predict(
    model_fn: Callable[[], Any],
    X_train: Any,
    y_train: Any,
    X: Any,
    *,
    n_members: int = 30,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    laplace_kwargs: Optional[Dict[str, Any]] = None,
    uncertainty: EnsembleUncertainty = None,
    uncertainty_eps: float = 1e-12,
) -> Union["EnsemblePredictions", Dict[str, Optional[np.ndarray]]]:
    """
    Convenience wrapper: train one model, then run a diagonal Laplace approximation ensemble.

    Notes
    -----
    - This function is model-agnostic and works for `TabularRegressorResNet`.
    - `laplace_kwargs` is forwarded to `laplace_diagonal_ensemble_predict` (except `n_members`).
    """
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    laplace_kwargs = {} if laplace_kwargs is None else dict(laplace_kwargs)
    laplace_kwargs.pop("n_members", None)
    laplace_kwargs.pop("uncertainty", None)
    laplace_kwargs.pop("uncertainty_eps", None)

    model = model_fn()
    model = fit_regressor_torch(model, X_train, y_train, **fit_kwargs)
    return laplace_diagonal_ensemble_predict(
        model,
        X,
        X_train,
        y_train,
        n_members=n_members,
        uncertainty=uncertainty,
        uncertainty_eps=uncertainty_eps,
        **laplace_kwargs,
    )


def laplace_diagonal_ensemble_predict(
    model: Any,
    X: Any,
    X_train: Any,
    y_train: Any,
    *,
    n_members: int = 30,
    prior_precision: float = 1.0,
    temperature: float = 1.0,
    batch_size_train: int = 512,
    batch_size_pred: int = 4096,
    device: Any = None,
    dtype: Any = None,
    loss: Literal["mse", "gaussian_nll", "edl_nig", "auto"] = "auto",
    edl_reg_weight: float = 1e-2,
    uncertainty: EnsembleUncertainty = None,
    uncertainty_eps: float = 1e-12,
) -> Union["EnsemblePredictions", Dict[str, Optional[np.ndarray]]]:
    """
    Diagonal Laplace approximation around a MAP-trained model.

    Uses a diagonal empirical-Fisher / Gauss-Newton style approximation:
      precision ≈ prior_precision + E[g^2],
    then samples parameter vectors ~ N(theta_map, temperature * precision^{-1}).
    """
    torch = _require_torch()
    if n_members <= 0:
        raise ValueError("n_members must be positive")
    if prior_precision <= 0:
        raise ValueError("prior_precision must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")

    model = model.to(device=device)
    model.eval()

    Xtr = _as_tensor(X_train, device=device, dtype=dtype)
    ytr = _as_tensor(y_train, device=device, dtype=dtype)
    if ytr.ndim == 1:
        ytr = ytr[:, None]

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Model has no trainable parameters")

    with torch.no_grad():
        map_vec = torch.nn.utils.parameters_to_vector(params).detach().clone()

    fisher_diag = torch.zeros_like(map_vec)
    n_data = Xtr.shape[0]
    for start in range(0, n_data, batch_size_train):
        xb = Xtr[start : start + batch_size_train]
        yb = ytr[start : start + batch_size_train]
        model.zero_grad(set_to_none=True)
        out = model(xb)
        parsed = _parse_model_output(out)
        mean_t = parsed.mean

        if loss == "auto":
            if parsed.nig is not None:
                loss_mode = "edl_nig"
            elif parsed.log_var is not None:
                loss_mode = "gaussian_nll"
            else:
                loss_mode = "mse"
        else:
            loss_mode = loss

        if loss_mode == "mse":
            # sum reduction for a more stable curvature proxy across batch sizes
            loss_t = torch.nn.functional.mse_loss(mean_t, yb, reduction="sum")
        elif loss_mode == "gaussian_nll":
            if parsed.log_var is None:
                raise ValueError("loss='gaussian_nll' requires model to output (mean, log_var)")
            loss_t = gaussian_nll_loss(mean_t, parsed.log_var, yb, reduction="sum")
        elif loss_mode == "edl_nig":
            if parsed.nig is None:
                raise ValueError("loss='edl_nig' requires model to output (mu, v, alpha, beta)")
            mu, v, alpha, beta = parsed.nig
            loss_t = edl_nig_loss(mu, v, alpha, beta, yb, reg_weight=edl_reg_weight, reduction="sum")
        else:
            raise ValueError(f"Unknown loss: {loss!r}")

        loss_t.backward()
        grads = []
        for p in params:
            if p.grad is None:
                grads.append(torch.zeros_like(p).reshape(-1))
            else:
                grads.append(p.grad.detach().reshape(-1))
        g = torch.cat(grads)
        fisher_diag += g * g

    fisher_diag = fisher_diag / float(n_data)
    precision = fisher_diag + float(prior_precision)
    var = (float(temperature) / precision).clamp_min(1e-16)
    std = torch.sqrt(var)

    Xt = _as_tensor(X, device=device, dtype=dtype)
    n = Xt.shape[0]

    member_means: list[np.ndarray] = []
    member_vars: list[np.ndarray] = []
    any_vars = False

    with torch.no_grad():
        for _ in range(n_members):
            sample_vec = map_vec + torch.randn_like(map_vec) * std
            torch.nn.utils.vector_to_parameters(sample_vec, params)

            means: list[torch.Tensor] = []
            vars_: list[torch.Tensor] = []
            for start in range(0, n, batch_size_pred):
                xb = Xt[start : start + batch_size_pred]
                out = model(xb)
                parsed = _parse_model_output(out)
                means.append(parsed.mean)
                if parsed.log_var is not None:
                    vars_.append(torch.exp(parsed.log_var))
                    any_vars = True
                elif parsed.nig is not None:
                    mu, v, alpha, beta = parsed.nig
                    vars_.append(edl_nig_aleatoric_var(mu, v, alpha, beta))
                    any_vars = True

            member_means.append(torch.cat(means, dim=0).cpu().numpy())
            if vars_:
                member_vars.append(torch.cat(vars_, dim=0).cpu().numpy())
            else:
                member_vars.append(None)

        torch.nn.utils.vector_to_parameters(map_vec, params)

    means_np = np.stack(member_means, axis=0)
    if any_vars:
        vars_np = np.stack(
            [v if v is not None else np.full_like(means_np[0], np.nan) for v in member_vars],
            axis=0,
        )
        preds = EnsemblePredictions(member_means=means_np, member_vars=vars_np)
        return _postprocess_ensemble_predictions(preds, uncertainty=uncertainty, eps=uncertainty_eps)
    preds = EnsemblePredictions(member_means=means_np, member_vars=None)
    return _postprocess_ensemble_predictions(preds, uncertainty=uncertainty, eps=uncertainty_eps)


UncertaintyMode = Literal["moment", "levi_simple", "levi_brier", "bayesian_kl", "levi_kl"]


def disentangle_regression_uncertainty(
    ensemble: Union[EnsemblePredictions, Tuple[Any, Optional[Any]], Dict[str, Any]],
    *,
    mode: UncertaintyMode = "moment",
    eps: float = 1e-12,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Turns an ensemble into predictive / aleatoric / epistemic uncertainties.

    Modes
    -----
    - 'moment': aleatoric = E[var], epistemic = Var[E[y]], predictive = sum
    - 'levi_simple': bounds on aleatoric scale + epistemic max |Δμ|
    - 'levi_brier': Levi-style set-valued uncertainty (quadratic-score inspired):
        epistemic = max_{i,j} (μ_i - μ_j)^2, aleatoric = [min var, max var] (interval)
    - 'bayesian_kl': entropy + mutual-information style decomposition (Gaussian)
    - 'levi_kl': bounds on entropy + epistemic max KL(N_i || N_j)
    """
    if isinstance(ensemble, EnsemblePredictions):
        member_means = ensemble.member_means
        member_vars = ensemble.member_vars
    elif isinstance(ensemble, tuple) and len(ensemble) == 2:
        member_means = _to_numpy(ensemble[0])
        member_vars = None if ensemble[1] is None else _to_numpy(ensemble[1])
    elif isinstance(ensemble, dict):
        member_means = _to_numpy(ensemble["member_means"])
        member_vars = None if ensemble.get("member_vars") is None else _to_numpy(ensemble["member_vars"])
    else:
        raise TypeError("ensemble must be EnsemblePredictions, (means, vars), or dict")

    means = np.asarray(member_means, dtype=float)
    if means.ndim == 2:  # (m, n) -> (m, n, 1)
        means = means[:, :, None]
    if means.ndim != 3:
        raise ValueError("member_means must have shape (m, n, d) or (m, n)")

    vars_ = None
    if member_vars is not None:
        vars_ = np.asarray(member_vars, dtype=float)
        if vars_.ndim == 2:
            vars_ = vars_[:, :, None]
        if vars_.shape != means.shape:
            raise ValueError(f"member_vars shape {vars_.shape} must match member_means shape {means.shape}")

    mean_prediction = means.mean(axis=0)

    if mode == "moment":
        ale = None if vars_ is None else np.nanmean(vars_, axis=0)
        epi = means.var(axis=0)
        pred = None if ale is None else ale + epi
        return {"mean": mean_prediction, "aleatoric": ale, "epistemic": epi, "predictive": pred}

    if mode == "levi_brier":
        # For 1D regression this is the squared range; for d>1 it's coordinate-wise squared range.
        epi = (np.nanmax(means, axis=0) - np.nanmin(means, axis=0)) ** 2
        if vars_ is None:
            return {"mean": mean_prediction, "aleatoric": None, "epistemic": epi, "predictive": None}

        ale_lb = np.nanmin(vars_, axis=0)
        ale_ub = np.nanmax(vars_, axis=0)
        return {
            "mean": mean_prediction,
            "aleatoric": None,
            "epistemic": epi,
            "predictive": None,
            "aleatoric_lower_bound": ale_lb,
            "aleatoric_upper_bound": ale_ub,
            "predictive_lower_bound": ale_lb + epi,
            "predictive_upper_bound": ale_ub + epi,
        }

    if vars_ is None:
        raise ValueError(f"mode={mode!r} requires member_vars to compute per-member scales")

    scales = np.sqrt(np.maximum(vars_, 0.0) + eps)

    if mode == "levi_simple":
        ale_lb = np.nanmin(scales, axis=0)
        ale_ub = np.nanmax(scales, axis=0)
        epi = np.nanmax(means, axis=0) - np.nanmin(means, axis=0)
        return {
            "mean": mean_prediction,
            "aleatoric": None,
            "epistemic": epi,
            "predictive": None,
            "aleatoric_lower_bound": ale_lb,
            "aleatoric_upper_bound": ale_ub,
            "predictive_lower_bound": ale_lb + epi,
            "predictive_upper_bound": ale_ub + epi,
        }

    if mode == "bayesian_kl":
        # Matches the formulas used in udboost.api.NGBEnsembleRegressor.pred_uncertainty for Gaussian params.
        log_s2 = np.log(np.maximum(scales * scales, eps))
        ale = 0.5 * (1.0 + np.log(2.0 * np.pi) + np.nanmean(log_s2, axis=0))

        mean_mu = np.nanmean(means, axis=0)
        mean_sigma = np.nanmean(scales, axis=0)
        log_mean_s2 = np.log(np.maximum(mean_sigma * mean_sigma, eps))

        mean_mu_deviation = np.nanmean((means * means - mean_mu * mean_mu), axis=0) / np.maximum(
            mean_sigma * mean_sigma, eps
        )
        epi = 0.5 * (log_mean_s2 - np.nanmean(log_s2, axis=0) + mean_mu_deviation)
        return {"mean": mean_prediction, "aleatoric": ale, "epistemic": epi, "predictive": ale + epi}

    if mode == "levi_kl":
        ent = 0.5 * np.log(np.maximum(2.0 * np.pi * np.e * (scales * scales), eps))
        ale_lb = np.nanmin(ent, axis=0)
        ale_ub = np.nanmax(ent, axis=0)

        m = means.shape[0]
        epi = np.zeros_like(mean_prediction)
        for i in range(m):
            mu_i = means[i]
            si = scales[i]
            # KL(N_i || N_j) over all j, take max
            kl_ij = (
                np.log(np.maximum(scales, eps) / np.maximum(si, eps))
                + (si * si + (mu_i - means) ** 2) / (2.0 * np.maximum(scales * scales, eps))
                - 0.5
            )
            epi = np.maximum(epi, np.nanmax(kl_ij, axis=0))

        return {
            "mean": mean_prediction,
            "aleatoric": None,
            "epistemic": epi,
            "predictive": None,
            "aleatoric_lower_bound": ale_lb,
            "aleatoric_upper_bound": ale_ub,
            "predictive_lower_bound": ale_lb + epi,
            "predictive_upper_bound": ale_ub + epi,
        }

    raise ValueError(f"Unknown mode: {mode!r}")
