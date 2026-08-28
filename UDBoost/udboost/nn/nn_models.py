from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "udboost.nn requires PyTorch. Install it (e.g. `pip install torch`) to use udboost's NN helpers."
    ) from e


ActivationName = Literal["relu", "gelu", "silu", "tanh"]
NormName = Literal["none", "batchnorm", "layernorm"]


def _make_activation(name: ActivationName) -> nn.Module:
    match name:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU()
        case "silu":
            return nn.SiLU()
        case "tanh":
            return nn.Tanh()
        case _:
            raise ValueError(f"Unknown activation: {name!r}")


def _make_norm(name: NormName, d: int) -> nn.Module:
    match name:
        case "none":
            return nn.Identity()
        case "batchnorm":
            return nn.BatchNorm1d(d)
        case "layernorm":
            return nn.LayerNorm(d)
        case _:
            raise ValueError(f"Unknown norm: {name!r}")


def _build_mlp_backbone(
    in_features: int,
    hidden_features: Sequence[int],
    *,
    activation: ActivationName,
    dropout: float,
    norm: NormName,
) -> Tuple[nn.Module, int]:
    act = _make_activation(activation)
    layers: list[nn.Module] = []
    d_in = in_features
    for d_out in hidden_features:
        layers.append(nn.Linear(d_in, d_out))
        layers.append(_make_norm(norm, d_out))
        layers.append(act.__class__())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d_in = d_out
    return (nn.Sequential(*layers) if layers else nn.Identity()), d_in


@dataclass(frozen=True)
class EvidentialNIGOutput:
    """
    Evidential Regression output (Amini et al., 2020) for a Normal-Inverse-Gamma (NIG).

    Constraints:
      - v > 0
      - alpha > 1
      - beta > 0
    """

    mu: torch.Tensor
    v: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


class TabularRegressorMLP(nn.Module):
    """
    Simple MLP for tabular regression.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int] = (256, 256),
        *,
        activation: ActivationName = "relu",
        dropout: float = 0.0,
        norm: NormName = "none",
        out_features: int = 1,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if dropout < 0.0:
            raise ValueError("dropout must be >= 0")
        self.backbone, d_backbone = _build_mlp_backbone(
            in_features,
            hidden_features,
            activation=activation,
            dropout=dropout,
            norm=norm,
        )
        self.head = nn.Linear(d_backbone, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TabularEDLRegressorMLP(nn.Module):
    """
    Evidential Regression MLP for tabular data (Amini et al., 2020).

    Outputs NIG parameters (mu, v, alpha, beta). Uncertainty is computed externally.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int] = (256, 256),
        *,
        activation: ActivationName = "relu",
        dropout: float = 0.0,
        norm: NormName = "none",
        out_features: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.backbone, d_out = _build_mlp_backbone(
            in_features,
            hidden_features,
            activation=activation,
            dropout=dropout,
            norm=norm,
        )
        self.mu_head = nn.Linear(d_out, out_features)
        self.v_head = nn.Linear(d_out, out_features)
        self.alpha_head = nn.Linear(d_out, out_features)
        self.beta_head = nn.Linear(d_out, out_features)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.mu_head(h)
        v = F.softplus(self.v_head(h)) + self.eps
        alpha = F.softplus(self.alpha_head(h)) + 1.0 + self.eps
        beta = F.softplus(self.beta_head(h)) + self.eps
        return mu, v, alpha, beta

    def forward_params(self, x: torch.Tensor) -> EvidentialNIGOutput:
        mu, v, alpha, beta = self.forward(x)
        return EvidentialNIGOutput(mu=mu, v=v, alpha=alpha, beta=beta)


class _ResNetBlock(nn.Module):
    def __init__(
        self,
        d: int,
        *,
        hidden_multiplier: int = 2,
        activation: ActivationName = "relu",
        dropout: float = 0.0,
        norm: NormName = "layernorm",
    ) -> None:
        super().__init__()
        if hidden_multiplier <= 0:
            raise ValueError("hidden_multiplier must be positive")
        if dropout < 0.0:
            raise ValueError("dropout must be >= 0")

        self.norm = _make_norm(norm, d)
        self.fc1 = nn.Linear(d, d * hidden_multiplier)
        self.act = _make_activation(activation)
        self.fc2 = nn.Linear(d * hidden_multiplier, d)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class TabularRegressorResNet(nn.Module):
    """
    ResNet-style model for tabular regression (numeric features).
    """

    def __init__(
        self,
        in_features: int,
        *,
        d: int = 256,
        n_blocks: int = 3,
        hidden_multiplier: int = 2,
        activation: ActivationName = "relu",
        dropout: float = 0.0,
        norm: NormName = "layernorm",
        out_features: int = 1,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if d <= 0:
            raise ValueError("d must be positive")
        if n_blocks < 0:
            raise ValueError("n_blocks must be >= 0")
        if out_features <= 0:
            raise ValueError("out_features must be positive")

        self.stem = nn.Linear(in_features, d)
        self.blocks = nn.Sequential(
            *[
                _ResNetBlock(
                    d,
                    hidden_multiplier=hidden_multiplier,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_norm = _make_norm(norm, d)
        self.head = nn.Linear(d, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        h = self.final_norm(h)
        return self.head(h)

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TabularEDLRegressorResNet(nn.Module):
    """
    Evidential Regression ResNet for tabular data (Amini et al., 2020).

    Outputs NIG parameters (mu, v, alpha, beta). Uncertainty is computed externally.
    """

    def __init__(
        self,
        in_features: int,
        *,
        d: int = 256,
        n_blocks: int = 3,
        hidden_multiplier: int = 2,
        activation: ActivationName = "relu",
        dropout: float = 0.0,
        norm: NormName = "layernorm",
        out_features: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.stem = nn.Linear(in_features, d)
        self.blocks = nn.Sequential(
            *[
                _ResNetBlock(
                    d,
                    hidden_multiplier=hidden_multiplier,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_norm = _make_norm(norm, d)

        self.mu_head = nn.Linear(d, out_features)
        self.v_head = nn.Linear(d, out_features)
        self.alpha_head = nn.Linear(d, out_features)
        self.beta_head = nn.Linear(d, out_features)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.blocks(h)
        h = self.final_norm(h)
        mu = self.mu_head(h)
        v = F.softplus(self.v_head(h)) + self.eps
        alpha = F.softplus(self.alpha_head(h)) + 1.0 + self.eps
        beta = F.softplus(self.beta_head(h)) + self.eps
        return mu, v, alpha, beta

    def forward_params(self, x: torch.Tensor) -> EvidentialNIGOutput:
        mu, v, alpha, beta = self.forward(x)
        return EvidentialNIGOutput(mu=mu, v=v, alpha=alpha, beta=beta)
