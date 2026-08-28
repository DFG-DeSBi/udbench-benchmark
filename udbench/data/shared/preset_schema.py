"""TypedDict definitions for the preset.json bundle format.

Both the generation exporter (`data/generation/exporter.py`) and the runtime
loader (`data/runtime/presets.py`) import from here so that both sides
validate against the same schema. Adding a new field to the bundle format
should always start here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore[assignment]


class FeatureSummary(TypedDict):
    """Per-feature statistics stored in the bundle."""

    name: str
    minimum: float
    maximum: float


class FullDataSummary(TypedDict):
    """Summary statistics for one dataset split (full / train / test)."""

    num_samples: int
    features: List[FeatureSummary]


class DatasetPayload(TypedDict, total=False):
    """Dataset provenance block written by the exporter."""

    name: str
    id: Optional[int]
    full_data_summary: FullDataSummary


class GpSpecPayload(TypedDict):
    """Fitted GP hyperparameters stored in the bundle."""

    fitted_kernel_expr: str
    fitted_mean_expr: str
    likelihood_obs_stddev: float
    original_kernel_expr: str
    original_mean_expr: str


class StandardizationPayload(TypedDict):
    """Standardization parameters needed to back-transform model-space quantities."""

    x_mean: List[float]
    x_std: List[float]
    y_mean: float
    y_std: float
    original_input_range: List[List[float]]


class SamplingCfgPayload(TypedDict, total=False):
    """Sampling configuration embedded in the bundle."""

    method: str
    quantile_edges: List[float]
    bin_fractions: List[float]
    shrinkage: float
    standardize: bool
    prefer_center_bins: bool


class PresetBundle(TypedDict, total=False):
    """Top-level structure of a ``preset.json`` file.

    All fields are optional at the TypedDict level because bundles may be
    either GPJax-backed (``gp_spec`` present) or runtime-callable-backed
    (``runtime`` present). Code that reads a bundle must check which variant
    is present before accessing variant-specific sub-keys.
    """

    dataset: DatasetPayload
    gp_spec: GpSpecPayload
    standardization: StandardizationPayload
    sampling_cfg: SamplingCfgPayload
    runtime: Dict[str, Any]
    preset: Dict[str, Any]


__all__ = [
    "DatasetPayload",
    "FeatureSummary",
    "FullDataSummary",
    "GpSpecPayload",
    "PresetBundle",
    "SamplingCfgPayload",
    "StandardizationPayload",
]
