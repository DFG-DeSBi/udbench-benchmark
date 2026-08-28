from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    name: str
    datasets: list[str]                          # preset names from DataSet.PRESETS; empty means use dataset_suite
    models: list[str]                            # keys into MODEL_REGISTRY
    dataset_suite: str = "training"              # "training" | "validation" | "all" | "custom"
    model_kwargs: dict[str, dict] = field(default_factory=dict)   # per-model constructor overrides
    sample_sizes: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    noise_mode: str = "heteroscedastic"          # "heteroscedastic" | "homoscedastic"
    seed: int = 42
    eval_size: int = 5000
    tune: bool = False
    tune_count: int = 160
    tune_project: str | None = None
    tune_objective: str = "val_rmse"             # key into udbench/tuning/objectives.py
    results_dir: Path = field(default_factory=lambda: Path("results"))
    max_pool_size: int | None = None             # cap UCI pool before GP gram matrix computation
    arrays_root: Path | None = None              # if set, save per-run .npz/.json arrays here
