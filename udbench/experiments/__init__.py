from udbench.experiments.config import ExperimentConfig
from udbench.experiments.datasets_registry import (
    DEFAULT_DATASET_NAMES,
    VALIDATION_DATASET_NAMES,
    resolve_dataset_names,
)

__all__ = [
    "DEFAULT_DATASET_NAMES",
    "ExperimentConfig",
    "VALIDATION_DATASET_NAMES",
    "resolve_dataset_names",
    "run_experiment",
]


def __getattr__(name: str):
    if name == "run_experiment":
        from udbench.experiments.runner import run_experiment

        return run_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
