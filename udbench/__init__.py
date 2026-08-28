"""Public API for udbench."""

__all__ = [
    "DataSet",
    "PRESETS",
]


def __getattr__(name: str):
    if name in {"DataSet", "PRESETS"}:
        from .datasets import DataSet, PRESETS

        return {"DataSet": DataSet, "PRESETS": PRESETS}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
