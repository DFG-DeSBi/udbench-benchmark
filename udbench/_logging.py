"""Centralised logging configuration for UDBench.

Usage in library modules:
    import logging
    logger = logging.getLogger(__name__)

Usage in experiment entry points:
    from udbench._logging import configure_logging, add_logging_args, apply_logging_args
    configure_logging()          # call once at startup with defaults
    add_logging_args(parser)     # attach --verbose / --debug / --quiet to argparse
    apply_logging_args(args)     # apply after parsing

Default levels (no flags):
    udbench.experiments  → INFO   (dataset/model progress, metrics)
    udbench.models       → WARNING (epoch-level noise hidden by default)
    everything else      → WARNING (third-party libraries stay quiet)
"""
from __future__ import annotations

import logging
import os
import sys

_NOISY_LIBS = ["jax", "jaxlib", "absl", "matplotlib", "PIL", "torch", "tensorflow", "flax", "optax"]


class _ANSIFormatter(logging.Formatter):
    """Colorised single-line formatter used when rich is not available."""

    _COLORS = {
        logging.DEBUG:    "\033[2m",      # dim
        logging.INFO:     "",              # default
        logging.WARNING:  "\033[33;1m",   # yellow bold
        logging.ERROR:    "\033[31;1m",   # red bold
        logging.CRITICAL: "\033[41;1m",   # bright red background
    }
    _RESET = "\033[0m"
    _DIM   = "\033[2m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        reset = self._RESET if color else ""
        ts = self.formatTime(record, "%H:%M:%S")
        parts = record.name.split(".")
        short_name = ".".join(parts[-2:]) if len(parts) >= 2 else record.name
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return (
            f"{self._DIM}{ts}{self._RESET}  "
            f"{color}{record.levelname:<8}{reset}  "
            f"{msg}  "
            f"{self._DIM}[{short_name}]{self._RESET}"
        )


def _make_handler(use_rich: bool) -> logging.Handler:
    if use_rich:
        try:
            from rich.logging import RichHandler
            return RichHandler(
                level=logging.DEBUG,
                rich_tracebacks=True,
                show_path=False,
                show_time=True,
                markup=False,
                log_time_format="[%H:%M:%S]",
            )
        except ImportError:
            pass
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ANSIFormatter())
    return handler


def _configure_wandb_silence(silent: bool) -> None:
    if silent:
        os.environ["WANDB_SILENT"] = "true"
    else:
        os.environ.pop("WANDB_SILENT", None)
    # If wandb is already imported, update its settings directly (env var alone
    # won't help once the module is loaded).
    if "wandb" in sys.modules:
        try:
            import wandb
            wandb.setup(settings=wandb.Settings(silent=silent, quiet=silent))
        except Exception:
            pass


def configure_logging(
    level: int = logging.INFO,
    model_level: int | None = None,
    *,
    use_rich: bool = True,
) -> None:
    """Configure UDBench loggers. Call once at experiment startup.

    Args:
        level: Log level for udbench.experiments (and __main__ entry points).
        model_level: Log level for udbench.models. Defaults to WARNING when
                     level <= INFO so epoch-level noise is hidden by default.
        use_rich: Try RichHandler first; fall back to ANSI formatter.
    """
    if model_level is None:
        model_level = logging.WARNING if level <= logging.INFO else level

    root = logging.getLogger()
    root.handlers.clear()
    handler = _make_handler(use_rich)
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)
    root.setLevel(logging.WARNING)

    logging.getLogger("udbench.experiments").setLevel(level)
    logging.getLogger("udbench.models").setLevel(model_level)
    logging.getLogger("udbench.datasets").setLevel(level)
    logging.getLogger("udbench.semi_synth_datasets").setLevel(level)
    logging.getLogger("udbench.tuning").setLevel(level)

    for lib in _NOISY_LIBS:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Suppress wandb's direct-to-stderr prints unless we're in full debug mode.
    _configure_wandb_silence(silent=level > logging.DEBUG)


def add_logging_args(parser: object) -> None:
    """Attach --verbose / --debug / --quiet to an argparse.ArgumentParser."""
    group = parser.add_mutually_exclusive_group()  # type: ignore[union-attr]
    group.add_argument(
        "--verbose", action="store_true",
        help="Show model INFO logs (epoch summaries, fit details).",
    )
    group.add_argument(
        "--debug", action="store_true",
        help="Show all DEBUG logs including per-epoch loss.",
    )
    group.add_argument(
        "--quiet", action="store_true",
        help="Only show WARNING and above.",
    )


def apply_logging_args(args: object) -> None:
    """Apply log levels based on --verbose / --debug / --quiet flags."""
    if getattr(args, "debug", False):
        configure_logging(level=logging.DEBUG, model_level=logging.DEBUG)
    elif getattr(args, "verbose", False):
        configure_logging(level=logging.INFO, model_level=logging.INFO)
    elif getattr(args, "quiet", False):
        configure_logging(level=logging.WARNING, model_level=logging.WARNING)
    else:
        configure_logging()
