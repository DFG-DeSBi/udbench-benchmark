"""Backward-compatibility shim. Import from udbench.data.shared.uci_loader instead."""
from udbench.data.shared.uci_loader import UCILoader  # noqa: F401

Utils = UCILoader  # deprecated alias

__all__ = ["Utils", "UCILoader"]
