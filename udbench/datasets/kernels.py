"""Backward-compatibility shim. Import from udbench.data.shared.kernels instead."""
from udbench.data.shared.kernels import (  # noqa: F401
    KernelSpec,
    ParamSpec,
    combine_kernel_specs,
    make_kernel_spec,
)

__all__ = ["KernelSpec", "ParamSpec", "combine_kernel_specs", "make_kernel_spec"]
