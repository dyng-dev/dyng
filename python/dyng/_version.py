# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The version of dynG: the VERSION file of the source tree, compiled into the native module.

``__version__`` is read from the native module on first access (reading it chooses the native
module of the process, see :mod:`dyng._backend`).
"""

from __future__ import annotations

from typing import Any

from ._backend import native

__all__ = ["__version__"]

__version__: str  # resolved by __getattr__ below


def __getattr__(name: str) -> Any:
    if name == "__version__":
        return str(native.__version__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
