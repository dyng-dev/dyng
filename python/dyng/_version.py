# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The version of dynG: the VERSION file of the source tree, compiled into the native module."""

from __future__ import annotations

from ._backend import native

__all__ = ["__version__"]

__version__: str = str(native.__version__)
"""The version string (PEP 440), for example ``"0.1.0"``."""
