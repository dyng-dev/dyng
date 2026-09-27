# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dynG: dynamic graph and hypergraph updates on GPUs.

This is the name-reservation release 0.0.1 of the ``dyng`` package. The library is under
development at https://github.com/dyng-dev/dyng; the first functional release is 0.1.0.
"""

__version__ = "0.0.1"
__all__ = ["__version__", "status"]

_STATUS = (
    "dynG (dynamic graph and hypergraph updates on GPUs) is under development. "
    "dyng 0.0.1 reserves the package name and provides no algorithms yet; "
    "see https://github.com/dyng-dev/dyng for progress."
)


def status() -> str:
    """Return a short description of the project status."""
    return _STATUS
