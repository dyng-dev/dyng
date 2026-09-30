# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.generators: seeded generators of graphs and batches (C++ ``dyng::generators``).

In 0.1 the generators are the bit-exact reproductions of the original tools' random streams,
:mod:`dyng.generators.legacy`, so the inputs of published experiments can be regenerated.
"""

from __future__ import annotations

from . import legacy

__all__ = ["legacy"]
