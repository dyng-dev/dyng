# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The registry of algorithms and the citations (C++ ``algorithms()``, ``citation()``)."""

from __future__ import annotations

from dataclasses import dataclass

from ._backend import native

__all__ = ["AlgorithmInfo", "algorithms", "citation", "citation_keys"]


@dataclass(frozen=True)
class AlgorithmInfo:
    """One algorithm of the registry (C++ ``dyng::algorithm_info``)."""

    name: str  #: the name: its module ``dyng.<name>`` and C++ namespace
    title: str  #: a one-line title
    family: str  #: ``"fixed_point"`` or ``"aggregate_delta"``
    container: str  #: ``"graph"`` or ``"hypergraph"``
    maturity: str  #: ``"experimental"``, ``"stable"`` or ``"deprecated"``
    determinism: str  #: ``"bitwise"``, ``"exact_value"`` or ``"tolerance"``
    oracle: str  #: ``"compute"`` or ``"reference"``
    backends: tuple[str, ...]  #: the backends it implements (see ``dyng.show_config()``)
    cite: tuple[str, ...]  #: its keys in references.bib


def algorithms() -> list[AlgorithmInfo]:
    """The algorithms of this build, in registry order.

    Example:
        >>> import dyng
        >>> [a.name for a in dyng.algorithms()]
        ['sssp', 'cycle_count']
    """
    return [AlgorithmInfo(**d) for d in native.algorithms()]


def citation(what: str = "dyng") -> str:
    """BibTeX entries to cite: ``"dyng"`` (the library), an algorithm name (its papers and the
    library) or a key of references.bib.

    Raises:
        InvalidArgumentError: if ``what`` names nothing.
    """
    return str(native.citation(what))


def citation_keys(what: str = "dyng") -> list[str]:
    """The references.bib keys :func:`citation` returns for ``what``."""
    return list(native.citation_keys(what))
