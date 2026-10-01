# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Internal helpers: enumerations given as strings, and keyword options."""

from __future__ import annotations

import dataclasses
import enum
import operator
from collections.abc import Mapping
from typing import Any

import numpy as np

from .errors import InvalidArgumentError


def as_int(value: Any, what: str) -> int:
    """An integer argument or option field: ``int`` or a NumPy integer, nothing else.

    ``operator.index`` semantics: floats (``1.9``), strings (``"1"``) and booleans are rejected
    with :class:`~dyng.InvalidArgumentError` instead of being truncated or parsed.
    """
    if isinstance(value, bool | np.bool_):
        raise InvalidArgumentError(f"{what}: expected an integer, got the bool {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise InvalidArgumentError(
            f"{what}: expected an integer, got {type(value).__name__} {value!r}"
        ) from None


def as_bool(value: Any, what: str) -> bool:
    """A boolean argument or option field: ``True`` / ``False`` (or a NumPy bool) only."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    raise InvalidArgumentError(
        f"{what}: expected True or False, got {type(value).__name__} {value!r}"
    )


def enum_member(enum_type: type[enum.Enum], value: Any, what: str) -> enum.Enum:
    """The member of a native enumeration named by ``value`` (a string or a member)."""
    if isinstance(value, enum_type):
        return value
    if isinstance(value, enum.Enum):
        value = value.name
    if isinstance(value, str):
        try:
            return enum_type[value]
        except KeyError:
            pass
    names = ", ".join(repr(m.name) for m in enum_type)
    raise InvalidArgumentError(f"{what}: {value!r} is not one of {names}")


def enum_name(value: Any) -> str:
    """The name of a native enumeration member (its C++ enumerator's name)."""
    return value.name if isinstance(value, enum.Enum) else str(value)


def with_options[T](cls: type[T], options: T | None, overrides: Mapping[str, Any], what: str) -> T:
    """``options`` (or ``cls()``) with the fields in ``overrides`` replaced.

    Unknown keywords raise TypeError, as for any Python function.
    """
    base = cls() if options is None else options
    if not isinstance(base, cls):
        raise TypeError(f"{what}: options must be a {cls.__module__}.{cls.__qualname__}")
    if not overrides:
        return base
    names = {f.name for f in dataclasses.fields(base)}  # type: ignore[arg-type]
    unknown = sorted(set(overrides) - names)
    if unknown:
        raise TypeError(
            f"{what}: unexpected keyword argument(s) {', '.join(unknown)} "
            f"(options are {', '.join(sorted(names))})"
        )
    return dataclasses.replace(base, **overrides)  # type: ignore[type-var]


def copy_fields(obj: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """The attributes ``names`` of a native object, with enumerations as strings."""
    out: dict[str, Any] = {}
    for n in names:
        v = getattr(obj, n)
        out[n] = enum_name(v) if isinstance(v, enum.Enum) else v
    return out
