# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Internal helpers: enumerations given as strings, and keyword options."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from typing import Any

from .errors import InvalidArgumentError


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
