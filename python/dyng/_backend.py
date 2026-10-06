# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Select the native module of this process (PLAN Section 5.4, "CPU core + CUDA plugins").

The ``dyng`` wheel always contains ``dyng._core`` (sequential and OpenMP backends). The CUDA
backends come as plugin wheels (``dyng-cu12``, ``dyng-cu13``; ADR 0030), each registering an entry
point in the group ``dyng.backends`` that names a module with ``available() -> bool`` and
``native`` (its extension module, a superset of ``_core``). Exactly one module is active per
process.

The module is chosen **on first use**, not at ``import dyng``: the first access to an attribute
of :data:`native` (the first graph, resources, reader, ``dyng.__version__``, ...) picks the first
available plugin, otherwise ``_core``. Only the chosen module is imported, so a plugin process
never loads ``_core`` (a second static libdyng and a second bundled OpenMP runtime).
``DYNG_CPU_ONLY=1`` in the environment, or :func:`use_cpu_only` before the first use, forces
``_core``.
"""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Callable
from importlib import metadata
from types import ModuleType
from typing import Any

__all__ = ["native", "active_module_name", "plugin_errors", "on_activate", "use_cpu_only"]

#: Errors met while loading plugins (name -> message), for dyng.show_config().
plugin_errors: dict[str, str] = {}

#: The name of the active native module, or None before the first use.
active_module_name: str | None = None

_lock = threading.RLock()
_active: ModuleType | None = None
_cpu_only = False
_hooks: list[Callable[[ModuleType], None]] = []


def _core() -> tuple[ModuleType, str]:
    return importlib.import_module("._core", __package__), "dyng._core"


def _select() -> tuple[ModuleType, str]:
    if _cpu_only or os.environ.get("DYNG_CPU_ONLY", "").strip() not in ("", "0"):
        return _core()
    try:
        entry_points = metadata.entry_points(group="dyng.backends")
    except Exception as e:  # pragma: no cover - broken installations only
        plugin_errors["<entry points>"] = str(e)
        return _core()
    for ep in sorted(entry_points, key=lambda e: e.name):
        try:
            plugin = ep.load()
            if plugin.available():
                return plugin.native, f"{ep.value} (plugin {ep.name})"
        except Exception as e:  # a broken plugin must not break the CPU package
            plugin_errors[ep.name] = f"{type(e).__name__}: {e}"
    return _core()


def active_module() -> ModuleType:
    """The active native module, chosen (and its hooks run) on the first call."""
    global _active, active_module_name
    module = _active
    if module is not None:
        return module
    with _lock:
        if _active is None:
            module, name = _select()
            for hook in _hooks:
                hook(module)
            active_module_name = name
            _active = module
        return _active


def on_activate(hook: Callable[[ModuleType], None]) -> None:
    """Run ``hook(module)`` when the native module is chosen (at once if it already is)."""
    with _lock:
        if _active is not None:
            hook(_active)
        else:
            _hooks.append(hook)


class _Native:
    """The active native module, resolved on first attribute access (see the module docs)."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(active_module(), name)

    def __repr__(self) -> str:
        return f"<dyng native module {active_module_name or '(not chosen yet)'}>"


#: The active native module (``dyng._core`` or a CUDA plugin's), chosen on first use.
native: Any = _Native()


def use_cpu_only() -> None:
    """Force the CPU module ``dyng._core`` (sequential and OpenMP) for this process.

    Call it before the first use of dynG (the first graph, resources, reader or
    ``dyng.__version__``; importing ``dyng`` does not count). Once ``_core`` is active, further
    calls do nothing. ``DYNG_CPU_ONLY=1`` in the environment does the same.

    Raises:
        RuntimeError: if a CUDA plugin module was already chosen by an earlier use.
    """
    global _cpu_only
    with _lock:
        if _active is None:
            _cpu_only = True
            active_module()
            return
        if active_module_name != "dyng._core":
            raise RuntimeError(
                f"dyng.use_cpu_only(): the native module {active_module_name} was already "
                "chosen by an earlier use of dynG; call use_cpu_only() before the first graph, "
                "resources or reader, or set DYNG_CPU_ONLY=1 in the environment"
            )
