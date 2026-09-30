# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Select the native module of this process (PLAN Section 5.4, "CPU core + CUDA plugins").

The ``dyng`` wheel always contains ``dyng._core`` (sequential and OpenMP backends). The CUDA
backends come as plugin wheels (``dyng-cu12``, ``dyng-cu13``; 0.1.x), each registering an entry
point in the group ``dyng.backends`` that names a module with ``available() -> bool`` and
``native`` (its extension module, a superset of ``_core``). At import the first available plugin
wins, otherwise ``_core``; exactly one module is active per process. ``DYNG_CPU_ONLY=1`` in the
environment (or :func:`dyng.use_cpu_only` before the first import of ``dyng``) forces ``_core``.
"""

from __future__ import annotations

import os
from importlib import metadata
from types import ModuleType

from . import _core

__all__ = ["native", "active_module_name", "plugin_errors"]

#: Errors met while loading plugins (name -> message), for dyng.show_config().
plugin_errors: dict[str, str] = {}


def _select() -> tuple[ModuleType, str]:
    if os.environ.get("DYNG_CPU_ONLY", "").strip() not in ("", "0"):
        return _core, "dyng._core"
    try:
        entry_points = metadata.entry_points(group="dyng.backends")
    except Exception as e:  # pragma: no cover - broken installations only
        plugin_errors["<entry points>"] = str(e)
        return _core, "dyng._core"
    for ep in sorted(entry_points, key=lambda e: e.name):
        try:
            plugin = ep.load()
            if plugin.available():
                return plugin.native, f"{ep.value} (plugin {ep.name})"
        except Exception as e:  # a broken plugin must not break the CPU package
            plugin_errors[ep.name] = f"{type(e).__name__}: {e}"
    return _core, "dyng._core"


native, active_module_name = _select()


def use_cpu_only() -> None:
    """Force the CPU module ``dyng._core`` (sequential and OpenMP).

    The module is chosen when ``dyng`` is first imported, so this only succeeds while the CPU
    module is the active one (always, in a process without CUDA plugins); otherwise set
    ``DYNG_CPU_ONLY=1`` in the environment before importing ``dyng``.

    Raises:
        RuntimeError: if a CUDA plugin module is already active.
    """
    if native is not _core:
        raise RuntimeError(
            f"dyng.use_cpu_only(): the native module {active_module_name} is already active; "
            "set DYNG_CPU_ONLY=1 in the environment before importing dyng"
        )
