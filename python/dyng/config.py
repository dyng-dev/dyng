# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The build configuration and logging: :func:`show_config`, :func:`set_log_level`."""

from __future__ import annotations

import dataclasses
import platform
import sys
from typing import IO, Any, Literal

from . import _backend
from ._backend import native, use_cpu_only
from ._convert import enum_member, enum_name

__all__ = ["config", "show_config", "set_log_level", "get_log_level", "use_cpu_only"]

LogLevelName = Literal["off", "error", "warn", "info", "debug", "trace"]


def _cuda_available(chosen: Any, build: dict[str, Any]) -> bool:
    """Whether the CUDA backend can run, without initializing CUDA in this process when a plugin
    is active: its probe (NVML or a child process; ADR 0031) already counted the devices, and a
    process that only looked at the configuration can still fork workers that use CUDA."""
    if not build.get("cuda"):
        return False
    report = next((p for p in chosen.plugins if p.state == "chosen"), None)
    status = getattr(sys.modules.get(report.module), "status", None) if report else None
    if callable(status):
        return bool(getattr(status(), "usable", False))
    return bool(native.backend_available(native.Backend.cuda))


def config() -> dict[str, Any]:
    """The configuration of this installation as a dict (what :func:`show_config` prints).

    ``native_module`` is the active native module, ``selection`` why it was chosen, and
    ``plugins`` one dict per installed CUDA plugin (``name``, ``module``, ``version``,
    ``cuda_major``, ``driver_version``, ``state``, ``reason``; see ADR 0031).
    """
    import numpy as np

    build = dict(native.build_config)
    chosen = _backend.selection()
    backends = {}
    for b in native.Backend:
        if b.name == "cuda":
            backends[b.name] = _cuda_available(chosen, build)
        else:
            backends[b.name] = bool(native.backend_available(b))
    # The native default_backend() (CUDA if available, else OpenMP, else sequential), from the
    # availability above: asking the native module would initialize CUDA.
    default = next(b for b in ("cuda", "openmp", "sequential") if backends.get(b))
    return {
        "version": native.__version__,
        "native_module": _backend.active_module_name,
        "selection": chosen.reason,
        "plugins": [dataclasses.asdict(p) for p in chosen.plugins],
        "backends": backends,
        "default_backend": default,
        "openmp_max_threads": int(native.openmp_max_threads()),
        "build": build,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "plugin_errors": dict(_backend.plugin_errors),
    }


def show_config(file: IO[str] | None = None) -> None:
    """Print the configuration: version, active native module, available backends, build.

    Args:
        file: Where to print (default: ``sys.stdout``).
    """
    c = config()
    out = file if file is not None else sys.stdout
    lines = [
        f"dynG {c['version']}",
        f"  native module     : {c['native_module']}",
        f"  chosen because    : {c['selection']}",
        "  backends          : "
        + ", ".join(f"{k} ({'yes' if v else 'no'})" for k, v in c["backends"].items()),
        f"  default backend   : {c['default_backend']}",
        f"  OpenMP threads    : {c['openmp_max_threads']} (OpenMP {c['build']['openmp_version']})",
        f"  built with        : {c['build']['compiler']}, {c['build']['build_type']}",
        f"  Python / NumPy    : {c['python']} / {c['numpy']}",
        f"  platform          : {c['platform']}",
    ]
    if not c["build"]["cuda"]:
        lines.append(
            "  CUDA              : not in this module (the CPU wheel); the CUDA plugins: "
            'pip install "dyng[cu13]" or "dyng[cu12]"'
        )
    else:
        b = c["build"]
        plugin = f"plugin {b['plugin']}, " if b.get("plugin") else ""
        lines.append(
            f"  CUDA              : {plugin}toolkit {b.get('cuda_toolkit', '?')}, "
            f"{b.get('cuda_runtime', '?')} runtime, architectures "
            f"{b.get('cuda_architectures', '?')}"
        )
    for plugin in c["plugins"]:
        driver = plugin["driver_version"]
        seen = "" if driver is None else f", driver CUDA {driver // 1000}.{(driver % 1000) // 10}"
        lines.append(
            f"  CUDA plugin       : {plugin['module']} {plugin['version'] or '?'} "
            f"(CUDA {plugin['cuda_major'] or '?'}{seen}): {plugin['state']}"
            + (f" ({plugin['reason']})" if plugin["state"] != "chosen" else "")
        )
    print("\n".join(lines), file=out)


def set_log_level(level: LogLevelName) -> None:
    """Set the library's log level (``"off"``, ``"error"``, ``"warn"`` (default), ``"info"``,
    ``"debug"``, ``"trace"``); messages go to standard error."""
    native.set_log_level(enum_member(native.LogLevel, level, "set_log_level"))


def get_log_level() -> LogLevelName:
    """The library's log level."""
    return enum_name(native.get_log_level())  # type: ignore[return-value]
