# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The build configuration and logging: :func:`show_config`, :func:`set_log_level`."""

from __future__ import annotations

import platform
import sys
from typing import IO, Any, Literal

from . import _backend
from ._backend import native, use_cpu_only
from ._convert import enum_member, enum_name

__all__ = ["config", "show_config", "set_log_level", "get_log_level", "use_cpu_only"]

LogLevelName = Literal["off", "error", "warn", "info", "debug", "trace"]


def config() -> dict[str, Any]:
    """The configuration of this installation as a dict (what :func:`show_config` prints)."""
    import numpy as np

    backends = {}
    for b in native.Backend:
        backends[b.name] = bool(native.backend_available(b))
    build = dict(native.build_config)
    return {
        "version": native.__version__,
        "native_module": _backend.active_module_name,
        "backends": backends,
        "default_backend": enum_name(native.default_backend()),
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
            "  CUDA              : not in this build (the CPU wheel); the CUDA plugins "
            "dyng-cu12 / dyng-cu13 follow in 0.1.x"
        )
    for name, error in c["plugin_errors"].items():
        lines.append(f"  plugin {name} failed to load: {error}")
    print("\n".join(lines), file=out)


def set_log_level(level: LogLevelName) -> None:
    """Set the library's log level (``"off"``, ``"error"``, ``"warn"`` (default), ``"info"``,
    ``"debug"``, ``"trace"``); messages go to standard error."""
    native.set_log_level(enum_member(native.LogLevel, level, "set_log_level"))


def get_log_level() -> LogLevelName:
    """The library's log level."""
    return enum_name(native.get_log_level())  # type: ignore[return-value]
