# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""A CUDA plugin of dynG: the import package ``dyng_cu12`` or ``dyng_cu13`` (PLAN Section 5.4).

This one source directory (``python/plugin/dyng_plugin``) becomes the import package of each
CUDA plugin distribution (``dyng-cu12``, ``dyng-cu13``; ADR 0030): ``ci/plugin_pyproject.py``
copies it to ``dyng_cu<N>`` in the plugin's source tree, and the wheel holds it next to the
extension module ``dyng_cu<N>/_core.abi3.so`` (every backend: sequential, OpenMP and CUDA;
nanobind domain ``dyng_cu<N>``; the CUDA runtime linked statically). The package learns which
plugin it is from its own name.

Users do not import it. ``import dyng`` finds it through the entry point group ``dyng.backends``
(``cu13 = dyng_cu13``) and, on first use, asks :func:`available`; when it says yes, the plugin's
:data:`native` module replaces ``dyng._core`` for the process. Nothing here loads the extension
module or the CUDA runtime before :data:`native` is read; :func:`status` only asks the driver
(``libcuda.so.1``, through ctypes) for its CUDA version and the number of devices.

Attributes:
    PLUGIN: The plugin's name, ``"cu12"`` or ``"cu13"``.
    CUDA_MAJOR: The CUDA major of the toolkit the module was built with (12 or 13).
    DISTRIBUTION: The distribution's name, ``"dyng-cu12"`` or ``"dyng-cu13"``.
    __version__: The distribution's version (it always equals the version of ``dyng`` it was
        built for: the plugin depends on ``dyng==<its version>``), or ``None`` when the package is
        not installed from a wheel.
    native: The extension module (``dyng_cu<N>._core``), imported on first access.
"""

from __future__ import annotations

import ctypes
import importlib
import re
import threading
from dataclasses import dataclass
from importlib import metadata
from types import ModuleType
from typing import Any

__all__ = [
    "PLUGIN",
    "CUDA_MAJOR",
    "DISTRIBUTION",
    "Status",
    "available",
    "status",
    "driver_version",
]

_name = re.fullmatch(r"dyng_(cu(\d+))", __name__)
if _name is None:  # pragma: no cover - the source directory imported under its own name
    raise ImportError(
        f"{__name__}: the CUDA plugin package must be installed as dyng_cu12 or dyng_cu13 "
        "(ci/plugin_pyproject.py renders the plugin's pyproject.toml)"
    )

PLUGIN: str = _name.group(1)
CUDA_MAJOR: int = int(_name.group(2))
DISTRIBUTION: str = f"dyng-{PLUGIN}"

try:
    __version__: str | None = metadata.version(DISTRIBUTION)
except metadata.PackageNotFoundError:  # pragma: no cover - a source tree, not a wheel
    __version__ = None

#: The CUDA driver API's success code.
_CUDA_SUCCESS = 0
#: The oldest Linux driver of each CUDA major (CUDA's release notes, table 3); for the messages.
_DRIVER_FLOOR = {12: "525.60.13", 13: "580.65.06"}


@dataclass(frozen=True)
class Status:
    """What :func:`status` found out about the CUDA driver of this machine.

    Attributes:
        usable: Whether this plugin's module can run CUDA work here (a driver of CUDA
            ``CUDA_MAJOR`` or newer and at least one visible device).
        driver_version: The driver's CUDA version as ``1000 * major + 10 * minor`` (13010 for
            CUDA 13.1), or ``None`` without a driver.
        device_count: The visible CUDA devices (0 without a driver).
        reason: Why the plugin is not usable (empty when it is).
    """

    usable: bool
    driver_version: int | None
    device_count: int
    reason: str


_lock = threading.Lock()
_status: Status | None = None


def _load_driver() -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None


def _probe() -> Status:
    lib = _load_driver()
    if lib is None:
        return Status(
            False,
            None,
            0,
            "no CUDA driver: libcuda.so.1 cannot be loaded (install the NVIDIA driver, "
            f"{_DRIVER_FLOOR.get(CUDA_MAJOR, 'a version')} or newer for CUDA {CUDA_MAJOR})",
        )
    version = ctypes.c_int(0)
    rc = lib.cuDriverGetVersion(ctypes.byref(version))
    if rc != _CUDA_SUCCESS:
        return Status(False, None, 0, f"cuDriverGetVersion failed (CUresult {rc})")
    v = int(version.value)
    if v < 1000 * CUDA_MAJOR:
        return Status(
            False,
            v,
            0,
            f"the CUDA driver supports CUDA {v // 1000}.{(v % 1000) // 10}, but {DISTRIBUTION} "
            f"needs CUDA {CUDA_MAJOR}.0 or newer (update the NVIDIA driver, or install the "
            f'plugin of the driver\'s CUDA major: pip install "dyng[cu{v // 1000}]")',
        )
    rc = lib.cuInit(0)
    if rc != _CUDA_SUCCESS:
        # 100 = CUDA_ERROR_NO_DEVICE (also with CUDA_VISIBLE_DEVICES masking every device)
        what = "no CUDA device is visible" if rc == 100 else f"cuInit failed (CUresult {rc})"
        return Status(False, v, 0, what)
    count = ctypes.c_int(0)
    rc = lib.cuDeviceGetCount(ctypes.byref(count))
    if rc != _CUDA_SUCCESS:
        return Status(False, v, 0, f"cuDeviceGetCount failed (CUresult {rc})")
    if count.value < 1:
        return Status(False, v, 0, "no CUDA device is visible")
    return Status(True, v, int(count.value), "")


def status() -> Status:
    """Probe the CUDA driver once per process (later calls return the first answer)."""
    global _status
    with _lock:
        if _status is None:
            _status = _probe()
        return _status


def available() -> bool:
    """Whether this plugin's module can run CUDA work on this machine (see :class:`Status`)."""
    return status().usable


def driver_version() -> int | None:
    """The CUDA driver's version (``1000 * major + 10 * minor``), or ``None`` without a driver."""
    return status().driver_version


def __getattr__(name: str) -> Any:
    if name == "native":
        module: ModuleType = importlib.import_module("._core", __name__)
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
