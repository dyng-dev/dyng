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
module or the CUDA runtime before :data:`native` is read, and nothing initializes CUDA in the
calling process (ADR 0031): :func:`status` asks the driver (``libcuda.so.1``, through ctypes) for
its CUDA version (``cuDriverGetVersion``, which needs no ``cuInit``), and NVML
(``libnvidia-ml.so.1``) for the devices and their compute capabilities, applying
``CUDA_VISIBLE_DEVICES``; when NVML cannot tell which devices CUDA will see, a short child process
asks the driver (``_devices.py``). A process that only chose the module can therefore still fork
workers that use CUDA.

Attributes:
    PLUGIN: The plugin's name, ``"cu12"`` or ``"cu13"``.
    CUDA_MAJOR: The CUDA major of the toolkit the module was built with (12 or 13).
    DISTRIBUTION: The distribution's name, ``"dyng-cu12"`` or ``"dyng-cu13"``.
    ARCHITECTURE_FLOOR: The oldest compute capability the module has code for, ``(7, 5)``; a
        machine whose visible GPUs are all older cannot use the plugin.
    __version__: The distribution's version (it always equals the version of ``dyng`` it was
        built for: the plugin depends on ``dyng==<its version>``), or ``None`` when the package is
        not installed from a wheel.
    native: The extension module (``dyng_cu<N>._core``), imported on first access.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from importlib import metadata
from types import ModuleType
from typing import Any

from . import _devices

__all__ = [
    "PLUGIN",
    "CUDA_MAJOR",
    "DISTRIBUTION",
    "ARCHITECTURE_FLOOR",
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

#: The CUDA driver API's (and NVML's) success code.
_CUDA_SUCCESS = 0
#: CUDA_ERROR_NO_DEVICE (also with CUDA_VISIBLE_DEVICES masking every device).
_CUDA_ERROR_NO_DEVICE = 100
#: The oldest Linux driver of each CUDA major (CUDA's release notes, table 3); for the messages.
_DRIVER_FLOOR = {12: "525.60.13", 13: "580.65.06"}
#: The oldest compute capability with code in the module: the first entry of the release list of
#: cmake/cuda_architectures.cmake (75-real on every toolkit; the PTX of the last entry covers
#: newer GPUs). test_plugin_package.py checks it against the CMake file.
ARCHITECTURE_FLOOR: tuple[int, int] = (7, 5)


@dataclass(frozen=True)
class Status:
    """What :func:`status` found out about the CUDA driver of this machine.

    Attributes:
        usable: Whether this plugin's module can run CUDA work here (a driver of CUDA
            ``CUDA_MAJOR`` or newer and at least one visible device of compute capability
            :data:`ARCHITECTURE_FLOOR` or newer).
        driver_version: The driver's CUDA version as ``1000 * major + 10 * minor`` (13010 for
            CUDA 13.1), or ``None`` without a driver.
        device_count: The visible CUDA devices (0 without a driver).
        reason: Why the plugin is not usable (empty when it is).
        capabilities: The compute capability ``(major, minor)`` of every visible device (empty
            when they were not read).
        probe: How the devices were found: ``"nvml"``, ``"child process"`` (a short process
            that asked the driver), ``"in process"`` (``cuInit`` in this process, when no child
            process could be started) or empty (not asked: no usable driver).
    """

    usable: bool
    driver_version: int | None
    device_count: int
    reason: str
    capabilities: tuple[tuple[int, int], ...] = ()
    probe: str = ""


_lock = threading.Lock()
_status: Status | None = None


def _load_driver() -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None


def _nvml_devices() -> list[tuple[str, tuple[int, int], bool]] | None:
    """``(UUID, compute capability, MIG enabled)`` of every GPU in NVML's (PCI bus) order, or
    None when NVML is missing or fails. NVML does not initialize CUDA (fork-safe)."""
    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError:
        return None
    if nvml.nvmlInit_v2() != _CUDA_SUCCESS:
        return None
    try:
        count = ctypes.c_uint(0)
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != _CUDA_SUCCESS:
            return None
        devices: list[tuple[str, tuple[int, int], bool]] = []
        for index in range(int(count.value)):
            handle = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != _CUDA_SUCCESS:
                return None
            major, minor = ctypes.c_int(0), ctypes.c_int(0)
            rc = nvml.nvmlDeviceGetCudaComputeCapability(
                handle, ctypes.byref(major), ctypes.byref(minor)
            )
            if rc != _CUDA_SUCCESS:
                return None
            uuid = ctypes.create_string_buffer(96)
            if nvml.nvmlDeviceGetUUID(handle, uuid, 96) != _CUDA_SUCCESS:
                return None
            current, pending = ctypes.c_uint(0), ctypes.c_uint(0)
            mig = (
                nvml.nvmlDeviceGetMigMode(handle, ctypes.byref(current), ctypes.byref(pending))
                == _CUDA_SUCCESS
                and current.value == 1
            )
            devices.append(
                (uuid.value.decode(errors="replace"), (int(major.value), int(minor.value)), mig)
            )
        return devices
    finally:
        nvml.nvmlShutdown()


def _visible_capabilities(
    devices: list[tuple[str, tuple[int, int], bool]],
) -> list[tuple[int, int]] | None:
    """The compute capabilities of the devices CUDA will see, from NVML's list and
    ``CUDA_VISIBLE_DEVICES``; None when this cannot be told without asking CUDA.

    ``CUDA_VISIBLE_DEVICES`` is read as CUDA reads it: comma-separated ordinals or ``GPU-``
    UUID prefixes, up to the first entry that names no device; a device named twice hides every
    device. CUDA numbers the devices fastest first unless ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, NVML by
    PCI bus, so ordinals are only mapped when the order cannot matter (one kind of GPU, every GPU
    visible, or PCI_BUS_ID). MIG devices are left to CUDA.
    """
    if any(mig for _, _, mig in devices):
        return None
    variable = os.environ.get("CUDA_VISIBLE_DEVICES")
    if variable is None:
        return [cc for _, cc, _ in devices]
    uuids = [uuid for uuid, _, _ in devices]
    chosen: list[int] = []
    by_ordinal = False
    for raw in variable.split(","):
        token = raw.strip()
        if token.startswith("MIG-"):
            return None
        if token.startswith("GPU-"):
            matches = [i for i, uuid in enumerate(uuids) if uuid.startswith(token)]
            if len(matches) != 1:
                break
            index = matches[0]
        else:
            m = re.match(r"[+-]?\d+", token)
            if m is None:
                break
            index = int(m.group())
            if not 0 <= index < len(devices):
                break
            by_ordinal = True
        if index in chosen:
            return []
        chosen.append(index)
    order_matters = (
        by_ordinal
        and len(chosen) < len(devices)
        and len({cc for _, cc, _ in devices}) > 1
        and os.environ.get("CUDA_DEVICE_ORDER", "").strip() != "PCI_BUS_ID"
    )
    if order_matters:
        return None
    return [devices[i][1] for i in chosen]


def _query_in_child() -> tuple[int, list[tuple[int, int]]] | None:
    """Run ``_devices.py`` in a child process (its ``cuInit`` stays there); None if it fails."""
    if not sys.executable:
        return None
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_devices.py")
    try:
        out = subprocess.run(
            [sys.executable, "-I", "-S", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        data = json.loads(out.stdout)
        rc = int(data["rc"])
        capabilities = [(int(c[0]), int(c[1])) for c in data["capabilities"]]
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError):
        return None
    if rc < 0:
        return None
    return rc, capabilities


def _arch(cc: tuple[int, int]) -> str:
    return f"sm_{cc[0]}{cc[1]}"


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
    rc = lib.cuDriverGetVersion(ctypes.byref(version))  # needs no cuInit
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
    devices = _nvml_devices()
    capabilities = _visible_capabilities(devices) if devices is not None else None
    probe = "nvml"
    if capabilities is None:
        answer = _query_in_child()
        probe = "child process"
        if answer is None:
            # The last resort: initializes CUDA in this process (a later fork cannot use CUDA).
            answer = _devices.query(lib)
            probe = "in process"
        rc, capabilities = answer
        if rc == _CUDA_ERROR_NO_DEVICE:
            capabilities = []
        elif rc != _CUDA_SUCCESS:
            return Status(False, v, 0, f"cuInit failed (CUresult {rc})", (), probe)
    caps = tuple(capabilities)
    if not caps:
        return Status(False, v, 0, "no CUDA device is visible", caps, probe)
    if not any(cc >= ARCHITECTURE_FLOOR for cc in caps):
        found = ", ".join(sorted({_arch(cc) for cc in caps}))
        return Status(
            False,
            v,
            len(caps),
            f"the visible GPU ({found}) is older than the oldest architecture of {DISTRIBUTION}, "
            f"{_arch(ARCHITECTURE_FLOOR)} (compute capability "
            f"{ARCHITECTURE_FLOOR[0]}.{ARCHITECTURE_FLOOR[1]} or newer: Turing and later)",
            caps,
            probe,
        )
    return Status(True, v, len(caps), "", caps, probe)


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
