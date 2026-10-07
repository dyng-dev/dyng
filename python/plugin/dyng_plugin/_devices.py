# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Ask the CUDA driver for the visible devices and their compute capabilities (internal).

The plugin package (``dyng_cu<N>/__init__.py``) learns about the devices without initializing
CUDA in the caller's process, so that the process can still fork workers that use CUDA (a child
forked after ``cuInit`` cannot initialize CUDA; ADR 0031). It asks NVML first; when NVML cannot
tell which devices CUDA will see (no NVML, MIG, ``CUDA_VISIBLE_DEVICES`` ordinals on GPUs of
different kinds), it runs this file as a script in a short child process
(``python -I -S _devices.py``), which calls :func:`query` and prints the answer as JSON. Only
when no child process can be started does the package call :func:`query` in its own process.

The file uses the standard library only (the child process runs without ``site``).
"""

from __future__ import annotations

import ctypes
import json
import sys
from typing import Any

#: CUdevice_attribute values of the driver API (cuda.h).
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


def query(lib: Any) -> tuple[int, list[tuple[int, int]]]:
    """Initialize the driver and read the compute capability of every visible device.

    Args:
        lib: ``libcuda.so.1`` (a ``ctypes.CDLL``).

    Returns:
        ``(CUresult, [(major, minor), ...])``: 0 and one entry per visible device, or the failing
        call's error code (100 = ``CUDA_ERROR_NO_DEVICE``) and an empty list.
    """
    rc = int(lib.cuInit(0))
    if rc != 0:
        return rc, []
    count = ctypes.c_int(0)
    rc = int(lib.cuDeviceGetCount(ctypes.byref(count)))
    if rc != 0:
        return rc, []
    capabilities: list[tuple[int, int]] = []
    for ordinal in range(int(count.value)):
        device = ctypes.c_int(0)
        rc = int(lib.cuDeviceGet(ctypes.byref(device), ordinal))
        if rc != 0:
            return rc, []
        major, minor = ctypes.c_int(0), ctypes.c_int(0)
        for value, attribute in (
            (major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
            (minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
        ):
            rc = int(lib.cuDeviceGetAttribute(ctypes.byref(value), attribute, device))
            if rc != 0:
                return rc, []
        capabilities.append((int(major.value), int(minor.value)))
    return 0, capabilities


def main() -> int:
    """The child process: print ``{"rc": ..., "capabilities": [[major, minor], ...]}``."""
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        print(json.dumps({"rc": -1, "capabilities": []}))
        return 0
    rc, capabilities = query(lib)
    print(json.dumps({"rc": rc, "capabilities": capabilities}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
