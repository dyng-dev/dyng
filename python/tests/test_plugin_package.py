# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The import package of the CUDA plugins (python/plugin/dyng_plugin; ADR 0030).

The package is loaded from the source tree under a plugin's name (``dyng_cu13``, ``dyng_cu12``)
with a fake CUDA driver in place of ``libcuda.so.1``, so every answer of ``status()`` is checked
on any machine. The real plugin wheels are tested by ci/plugin_wheels.sh.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "plugin" / "dyng_plugin" / "__init__.py"

pytestmark = pytest.mark.skipif(not SOURCE.is_file(), reason="needs the source tree")


class FakeDriver:
    """The driver entry points the package calls, with chosen answers."""

    def __init__(
        self,
        version: int = 13010,
        init: int = 0,
        devices: int = 2,
        capabilities: list[tuple[int, int]] | None = None,
    ) -> None:
        self.version, self.init, self.devices = version, init, devices
        self.capabilities = capabilities or [(8, 6)] * devices
        self.calls: list[str] = []

    @staticmethod
    def _set(ref: Any, value: int) -> None:
        ref._obj.value = value  # ctypes.byref(c_int) keeps the object in _obj

    def cuDriverGetVersion(self, ref: Any) -> int:  # noqa: N802 (the driver's names)
        self.calls.append("cuDriverGetVersion")
        self._set(ref, self.version)
        return 0

    def cuInit(self, flags: int) -> int:  # noqa: N802
        self.calls.append("cuInit")
        return self.init

    def cuDeviceGetCount(self, ref: Any) -> int:  # noqa: N802
        self.calls.append("cuDeviceGetCount")
        self._set(ref, self.devices)
        return 0

    def cuDeviceGet(self, ref: Any, ordinal: int) -> int:  # noqa: N802
        self._set(ref, ordinal)
        return 0

    def cuDeviceGetAttribute(self, ref: Any, attribute: int, device: Any) -> int:  # noqa: N802
        cc = self.capabilities[int(getattr(device, "value", device))]
        self._set(ref, cc[0] if attribute == 75 else cc[1])
        return 0


class FakeNvml:
    """NVML's entry points: GPUs as (UUID, compute capability, MIG enabled), in PCI bus order."""

    def __init__(
        self, devices: list[tuple[str, tuple[int, int], bool]] | None = None, init: int = 0
    ) -> None:
        self.devices = devices if devices is not None else [GPU_A, GPU_B]
        self.init = init
        self.calls: list[str] = []

    def nvmlInit_v2(self) -> int:  # noqa: N802
        self.calls.append("init")
        return self.init

    def nvmlShutdown(self) -> int:  # noqa: N802
        self.calls.append("shutdown")
        return 0

    def nvmlDeviceGetCount_v2(self, ref: Any) -> int:  # noqa: N802
        ref._obj.value = len(self.devices)
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, index: int, ref: Any) -> int:  # noqa: N802
        ref._obj.value = index + 1  # a c_void_p of 0 reads back as None
        return 0

    def _device(self, handle: Any) -> tuple[str, tuple[int, int], bool]:
        return self.devices[int(handle.value) - 1]

    def nvmlDeviceGetCudaComputeCapability(  # noqa: N802
        self, handle: Any, major: Any, minor: Any
    ) -> int:
        cc = self._device(handle)[1]
        major._obj.value, minor._obj.value = cc
        return 0

    def nvmlDeviceGetUUID(self, handle: Any, buffer: Any, size: int) -> int:  # noqa: N802
        buffer.value = self._device(handle)[0].encode()
        return 0

    def nvmlDeviceGetMigMode(self, handle: Any, current: Any, pending: Any) -> int:  # noqa: N802
        if not self._device(handle)[2]:
            return 3  # NVML_ERROR_NOT_SUPPORTED: a GPU without MIG
        current._obj.value = pending._obj.value = 1
        return 0


GPU_A = ("GPU-c8f1454c-ae9c-f9ee-1002-0d11da716b01", (8, 6), False)
GPU_B = ("GPU-e145f857-c145-9a57-bf52-723eb3016d9a", (8, 6), False)
VOLTA = ("GPU-70700000-0000-0000-0000-000000000000", (7, 0), False)


class Child:
    """_query_in_child: the answer of the child process (None: none could be started)."""

    def __init__(self, answer: tuple[int, list[tuple[int, int]]] | None = None) -> None:
        self.answer, self.calls = answer, 0

    def __call__(self) -> tuple[int, list[tuple[int, int]]] | None:
        self.calls += 1
        return self.answer


@pytest.fixture(autouse=True)
def _no_cuda_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)


@pytest.fixture
def load(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., ModuleType]]:
    def _load(
        name: str = "dyng_cu13",
        driver: FakeDriver | None = None,
        nvml: FakeNvml | None = None,
        child: Child | None = None,
    ) -> ModuleType:
        opened: list[str] = []

        def cdll(path: str, *args: Any, **kwargs: Any) -> Any:
            opened.append(path)
            library = nvml if "nvidia-ml" in path else driver
            if library is None:
                raise OSError(f"{path}: cannot open shared object file")
            return library

        monkeypatch.setattr(ctypes, "CDLL", cdll)
        spec = importlib.util.spec_from_file_location(
            name, SOURCE, submodule_search_locations=[str(SOURCE.parent)]
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # monkeypatch restores sys.modules: an installed plugin (dyng_cu13 and its module,
        # imported when the plugin is active) is back in place after the test.
        monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.delitem(sys.modules, f"{name}._core", raising=False)
        monkeypatch.delitem(sys.modules, f"{name}._devices", raising=False)
        spec.loader.exec_module(module)
        module._test_real_query_in_child = module._query_in_child  # type: ignore[attr-defined]
        module._query_in_child = child or Child(None)  # type: ignore[attr-defined]
        module._test_opened = opened  # type: ignore[attr-defined]
        return module

    yield _load


def test_identity_comes_from_the_package_name(load: Callable[..., ModuleType]) -> None:
    for name, major in (("dyng_cu13", 13), ("dyng_cu12", 12)):
        m = load(name)
        assert (m.PLUGIN, m.CUDA_MAJOR, m.DISTRIBUTION) == (name[5:], major, f"dyng-{name[5:]}")


def test_another_name_is_refused(load: Callable[..., ModuleType]) -> None:
    with pytest.raises(ImportError, match="dyng_cu12 or dyng_cu13"):
        load("dyng_plugin")


def test_importing_probes_nothing(load: Callable[..., ModuleType]) -> None:
    m = load(driver=FakeDriver())
    assert m._test_opened == []  # the driver is asked on the first status() only


def test_usable_through_nvml_without_initializing_cuda(load: Callable[..., ModuleType]) -> None:
    driver, nvml = FakeDriver(version=13010), FakeNvml()
    m = load(driver=driver, nvml=nvml)
    s = m.status()
    assert s == m.Status(True, 13010, 2, "", ((8, 6), (8, 6)), "nvml")
    assert m.available() and m.driver_version() == 13010
    assert m._test_opened == ["libcuda.so.1", "libnvidia-ml.so.1"]
    m.status()
    # cuDriverGetVersion needs no cuInit: this process can still fork workers that use CUDA.
    assert driver.calls == ["cuDriverGetVersion"]  # probed once, never cuInit
    assert nvml.calls == ["init", "shutdown"]


def test_without_nvml_a_child_process_asks_the_driver(load: Callable[..., ModuleType]) -> None:
    driver, child = FakeDriver(), Child((0, [(8, 9)]))
    m = load(driver=driver, child=child)
    assert m.status() == m.Status(True, 13010, 1, "", ((8, 9),), "child process")
    assert child.calls == 1 and driver.calls == ["cuDriverGetVersion"]
    m = load(driver=FakeDriver(), nvml=FakeNvml(init=9), child=Child((0, [(8, 6)])))
    assert m.status().probe == "child process"  # NVML fails (e.g. the driver is not loaded)


def test_without_a_child_process_the_driver_is_asked_here(load: Callable[..., ModuleType]) -> None:
    driver = FakeDriver(devices=2, capabilities=[(8, 6), (9, 0)])
    m = load(driver=driver)
    assert m.status() == m.Status(True, 13010, 2, "", ((8, 6), (9, 0)), "in process")
    assert driver.calls == ["cuDriverGetVersion", "cuInit", "cuDeviceGetCount"]


def test_no_driver(load: Callable[..., ModuleType]) -> None:
    m = load(driver=None)
    s = m.status()
    assert not s.usable and s.driver_version is None and s.device_count == 0
    assert "libcuda.so.1" in s.reason and "580.65.06" in s.reason
    assert not m.available()


def test_a_driver_of_an_older_cuda_major(load: Callable[..., ModuleType]) -> None:
    m = load("dyng_cu13", FakeDriver(version=12080), FakeNvml())
    s = m.status()
    assert not s.usable and s.driver_version == 12080
    assert "CUDA 12.8" in s.reason and 'pip install "dyng[cu12]"' in s.reason


def test_cu12_runs_on_a_cuda_13_driver(load: Callable[..., ModuleType]) -> None:
    # CUDA's backward compatibility: a newer driver runs code built with an older toolkit.
    assert load("dyng_cu12", FakeDriver(version=13010), FakeNvml()).available()


def test_no_visible_device(load: Callable[..., ModuleType]) -> None:
    m = load(driver=FakeDriver(), nvml=FakeNvml([]))
    s = m.status()
    assert not s.usable and s.reason == "no CUDA device is visible" and s.driver_version == 13010
    m = load(driver=FakeDriver(), child=Child((100, [])))  # CUDA_ERROR_NO_DEVICE in the child
    assert m.status().reason == "no CUDA device is visible"
    m = load(driver=FakeDriver(init=100))  # ... and in this process
    assert m.status().reason == "no CUDA device is visible"
    m = load("dyng_cu12", FakeDriver(devices=0))
    assert m.status().reason == "no CUDA device is visible"


def test_driver_errors_are_reasons(load: Callable[..., ModuleType]) -> None:
    m = load(driver=FakeDriver(init=3))
    assert m.status().reason == "cuInit failed (CUresult 3)"
    m = load(driver=FakeDriver(), child=Child((999, [])))
    assert m.status().reason == "cuInit failed (CUresult 999)"


@pytest.mark.parametrize(
    "where",
    [
        {"nvml": FakeNvml([VOLTA])},
        {"child": Child((0, [(7, 0), (6, 1)]))},
        {"driver": FakeDriver(devices=1, capabilities=[(7, 0)])},
    ],
    ids=["nvml", "child", "in-process"],
)
def test_a_gpu_below_the_architecture_floor_is_unusable(
    load: Callable[..., ModuleType], where: dict[str, Any]
) -> None:
    m = load(**{"driver": FakeDriver(), **where})
    s = m.status()
    assert not s.usable and s.device_count >= 1
    assert (
        "sm_70" in s.reason and "older than the oldest architecture of dyng-cu13, sm_75" in s.reason
    )


def test_one_supported_gpu_is_enough(load: Callable[..., ModuleType]) -> None:
    m = load(driver=FakeDriver(), nvml=FakeNvml([VOLTA, GPU_A]))
    assert m.status() == m.Status(True, 13010, 2, "", ((7, 0), (8, 6)), "nvml")


def test_the_architecture_floor_is_the_release_lists_first_entry(
    load: Callable[..., ModuleType],
) -> None:
    cmake = SOURCE.parents[3] / "cmake" / "cuda_architectures.cmake"
    lists = re.findall(r'set\(_archs "([^"]+)"\)', cmake.read_text())
    assert lists, "no release list found in cmake/cuda_architectures.cmake"
    firsts = {archs.split(";")[0].split("-")[0] for archs in lists}
    major, minor = load().ARCHITECTURE_FLOOR
    assert firsts == {f"{major}{minor}"}


@pytest.mark.parametrize(
    ("visible", "order", "devices", "expected"),
    [
        ("", None, [GPU_A, GPU_B], []),
        ("1", None, [GPU_A, GPU_B], [(8, 6)]),  # one kind of GPU: the order cannot matter
        ("0,1", None, [VOLTA, GPU_A], [(7, 0), (8, 6)]),  # every GPU: the order cannot matter
        ("1", "PCI_BUS_ID", [VOLTA, GPU_A], [(8, 6)]),
        (" 1 ", "PCI_BUS_ID", [VOLTA, GPU_A], [(8, 6)]),
        ("0", "PCI_BUS_ID", [VOLTA, GPU_A], [(7, 0)]),
        ("1,0", "PCI_BUS_ID", [VOLTA, GPU_A], [(8, 6), (7, 0)]),
        ("0,7,1", None, [GPU_A, GPU_B], [(8, 6)]),  # stops at the first ordinal without a device
        ("0,0", None, [GPU_A, GPU_B], []),  # a device named twice hides every device
        ("-1", None, [GPU_A, GPU_B], []),
        ("none", None, [GPU_A, GPU_B], []),
        ("GPU-e145", None, [VOLTA, GPU_A, GPU_B], [(8, 6)]),  # a UUID prefix
        ("GPU-70", None, [VOLTA, GPU_A], [(7, 0)]),
        ("GPU-", None, [GPU_A, GPU_B], []),  # ambiguous: names no single device
        ("GPU-e145,0", "PCI_BUS_ID", [VOLTA, GPU_A, GPU_B], [(8, 6), (7, 0)]),
    ],
)
def test_cuda_visible_devices(
    load: Callable[..., ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    visible: str,
    order: str | None,
    devices: list[Any],
    expected: list[tuple[int, int]],
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    if order is not None:
        monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    child = Child((0, [(1, 0)]))  # must not be asked
    m = load(driver=FakeDriver(), nvml=FakeNvml(devices), child=child)
    s = m.status()
    assert child.calls == 0
    assert list(s.capabilities) == expected and s.probe == "nvml"
    assert s.usable == any(cc >= (7, 5) for cc in expected)


@pytest.mark.parametrize(
    ("visible", "devices"),
    [
        ("1", [VOLTA, GPU_A]),  # ordinals on two kinds of GPU: CUDA numbers them fastest first
        ("MIG-0d4f", [GPU_A]),
        (None, [(GPU_A[0], (8, 0), True)]),  # MIG enabled on a GPU
    ],
)
def test_what_nvml_cannot_tell_is_asked_in_a_child_process(
    load: Callable[..., ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    visible: str | None,
    devices: list[Any],
) -> None:
    if visible is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    child = Child((0, [(8, 0)]))
    driver = FakeDriver()
    m = load(driver=driver, nvml=FakeNvml(devices), child=child)
    assert m.status() == m.Status(True, 13010, 1, "", ((8, 0),), "child process")
    assert child.calls == 1 and "cuInit" not in driver.calls


def test_the_device_query_of_the_child_process() -> None:
    spec = importlib.util.spec_from_file_location("_devices", SOURCE.parent / "_devices.py")
    assert spec is not None and spec.loader is not None
    devices = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(devices)
    assert devices.query(FakeDriver(devices=2, capabilities=[(7, 5), (12, 0)])) == (
        0,
        [(7, 5), (12, 0)],
    )
    assert devices.query(FakeDriver(init=100)) == (100, [])


def test_the_child_process_answer_is_parsed(
    load: Callable[..., ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    m = load(driver=FakeDriver())
    real = m._test_real_query_in_child
    runs: list[list[str]] = []
    answer = ""

    def run(args: list[str], **kwargs: Any) -> Any:
        runs.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=answer, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    answer = '{"rc": 0, "capabilities": [[8, 6], [12, 0]]}\n'
    assert real() == (0, [(8, 6), (12, 0)])
    assert runs[0][1:3] == ["-I", "-S"] and runs[0][3].endswith("_devices.py")
    answer = '{"rc": -1, "capabilities": []}'  # the child could not load the driver
    assert real() is None
    answer = "Traceback (most recent call last): ..."
    assert real() is None
    monkeypatch.setattr(sys, "executable", "")
    assert real() is None


def test_native_is_imported_on_access(load: Callable[..., ModuleType]) -> None:
    m = load()
    # The source tree has no compiled module: the access tries dyng_cu13._core.
    with pytest.raises(ModuleNotFoundError, match="dyng_cu13._core"):
        _ = m.native
    with pytest.raises(AttributeError):
        _ = m.no_such_attribute
