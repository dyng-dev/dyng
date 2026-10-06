# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The import package of the CUDA plugins (python/plugin/dyng_plugin; ADR 0030).

The package is loaded from the source tree under a plugin's name (``dyng_cu13``, ``dyng_cu12``)
with a fake CUDA driver in place of ``libcuda.so.1``, so every answer of ``status()`` is checked
on any machine. The real plugin wheels are tested by ci/plugin_wheels.sh.
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "plugin" / "dyng_plugin" / "__init__.py"

pytestmark = pytest.mark.skipif(not SOURCE.is_file(), reason="needs the source tree")


class FakeDriver:
    """The four driver entry points the package calls, with chosen answers."""

    def __init__(self, version: int = 13010, init: int = 0, devices: int = 2) -> None:
        self.version, self.init, self.devices = version, init, devices
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


@pytest.fixture
def load(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., ModuleType]]:
    names: list[str] = []

    def _load(name: str = "dyng_cu13", driver: FakeDriver | None = None) -> ModuleType:
        opened: list[str] = []

        def cdll(path: str, *args: Any, **kwargs: Any) -> Any:
            opened.append(path)
            if driver is None:
                raise OSError(f"{path}: cannot open shared object file")
            return driver

        monkeypatch.setattr(ctypes, "CDLL", cdll)
        spec = importlib.util.spec_from_file_location(
            name, SOURCE, submodule_search_locations=[str(SOURCE.parent)]
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        names.append(name)
        spec.loader.exec_module(module)
        module._test_opened = opened  # type: ignore[attr-defined]
        return module

    yield _load
    for name in names:
        sys.modules.pop(name, None)


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


def test_usable_with_a_driver_and_a_device(load: Callable[..., ModuleType]) -> None:
    driver = FakeDriver(version=13010, devices=2)
    m = load(driver=driver)
    s = m.status()
    assert s == m.Status(True, 13010, 2, "")
    assert m.available() and m.driver_version() == 13010
    assert m._test_opened == ["libcuda.so.1"]
    m.status()
    assert driver.calls == ["cuDriverGetVersion", "cuInit", "cuDeviceGetCount"]  # probed once


def test_no_driver(load: Callable[..., ModuleType]) -> None:
    m = load(driver=None)
    s = m.status()
    assert not s.usable and s.driver_version is None and s.device_count == 0
    assert "libcuda.so.1" in s.reason and "580.65.06" in s.reason
    assert not m.available()


def test_a_driver_of_an_older_cuda_major(load: Callable[..., ModuleType]) -> None:
    m = load("dyng_cu13", FakeDriver(version=12080))
    s = m.status()
    assert not s.usable and s.driver_version == 12080
    assert "CUDA 12.8" in s.reason and 'pip install "dyng[cu12]"' in s.reason


def test_cu12_runs_on_a_cuda_13_driver(load: Callable[..., ModuleType]) -> None:
    # CUDA's backward compatibility: a newer driver runs code built with an older toolkit.
    assert load("dyng_cu12", FakeDriver(version=13010)).available()


def test_no_visible_device(load: Callable[..., ModuleType]) -> None:
    m = load(driver=FakeDriver(init=100))  # CUDA_ERROR_NO_DEVICE
    s = m.status()
    assert not s.usable and s.reason == "no CUDA device is visible" and s.driver_version == 13010
    m = load("dyng_cu12", FakeDriver(devices=0))
    assert m.status().reason == "no CUDA device is visible"


def test_driver_errors_are_reasons(load: Callable[..., ModuleType]) -> None:
    m = load(driver=FakeDriver(init=3))
    assert m.status().reason == "cuInit failed (CUresult 3)"


def test_native_is_imported_on_access(load: Callable[..., ModuleType]) -> None:
    m = load()
    # The source tree has no compiled module: the access tries dyng_cu13._core.
    with pytest.raises(ModuleNotFoundError, match="dyng_cu13._core"):
        _ = m.native
    with pytest.raises(AttributeError):
        _ = m.no_such_attribute
