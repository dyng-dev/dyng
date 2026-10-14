# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The choice of the native module: CUDA plugins, the CPU fallback and use_cpu_only() (ADR 0031).

No GPU is needed: the plugins are fakes. The rules are checked on ``dyng._backend._select()``
in this process (fake entry points, a fake CPU module), and the first use, the
:class:`dyng.BackendWarning`, ``DYNG_CPU_ONLY``, ``use_cpu_only()``, ``show_config()`` and
``Resources.cuda()`` in fresh processes whose entry points are replaced before dynG is used.
The real plugins are tested by test_cuda.py (marker gpu) and ci/plugin_wheels.sh.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from typing import Any

import dyng
import dyng._backend as backend
import pytest

VERSION = "9.8.7"
CORE = types.ModuleType("fake_core")


def _status(usable: bool, driver: int | None, reason: str) -> Any:
    return types.SimpleNamespace(
        usable=usable, driver_version=driver, device_count=1 if usable else 0, reason=reason
    )


class FakePlugin(types.ModuleType):
    """A plugin package: CUDA_MAJOR, __version__, available(), status() and a lazy ``native``."""

    def __init__(
        self,
        major: int | None,
        *,
        version: str | None = VERSION,
        usable: bool = True,
        driver: int | None = 13010,
        reason: str = "",
        native_error: Exception | None = None,
        module_version: str | None = None,
        module_cuda: bool = True,
    ) -> None:
        super().__init__(f"fake_cu{major}")
        if major is not None:
            self.CUDA_MAJOR = major
        if version is not None:
            self.__version__ = version
        self._usable, self._driver, self._reason = usable, driver, reason
        self._native_error = native_error
        self.module = types.ModuleType(f"fake_cu{major}._core")
        self.module.__version__ = module_version or version or VERSION  # type: ignore[attr-defined]
        self.module.build_config = {"cuda": module_cuda}  # type: ignore[attr-defined]
        self.native_reads = 0

    def available(self) -> bool:
        return self._usable

    def status(self) -> Any:
        return _status(self._usable, self._driver, self._reason)

    @property
    def native(self) -> types.ModuleType:
        self.native_reads += 1
        if self._native_error is not None:
            raise self._native_error
        return self.module


class EntryPoint:
    """An entry point of the group dyng.backends."""

    def __init__(self, name: str, plugin: Any = None, error: Exception | None = None) -> None:
        self.name, self.value = name, f"dyng_{name}"
        self.plugin, self.error, self.loads = plugin, error, 0

    def load(self) -> Any:
        self.loads += 1
        if self.error is not None:
            raise self.error
        return self.plugin


def select(*eps: EntryPoint, cpu_only: bool = False, version: str | None = VERSION) -> Any:
    module, chosen = backend._select(
        list(eps), cpu_only=cpu_only, dyng_version=version, load_core=lambda: CORE
    )
    return module, chosen


def states(chosen: Any) -> dict[str, str]:
    return {p.name: p.state for p in chosen.plugins}


# -------------------------------------------------------------------------------------------------
# The rules, on _select()
# -------------------------------------------------------------------------------------------------


def test_no_plugin_chooses_the_cpu_module_silently() -> None:
    module, chosen = select()
    assert module is CORE and chosen.module_name == "dyng._core" and chosen.plugin is None
    assert chosen.reason == "no CUDA plugin is installed"
    assert chosen.warning is None  # the CPU-only install behaves as in 0.1.0


def test_a_usable_plugin_of_the_same_version_is_chosen() -> None:
    cu13 = FakePlugin(13)
    module, chosen = select(EntryPoint("cu13", cu13))
    assert module is cu13.module
    assert chosen.module_name == "dyng_cu13 (plugin cu13)" and chosen.plugin == "cu13"
    assert "matches the driver's CUDA major (CUDA 13.1)" in chosen.reason
    assert states(chosen) == {"cu13": "chosen"} and chosen.warning is None
    report = chosen.plugins[0]
    assert (report.version, report.cuda_major, report.driver_version) == (VERSION, 13, 13010)


def test_the_plugin_of_the_drivers_major_wins() -> None:
    cu12, cu13 = FakePlugin(12), FakePlugin(13)
    module, chosen = select(EntryPoint("cu12", cu12), EntryPoint("cu13", cu13))
    assert module is cu13.module and chosen.plugin == "cu13"
    assert states(chosen) == {"cu12": "available", "cu13": "chosen"}
    assert "dyng_cu13 was chosen" in chosen.plugins[0].reason
    assert cu12.native_reads == 0  # only the chosen module is imported


def test_a_cuda_12_driver_chooses_the_cu12_plugin() -> None:
    cu12 = FakePlugin(12, driver=12080)
    cu13 = FakePlugin(13, usable=False, driver=12080, reason="the driver supports CUDA 12.8")
    module, chosen = select(EntryPoint("cu13", cu13), EntryPoint("cu12", cu12))
    assert module is cu12.module and chosen.plugin == "cu12"
    assert states(chosen) == {"cu12": "chosen", "cu13": "unusable"}
    assert [p.name for p in chosen.plugins] == ["cu12", "cu13"]  # name order


def test_a_newer_driver_chooses_the_newest_older_major() -> None:
    cu12, cu13 = FakePlugin(12, driver=14000), FakePlugin(13, driver=14000)
    module, chosen = select(EntryPoint("cu12", cu12), EntryPoint("cu13", cu13))
    assert module is cu13.module
    assert "no CUDA 14 plugin is installed, and CUDA 13 runs on it" in chosen.reason


def test_a_newer_drivers_plugin_that_cannot_be_used_is_named() -> None:
    cu13 = FakePlugin(13, driver=14000)
    cu14 = FakePlugin(14, version="9.8.6", driver=14000)
    module, chosen = select(EntryPoint("cu13", cu13), EntryPoint("cu14", cu14))
    assert module is cu13.module
    assert "the CUDA 14 plugin cannot be used, and CUDA 13 runs on it" in chosen.reason


def test_a_version_mismatch_falls_back_with_a_warning() -> None:
    cu13 = FakePlugin(13, version="9.8.6")
    module, chosen = select(EntryPoint("cu13", cu13))
    assert module is CORE and states(chosen) == {"cu13": "version mismatch"}
    assert cu13.native_reads == 0  # a plugin of another version is never imported
    assert chosen.warning is not None
    assert "version 9.8.6 does not match dyng 9.8.7" in chosen.warning
    assert 'pip install "dyng[cu13]==9.8.7"' in chosen.warning


def test_the_modules_version_is_checked_too() -> None:
    cu13 = FakePlugin(13, version=None, module_version="1.0")
    module, chosen = select(EntryPoint("cu13", cu13))
    assert module is CORE and states(chosen) == {"cu13": "failed"}
    assert "its module is version 1.0, dyng is 9.8.7" in chosen.plugins[0].reason


def test_without_dyng_metadata_the_version_is_not_checked() -> None:
    cu13 = FakePlugin(13, version="1.0")
    module, _ = select(EntryPoint("cu13", cu13), version=None)
    assert module is cu13.module


@pytest.mark.parametrize(
    ("driver", "reason"),
    [
        (None, "no CUDA driver: libcuda.so.1 cannot be loaded"),
        (12080, "the CUDA driver supports CUDA 12.8, but dyng-cu13 needs CUDA 13.0 or newer"),
        (13010, "no CUDA device is visible"),
    ],
)
def test_an_unusable_plugin_falls_back_with_its_reason(driver: int | None, reason: str) -> None:
    cu13 = FakePlugin(13, usable=False, driver=driver, reason=reason)
    module, chosen = select(EntryPoint("cu13", cu13))
    assert module is CORE and chosen.plugin is None
    assert chosen.reason == "no installed CUDA plugin can be used (see the plugins)"
    assert states(chosen) == {"cu13": "unusable"}
    warning = chosen.warning
    assert warning is not None and reason in warning and "dyng._core" in warning
    assert "DYNG_CPU_ONLY=1" in warning


def test_a_broken_plugin_is_reported_and_skipped() -> None:
    cu12 = FakePlugin(12, driver=13010)
    module, chosen = select(
        EntryPoint("cu13", error=ImportError("no module named dyng_cu13")),
        EntryPoint("cu12", cu12),
    )
    assert module is cu12.module and states(chosen) == {"cu12": "chosen", "cu13": "failed"}
    assert "ImportError: no module named dyng_cu13" in backend.plugin_errors["cu13"]


def test_a_module_that_fails_to_import_passes_to_the_next_candidate() -> None:
    cu12 = FakePlugin(12)
    cu13 = FakePlugin(13, native_error=ImportError("GLIBC_2.99 not found"))
    module, chosen = select(EntryPoint("cu12", cu12), EntryPoint("cu13", cu13))
    assert module is cu12.module and states(chosen) == {"cu12": "chosen", "cu13": "failed"}
    assert "GLIBC_2.99 not found" in chosen.plugins[1].reason


def test_a_module_without_cuda_is_refused() -> None:
    cu13 = FakePlugin(13, module_cuda=False)
    module, chosen = select(EntryPoint("cu13", cu13))
    assert module is CORE and "built without CUDA" in chosen.plugins[0].reason


def test_a_plugin_whose_available_raises_is_reported() -> None:
    cu13 = FakePlugin(13)
    cu13.available = lambda: 1 / 0  # type: ignore[method-assign]
    module, chosen = select(EntryPoint("cu13", cu13))
    assert module is CORE and states(chosen) == {"cu13": "failed"}
    assert "ZeroDivisionError" in chosen.plugins[0].reason


def test_the_minimal_contract_is_enough() -> None:
    # ADR 0011 item 14: an entry point names a module with available() and native.
    module = types.ModuleType("minimal._core")
    module.build_config = {"cuda": True}  # type: ignore[attr-defined]
    minimal = types.SimpleNamespace(available=lambda: True, native=module)
    chosen_module, chosen = select(EntryPoint("cu99", minimal))
    assert chosen_module is module and chosen.reason == "plugin cu99: it is available"


def test_cpu_only_loads_no_plugin_and_does_not_warn() -> None:
    ep = EntryPoint("cu13", FakePlugin(13, usable=False, reason="no driver"))
    module, chosen = select(ep, cpu_only=True)
    assert module is CORE and ep.loads == 0 and chosen.warning is None
    assert states(chosen) == {"cu13": "not considered"}
    assert "DYNG_CPU_ONLY=1 or dyng.use_cpu_only()" in chosen.reason


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        (" TRUE ", True),
        ("Yes", True),
        ("on", True),
        ("", False),
        ("0", False),
        ("false", False),
        ("False", False),
        ("no", False),
        ("OFF", False),
        (" off ", False),
    ],
)
def test_the_values_of_dyng_cpu_only(
    value: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backend, "_cpu_only", False)
    monkeypatch.setenv("DYNG_CPU_ONLY", value)
    assert backend._cpu_only_requested() == (expected, None)


@pytest.mark.parametrize("value", ["2", "maybe", "y", "enable"])
def test_an_unknown_dyng_cpu_only_is_ignored_with_a_warning(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backend, "_cpu_only", False)
    monkeypatch.setenv("DYNG_CPU_ONLY", value)
    cpu_only, warning = backend._cpu_only_requested()
    assert not cpu_only
    assert warning is not None and f"DYNG_CPU_ONLY={value!r} is ignored" in warning


def test_unset_dyng_cpu_only_and_use_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYNG_CPU_ONLY", raising=False)
    monkeypatch.setattr(backend, "_cpu_only", False)
    assert backend._cpu_only_requested() == (False, None)
    monkeypatch.setattr(backend, "_cpu_only", True)
    monkeypatch.setenv("DYNG_CPU_ONLY", "maybe")  # use_cpu_only() wins, nothing to warn about
    assert backend._cpu_only_requested() == (True, None)


def test_this_process_reports_its_choice() -> None:
    chosen = backend.selection()
    assert chosen.module_name == backend.active_module_name
    c = dyng.config()
    assert c["selection"] == chosen.reason
    assert [p["name"] for p in c["plugins"]] == [p.name for p in chosen.plugins]


# -------------------------------------------------------------------------------------------------
# A fresh process: the first use, the warning, DYNG_CPU_ONLY, use_cpu_only(), show_config()
# -------------------------------------------------------------------------------------------------

_PRELUDE = """
import sys, types, warnings
warnings.simplefilter("always")
import dyng._backend as backend

loads = []
version = backend._dyng_version()
fake_native = types.ModuleType("fake_native")
fake_native.__version__ = version or "fake"
fake_native.build_config = {"cuda": True}
fake_native._set_error_types = lambda classes: None


class Plugin:
    CUDA_MAJOR = 13
    __version__ = version

    def __init__(self, usable, reason):
        self.usable, self.reason = usable, reason

    def available(self):
        return self.usable

    def status(self):
        driver = 13010 if self.usable else None
        return types.SimpleNamespace(usable=self.usable, driver_version=driver,
                                     device_count=int(self.usable), reason=self.reason)

    @property
    def native(self):
        return fake_native


class EntryPoint:
    name, value = "cu13", "dyng_cu13"

    def load(self):
        loads.append(self.name)
        return PLUGIN


PLUGIN = Plugin(USABLE, "no CUDA driver: libcuda.so.1 cannot be loaded")
backend._entry_points = lambda: [EntryPoint()]
import dyng
assert backend.active_module_name is None and "dyng._core" not in sys.modules, "chosen at import"
"""


def _run(code: str, *, usable: bool, env: dict[str, str] | None = None) -> str:
    full = {k: v for k, v in os.environ.items() if k != "DYNG_CPU_ONLY"}
    full.update(env or {})
    prelude = _PRELUDE.replace("USABLE", repr(usable))
    out = subprocess.run(
        [sys.executable, "-c", prelude + code], capture_output=True, text=True, env=full
    )
    assert out.returncode == 0, out.stderr
    return out.stdout + out.stderr


def test_the_native_module_is_chosen_on_first_use() -> None:
    # A plugin process imports only the plugin's module, never dyng._core.
    out = _run(
        "print(dyng.__version__ == fake_native.__version__, backend.active_module_name, loads,\n"
        "      'dyng._core' in sys.modules)\n"
        "try:\n"
        "    dyng.use_cpu_only()\n"
        "except RuntimeError as e:\n"
        "    print('refused:', 'already' in str(e))\n",
        usable=True,
    )
    assert out.split("\n")[:2] == ["True dyng_cu13 (plugin cu13) ['cu13'] False", "refused: True"]


def test_an_unusable_plugin_warns_once_and_runs_on_the_cpu_module() -> None:
    out = _run(
        "with warnings.catch_warnings(record=True) as w:\n"
        "    g = dyng.Graph.from_edges([0], [1], [1])\n"
        "    dyng.sssp.compute(g, 0)\n"
        "print('warnings:', len(w), w[0].category.__name__, w[0].filename)\n"
        "print('module:', backend.active_module_name, loads)\n"
        "assert 'no CUDA driver' in str(w[0].message), w[0].message\n"
        "try:\n"
        "    dyng.Resources.cuda()\n"
        "except dyng.NotSupportedError as e:\n"
        "    print('cuda:', 'no CUDA driver' in str(e))\n"
        "dyng.show_config()\n",
        usable=False,
    )
    assert "warnings: 1 BackendWarning <string>" in out  # points at the caller, not at dyng
    assert "module: dyng._core ['cu13']" in out
    assert "cuda: True" in out
    assert "chosen because    : no installed CUDA plugin can be used" in out
    assert "CUDA plugin       : dyng_cu13" in out and "unusable (no CUDA driver" in out
    # The plugin is installed: show_config() points at its line, not at `pip install`.
    assert "the installed CUDA plugins are not used (see the plugin lines below)" in out
    assert "pip install" not in out


@pytest.mark.parametrize(
    ("how", "env"),
    [("dyng.use_cpu_only()\n", None), ("", {"DYNG_CPU_ONLY": "1"})],
    ids=["use_cpu_only", "DYNG_CPU_ONLY"],
)
def test_asking_for_the_cpu_module_does_not_warn(how: str, env: dict[str, str] | None) -> None:
    out = _run(
        "with warnings.catch_warnings(record=True) as w:\n"
        f"    {how.strip() or 'pass'}\n"
        "    dyng.__version__\n"
        "print('warnings:', len(w), backend.active_module_name, loads)\n"
        "dyng.use_cpu_only()  # again: a no-op\n"
        "dyng.show_config()\n",
        usable=False,
        env=env,
    )
    assert "warnings: 0 dyng._core []" in out
    assert "not considered" in out


@pytest.mark.parametrize("value", ["false", "No", "off", "0"])
def test_a_false_dyng_cpu_only_keeps_the_plugin(value: str) -> None:
    out = _run(
        "with warnings.catch_warnings(record=True) as w:\n"
        "    dyng.__version__\n"
        "print('warnings:', len(w), backend.active_module_name, loads)\n",
        usable=True,
        env={"DYNG_CPU_ONLY": value},
    )
    assert "warnings: 0 dyng_cu13 (plugin cu13) ['cu13']" in out


def test_an_unknown_dyng_cpu_only_warns_and_keeps_the_plugin() -> None:
    out = _run(
        "with warnings.catch_warnings(record=True) as w:\n"
        "    dyng.__version__\n"
        "print('warnings:', len(w), w[0].category.__name__, backend.active_module_name)\n"
        "print(w[0].message)\n",
        usable=True,
        env={"DYNG_CPU_ONLY": "maybe"},
    )
    assert "warnings: 1 BackendWarning dyng_cu13 (plugin cu13)" in out
    assert "DYNG_CPU_ONLY='maybe' is ignored" in out


def test_the_warning_can_be_filtered_by_its_class() -> None:
    out = _run(
        "warnings.filterwarnings('error', category=dyng.BackendWarning)\n"
        "try:\n"
        "    dyng.__version__\n"
        "except dyng.BackendWarning as e:\n"
        "    print('raised:', 'dyng_cu13' in str(e))\n",
        usable=False,
    )
    assert "raised: True" in out
