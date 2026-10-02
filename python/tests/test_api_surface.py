# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The public names of PLAN Section 5.4 and the uniform algorithm contract."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.resources
import inspect
import os

import dyng
import pytest
from dyng._algorithms import ALGORITHMS

PUBLIC = [
    "Resources",
    "get_default_resources",
    "set_default_resources",
    "Graph",
    "GraphProperties",
    "BatchSemantics",
    "ApplySummary",
    "EdgeBatch",
    "Array",
    "update",
    "sssp",
    "cycle_count",
    "mosp",
    "io",
    "generators",
    "testing",
    "profile",
    "Profiler",
    "citation",
    "show_config",
    "algorithms",
    "use_cpu_only",
    "Error",
    "InvalidArgumentError",
    "StaleResultError",
    "FileFormatError",
    "CapacityError",
    "NotSupportedError",
    "ConvergenceError",
    "CudaError",
    "OutOfMemoryError",
    "InternalError",
]


@pytest.mark.parametrize("name", PUBLIC)
def test_public_name(name: str) -> None:
    assert hasattr(dyng, name)
    assert name in dyng.__all__


def test_all_is_importable() -> None:
    for name in dyng.__all__:
        assert getattr(dyng, name) is not None


def test_version() -> None:
    assert dyng.__version__ == dyng._backend.native.library_version()
    try:
        assert importlib.metadata.version("dyng") == dyng.__version__
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - not installed
        pytest.skip("dyng is not installed as a distribution")


def test_registry_matches_modules() -> None:
    names = [a.name for a in dyng.algorithms()]
    assert names == list(ALGORITHMS)
    for a in dyng.algorithms():
        assert a.backends[0] == "sequential"
        assert a.cite
        assert a.title == ALGORITHMS[a.name]["title"]


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_uniform_contract(name: str) -> None:
    module = importlib.import_module(f"dyng.{name}")
    assert getattr(dyng, name) is module
    for attr in ("Options", "Result", "Stats", "compute", "update"):
        assert hasattr(module, attr), f"dyng.{name}.{attr}"
    update = inspect.signature(module.update)
    assert list(update.parameters)[:3] == ["graph", "batch", "result"]
    assert update.parameters["resources"].kind is inspect.Parameter.KEYWORD_ONLY
    compute = inspect.signature(module.compute)
    assert compute.parameters["resources"].kind is inspect.Parameter.KEYWORD_ONLY
    assert compute.parameters["options"].kind is inspect.Parameter.KEYWORD_ONLY
    stats_fields = {f for f in module.Stats.__dataclass_fields__}
    assert {"affected", "iterations", "engine_used", "batch"} <= stats_fields


def test_citation() -> None:
    assert "@software{dyng" in dyng.citation()
    text = dyng.citation("sssp")
    assert "dynamosp2025" in text
    assert "trucy2026" in dyng.citation("cycle_count")
    assert "dynamosp2025" in dyng.citation("mosp")
    assert "dynamosp2025" in dyng.citation_keys("sssp")
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.citation("no_such_algorithm")


def test_show_config(capsys: pytest.CaptureFixture[str]) -> None:
    dyng.show_config()
    out = capsys.readouterr().out
    assert out.startswith(f"dynG {dyng.__version__}")
    assert "sequential (yes)" in out
    c = dyng.config()
    assert c["native_module"] == "dyng._core"
    assert c["backends"]["sequential"] is True


def test_use_cpu_only_is_a_no_op_with_the_cpu_module() -> None:
    dyng.use_cpu_only()


_FAKE_PLUGIN = """
import sys, types
from importlib import metadata

loads = []
fake_native = types.SimpleNamespace(__version__="fake", _set_error_types=lambda classes: None)


class EntryPoint:
    name = "cu99"
    value = "fake_plugin"

    def load(self):
        loads.append(self.name)
        return types.SimpleNamespace(available=lambda: True, native=fake_native)


real = metadata.entry_points


def fake_entry_points(**kw):
    return [EntryPoint()] if kw.get("group") == "dyng.backends" else real(**kw)


metadata.entry_points = fake_entry_points
import dyng, dyng._backend as backend
assert backend.active_module_name is None and "dyng._core" not in sys.modules, "chosen at import"
"""


def _run(code: str) -> str:
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "DYNG_CPU_ONLY"}
    out = subprocess.run(
        [sys.executable, "-c", _FAKE_PLUGIN + code], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_the_native_module_is_chosen_on_first_use() -> None:
    # A plugin process imports only the plugin's module, never dyng._core.
    out = _run(
        "print(dyng.__version__, backend.active_module_name, loads, 'dyng._core' in sys.modules)\n"
        "try:\n"
        "    dyng.use_cpu_only()\n"
        "except RuntimeError as e:\n"
        "    print('refused:', 'already' in str(e))\n"
    )
    assert out.split("\n")[:2] == ["fake fake_plugin (plugin cu99) ['cu99'] False", "refused: True"]


def test_use_cpu_only_before_first_use_forces_the_cpu_module() -> None:
    out = _run(
        "dyng.use_cpu_only()\n"
        "print(backend.active_module_name, loads, dyng.__version__ != 'fake')\n"
        "dyng.use_cpu_only()  # again: a no-op\n"
    )
    assert out.strip() == "dyng._core [] True"


def test_log_level_round_trip() -> None:
    old = dyng.get_log_level()
    try:
        dyng.set_log_level("error")
        assert dyng.get_log_level() == "error"
        with pytest.raises(dyng.InvalidArgumentError):
            dyng.set_log_level("loud")  # type: ignore[arg-type]
    finally:
        dyng.set_log_level(old)


def test_stubs_are_committed() -> None:
    import dyng as pkg

    stub = importlib.resources.files(pkg) / "_core.pyi"
    assert stub.is_file()
    text = stub.read_text(encoding="utf-8")
    assert "def sssp_compute" in text and "class GraphI32I32I32" in text
    assert (importlib.resources.files(pkg) / "py.typed").is_file()


@pytest.mark.parametrize(
    ("cls", "native_name"),
    [
        (dyng.sssp.Options, "SsspOptions"),
        (dyng.cycle_count.Options, "CycleCountOptions"),
        (dyng.mosp.Options, "MospOptions"),
        (dyng.GraphProperties, "GraphProperties"),
        (dyng.BatchSemantics, "BatchSemantics"),
    ],
)
def test_python_defaults_equal_the_cpp_defaults(cls: type, native_name: str) -> None:
    # ADR 0011 item 9: the dataclasses carry the C++ field names and defaults; a changed C++
    # default must not drift silently from the Python one.
    from dyng._backend import native

    assert cls() == cls._from_native(getattr(native, native_name)())
