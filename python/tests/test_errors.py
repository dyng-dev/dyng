# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The exception hierarchy of PLAN Section 5.4 and the translation of the C++ exceptions."""

from __future__ import annotations

import pickle
from pathlib import Path

import dyng
import numpy as np
import pytest
from dyng._backend import native

HIERARCHY = [
    (dyng.InvalidArgumentError, (dyng.Error, ValueError)),
    (dyng.StaleResultError, (dyng.InvalidArgumentError, ValueError)),
    (dyng.FileFormatError, (dyng.Error, OSError)),
    (dyng.CapacityError, (dyng.Error, MemoryError)),
    (dyng.NotSupportedError, (dyng.Error, NotImplementedError)),
    (dyng.ConvergenceError, (dyng.Error, RuntimeError)),
    (dyng.CudaError, (dyng.Error, RuntimeError)),
    (dyng.OutOfMemoryError, (dyng.Error, MemoryError)),
    (dyng.InternalError, (dyng.Error, RuntimeError)),
]


@pytest.mark.parametrize(("cls", "bases"), HIERARCHY)
def test_hierarchy(cls: type, bases: tuple[type, ...]) -> None:
    for base in bases:
        assert issubclass(cls, base)
    assert issubclass(cls, dyng.Error)


TRANSLATIONS = [
    ("stale_result", dyng.StaleResultError),
    ("invalid_argument", dyng.InvalidArgumentError),
    ("io", dyng.FileFormatError),
    ("capacity", dyng.CapacityError),
    ("not_supported", dyng.NotSupportedError),
    ("convergence", dyng.ConvergenceError),
    ("cuda", dyng.CudaError),
    ("out_of_memory", dyng.OutOfMemoryError),
    ("internal", dyng.InternalError),
    ("error", dyng.Error),
]


@pytest.mark.parametrize(("which", "cls"), TRANSLATIONS)
def test_translation(which: str, cls: type) -> None:
    with pytest.raises(cls, match="the message") as e:
        native._raise_for_test(which, "the message")
    assert type(e.value) is cls


def test_file_format_error_attributes() -> None:
    with pytest.raises(dyng.FileFormatError) as e:
        native._raise_for_test("io", "bad line")
    assert (e.value.path, e.value.line, e.value.column) == ("some/file.txt", 7, 3)
    assert str(e.value) == "bad line"
    copy = pickle.loads(pickle.dumps(e.value))
    assert (copy.path, copy.line, copy.column) == ("some/file.txt", 7, 3)


def test_cuda_error_code() -> None:
    with pytest.raises(dyng.CudaError) as e:
        native._raise_for_test("cuda", "boom")
    assert e.value.code == 2
    assert pickle.loads(pickle.dumps(e.value)).code == 2


def test_real_file_format_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("0 1\n2 x\n", encoding="utf-8")
    with pytest.raises(dyng.FileFormatError) as e:
        dyng.io.read_edge_list(bad)
    assert e.value.path == str(bad) and e.value.line == 2
    assert isinstance(e.value, OSError)
    with pytest.raises(OSError):
        dyng.io.read_edge_list(tmp_path / "missing.txt")


def test_value_error_is_caught_as_such() -> None:
    g = dyng.Graph.from_edges([0], [1], [1])
    with pytest.raises(ValueError):
        dyng.sssp.compute(g, 5)


@pytest.mark.parametrize("bad", [1.9, "1", True, None, 1.0])
def test_integer_arguments_are_never_truncated_or_parsed(bad: object) -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], [1, 1])
    with pytest.raises(dyng.InvalidArgumentError, match="source: expected an integer"):
        dyng.sssp.compute(g, bad)  # type: ignore[arg-type]
    with pytest.raises(dyng.InvalidArgumentError, match="delta: expected an integer"):
        dyng.sssp.compute(g, 0, delta=bad)
    with pytest.raises(dyng.InvalidArgumentError, match="max_length: expected an integer"):
        dyng.cycle_count.compute(g, max_length=bad)


def test_integer_arguments_accept_numpy_integers() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], [1, 1])
    assert dyng.sssp.compute(g, np.int64(1)).source == 1
    assert dyng.cycle_count.compute(g, max_length=np.int32(3)).bound == 3


def test_boolean_options_must_be_booleans() -> None:
    g = dyng.Graph.from_edges([0, 1], [1, 2], [1, 1])
    with pytest.raises(dyng.InvalidArgumentError, match="validate_inputs: expected True or False"):
        dyng.sssp.compute(g, 0, validate_inputs="no")
    with pytest.raises(dyng.InvalidArgumentError, match="directed: expected True or False"):
        dyng.Graph.from_edges([0], [1], directed=1)  # type: ignore[arg-type]
