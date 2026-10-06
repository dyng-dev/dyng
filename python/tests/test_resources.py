# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.Resources and the default resources."""

from __future__ import annotations

import dyng
import pytest


def test_sequential() -> None:
    r = dyng.Resources.sequential()
    assert r.backend == "sequential"
    assert r.num_threads == 1
    assert r.device == -1
    assert r.default_space == "host"
    assert "sequential" in repr(r)
    assert dyng.Resources("sequential").backend == "sequential"


def test_openmp_threads() -> None:
    if not dyng.config()["backends"]["openmp"]:
        pytest.skip("built without OpenMP")
    r = dyng.Resources.openmp(3)
    assert r.backend == "openmp"
    assert r.num_threads == 3
    assert dyng.Resources("openmp", num_threads=2).num_threads == 2
    with pytest.raises(dyng.InvalidArgumentError):
        dyng.Resources.openmp(-1)


def test_cuda_in_the_cpu_wheel_points_to_the_plugins() -> None:
    if dyng.config()["build"]["cuda"]:
        pytest.skip("a CUDA build")
    with pytest.raises(dyng.NotSupportedError, match="dyng-cu12 and dyng-cu13") as e:
        dyng.Resources.cuda()
    assert isinstance(e.value, NotImplementedError)
    assert "from source" not in str(e.value) and 'pip install "dyng[cu13]"' in str(e.value)
    with pytest.raises(dyng.NotSupportedError):
        dyng.Resources("cuda")


def test_unknown_backend() -> None:
    with pytest.raises(dyng.InvalidArgumentError, match="'openmp'"):
        dyng.Resources("tpu")  # type: ignore[arg-type]


def test_copy_policy() -> None:
    r = dyng.Resources.sequential()
    assert r.copy_policy == "allow"
    r.copy_policy = "error"
    assert r.copy_policy == "error"
    with pytest.raises(dyng.InvalidArgumentError):
        r.copy_policy = "sometimes"  # type: ignore[assignment]


def test_default_resources() -> None:
    try:
        d = dyng.get_default_resources()
        assert d is dyng.get_default_resources()
        assert d.backend == dyng.config()["default_backend"]
        dyng.set_default_resources("sequential")
        assert dyng.get_default_resources().backend == "sequential"
        g = dyng.Graph.from_edges([0], [1], [1])
        assert g.resources.backend == "sequential"
        with pytest.raises(TypeError):
            dyng.set_default_resources(42)  # type: ignore[arg-type]
    finally:
        dyng.set_default_resources(None)


def test_workspace_and_lifecycle() -> None:
    r = dyng.Resources.sequential()
    g = dyng.Graph.from_edges([0, 1], [1, 2], [1, 1], resources=r)
    dyng.sssp.compute(g, 0)
    assert r.workspace_bytes >= 0
    r.release_workspaces()
    r.warm_up()
    r.synchronize()


def test_stream_argument_forms() -> None:
    from dyng.resources import _stream_handle

    class Protocol:
        def __cuda_stream__(self) -> tuple[int, int]:
            return (0, 1234)

    class Cupy:
        ptr = 99

    assert _stream_handle(None) == 0
    assert _stream_handle(7) == 7
    assert _stream_handle(Protocol()) == 1234
    assert _stream_handle(Cupy()) == 99
    with pytest.raises(TypeError):
        _stream_handle("stream")
