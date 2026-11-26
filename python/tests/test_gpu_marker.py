# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Every test case that runs on the CUDA backend carries the ``gpu`` marker.

The GPU tests of the plugin wheels select the CUDA tests with ``pytest -m gpu`` (ci/plugin_wheels.sh
and the check of the published wheels, docs/developer/release.md steps 8 and 10), so a CUDA case
without the marker is deselected there and never runs against the wheels that are published: the
hyper_sssp example's ``cuda`` case of test_examples.py was such a case in 0.4.0rc1 (R041). The
check reads the test cases of this session (the whole suite in a plain run).
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest


def _unmarked_cuda_cases(items: Iterable[Any]) -> list[str]:
    """The node ids of the cases parametrized with ``backend="cuda"`` that lack the gpu marker."""
    out = []
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None or callspec.params.get("backend") != "cuda":
            continue
        if item.get_closest_marker("gpu") is None:
            out.append(item.nodeid)
    return out


def test_every_cuda_case_is_marked_gpu(request: pytest.FixtureRequest) -> None:
    unmarked = _unmarked_cuda_cases(request.session.items)
    assert not unmarked, f"CUDA test cases without the gpu marker (-m gpu skips them): {unmarked}"


def _case(nodeid: str, params: dict[str, object] | None, marks: set[str]) -> SimpleNamespace:
    case = SimpleNamespace(nodeid=nodeid, get_closest_marker=lambda n: n if n in marks else None)
    if params is not None:
        case.callspec = SimpleNamespace(params=params)
    return case


def test_the_check_reports_only_unmarked_cuda_cases() -> None:
    cases = [
        _case("t.py::a[cuda]", {"backend": "cuda"}, set()),  # reported
        _case("t.py::b[cuda]", {"backend": "cuda"}, {"gpu"}),
        _case("t.py::c[sequential]", {"backend": "sequential"}, set()),
        _case("t.py::d[cuda-CudaError]", {"which": "cuda"}, set()),  # not a backend
        _case("t.py::e", None, set()),
    ]
    assert _unmarked_cuda_cases(cases) == ["t.py::a[cuda]"]
