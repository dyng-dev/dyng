# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures of the Python tests: the backends, the test data and the Hypothesis profiles.

The tests run against the installed ``dyng`` (an editable install in development, the wheel in a
clean venv); the fixture data are read from the source tree (cpp/tests/data).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

# The default resources use every OpenMP thread; a shared or small machine runs the suite faster
# with a few (set before the OpenMP runtime is loaded with dyng._core; OMP_NUM_THREADS wins).
os.environ.setdefault("OMP_NUM_THREADS", "4")

import dyng  # noqa: E402
import pytest  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "cpp" / "tests" / "data"

try:
    from hypothesis import HealthCheck, settings

    # "dyng": the small profile of PLAN 11.2 (M5); "ci": more examples; "dev": a quick run;
    # "full": the long run of PLAN 8.1 (.github/workflows/property.yml, weekly and on demand):
    # many more examples, NOT derandomized (every run explores new inputs; a failure prints the
    # @reproduce_failure blob to replay it), without the example database.
    settings.register_profile(
        "dyng",
        max_examples=60,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
        derandomize=True,
    )
    settings.register_profile("ci", parent=settings.get_profile("dyng"), max_examples=200)
    settings.register_profile("dev", parent=settings.get_profile("dyng"), max_examples=15)
    settings.register_profile(
        "full",
        parent=settings.get_profile("dyng"),
        max_examples=int(os.environ.get("DYNG_HYPOTHESIS_EXAMPLES", "10000")),
        derandomize=False,
        database=None,
        print_blob=True,
    )
    settings.load_profile(os.environ.get("DYNG_HYPOTHESIS_PROFILE", "dyng"))
except ImportError:  # pragma: no cover - hypothesis is a test dependency
    pass


def host_backends() -> list[str]:
    """The host backends of this build."""
    out = ["sequential"]
    if dyng.config()["backends"]["openmp"]:
        out.append("openmp")
    return out


@pytest.fixture(params=host_backends())
def backend(request: pytest.FixtureRequest) -> str:
    """Each host backend."""
    return request.param


@pytest.fixture
def res(backend: str) -> dyng.Resources:
    """Resources of each host backend (OpenMP with 3 threads, an odd team size)."""
    if backend == "openmp":
        return dyng.Resources.openmp(3)
    return dyng.Resources.sequential()


@pytest.fixture
def data() -> Path:
    """cpp/tests/data of the source tree."""
    if not DATA.is_dir():
        pytest.skip("the test data of the source tree (cpp/tests/data) are not available")
    return DATA


@pytest.fixture
def host_default_resources() -> Iterator[None]:
    """The sequential backend as the default resources for the test, also when a CUDA plugin's
    module is active (whose default backend is cuda): for tests of host-memory behaviour."""
    old = dyng.get_default_resources()
    dyng.set_default_resources("sequential")
    yield
    dyng.set_default_resources(old)
