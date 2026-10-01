# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The Python examples (examples/python, PLAN Section 9.6) run and give the originals' outputs.

The documentation quotes these programs, so they are tested like the C++ examples: the first
update prints what its page says, sssp_update and mosp_update write MOSP's files byte for byte,
and cycle_count_update prints the histogram of the original `cycle-enum --task update`. Skipped
where the repository files are not present (the sdist leaves the examples out).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples" / "python"
DATA = REPO / "cpp" / "tests" / "data"

if not EXAMPLES.is_dir():
    pytest.skip("examples/python is not present", allow_module_level=True)

BACKENDS = ["sequential", "openmp"]


def _run(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *map(str, args)], capture_output=True, text=True, check=True, timeout=300
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_first_update(backend: str) -> None:
    out = _run(EXAMPLES / "first_update.py", backend).stdout
    assert out == "invalidated 2, affected 2\ndistances 0 4 1 2\nparents -1 0 0 2\n"


@pytest.mark.parametrize("backend", BACKENDS)
def test_sssp_update_writes_mosps_files(backend: str, tmp_path: Path) -> None:
    case = DATA / "mosp_graph_io" / "testCase0"
    expected = DATA / "mosp_sssp" / "testCase0" / "updated" / "obj0"
    out = _run(
        EXAMPLES / "sssp_update.py",
        case / "graphCsr",
        case / "insert.txt",
        case / "delete.txt",
        tmp_path,
        backend,
    ).stdout
    assert out.startswith("+0 -1 edges, invalidated 0, affected 2, engine ")
    assert "sssp.update" in out
    for name in ("SSSPTreeUpdated.txt", "distancesUpdated.txt"):
        assert (tmp_path / name).read_bytes() == (expected / name).read_bytes(), name


@pytest.mark.parametrize("backend", BACKENDS)
def test_cycle_count_update_equals_the_original(backend: str) -> None:
    data = DATA / "cycle_enum"
    out = _run(
        EXAMPLES / "cycle_count_update.py",
        data / "parser" / "tudataset_A.txt",
        4,
        40,
        40,
        1,
        backend,
    ).stdout
    assert out == (data / "cli" / "c06.out").read_text()


@pytest.mark.parametrize("backend", BACKENDS)
def test_mosp_update_writes_the_originals_combined_files(backend: str, tmp_path: Path) -> None:
    case = DATA / "mosp_combined" / "thesis" / "input"
    expected = DATA / "mosp_combined" / "thesis_414" / "combined"
    out = _run(
        EXAMPLES / "mosp_update.py",
        case / "graphCsr",
        case / "insert.txt",
        case / "delete.txt",
        tmp_path,
        "4,1,4",
        backend,
    ).stdout
    assert out.startswith("K=3 L=4: invalidated [4, 4, 0], combined edges 9, affected 4\n")
    assert "vertex 6: path costs [15, 3, 20]" in out  # the thesis' worked example
    for name in ("distancesCsr.txt", "SSSPTreeCsr.txt", "mospCosts.txt"):
        got = (tmp_path / "combinedGraph" / name).read_bytes()
        assert got == (expected / name).read_bytes(), name
