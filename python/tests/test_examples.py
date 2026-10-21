# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The Python examples (examples/python, PLAN Section 9.6) run and give the originals' outputs.

The documentation quotes these programs, so they are tested like the C++ examples: the first
update prints what its page says, sssp_update and mosp_update write MOSP's files byte for byte,
and cycle_count_update prints the histogram of the original `cycle-enum --task update`. Skipped
where the repository files are not present (the sdist leaves the examples out).
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from math import comb
from pathlib import Path

import dyng
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


_CUDA = pytest.param(
    "cuda",
    marks=[
        pytest.mark.gpu,
        pytest.mark.skipif(
            not dyng.config()["backends"]["cuda"],
            reason="needs a CUDA plugin module and a visible CUDA device",
        ),
    ],
)


@pytest.mark.parametrize("backend", [*BACKENDS, _CUDA])
def test_triad_count_prints_the_expected_counts(backend: str) -> None:
    # examples/data/triad_count_update.expected.txt: the counts of a brute force over every
    # triple of hyperedges (test_the_triad_count_expected_output_is_the_brute_force);
    # examples/cpp/triad_count_update.cpp prints the same bytes (CTest
    # example.triad_count_update.*, and ci/cmake_consumer.sh against the installed package).
    data = REPO / "examples" / "data"
    out = _run(
        EXAMPLES / "triad_count.py", data / "triad_count.hg", data / "triad_count.dgt", backend
    ).stdout
    assert out == (data / "triad_count_update.expected.txt").read_text()


# The canonical Venn id of every h-motif class 1..30 (bin k - 1): the table of
# docs/algorithms/triad_count.md section 1 (bit 0, 1, 2 "A only", "B only", "C only"; bit 3, 4, 5
# "A and B only", "B and C only", "C and A only"; bit 6 "A, B and C").
_CANONICAL = [64, 65, 67, 71, 72, 73, 76, 75, 77, 79, 88, 90, 89, 91, 93, 95, 120, 121, 123, 127,
              24, 26, 25, 27, 29, 31, 56, 57, 59, 63]  # fmt: skip


def _triads(live: dict[int, frozenset[int]], through: set[int] | None = None) -> list[int]:
    """The 30 bins of the closed triads of `live` (id -> members), by definition; with `through`,
    only the triads with a hyperedge among those ids."""
    bins = [0] * 30
    for ids in itertools.combinations(sorted(live), 3):
        if through is not None and not through.intersection(ids):
            continue
        a, b, c = (live[i] for i in ids)
        if not (a & b and b & c and c & a):
            continue
        venn = min(
            sum(1 << k for k, r in enumerate(regions) if r)
            for x, y, z in itertools.permutations((a, b, c))
            for regions in [
                [x - y - z, y - x - z, z - x - y, x & y - z, y & z - x, z & x - y, x & y & z]
            ]
        )
        bins[_CANONICAL.index(venn)] += 1
    return bins


def _nonzero(label: str, values: list[int]) -> str:
    return label + "".join(f" {k}:{v}" for k, v in enumerate(values) if v)


def test_the_triad_count_expected_output_is_the_brute_force() -> None:
    # The expected output the examples are compared with, rebuilt by a brute force over every
    # triple of hyperedges after every batch (no dynG count is used). The inserted ids are the
    # store's choice (best-fit reuse): they are taken from the expected file itself.
    data = REPO / "examples" / "data"
    expected = (data / "triad_count_update.expected.txt").read_text().splitlines()
    rows: list[frozenset[int]] = []
    for line in (data / "triad_count.hg").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            row = frozenset(int(v) - 1 for v in line.split())  # file ids 1..n
            if row not in rows:
                rows.append(row)  # dedupe_hyperedges: the first of every member set
    live = dict(enumerate(rows))
    batches: list[tuple[list[int], list[frozenset[int]]]] = []
    for line in (data / "triad_count.dgt").read_text().splitlines():
        if line.startswith("%batch"):
            batches.append(([], []))
        elif line.startswith("-h "):
            batches[-1][0].append(int(line.split()[1]))
        elif line.startswith("+h "):
            batches[-1][1].append(frozenset(int(v) for v in line.split()[1:]))
    bins = _triads(live)
    type1 = sum(comb(len(e), 3) for e in live.values())
    got = [f"{len(live)} hyperedges: {sum(bins)} triads, type1 {type1}", _nonzero("  bins", bins)]
    for i, (deleted, inserted) in enumerate(batches):
        ids = [int(x) for x in expected[len(got)].split("inserted ids")[1].split()]
        deleted = sorted(set(d for d in deleted if d in live))
        removed = sum(_triads(live, set(deleted)))
        removed1 = sum(comb(len(live[d]), 3) for d in deleted)
        for d in deleted:
            del live[d]
        assert len(ids) == len(inserted) and not set(ids) & set(live)
        live.update(zip(ids, inserted, strict=True))
        after = _triads(live)
        added = sum(_triads(live, set(ids)))
        added1 = sum(comb(len(e), 3) for e in inserted)
        type1 += added1 - removed1
        assert sum(after) == sum(bins) - removed + added
        got.append(
            f"batch {i}: {len(deleted)} deleted, inserted ids" + "".join(f" {x}" for x in ids)
        )
        got.append(
            f"  h_motif -{removed} +{added} = {sum(after)}, type1 -{removed1} +{added1} = {type1}"
        )
        got.append(_nonzero("  delta", [x - y for x, y in zip(after, bins, strict=True)]))
        bins = after
    assert got == expected


def test_the_triad_count_examples_fit_plan_9_6() -> None:
    # PLAN 9.6: one runnable example per algorithm, at most 60 lines.
    for path in (REPO / "examples" / "cpp" / "triad_count_update.cpp", EXAMPLES / "triad_count.py"):
        assert len(path.read_text().splitlines()) <= 60, path
