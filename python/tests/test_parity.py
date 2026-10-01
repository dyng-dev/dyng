# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Python-level parity with the pinned originals on a subset of the committed goldens.

The same fixtures as the C++ parity tests (cpp/tests/algorithms/*/..._fixture_test.cpp), produced
by the pinned originals (parity/fixtures/): the files written through the Python package must be
byte-identical to MOSP-OpenMP@c352151's, and the histograms bit-identical to
CycleEnumeration-GPU@0a976ad's, on both host backends and every graph type.
"""

from __future__ import annotations

from pathlib import Path

import dyng
import pytest

SSSP_CASES = ["testCase0", "testCase3", "testCase9", "r04", "r10", "h0", "c2i_0", "escher_ties"]
VERTEX_TYPES = [("int32", "int32"), ("int32", "int64"), ("int64", "int64")]


def _sssp_case(data: Path, name: str) -> tuple[Path, int]:
    for line in (data / "mosp_sssp" / "cases.txt").read_text().splitlines():
        case, rel, k = line.split()
        if case == name:
            return data / rel, int(k)
    raise KeyError(name)


def _invalidated(data: Path, name: str, k: int) -> int:
    for line in (data / "mosp_sssp" / name / "stats.txt").read_text().splitlines():
        obj, _, value = line.split()
        if obj == f"obj{k}":
            return int(value)
    raise KeyError((name, k))


@pytest.mark.parametrize("types", VERTEX_TYPES, ids=["-".join(t) for t in VERTEX_TYPES])
@pytest.mark.parametrize("name", SSSP_CASES)
def test_sssp_bytes_equal_mosp(
    data: Path, tmp_path: Path, res: dyng.Resources, name: str, types: tuple[str, str]
) -> None:
    vertex, edge = types
    inp, num_weights = _sssp_case(data, name)
    expected = data / "mosp_sssp" / name
    batch = None
    for k in range(num_weights):
        g = dyng.io.read_csr_triplet(
            inp / "graphCsr",
            num_weights=num_weights,
            vertex_dtype=vertex,
            edge_dtype=edge,
            properties="mosp_compatible",
            resources=res,
        )
        if batch is None:
            batch = dyng.io.read_legacy_batch(
                inp / "insert.txt",
                inp / "delete.txt",
                num_weights=num_weights,
                num_vertices=g.num_vertices,
                vertex_dtype=vertex,
            )
        obj = f"obj{k}"
        init_d = expected / "init" / obj / "distancesOriginal.txt"
        init_t = expected / "init" / obj / "SSSPTreeOriginal.txt"
        # compute() == mospPrep init (Dijkstra)
        tree = dyng.sssp.compute(g, 0, objective=k)
        dyng.io.write_distances(tmp_path / "cd.txt", tree.distances)
        dyng.io.write_parents(tmp_path / "ct.txt", tree.parents)
        assert (tmp_path / "cd.txt").read_bytes() == init_d.read_bytes()
        assert (tmp_path / "ct.txt").read_bytes() == init_t.read_bytes()
        # update() from the original's files == the `mosp` driver
        dist = dyng.io.read_distances(init_d, g.num_vertices)
        parents = dyng.io.read_parents(init_t, g.num_vertices, vertex_dtype=vertex)
        r = dyng.sssp.Result.from_arrays(g, 0, dist, parents, canonicalize=False, objective=k)
        st = dyng.sssp.update(g, batch, r)
        dyng.io.write_distances(tmp_path / "ud.txt", r.distances)
        dyng.io.write_parents(tmp_path / "ut.txt", r.parents)
        upd = expected / "updated" / obj
        assert (tmp_path / "ud.txt").read_bytes() == (upd / "distancesUpdated.txt").read_bytes()
        assert (tmp_path / "ut.txt").read_bytes() == (upd / "SSSPTreeUpdated.txt").read_bytes()
        assert st.invalidated == _invalidated(data, name, k)


# -------------------------------------------------------------------------------------------------
# cycle_count
# -------------------------------------------------------------------------------------------------


def _hist_text(counts: list[int]) -> str:
    return " ".join(f"{i}:{c}" for i, c in enumerate(counts) if i >= 2 and c)


def _read_case(path: Path) -> tuple[int, list[tuple[int, int]], dyng.EdgeBatch]:
    n = 0
    edges: list[tuple[int, int]] = []
    dels: list[tuple[int, int]] = []
    ins: list[tuple[int, int]] = []
    for line in path.read_text().splitlines():
        tag, *rest = line.split()
        if tag == "n":
            n = int(rest[0])
        elif tag == "e":
            edges.append((int(rest[0]), int(rest[1])))
        elif tag == "-":
            dels.append((int(rest[0]), int(rest[1])))
        elif tag == "+":
            ins.append((int(rest[0]), int(rest[1])))
    batch = dyng.EdgeBatch(
        insert=([u for u, _ in ins], [v for _, v in ins]),
        delete=([u for u, _ in dels], [v for _, v in dels]),
    )
    return n, edges, batch


def _expectations(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    section = ""
    for line in path.read_text().splitlines():
        name, _, rest = line.partition(" ")
        if name == "k":
            section = rest
            continue
        out[f"{section}/{name}"] = rest
    return out


@pytest.mark.parametrize("edge", ["int32", "int64"])
@pytest.mark.parametrize("case", [f"case_{i:03d}" for i in range(0, 80, 4)])
def test_cycle_count_random_cases_equal_the_original(
    data: Path, res: dyng.Resources, case: str, edge: str
) -> None:
    cases = data / "cycle_enum" / "cases"
    n, edges, batch = _read_case(cases / f"{case}.txt")
    expected = _expectations(cases / f"{case}.counts")
    prefix = "omp_" if res.backend == "openmp" else "seq_"
    for k in (2, 3, 4, 5, 6, 7, -1):
        g = dyng.Graph.from_edges(
            [u for u, _ in edges],
            [v for _, v in edges],
            num_vertices=n,
            properties="cycle_enum_compatible",
            vertex_dtype="int32",
            edge_dtype=edge,
            resources=res,
        )
        h = dyng.cycle_count.compute(g, max_length=k)
        assert _hist_text(h.counts.tolist()) == expected[f"{k}/{prefix}before"], k
        dyng.cycle_count.update(g, batch, h)
        key = f"{k}/{prefix}" + ("update" if k > 0 else "after")
        assert _hist_text(h.counts.tolist()) == expected[key], k
        assert (
            _hist_text(dyng.cycle_count.compute(g, max_length=k).counts.tolist())
            == expected[f"{k}/{prefix}after"]
        )


def _line(name: str, counts: list[int]) -> str:
    text = _hist_text(counts)
    return f"{name} {text}" if text else name


def test_cycle_count_fixture_graphs_equal_the_original(data: Path, res: dyng.Resources) -> None:
    root = data / "cycle_enum"
    jobs = (root / "counts" / "cases.txt").read_text().splitlines()
    assert len(jobs) == 23
    mine = "omp" if res.backend == "openmp" else "seq"
    for i, job in enumerate(jobs):
        what, file, k, *rest = job.split()
        expected = (root / "counts" / f"c{i:02d}.expected").read_text().splitlines()
        g = dyng.io.read_edge_list(root / file, properties="cycle_enum_compatible", resources=res)
        h = dyng.cycle_count.compute(g, max_length=int(k))
        if what == "count":
            assert _line(mine, h.counts.tolist()) == expected[1 if mine == "omp" else 0], job
            continue
        dels, ins, seed = (int(x) for x in rest[:3])
        window = int(rest[3]) if len(rest) > 3 else None
        assert _line("prior", h.counts.tolist()) == expected[1], job
        b = dyng.generators.legacy.cycle_enum_batch(
            g, num_deletions=dels, num_insertions=ins, seed=seed, locality_window=window
        )
        st = dyng.cycle_count.update(g, b, h)
        assert f"deletions {st.deletions} insertions {st.insertions}" == expected[0], job
        assert _line(f"{mine}_update", h.counts.tolist()) == expected[2 if mine == "seq" else 3], (
            job
        )
