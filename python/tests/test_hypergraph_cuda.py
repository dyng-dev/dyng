# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.Hypergraph on the CUDA backend, through a CUDA plugin wheel (marker gpu): the same ids,
rows and line graph as the sequential backend, the placement rule, copies between backends, and
the dump replays of test_hypergraph_parity.py on the device.
"""

from __future__ import annotations

import os

import dyng
import numpy as np
import pytest
import test_hypergraph_parity as parity

_AVAILABLE = bool(dyng.config()["backends"]["cuda"])
if os.environ.get("DYNG_REQUIRE_CUDA", "") == "1" and not _AVAILABLE:
    raise RuntimeError(
        "DYNG_REQUIRE_CUDA=1, but the active module has no usable CUDA backend: "
        f"{dyng.config()['native_module']} ({dyng.config()['selection']})"
    )

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _AVAILABLE, reason="needs a CUDA plugin module and a visible CUDA device"
    ),
]


def _rows(h: dyng.Hypergraph) -> list[list[int]]:
    v = h.h2v()
    return [v.row(i).tolist() for i in range(v.num_rows)]


@pytest.mark.parametrize("reuse", ["erase_first", "insert_first"])
def test_cuda_equals_sequential(reuse: str) -> None:
    states = []
    for res in (dyng.Resources.sequential(), dyng.Resources.cuda(0)):
        h = dyng.generators.uniform_hypergraph(
            num_vertices=300,
            num_hyperedges=400,
            max_cardinality=8,
            seed=11,
            line_graph="sorted_merge",
            id_reuse=reuse,  # type: ignore[arg-type]
            resources=res,
        )
        ids = []
        for i, mix in enumerate(("mixed", "reuse", "skewed", "duplicates", "delete_all")):
            b = dyng.generators.random_hyperedge_batch(h, mix=mix, size=40, seed=i)  # type: ignore[arg-type]
            ids.append(h.apply(b).inserted_ids.tolist())
            h.check_integrity()
        lg = h.h2h()
        states.append(
            (ids, _rows(h), lg.col_ind.tolist(), lg.out.tolist(), h.v2h().col_ind.tolist())
        )
        assert h.space == ("device" if res.backend == "cuda" else "host")
    assert states[0] == states[1]


def test_placement_and_copies() -> None:
    cuda, seq = dyng.Resources.cuda(0), dyng.Resources.sequential()
    h = dyng.Hypergraph.from_lists([[0, 1, 2], [2, 3], [4]], resources=cuda)
    assert h.store_stats().device_bytes > 0
    with pytest.raises(dyng.InvalidArgumentError):
        h.apply(dyng.HyperedgeBatch(insert_hyperedges=[[0]]), resources=seq)  # nothing is copied
    host = h.to_backend(seq)
    back = host.to_backend(cuda)
    batch = dyng.HyperedgeBatch([1], [[3, 4], [0]])
    want = h.apply(batch).inserted_ids.tolist()
    assert host.apply(batch).inserted_ids.tolist() == want
    assert back.apply(batch).inserted_ids.tolist() == want
    assert _rows(back) == _rows(h) == _rows(host)


def test_repeated_cuda_runs_give_identical_ids() -> None:
    runs = []
    for _ in range(3):
        h = dyng.generators.uniform_hypergraph(
            num_vertices=200, num_hyperedges=300, seed=4, resources=dyng.Resources.cuda(0)
        )
        b = dyng.generators.random_hyperedge_batch(h, mix="heavy_delete", size=60, seed=5)
        runs.append(h.apply(b).inserted_ids.tolist())
    assert runs[0] == runs[1] == runs[2]


def test_the_dumps_on_cuda() -> None:
    parity.test_escher_gpu_dump_gives_its_ids("cuda")
    parity.test_mosp_escher_dump_gives_its_ids("cuda")


def test_dgb_batches_on_cuda(tmp_path: object) -> None:
    import pathlib

    path = pathlib.Path(str(tmp_path)) / "b.dgb"
    h = dyng.Hypergraph.from_lists([[0, 1], [1, 2]], resources=dyng.Resources.cuda(0))
    dyng.io.write_binary_batches(path, [dyng.HyperedgeBatch([0], [[2, 0]])])
    for r in dyng.io.BatchReader(path):
        s = h.apply(r.batch)
        assert np.array_equal(s.inserted_ids, [0])
