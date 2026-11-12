# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.hyper_sssp on the CUDA backend, through a CUDA plugin wheel (marker gpu): compute and
update on both CUDA engines (MOSP_ESCHER's kernels, engine "fused", and the shared sssp engine,
"operators") equal the sequential backend bit for bit, distances, parents and the deterministic
counters, under hyperedge batches and every incidence-edit mix, through hyper_sssp.update and
through dyng.update with two results; the device arrays; clones between backends; the CLI with
--backend cuda.
"""

from __future__ import annotations

import os
from pathlib import Path

import dyng
import numpy as np
import pytest

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

SEQ = dyng.Resources("sequential")
MIXES = (
    "mixed",
    "incidence_removals",
    "incidence_additions",
    "incidence_mixed",
    "incidence_emptying",
    "incidence_same_batch",
    "incidence_repeats",
    "incidence_out_of_range",
    "reuse",
    "delete_all",
)


def _replay(res: dyng.Resources, engine: str, together: bool, reuse: str) -> list[object]:
    """Every mix in turn on a uniform weighted slack_csr hypergraph; together: one dyng.update()
    of two results (sources 0 and 1) per batch, else hyper_sssp.update() of one."""
    props = dyng.HypergraphProperties(storage="slack_csr", line_graph="slack_csr", reuse=reuse)  # type: ignore[arg-type]
    h = dyng.generators.uniform_hypergraph(
        num_vertices=200,
        num_hyperedges=400,
        max_cardinality=6,
        weighted=True,
        max_weight=40,
        seed=9,
        properties=props,
        resources=res,
    )
    results = [
        dyng.hyper_sssp.compute(h, s, cuda_engine=engine) for s in ((0, 1) if together else (0,))
    ]  # type: ignore[arg-type]
    out: list[object] = []
    for i, mix in enumerate(MIXES):
        b = dyng.generators.random_hyperedge_batch(h, mix=mix, size=30, seed=i, resources=res)  # type: ignore[arg-type]
        stats = (
            list(dyng.update(h, b, *results))
            if together
            else [dyng.hyper_sssp.update(h, b, results[0])]
        )
        for r, s in zip(results, stats, strict=True):
            out.append(
                (
                    r.hyperedge_distances.to_numpy().tolist(),
                    r.hyperedge_parents.to_numpy().tolist(),
                    r.vertex_distances().to_numpy().tolist(),
                    s.batch.inserted_ids.tolist(),
                    s.batch.inserted_incidences,
                    s.batch.deleted_incidences,
                    s.batch.skipped_ops,
                    s.line_graph_insertions,
                    s.line_graph_deletions,
                    s.invalidated,
                )
            )
        h.check_integrity()
    return out


@pytest.mark.parametrize("engine", ["fused", "operators"])
@pytest.mark.parametrize("reuse", ["erase_first", "insert_first"])
def test_cuda_equals_sequential(engine: str, reuse: str) -> None:
    seq = _replay(SEQ, engine, together=False, reuse=reuse)
    cuda = _replay(dyng.Resources.cuda(0), engine, together=False, reuse=reuse)
    assert seq == cuda


@pytest.mark.parametrize("engine", ["fused", "operators"])
def test_dyng_update_with_two_sources_on_cuda(engine: str) -> None:
    seq = _replay(SEQ, engine, together=True, reuse="insert_first")
    cuda = _replay(dyng.Resources.cuda(0), engine, together=True, reuse="insert_first")
    assert seq == cuda


def test_results_on_cuda() -> None:
    res = dyng.Resources.cuda(0)
    lists = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5]]
    h = dyng.Hypergraph.from_lists(
        lists, [2, 3, 1, 9, 4], storage="slack_csr", line_graph="slack_csr", resources=res
    )
    sp = dyng.hyper_sssp.compute(h, 0)
    assert sp.space == "device" and sp.hyperedge_distances.device == "cuda:0"
    assert sp.hyperedge_distances.to_numpy().tolist()[:4] == [2, 5, 6, 9]
    assert sp.distance_to(2) == 5 and sp.vertex_distances().to_numpy().tolist()[:4] == [0, 2, 5, 6]
    d, p = dyng.testing.hyper_sssp_reference(h, 0)
    assert sp.hyperedge_parents.to_numpy().tolist() == p.tolist()
    st = dyng.hyper_sssp.update(h, dyng.HyperedgeBatch(insert_incidences=([0], [3])), sp)
    assert st.engine_used == "fused" and st.line_graph_insertions == 1
    assert dyng.testing.check_hyper_sssp_tree(h, sp) == (0, 0)
    ops = dyng.hyper_sssp.compute(h, 0, cuda_engine="operators")
    assert ops.hyperedge_distances.to_numpy().tolist() == sp.hyperedge_distances.to_numpy().tolist()
    # a result moves between backends with its hypergraph
    host_h = h.to_backend(SEQ)
    host_sp = sp.clone(SEQ)
    assert host_sp.space == "host"
    batch = dyng.HyperedgeBatch(delete_hyperedges=[1], delete_incidences=([2], [2]))
    dyng.hyper_sssp.update(host_h, batch, host_sp)
    dyng.hyper_sssp.update(h, batch, sp)
    assert host_sp.hyperedge_distances.tolist() == sp.hyperedge_distances.to_numpy().tolist()
    assert host_sp.distance_to(3) == sp.distance_to(3)
    host = dyng.Hypergraph.from_lists(lists, [1, 1, 1, 1, 1], storage="slack_csr", resources=SEQ)
    with pytest.raises(dyng.InvalidArgumentError, match="to_backend"):
        dyng.hyper_sssp.compute(host, 0, resources=res)


def test_the_cli_on_cuda(tmp_path: Path) -> None:
    from dyng.cli import main

    hg = tmp_path / "h.hg"
    hg.write_text("1 2 ; 2\n2 3 ; 3\n3 4 ; 1\n1 4 ; 9\n5 6 ; 4\n", encoding="utf-8")
    batches = tmp_path / "b.dgt"
    batches.write_text("%batch 0\n+i 0 3\n-h 4\n", encoding="utf-8")
    outs = {}
    for backend in ("sequential", "cuda"):
        out = tmp_path / backend
        status = main(
            [
                "hyper_sssp", "update", "--input", str(hg), "--batches", str(batches),
                "--backend", backend, "--cuda-engine", "operators", "--out", str(out),
                "--compare-recompute", "--quiet",
            ]
        )  # fmt: skip
        assert status == 0
        outs[backend] = [
            (out / f).read_bytes() for f in ("hyperedge_distances.txt", "hyperedge_parents.txt")
        ]
    assert outs["sequential"] == outs["cuda"]
    assert np.array_equal(
        dyng.io.read_distances(tmp_path / "cuda" / "hyperedge_distances.txt", 5)[:4], [2, 5, 3, 9]
    )
