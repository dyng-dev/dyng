# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.triad_count on the CUDA backend, through a CUDA plugin wheel (marker gpu): compute and
update equal the sequential backend bit for bit (counts, deltas, the two phases, the frontiers)
for every pattern, through triad_count.update and through dyng.update with several results; the
make-test counts of ESCHER-GPU@abcf9b2 on the device; the CLI with --backend cuda; the engine
option.
"""

from __future__ import annotations

import os
from pathlib import Path

import dyng
import numpy as np
import pytest
from triad_fixture import EXES, KINDS, read_fixture, regenerate

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


def _replay(res: dyng.Resources, kinds: list[str], together: bool) -> list[object]:
    """Five batches of the hypergraph generators' mixes; together: one dyng.update() of every
    result per batch, else triad_count.update() of the one result."""
    h = dyng.generators.uniform_hypergraph(
        num_vertices=300,
        num_hyperedges=500,
        max_cardinality=8,
        seed=5,
        line_graph="sorted_merge",
        resources=res,
    )
    results = [dyng.triad_count.compute(h, kind=k) for k in kinds]
    out: list[object] = [[r.counts.tolist() for r in results]]
    for i, mix in enumerate(("mixed", "reuse", "skewed", "duplicates", "delete_all")):
        b = dyng.generators.random_hyperedge_batch(h, mix=mix, size=60, seed=i)  # type: ignore[arg-type]
        if together:
            stats = list(dyng.update(h, b, *results))
        else:
            stats = [dyng.triad_count.update(h, b, results[0])]
        for r, s in zip(results, stats, strict=True):
            out.append(
                (
                    r.counts.tolist(),
                    s.delta.tolist(),
                    s.triads_removed,
                    s.triads_added,
                    s.deleted_frontier,
                    s.inserted_frontier,
                    s.batch.inserted_ids.tolist(),
                )
            )
        h.check_integrity()
    return out


@pytest.mark.parametrize("kind", KINDS)
def test_cuda_equals_sequential(kind: str) -> None:
    seq = _replay(dyng.Resources.sequential(), [kind], together=False)
    cuda = _replay(dyng.Resources.cuda(0), [kind], together=False)
    assert seq == cuda


def test_dyng_update_with_every_pattern_on_cuda() -> None:
    seq = _replay(dyng.Resources.sequential(), KINDS, together=True)
    cuda = _replay(dyng.Resources.cuda(0), KINDS, together=True)
    assert seq == cuda


def test_results_on_cuda() -> None:
    res = dyng.Resources.cuda(0)
    h = dyng.Hypergraph.from_lists([[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 2, 4]], resources=res)
    r = dyng.triad_count.compute(h)
    assert r.space == "host" and r.counts.dtype == np.uint64 and r.total == 4
    assert r.counts.tolist() == dyng.testing.triad_reference(h).tolist()
    twin = r.clone()
    st = dyng.triad_count.update(h, dyng.HyperedgeBatch(delete_hyperedges=[3]), r)
    assert st.engine_used == "fused" and st.triads_removed == 3 and r.total == 1
    assert twin.total == 4
    with pytest.raises(dyng.NotSupportedError, match="operators"):
        dyng.triad_count.compute(h, cuda_engine="operators")
    assert dyng.triad_count.compute(h, cuda_engine="fused").total == 1
    host = dyng.Hypergraph.from_lists([[0, 1, 2]])
    with pytest.raises(dyng.InvalidArgumentError, match="to_backend"):
        dyng.triad_count.compute(host, resources=res)


def test_make_test_counts_on_cuda(data: Path) -> None:
    if not dyng.generators.legacy.standard_library_matches_legacy_streams():
        pytest.skip("this standard library does not reproduce the legacy generator streams")
    groups, cases = read_fixture(data)
    res = dyng.Resources.cuda(0)
    for g in groups:
        h, gen = regenerate(g, res)
        results = [dyng.triad_count.compute(h, kind=k) for k in EXES.values()]
        for e, r in zip(EXES, results, strict=True):
            assert r.counts.tolist() == cases[(g["name"], e)]["full"], (g["name"], e)
        for i in range(g["batches"]):
            b = gen.next(h)
            stats = dyng.update(h, b, *results)
            for e, s in zip(EXES, stats, strict=True):
                want = cases[(g["name"], e)]["batches"][i]
                assert (b.num_deletions, b.num_insertions, s.delta.tolist()) == want


def test_cli_on_cuda(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    from dyng.cli import main

    (tmp_path / "s.hg").write_text("1 2 3\n2 3 4\n3 4 5\n1 3 5\n")
    (tmp_path / "s.dgt").write_text("%dgt 1 hypergraph\n%batch 0\n-h 3\n+h 1 2 4\n")
    outs = []
    for backend in ("sequential", "cuda"):
        argv = ["triad_count", "update", "--input", str(tmp_path / "s.hg"), "--batches"]
        argv += [str(tmp_path / "s.dgt"), "--format", "escher", "--backend", backend]
        assert main(argv) == 0
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1] and outs[0].startswith("Hypergraph: 4 hyperedges")
