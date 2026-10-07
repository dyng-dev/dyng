# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The CUDA backend from Python, through a CUDA plugin wheel (marker gpu; ADR 0030 and 0031).

These tests run when the active native module has CUDA and a device is visible: with a plugin
wheel installed (``pip install "dyng[cu13]"``; ci/plugin_wheels.sh and the plugin step of
ci/gpu_local.sh run them on GPU 1). Elsewhere they skip.

- sssp, cycle_count and mosp on CUDA equal the sequential backend element by element (compute,
  update, ``dyng.update``), on random graphs and both CUDA engines;
- a subset of the committed goldens of the pinned originals (cpp/tests/data, the same fixtures as
  the C++ parity tests): MOSP's files byte for byte, CycleEnumeration-GPU's CUDA histograms;
- ``dyng.Array`` in device memory: metadata, host copies, ``__cuda_array_interface__``,
  ``__dlpack__`` with stream ordering (PLAN Section 5.4 rule 4), staleness;
- inputs in device memory, and round trips through PyTorch and CuPy when they are installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dyng
import numpy as np
import pytest

_AVAILABLE = bool(dyng.config()["backends"]["cuda"])
if os.environ.get("DYNG_REQUIRE_CUDA", "") == "1" and not _AVAILABLE:
    # ci/plugin_wheels.sh and ci/gpu_local.sh: a run that should test the plugin must not pass by
    # skipping every test.
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

ENGINES = ["fused", "operators"]


@pytest.fixture(scope="module")
def cuda() -> dyng.Resources:
    """CUDA resources of device 0 (CUDA_VISIBLE_DEVICES selects the GPU)."""
    return dyng.Resources.cuda(0)


@pytest.fixture(scope="module")
def seq() -> dyng.Resources:
    """The sequential reference backend."""
    return dyng.Resources.sequential()


def random_edges(seed: int, n: int = 300, m: int = 2400, k: int = 1) -> tuple[Any, Any, Any]:
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, m, dtype=np.int32)
    dst = rng.integers(0, n, m, dtype=np.int32)
    keep = src != dst
    w = rng.integers(1, 50, (int(keep.sum()), k), dtype=np.int32)
    return src[keep], dst[keep], (w[:, 0] if k == 1 else w)


def random_batch(seed: int, n: int, k: int = 1, size: int = 120) -> dyng.EdgeBatch:
    rng = np.random.default_rng(seed + 1000)
    s = rng.integers(0, n, size, dtype=np.int32)
    d = rng.integers(0, n, size, dtype=np.int32)
    keep = s != d
    s, d = s[keep], d[keep]
    w = rng.integers(1, 50, (s.size, k), dtype=np.int32)
    half = s.size // 2
    return dyng.EdgeBatch(
        insert=(s[:half], d[:half], w[:half, 0] if k == 1 else w[:half]),
        delete=(s[half:], d[half:]),
    )


def host(a: dyng.Array) -> np.ndarray:
    return a.to_numpy()


# -------------------------------------------------------------------------------------------------
# The plugin is active and Resources.cuda() works
# -------------------------------------------------------------------------------------------------


def test_the_plugin_module_is_active(
    cuda: dyng.Resources, capsys: pytest.CaptureFixture[str]
) -> None:
    c = dyng.config()
    assert c["build"]["cuda"] and c["build"]["plugin"] in ("cu12", "cu13")
    assert c["native_module"].startswith(f"dyng_{c['build']['plugin']} ")
    assert [p["state"] for p in c["plugins"] if p["name"] == c["build"]["plugin"]] == ["chosen"]
    assert c["default_backend"] == "cuda"
    dyng.show_config()
    out = capsys.readouterr().out
    assert f"plugin {c['build']['plugin']}" in out and "chosen because    : plugin" in out
    assert cuda.backend == "cuda" and cuda.device == 0 and cuda.default_space == "device"
    cuda.warm_up()
    cuda.synchronize()


# -------------------------------------------------------------------------------------------------
# CUDA results equal the sequential backend's
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sssp_on_cuda_equals_sequential(
    cuda: dyng.Resources, seq: dyng.Resources, seed: int, engine: str
) -> None:
    src, dst, w = random_edges(seed)
    graphs = {
        r.backend: dyng.Graph.from_edges(src, dst, w, num_vertices=300, resources=r)
        for r in (seq, cuda)
    }
    trees = {b: dyng.sssp.compute(g, 0, cuda_engine=engine) for b, g in graphs.items()}
    assert trees["cuda"].distances.device == "cuda:0"
    assert np.array_equal(host(trees["cuda"].distances), host(trees["sequential"].distances))
    assert np.array_equal(host(trees["cuda"].parents), host(trees["sequential"].parents))
    for step in range(3):
        b = random_batch(seed * 10 + step, 300)
        st = {k: dyng.sssp.update(graphs[k], b, trees[k]) for k in graphs}
        assert st["cuda"].invalidated == st["sequential"].invalidated
        assert st["cuda"].engine_used == engine
        assert np.array_equal(host(trees["cuda"].distances), host(trees["sequential"].distances))
        assert np.array_equal(host(trees["cuda"].parents), host(trees["sequential"].parents))
    assert dyng.testing.check_sssp_tree(graphs["cuda"], trees["cuda"])


@pytest.mark.parametrize(("n", "m", "max_length"), [(40, 160, 3), (40, 160, 5), (12, 30, -1)])
def test_cycle_count_on_cuda_equals_sequential(
    cuda: dyng.Resources, seq: dyng.Resources, n: int, m: int, max_length: int
) -> None:
    src, dst, _ = random_edges(7, n=n, m=m)
    graphs = {
        r.backend: dyng.Graph.from_edges(
            src, dst, num_vertices=n, properties="cycle_enum_compatible", resources=r
        )
        for r in (seq, cuda)
    }
    hists = {b: dyng.cycle_count.compute(g, max_length=max_length) for b, g in graphs.items()}
    assert np.array_equal(host(hists["cuda"].counts), host(hists["sequential"].counts))
    assert hists["cuda"].total == hists["sequential"].total > 0
    b = dyng.EdgeBatch(insert=([0, 5, 9], [5, 9, 0]), delete=(src[:2], dst[:2]))
    for k in graphs:
        dyng.cycle_count.update(graphs[k], b, hists[k])
    assert np.array_equal(host(hists["cuda"].counts), host(hists["sequential"].counts))


@pytest.mark.parametrize("engine", ENGINES)
def test_mosp_on_cuda_equals_sequential(
    cuda: dyng.Resources, seq: dyng.Resources, engine: str
) -> None:
    src, dst, w = random_edges(11, k=3)
    graphs = {
        r.backend: dyng.Graph.from_edges(
            src, dst, w, num_vertices=300, properties="mosp_compatible", resources=r
        )
        for r in (seq, cuda)
    }
    paths = {
        b: dyng.mosp.compute(g, 0, preferences=[4, 1, 4], cuda_engine=engine)
        for b, g in graphs.items()
    }

    def same() -> None:
        a, e = paths["cuda"], paths["sequential"]
        for k in range(3):
            assert a.distances(k).device == "cuda:0"
            assert np.array_equal(host(a.distances(k)), host(e.distances(k)))
            assert np.array_equal(host(a.parents(k)), host(e.parents(k)))
        assert np.array_equal(host(a.combined_distances), host(e.combined_distances))
        assert np.array_equal(host(a.combined_parents), host(e.combined_parents))
        assert np.array_equal(host(a.path_costs), host(e.path_costs))

    same()
    for step in range(2):
        b = random_batch(step, 300, k=3)
        st = {k: dyng.mosp.update(graphs[k], b, paths[k]) for k in graphs}
        assert st["cuda"].affected == st["sequential"].affected
        same()


def test_one_batch_updates_several_results_on_cuda(
    cuda: dyng.Resources, seq: dyng.Resources
) -> None:
    src, dst, w = random_edges(5)
    out = {}
    for r in (seq, cuda):
        g = dyng.Graph.from_edges(src, dst, w, num_vertices=300, resources=r)
        tree, hist = dyng.sssp.compute(g, 0), dyng.cycle_count.compute(g, max_length=3)
        st_tree, st_hist = dyng.update(g, random_batch(5, 300), tree, hist)
        out[r.backend] = (host(tree.distances), host(hist.counts), st_tree.invalidated)
    assert all(np.array_equal(a, b) for a, b in zip(out["cuda"], out["sequential"], strict=True))


# -------------------------------------------------------------------------------------------------
# The goldens of the pinned originals (a subset of the C++ parity fixtures)
# -------------------------------------------------------------------------------------------------

SSSP_CASES = ["testCase0", "testCase9", "r04", "h0", "c2i_0", "escher_ties"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", SSSP_CASES)
def test_sssp_on_cuda_writes_mosps_bytes(
    data: Path, tmp_path: Path, cuda: dyng.Resources, name: str, engine: str
) -> None:
    from test_parity import _invalidated, _sssp_case

    inp, num_weights = _sssp_case(data, name)
    expected = data / "mosp_sssp" / name
    for k in range(num_weights):
        g = dyng.io.read_csr_triplet(
            inp / "graphCsr",
            num_weights=num_weights,
            properties="mosp_compatible",
            resources=cuda,
        )
        batch = dyng.io.read_legacy_batch(
            inp / "insert.txt",
            inp / "delete.txt",
            num_weights=num_weights,
            num_vertices=g.num_vertices,
        )
        obj = f"obj{k}"
        init_d = expected / "init" / obj / "distancesOriginal.txt"
        init_t = expected / "init" / obj / "SSSPTreeOriginal.txt"
        tree = dyng.sssp.compute(g, 0, objective=k, cuda_engine=engine)
        dyng.io.write_distances(tmp_path / "cd.txt", tree.distances)
        dyng.io.write_parents(tmp_path / "ct.txt", tree.parents)
        assert (tmp_path / "cd.txt").read_bytes() == init_d.read_bytes()
        assert (tmp_path / "ct.txt").read_bytes() == init_t.read_bytes()
        dist = dyng.io.read_distances(init_d, g.num_vertices)
        parents = dyng.io.read_parents(init_t, g.num_vertices)
        r = dyng.sssp.Result.from_arrays(
            g, 0, dist, parents, canonicalize=False, objective=k, cuda_engine=engine
        )
        st = dyng.sssp.update(g, batch, r)
        upd = expected / "updated" / obj
        dyng.io.write_distances(tmp_path / "ud.txt", r.distances)
        dyng.io.write_parents(tmp_path / "ut.txt", r.parents)
        assert (tmp_path / "ud.txt").read_bytes() == (upd / "distancesUpdated.txt").read_bytes()
        assert (tmp_path / "ut.txt").read_bytes() == (upd / "SSSPTreeUpdated.txt").read_bytes()
        assert st.invalidated == _invalidated(data, name, k)


def test_cycle_count_on_cuda_equals_the_originals_cuda_histograms(
    data: Path, cuda: dyng.Resources
) -> None:
    from test_parity import _line

    root = data / "cycle_enum"
    jobs = (root / "counts" / "cases.txt").read_text().splitlines()
    checked = 0
    for i, job in enumerate(jobs):
        what, file, k, *rest = job.split()
        expected = (root / "counts" / f"c{i:02d}.cuda").read_text().splitlines()
        g = dyng.io.read_edge_list(root / file, properties="cycle_enum_compatible", resources=cuda)
        h = dyng.cycle_count.compute(g, max_length=int(k))
        if what == "count":
            assert _line("cuda", h.counts.tolist()) == expected[0], job
            checked += 1
            continue
        dels, ins, seed = (int(x) for x in rest[:3])
        window = int(rest[3]) if len(rest) > 3 else None
        assert _line("prior", h.counts.tolist()) == expected[1], job
        b = dyng.generators.legacy.cycle_enum_batch(
            g, num_deletions=dels, num_insertions=ins, seed=seed, locality_window=window
        )
        st = dyng.cycle_count.update(g, b, h)
        assert f"deletions {st.deletions} insertions {st.insertions}" == expected[0], job
        assert _line("cuda_update", h.counts.tolist()) == expected[2], job
        checked += 1
    assert checked == len(jobs) == 23


@pytest.mark.parametrize("case", [f"case_{i:03d}" for i in range(0, 80, 8)])
def test_cycle_count_random_cases_on_cuda(data: Path, cuda: dyng.Resources, case: str) -> None:
    from test_parity import _expectations, _hist_text, _read_case

    cases = data / "cycle_enum" / "cases"
    n, edges, batch = _read_case(cases / f"{case}.txt")
    expected = _expectations(cases / f"{case}.cuda")
    for k in (2, 3, 4, 5, 6, 7, -1):
        g = dyng.Graph.from_edges(
            [u for u, _ in edges],
            [v for _, v in edges],
            num_vertices=n,
            properties="cycle_enum_compatible",
            vertex_dtype="int32",
            resources=cuda,
        )
        h = dyng.cycle_count.compute(g, max_length=k)
        assert _hist_text(h.counts.tolist()) == expected[f"{k}/cuda_before"], k
        dyng.cycle_count.update(g, batch, h)
        key = f"{k}/cuda_" + ("update" if k > 0 else "after")
        assert _hist_text(h.counts.tolist()) == expected[key], k


# -------------------------------------------------------------------------------------------------
# dyng.Array in device memory
# -------------------------------------------------------------------------------------------------


@pytest.fixture
def tree(cuda: dyng.Resources) -> Iterator[tuple[dyng.Graph, dyng.sssp.Result, np.ndarray]]:
    src, dst, w = random_edges(21)
    g = dyng.Graph.from_edges(src, dst, w, num_vertices=300, resources=cuda)
    t = dyng.sssp.compute(g, 0)
    ref = dyng.sssp.compute(
        dyng.Graph.from_edges(src, dst, w, num_vertices=300, resources=dyng.Resources.sequential()),
        0,
    )
    yield g, t, ref.distances.to_numpy()


def test_device_array_metadata_and_host_copies(tree: Any) -> None:
    _, t, ref = tree
    d = t.distances
    assert d.device == "cuda:0" and d.__dlpack_device__() == (2, 0)
    assert (d.shape, d.dtype, d.ndim, d.size, len(d)) == ((300,), np.dtype(np.int64), 1, 300, 300)
    assert t.parents.dtype == np.dtype(np.int32)
    out = d.to_numpy()
    assert out.flags.writeable and np.array_equal(out, ref)
    view = d.to_numpy(copy=None)
    assert not view.flags.writeable and np.array_equal(view, ref)
    assert d.to_numpy(copy=None) is view  # one host copy per Array
    with pytest.raises(ValueError, match="device memory"):
        d.to_numpy(copy=False)
    assert np.array_equal(np.asarray(d), ref) and d.tolist() == ref.tolist()
    assert d[5] == ref[5] and list(d)[:3] == ref[:3].tolist()
    assert bool(np.all(d == ref)) and "device='cuda:0'" in repr(d)
    with pytest.raises(ValueError, match="needs a copy"):
        np.asarray(d, copy=False)
    assert not hasattr(d, "__array_interface__")


def test_cuda_array_interface(cuda: dyng.Resources, tree: Any) -> None:
    _, t, _ = tree
    cai = t.distances.__cuda_array_interface__
    assert cai["version"] == 3 and cai["shape"] == (300,) and cai["typestr"] == "<i8"
    assert cai["strides"] is None and cai["data"][1] is True and cai["data"][0] != 0
    assert cai["stream"] == 2  # the per-thread default stream of Resources.cuda()
    host = dyng.Resources.sequential()
    assert not hasattr(
        dyng.sssp.compute(dyng.Graph.from_edges([0], [1], [1], resources=host), 0).distances,
        "__cuda_array_interface__",
    )


def test_dlpack_exports_order_the_consumers_stream(
    tree: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dyng._backend as backend

    _, t, ref = tree
    calls: list[int] = []
    real = backend.active_module().order_stream
    monkeypatch.setattr(
        backend.active_module(), "order_stream", lambda res, s: (calls.append(s), real(res, s))
    )
    d = t.distances
    for stream, expected in ((None, [1]), (1, [1]), (2, [2]), (-1, [])):
        calls.clear()
        capsule = d.__dlpack__(stream=stream, max_version=(1, 0))
        assert type(capsule).__name__ == "PyCapsule" and calls == expected, stream
    with pytest.raises(ValueError, match="ambiguous"):
        d.__dlpack__(stream=0, max_version=(1, 0))
    with pytest.raises(BufferError, match="DLPack >= 1.0"):
        d.__dlpack__()  # the legacy capsule cannot be read-only
    with pytest.raises(BufferError, match="copy=True"):
        d.__dlpack__(max_version=(1, 0), copy=True)
    # dl_device=(1, 0): a host copy (NumPy's from_dlpack(..., device="cpu"))
    assert np.array_equal(np.from_dlpack(d, device="cpu"), ref)


def test_device_arrays_go_stale(tree: Any) -> None:
    g, t, _ = tree
    d = t.distances
    dyng.sssp.update(g, random_batch(3, 300), t)
    with pytest.raises(dyng.StaleResultError):
        d.to_numpy()
    with pytest.raises(dyng.StaleResultError):
        d.__cuda_array_interface__  # noqa: B018
    with pytest.raises(dyng.StaleResultError):
        d.__dlpack__(max_version=(1, 0))
    assert t.distances.to_numpy().shape == (300,)


def test_device_results_feed_inputs(cuda: dyng.Resources, tree: Any) -> None:
    g, t, ref = tree
    # dyng.Array (device) as input: from_arrays copies it to the host once
    r = dyng.sssp.Result.from_arrays(g, 0, t.distances, t.parents)
    assert np.array_equal(r.distances.to_numpy(), ref)
    dyng.io.write_distances(Path(os.devnull), t.distances)


# -------------------------------------------------------------------------------------------------
# PyTorch and CuPy (when installed)
# -------------------------------------------------------------------------------------------------


def test_torch_round_trip(cuda: dyng.Resources, tree: Any) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch without CUDA")
    g, t, ref = tree
    x = torch.from_dlpack(t.distances)
    assert x.device == torch.device("cuda", 0) and x.dtype == torch.int64
    assert np.array_equal(x.cpu().numpy(), ref)
    y = t.parents.to_torch()
    assert y.is_cuda and np.array_equal(y.cpu().numpy(), t.parents.to_numpy())
    # on a side stream: PyTorch passes it to __dlpack__, which orders it after dynG's stream
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        z = torch.from_dlpack(t.distances) + 0
    side.synchronize()
    assert np.array_equal(z.cpu().numpy(), ref)
    # back into dynG: CUDA tensors as graph and batch inputs (one host copy each)
    src, dst, w = random_edges(21)
    tg = dyng.Graph.from_edges(
        torch.from_numpy(src).cuda(),
        torch.from_numpy(dst).cuda(),
        torch.from_numpy(w).cuda(),
        num_vertices=300,
        resources=cuda,
    )
    assert tg.vertex_dtype == np.dtype(np.int32)
    assert np.array_equal(dyng.sssp.compute(tg, 0).distances.to_numpy(), ref)
    b = random_batch(4, 300)
    w_ins = b.insert_weights
    assert w_ins is not None
    tb = dyng.EdgeBatch(
        insert=tuple(
            torch.from_numpy(np.ascontiguousarray(a)).cuda()
            for a in (b.insert_src, b.insert_dst, w_ins.reshape(-1))
        ),
        delete=tuple(
            torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in (b.delete_src, b.delete_dst)
        ),
    )
    st1 = dyng.sssp.update(g, b, t)
    tt = dyng.sssp.compute(tg, 0)
    st2 = dyng.sssp.update(tg, tb, tt)
    assert st1.invalidated == st2.invalidated
    assert np.array_equal(tt.distances.to_numpy(), t.distances.to_numpy())


def test_cupy_round_trip(cuda: dyng.Resources, tree: Any) -> None:
    cupy = pytest.importorskip("cupy")
    _, t, ref = tree
    a = cupy.from_dlpack(t.distances)
    assert int(a.device.id) == 0 and np.array_equal(cupy.asnumpy(a), ref)
    b = cupy.asarray(t.distances)  # through __cuda_array_interface__
    assert np.array_equal(cupy.asnumpy(b), ref)
    assert np.array_equal(cupy.asnumpy(t.parents.to_cupy()), t.parents.to_numpy())
    src, dst, w = random_edges(21)
    cg = dyng.Graph.from_edges(
        cupy.asarray(src), cupy.asarray(dst), cupy.asarray(w), num_vertices=300, resources=cuda
    )
    assert np.array_equal(dyng.sssp.compute(cg, 0).distances.to_numpy(), ref)


@pytest.mark.parametrize(
    ("stream", "native", "interface"),
    [(None, 2, 2), (0, 0, 1), (1, 1, 1), (2, 2, 2)],
    ids=["None-per-thread", "0-legacy", "1-legacy", "2-per-thread"],
)
def test_stream_handles_are_read_as_in_cpp(
    stream: int | None, native: int, interface: int, tree: Any
) -> None:
    # None: the per-thread default stream; an integer is a cudaStream_t, 0 the legacy stream
    # (C++ stream_ref(0)); the CUDA array interface names the legacy stream 1 (0 is ambiguous).
    res = dyng.Resources.cuda(0, stream=stream)
    assert int(res._native.stream) == native
    src, dst, w = random_edges(21)
    g = dyng.Graph.from_edges(src, dst, w, num_vertices=300, resources=res)
    t = dyng.sssp.compute(g, 0)
    assert t.distances.__cuda_array_interface__["stream"] == interface
    assert np.array_equal(t.distances.to_numpy(), tree[2])


def test_framework_default_streams_are_the_legacy_stream() -> None:
    streams = []
    torch = _import_or_none("torch")
    if torch is not None and torch.cuda.is_available():
        streams.append(torch.cuda.default_stream(0))
    cupy = _import_or_none("cupy")
    if cupy is not None:
        streams.append(cupy.cuda.Stream.null)
    if not streams:
        pytest.skip("needs PyTorch or CuPy")
    for s in streams:
        res = dyng.Resources.cuda(0, stream=s)
        assert int(res._native.stream) == 0, s  # the legacy stream, not the per-thread one


def _import_or_none(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError:
        return None


def test_a_user_stream_orders_the_results(tree: Any) -> None:
    cupy = pytest.importorskip("cupy")
    _, _, ref = tree
    stream = cupy.cuda.Stream(non_blocking=True)
    res = dyng.Resources.cuda(0, stream=stream)
    src, dst, w = random_edges(21)
    g = dyng.Graph.from_edges(src, dst, w, num_vertices=300, resources=res)
    t = dyng.sssp.compute(g, 0)
    assert t.distances.__cuda_array_interface__["stream"] == stream.ptr
    with cupy.cuda.Stream(non_blocking=True):
        # dynG's calls are complete when they return, so this alone does not show the event's
        # ordering (test_dlpack_orders_the_consumer_after_the_writers_later_work does).
        x = cupy.from_dlpack(t.distances) + 0
    cupy.cuda.Device(0).synchronize()
    assert np.array_equal(cupy.asnumpy(x), ref)


_SPIN = r"""
extern "C" __global__ void spin(long long cycles) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
}
"""


@pytest.mark.parametrize("ordered", [True, False], ids=["ordered", "control-without-event"])
def test_dlpack_orders_the_consumer_after_the_writers_later_work(
    tree: Any, ordered: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PLAN 5.4 rule 4 as a behaviour: work enqueued on the writer's stream after the dynG call
    # (here a kernel that spins for about half a second) must finish before the consumer's work
    # that reads the export. The control replaces the event by nothing and must see the consumer
    # finish while the writer still spins, which shows the test can detect missing ordering.
    cupy = pytest.importorskip("cupy")
    import dyng._backend as backend

    _, _, ref = tree
    writer = cupy.cuda.Stream(non_blocking=True)
    res = dyng.Resources.cuda(0, stream=writer)
    src, dst, w = random_edges(21)
    t = dyng.sssp.compute(dyng.Graph.from_edges(src, dst, w, num_vertices=300, resources=res), 0)
    if not ordered:
        monkeypatch.setattr(backend.active_module(), "order_stream", lambda res, s: None)
    spin = cupy.RawKernel(_SPIN, "spin")
    rate = cupy.cuda.runtime.deviceGetAttribute(cupy.cuda.runtime.cudaDevAttrClockRate, 0)
    rate = rate or 2_000_000  # kHz; 0 where the attribute is no longer reported
    spin((1,), (1,), (np.int64(rate * 1000 // 2),), stream=writer)  # kHz * 500 = 0.5 s of cycles
    consumer = cupy.cuda.Stream(non_blocking=True)
    with consumer:
        x = cupy.from_dlpack(t.distances) + 0
    consumer.synchronize()
    writer_done_when_consumer_done = writer.done
    writer.synchronize()
    assert np.array_equal(cupy.asnumpy(x), ref)
    if ordered:
        assert writer_done_when_consumer_done, "the consumer did not wait for the writer's stream"
    else:
        assert not writer_done_when_consumer_done, "the spin was too short to show the ordering"


def test_an_update_on_another_stream_keeps_that_stream_alive(cuda: dyng.Resources) -> None:
    # Memory an update adds to the graph and the results is released on the update's stream,
    # possibly after the caller dropped it: the graph and the results keep the stream objects of
    # every resources used on them, the result's Arrays keep them too, and the result's
    # last writer names its stream (ADR 0031).
    cupy = pytest.importorskip("cupy")
    import gc
    import weakref

    src, dst, w = random_edges(23)
    seq = dyng.Resources.sequential()
    b = random_batch(23, 300)
    b2 = random_batch(24, 300)
    expected = []
    for algorithm in ("sssp", "cycle_count", "update", "apply"):
        g = dyng.Graph.from_edges(src, dst, w, num_vertices=300, resources=cuda)
        t = dyng.sssp.compute(g, 0) if algorithm in ("sssp", "update") else None
        h = (
            dyng.cycle_count.compute(g, max_length=3)
            if algorithm in ("cycle_count", "update")
            else None
        )
        sides = [cupy.cuda.Stream(non_blocking=True) for _ in range(2)]
        alive = [weakref.ref(s) for s in sides]
        handle = sides[1].ptr
        for side, batch in zip(sides, (b, b2), strict=True):
            res = dyng.Resources.cuda(0, stream=side)
            if algorithm == "sssp":
                dyng.sssp.update(g, batch, t, resources=res)
            elif algorithm == "cycle_count":
                dyng.cycle_count.update(g, batch, h, resources=res)
            elif algorithm == "update":
                dyng.update(g, batch, t, h, resources=res)
            else:
                g.apply(batch, resources=res)
        del side, sides, res
        gc.collect()
        assert all(a() is not None for a in alive), algorithm  # kept by the graph and results
        arrays = []
        if t is not None:
            assert t.distances.__cuda_array_interface__["stream"] == handle  # the last writer
            arrays.append(t.distances)
            expected.append(t.distances.to_numpy())
        if h is not None:
            arrays.append(h.counts)
        del t, h
        gc.collect()
        assert all(a() is not None for a in alive), algorithm  # the graph keeps them
        del g
        gc.collect()
        if arrays:
            assert all(a() is not None for a in alive), algorithm  # the Arrays keep them
            assert all(x.size > 0 for x in arrays)
        del arrays
        gc.collect()
        assert all(a() is None for a in alive), algorithm  # and released with the last of them
    seq_g = dyng.Graph.from_edges(src, dst, w, num_vertices=300, resources=seq)
    seq_t = dyng.sssp.compute(seq_g, 0)
    dyng.sssp.update(seq_g, b, seq_t)
    dyng.sssp.update(seq_g, b2, seq_t)
    for got in expected:
        assert np.array_equal(got, seq_t.distances.to_numpy())


# -------------------------------------------------------------------------------------------------
# The fallback of an installed plugin (a fresh process)
# -------------------------------------------------------------------------------------------------


def _fresh(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    full = {k: v for k, v in os.environ.items() if k != "DYNG_CPU_ONLY"}
    full.update(env)
    return subprocess.run(
        [sys.executable, "-W", "always", "-c", code], capture_output=True, text=True, env=full
    )


def test_no_visible_device_falls_back_with_a_warning() -> None:
    out = _fresh(
        "import dyng, dyng._backend as b; dyng.__version__; print(b.active_module_name)",
        CUDA_VISIBLE_DEVICES="",
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "dyng._core"
    assert "BackendWarning" in out.stderr and "no CUDA device is visible" in out.stderr


def test_use_cpu_only_with_a_plugin_installed() -> None:
    out = _fresh(
        "import dyng, dyng._backend as b; dyng.use_cpu_only(); "
        "print(b.active_module_name, dyng.config()['backends']['cuda'])"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "dyng._core False" and "Warning" not in out.stderr


# -------------------------------------------------------------------------------------------------
# fork: choosing the plugin initializes no CUDA in the parent (ADR 0031)
# -------------------------------------------------------------------------------------------------

_FORK = """
import multiprocessing as mp
import dyng


def work(_):
    try:
        res = dyng.Resources.cuda(0)
        g = dyng.Graph.from_edges([0, 0, 1], [1, 2, 2], [4, 1, 1], resources=res)
        return ("ok", dyng.sssp.compute(g, 0).distances.to_numpy().tolist())
    except Exception as e:
        return ("err", f"{type(e).__name__}: {e}")


PARENT
with mp.get_context("fork").Pool(1) as pool:
    print(pool.map(work, [0])[0])
"""


@pytest.mark.parametrize(
    "parent",
    [
        "pass",
        "dyng.__version__",
        "dyng.show_config()",
        "dyng.sssp.compute(dyng.Graph.from_edges([0], [1], [1], "
        "resources=dyng.Resources.sequential()), 0)",
    ],
    ids=["nothing", "version", "show_config", "cpu_work"],
)
def test_a_forked_worker_can_use_cuda_after_the_parent_chose_the_plugin(parent: str) -> None:
    out = _fresh(_FORK.replace("PARENT", parent))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "('ok', [0, 4, 1])", out.stdout + out.stderr


def test_a_forked_worker_of_a_cuda_parent_is_told_about_fork() -> None:
    # The parent used CUDA itself: the child cannot, and the error names fork and the remedy.
    out = _fresh(_FORK.replace("PARENT", "dyng.Resources.cuda(0).synchronize()"))
    assert out.returncode == 0, out.stderr
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("('err'") and "fork" in last and "spawn" in last, last


def test_a_device_without_code_names_its_compute_capability() -> None:
    # CUDA_FORCE_PTX_JIT=1 ignores the SASS: a device older than the PTX's architecture (the
    # release list embeds PTX for the newest one only) then has no code, as a GPU below the
    # plugin's floor would.
    out = _fresh(
        "import dyng\n"
        "try:\n"
        "    g = dyng.Graph.from_edges([0, 0, 1], [1, 2, 2], [4, 1, 1],\n"
        "                              resources=dyng.Resources.cuda(0))\n"
        "    print('ok', dyng.sssp.compute(g, 0).distances.to_numpy().tolist())\n"
        "except dyng.CudaError as e:\n"
        "    print('err', e)\n",
        CUDA_FORCE_PTX_JIT="1",
    )
    assert out.returncode == 0, out.stderr
    if out.stdout.startswith("ok"):
        pytest.skip("the module's PTX runs on this device (not a release-list build)")
    assert "cudaErrorNoKernelImageForDevice" in out.stdout
    assert "compute capability" in out.stdout and "7.5 or newer" in out.stdout, out.stdout
