# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The GIL is released around native work, and concurrent Python threads stay safe."""

from __future__ import annotations

import threading
import time

import dyng
import numpy as np


def _dense_graph(n: int) -> dyng.Graph:
    src, dst = np.nonzero(~np.eye(n, dtype=bool))
    return dyng.Graph.from_edges(
        src.astype(np.int32), dst.astype(np.int32), resources=dyng.Resources.sequential()
    )


def test_gil_is_released_during_a_long_count() -> None:
    g = _dense_graph(12)  # counting the cycles of K12 up to length 8 takes a noticeable time
    ticks = 0
    done = threading.Event()

    def count() -> None:
        dyng.cycle_count.compute(g, max_length=8)
        done.set()

    worker = threading.Thread(target=count)
    started = time.perf_counter()
    worker.start()
    while not done.is_set():
        ticks += 1
        time.sleep(0.0005)
    worker.join()
    elapsed = time.perf_counter() - started
    assert elapsed > 0.02, "the count was too fast to observe the GIL; make the graph larger"
    assert ticks > 5  # the main thread ran while the native count was running


def test_concurrent_updates_of_one_graph_are_serialized() -> None:
    rng = np.random.default_rng(5)
    n = 200
    src, dst, w = rng.integers(0, n, 2000), rng.integers(0, n, 2000), rng.integers(1, 9, 2000)
    g = dyng.Graph.from_edges(src, dst, w, vertex_dtype="int32")
    trees = [dyng.sssp.compute(g, s) for s in range(4)]
    batches = [
        dyng.EdgeBatch(
            insert=(rng.integers(0, n, 20), rng.integers(0, n, 20), rng.integers(1, 9, 20))
        )
        for _ in range(8)
    ]
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            for b in batches[i::4]:
                dyng.update(g, b, *trees)
        except dyng.StaleResultError:
            pass  # another thread's update may win the race: a clean, detected outcome
        except BaseException as e:  # pragma: no cover - a failure
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for s, tree in enumerate(trees):
        assert tree.graph_version == g.version
        assert tree.distances.tolist() == dyng.sssp.compute(g, s).distances.tolist()


def test_profiler_reads_while_other_threads_compute() -> None:
    r = dyng.Resources.sequential()
    g = _dense_graph(9)
    g2 = dyng.Graph.from_edges([0, 1], [1, 0], resources=r)
    with dyng.profile(r) as p:
        t = threading.Thread(
            target=lambda: [dyng.cycle_count.compute(g2, max_length=2) for _ in range(200)]
        )
        t.start()
        for _ in range(50):
            _ = p.stages  # never overlaps a recording call
        t.join()
    assert any(s.name.startswith("cycle_count.") for s in p.stages)
    del g


def test_reading_arrays_while_another_thread_updates() -> None:
    # A reader's Array either shows a complete state or raises StaleResultError, never memory
    # an update is changing or has freed (the state is copied while a view is alive).
    n = 1_000
    src = np.arange(n - 1)
    g = dyng.Graph.from_edges(src, src + 1, np.ones(n - 1, dtype=np.int32))
    tree = dyng.sssp.compute(g, 0)
    done = threading.Event()
    bad: list[str] = []

    def reader() -> None:
        while not done.is_set():
            a = tree.distances
            try:
                v = a.to_numpy(copy=False)
            except dyng.StaleResultError:
                continue
            if not np.array_equal(v, np.arange(v.size)):
                bad.append(f"torn read of {v.size} distances")
                return

    t = threading.Thread(target=reader)
    t.start()
    try:
        size = n
        for _ in range(40):
            s = np.arange(size - 1, size + 499, dtype=np.int64)
            dyng.sssp.update(g, dyng.EdgeBatch(insert=(s, s + 1, np.ones(500, np.int32))), tree)
            size += 500
    finally:
        done.set()
        t.join()
    assert not bad, bad
    assert tree.distances.to_numpy().tolist() == list(range(size))


def test_profile_and_copy_policy_while_other_threads_compute() -> None:
    # Attaching a profiler and setting the copy policy of a handle wait for running calls
    # (they are plain fields of the shared handle that every call reads).
    res = dyng.Resources.openmp(2)
    src = np.arange(2_000)
    g = dyng.Graph.from_edges(src, src + 1, np.ones(2_000, np.int32), resources=res)
    done = threading.Event()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            while not done.is_set():
                dyng.sssp.compute(g, 0)
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    try:
        for i in range(50):
            with dyng.profile(res) as p:
                dyng.sssp.compute(g, 0)
            assert p.total_host_ms("sssp.compute") >= 0.0
            res.copy_policy = "warn" if i % 2 else "allow"
            assert res.copy_policy in ("warn", "allow")
    finally:
        done.set()
        for t in threads:
            t.join()
    assert not errors, errors
    assert not res._native.has_profiler()
