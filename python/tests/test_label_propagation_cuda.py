# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.label_propagation on the CUDA backend, through a CUDA plugin wheel (marker gpu; M10b step
3; ADR 0108, ADR 0031).

- The published setup on LabelPropagation-CUDA's 5k5 stream, damped: the labels are the
  original's own dynlp_labels.txt byte for byte (the fused engine, AD6; the committed digest of
  examples/data); the operators engine gives the same bits (AD6, ADR 0101); the harmonic oracle.
- Result.labels in device memory: ``__cuda_array_interface__``, zero-copy
  ``torch.from_dlpack`` on cuda:0 (pointer equality), copy-on-write.
- The PLAN 5.5 example on cuda.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import dyng
import numpy as np
import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not dyng.config()["backends"]["cuda"],
        reason="needs a CUDA plugin module and a visible CUDA device",
    ),
]

REPO = Path(__file__).resolve().parents[2]
DIGESTS = REPO / "examples" / "data" / "label_propagation_5k5_damped.sha256"
SEQ = dyng.Resources("sequential")


@pytest.fixture(scope="module")
def cuda() -> dyng.Resources:
    return dyng.Resources.cuda(0)


@pytest.fixture(scope="module")
def stream() -> dyng.generators.VertexArrivalStream:
    return dyng.generators.legacy.dynlp_vertex_arrival_stream(
        total_vertices=5000, num_batches=5, seed=42, resources=SEQ
    )


def digest(name: str) -> str:
    for line in DIGESTS.read_text().splitlines():
        if line and not line.startswith("#") and line.split()[1] == name:
            return line.split()[0]
    raise KeyError(name)


def run(
    stream: dyng.generators.VertexArrivalStream, res: dyng.Resources, **options: object
) -> tuple[dyng.Graph, dyng.label_propagation.Result, list[dyng.label_propagation.Stats]]:
    g = dyng.Graph.with_capacity(stream.row_capacity, resources=res)
    r = dyng.label_propagation.compute(
        g, similarity_threshold=stream.similarity_threshold, **options
    )
    r.reserve(stream.max_batch_vertices, stream.max_batch_edges)
    return g, r, [dyng.label_propagation.update(g, b, r) for b in stream.batches]


def test_damped_labels_are_the_originals(
    cuda: dyng.Resources, stream: dyng.generators.VertexArrivalStream, tmp_path: Path
) -> None:
    g, r, stats = run(stream, cuda, update_schedule="damped")
    assert r.space == "device" and g.space == "device"
    assert [s.iterations for s in stats] == [0, 1217, 548, 327, 65]  # the original's results.csv
    assert [s.components for s in stats] == [50, 829, 850, 836, 858]
    assert all(s.engine_used == "fused" for s in stats)
    dyng.io.write_labels(tmp_path / "dynlp_labels.txt", r.labels)  # one host copy
    assert hashlib.sha256((tmp_path / "dynlp_labels.txt").read_bytes()).hexdigest() == digest(
        "cuda"
    )
    # the operators engine: the same bits and counts (AD6 met at max |dF| = 0, ADR 0101)
    _, ro, so = run(stream, cuda, update_schedule="damped", cuda_engine="operators")
    assert all(s.engine_used == "operators" for s in so)
    assert [(s.iterations, s.components, s.frontier_visits) for s in so] == [
        (s.iterations, s.components, s.frontier_visits) for s in stats
    ]
    assert ro.labels.to_numpy().tobytes() == r.labels.to_numpy().tobytes()
    f = dyng.testing.harmonic_solution(g, r.seed_labels)
    mean, _ = f.error(r.labels)
    assert f.converged and mean <= 2000 * 1e-4


def test_device_labels_are_zero_copy_and_copy_on_write(
    cuda: dyng.Resources, stream: dyng.generators.VertexArrivalStream
) -> None:
    g = dyng.Graph.with_capacity(stream.row_capacity, resources=cuda)
    r = dyng.label_propagation.compute(
        g, similarity_threshold=stream.similarity_threshold, update_schedule="damped"
    )
    for b in stream.batches[:2]:
        dyng.label_propagation.update(g, b, r)
    a = r.labels
    assert a.device == "cuda" and a.dtype == np.float32 and a.shape == (5000,)
    ptr = int(a.__cuda_array_interface__["data"][0])
    assert a.__cuda_array_interface__["data"][1] is True  # read-only
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch without CUDA")
    t = torch.from_dlpack(a)
    assert t.device == torch.device("cuda", 0) and t.dtype == torch.float32
    assert t.data_ptr() == ptr  # the same device memory: no copy
    before = t.cpu().numpy().copy()
    dyng.label_propagation.update(g, stream.batches[2], r)  # copies once: t is live
    torch.cuda.synchronize()
    assert np.array_equal(t.cpu().numpy(), before)
    assert int(r.labels.__cuda_array_interface__["data"][0]) != ptr
    del a, t
    ptr2 = int(r.labels.__cuda_array_interface__["data"][0])
    dyng.label_propagation.update(g, stream.batches[3], r)  # no export: in place
    assert int(r.labels.__cuda_array_interface__["data"][0]) == ptr2


def test_plan_example_on_cuda(cuda: dyng.Resources) -> None:
    """PLAN 5.5's label_propagation example (test_plan_examples.py) with dyng.Resources.cuda."""
    stream = dyng.generators.vertex_arrival_stream(
        total_vertices=5_000, average_degree=5, num_batches=5, seed=42, resources=SEQ
    )
    lg = dyng.Graph.with_capacity(
        stream.row_capacity, directed=False, layout="slotted", semantics="net_effect",
        resources=cuda,
    )  # fmt: skip
    labels = dyng.label_propagation.compute(
        lg, tolerance=1e-4, update_schedule="damped",
        similarity_threshold=stream.similarity_threshold,
    )  # fmt: skip
    for b in stream.batches:
        st = dyng.label_propagation.update(lg, b, labels)
    assert st.converged and labels.space == "device"
    _, seq_r, _ = run(stream, SEQ, update_schedule="damped")
    # damped is bitwise per backend; across backends the labels agree within the bound of AD7
    diff = np.abs(labels.labels.to_numpy() - seq_r.labels.to_numpy())
    assert float(diff.max()) <= 2000 * 1e-4
