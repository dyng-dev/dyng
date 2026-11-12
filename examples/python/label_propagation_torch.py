# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""label_propagation_torch: DynLP on LabelPropagation-CUDA's vertex-arrival stream, the published
setup (a graph with the stream's row capacities, compute on the empty graph, one update per
batch), then the labels as a PyTorch tensor without a copy (NumPy when PyTorch is missing).

    python label_propagation_torch.py <out_dir> [sequential|cuda] [total_vertices]

It writes <out_dir>/dynlp_labels.txt, the bytes examples/cpp/label_propagation_stream.cpp writes.
"""

import sys
from pathlib import Path

import dyng

out = Path(sys.argv[1])
backend = sys.argv[2] if len(sys.argv) > 2 else "sequential"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
res = dyng.Resources.cuda() if backend == "cuda" else dyng.Resources.sequential()  # no OpenMP

stream = dyng.generators.legacy.dynlp_vertex_arrival_stream(
    total_vertices=n, num_batches=5, seed=42, resources=dyng.Resources.sequential()
)
g = dyng.Graph.with_capacity(stream.row_capacity, resources=res)  # slot rows, float32 weights
lp = dyng.label_propagation.compute(
    g, update_schedule="damped", similarity_threshold=stream.similarity_threshold
)
lp.reserve(stream.max_batch_vertices, stream.max_batch_edges)
for i, batch in enumerate(stream.batches):
    st = dyng.label_propagation.update(g, batch, lp)
    print(
        f"batch {i}: {st.iterations} iterations, {st.components} components, "
        f"{g.num_active_vertices} active"
    )

active = g.vertex_status() == dyng.VertexStatus.active
unseeded = active & (lp.seed_labels.to_numpy() < 0)
truth = stream.true_labels
try:
    import torch

    F = torch.from_dlpack(lp.labels)  # float32 on the result's device, no copy
    predicted = (F >= 0.5).cpu().numpy()
    del F  # no export outlives the result's next update (it would copy the labels once)
except ImportError:
    predicted = lp.labels.to_numpy() >= 0.5  # a host copy (a view on the host backends)
accuracy = float((predicted[unseeded] == (truth[unseeded] == 1)).mean())
print(f"accuracy {accuracy:.4f} over {int(unseeded.sum())} active unlabelled vertices")
out.mkdir(parents=True, exist_ok=True)
dyng.io.write_labels(out / "dynlp_labels.txt", lp.labels)
