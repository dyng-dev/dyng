# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The code of the tutorial "Streaming label propagation" (docs/tutorials/
streaming_label_propagation.md, part 2), which quotes this file between its `# [tutorial: ...]`
markers (ci/tutorial_quotes.py pins every quote): your own feature vectors through the kNN recipe
(datasets/preprocess/knn_graph.py, AD18), the vertex-arrival batches it writes run in the
streaming mode (slack rows without capacity hints, tau of the current graph; AD16), the labels
read by PyTorch without a copy (importorskip) and by NumPy. Explicit resources (AD19). Skipped
where the repository script is not present (the sdist leaves datasets/ out).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import dyng
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "datasets" / "preprocess" / "knn_graph.py"
if not RECIPE.is_file():
    pytest.skip("datasets/preprocess/knn_graph.py is not present", allow_module_level=True)


def _features(out: Path) -> tuple[Path, Path, np.ndarray]:
    # [tutorial: features]
    rng = np.random.default_rng(2026)
    n, d = 2000, 16
    y = (np.arange(n) % 2).astype(np.int8)  # two classes
    centre = rng.standard_normal(d)
    centre *= 2.0 / np.linalg.norm(centre)
    x = np.where(y[:, None] == 1, centre, -centre) + 0.5 * rng.standard_normal((n, d))
    np.save(out / "features.npy", x)  # n x d feature vectors, one per row
    (out / "labels.txt").write_text("".join(f"{c}\n" for c in y))  # their classes
    # [tutorial: features end]
    return out / "features.npy", out / "labels.txt", y


def _stream(out: Path, features: Path, labels: Path) -> None:
    # [tutorial: knn]
    subprocess.run(
        [sys.executable, RECIPE, features, out / "g.txt", "--labels", labels, "-k", "5",
         "--arrival-batches", "5", "--seed-fraction", "0.01", "--seed", "7"],
        check=True,
    )  # fmt: skip
    # [tutorial: knn end]


def _run(out: Path, res: dyng.Resources) -> tuple[dyng.label_propagation.Result, dyng.Graph]:
    # [tutorial: streaming]
    n = len((out / "g.true_labels.txt").read_text().splitlines())
    g = dyng.Graph.with_capacity([], vertex_capacity=n, layout="slack", resources=res)
    lp = dyng.label_propagation.compute(g, update_schedule="damped")  # tau: the current graph's
    for i, batch in enumerate(dyng.io.read_binary_batches(out / "g.dgb")):
        st = dyng.label_propagation.update(g, batch, lp)
        print(f"batch {i}: {g.num_active_vertices} active, {st.iterations} iterations, "
              f"{st.batch.relocated_rows} rows moved, {st.batch.compactions} compactions")
    # [tutorial: streaming end]
    return lp, g


def test_the_streaming_tutorial(tmp_path: Path) -> None:
    features, labels, y = _features(tmp_path)
    _stream(tmp_path, features, labels)
    res = dyng.Resources("sequential")  # or dyng.Resources.cuda()
    lp, g = _run(tmp_path, res)
    # [tutorial: labels]
    active = g.vertex_status() == dyng.VertexStatus.active
    unseeded = active & (lp.seed_labels.to_numpy() < 0)
    F = np.from_dlpack(lp.labels)  # float32, read-only, no copy (on the host backends)
    accuracy = ((F[unseeded] >= 0.5) == (y[unseeded] == 1)).mean()
    print(f"accuracy {accuracy:.4f} over {unseeded.sum()} vertices")
    # [tutorial: labels end]
    assert not F.flags.writeable and F.dtype == np.float32
    assert unseeded.sum() > len(y) // 2
    assert accuracy >= 0.95
    # The streaming mode's labels meet the harmonic oracle of the final graph.
    mean, _ = dyng.testing.harmonic_solution(g, lp.seed_labels).error(lp.labels)
    assert mean <= 2000 * lp.options.tolerance


def test_the_streaming_tutorial_with_torch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    features, labels, y = _features(tmp_path)
    _stream(tmp_path, features, labels)
    res = dyng.Resources("sequential")
    lp, _ = _run(tmp_path, res)
    # [tutorial: torch]
    F = torch.from_dlpack(lp.labels)  # a tensor over the result's memory: no copy
    assert F.data_ptr() == lp.labels.__array_interface__["data"][0]
    classes = (F >= 0.5).to(torch.int8)  # what a model downstream would read
    del F  # drop the view before the next update (or it copies the labels once)
    # [tutorial: torch end]
    assert classes.shape == (len(y),)
