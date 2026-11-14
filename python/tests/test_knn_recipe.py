# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""datasets/preprocess/knn_graph.py, the kNN recipe of label_propagation (PLAN O10; ADR 0108,
AD18): on a two-blob set (2000 points, d = 16, 1 % seeds, a fixed seed) compute and the update
chain reach 0.95 accuracy against the true labels; the scikit-learn and NumPy paths write
identical files (skipped without scikit-learn); the file is symmetric with weights in (0, 1],
read by the float reader, and the CLI equals the Python API. Explicit resources (AD19). Skipped
where the repository script is not present (the sdist leaves datasets/ out).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import dyng
import numpy as np
import pytest
from dyng.cli import main

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "datasets" / "preprocess" / "knn_graph.py"
if not SCRIPT.is_file():
    pytest.skip("datasets/preprocess/knn_graph.py is not present", allow_module_level=True)

SEQ = dyng.Resources("sequential")
N, D, SEED = 2000, 16, 20261028


def _recipe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("knn_graph", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def knn() -> ModuleType:
    return _recipe()


@pytest.fixture(scope="module")
def blobs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, np.ndarray]:
    """Two Gaussian blobs around opposite centres (features.npy, labels.txt, labels)."""
    rng = np.random.default_rng(SEED)
    centre = rng.standard_normal(D)
    centre *= 2.0 / np.linalg.norm(centre)
    y = (np.arange(N) % 2).astype(np.int8)
    x = np.where(y[:, None] == 1, centre, -centre) + rng.standard_normal((N, D)) * 0.5
    d = tmp_path_factory.mktemp("blobs")
    np.save(d / "features.npy", x)
    (d / "labels.txt").write_text("".join(f"{int(c)}\n" for c in y))
    return d / "features.npy", d / "labels.txt", y


def _accuracy(labels: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float(((labels[mask] >= 0.5) == (truth[mask] == 1)).mean())


def test_the_graph_file(
    knn: ModuleType, blobs: tuple[Path, Path, np.ndarray], tmp_path: Path
) -> None:
    features, _, _ = blobs
    out = tmp_path / "g.txt"
    assert knn.main([str(features), str(out), "--method", "numpy"]) == 0
    rows = [line.split() for line in out.read_text().splitlines()]
    u = np.array([int(r[0]) for r in rows])
    v = np.array([int(r[1]) for r in rows])
    w = np.array([float(r[2]) for r in rows])
    assert (u < v).all() and len(set(zip(u, v, strict=True))) == len(rows)  # each edge once
    assert ((w > 0) & (w <= 1)).all()
    deg = np.bincount(np.concatenate([u, v]), minlength=N)
    assert deg.min() >= 1 and len(rows) >= N * 5 // 2  # every point has its 5 neighbours
    g = dyng.io.read_edge_list(
        out, weight_dtype="float32", ids="as_is", directed=False, resources=SEQ
    )
    assert g.num_vertices == N and g.num_edges == 2 * len(rows)
    src, dst, ww = g.edges()
    assert ww is not None and ww.dtype == np.float32
    back = {(int(a), int(b)): float(c) for a, b, c in zip(src, dst, ww[:, 0], strict=True)}
    assert all(back[(int(a), int(b))] == back[(int(b), int(a))] for a, b in zip(u, v, strict=True))


def test_compute_and_the_update_chain_reach_095(
    knn: ModuleType, blobs: tuple[Path, Path, np.ndarray], tmp_path: Path
) -> None:
    features, labels_file, y = blobs
    out = tmp_path / "g.txt"
    args = [str(features), str(out), "--labels", str(labels_file), "--method", "numpy"]
    assert knn.main([*args, "--arrival-batches", "5", "--seed", "7"]) == 0
    # compute on the whole graph with 1 % seeds
    g = dyng.io.read_edge_list(
        out, weight_dtype="float32", ids="as_is", directed=False, resources=SEQ
    )
    rng = np.random.default_rng(SEED)
    seeds = rng.choice(N, size=N // 100, replace=False)
    r = dyng.label_propagation.compute(g, seeds, y[seeds], tolerance=1e-6)
    mask = np.ones(N, dtype=bool)
    mask[seeds] = False
    acc = _accuracy(r.labels.to_numpy(), y, mask)
    print(f"compute accuracy {acc:.4f}")
    assert acc >= 0.95
    # the update chain over the arrival batches (the published setup)
    meta = json.loads(tmp_path.joinpath("g.stream.json").read_text())
    cap = np.loadtxt(tmp_path / "g.row_capacity.txt", dtype=np.int64)
    lg = dyng.Graph.with_capacity(cap, resources=SEQ)
    lr = dyng.label_propagation.compute(
        lg, similarity_threshold=meta["similarity_threshold"], update_schedule="damped",
        tolerance=1e-6,
    )  # fmt: skip
    batches = dyng.io.read_binary_batches(tmp_path / "g.dgb")
    assert len(batches) == 5 and meta["num_batches"] == 5
    for b in batches:
        assert isinstance(b, dyng.EdgeBatch)
        dyng.label_propagation.update(lg, b, lr)
    active = lg.vertex_status() == dyng.VertexStatus.active
    unseeded = active & (lr.seed_labels.to_numpy() < 0)
    assert unseeded.sum() > N // 2
    acc = _accuracy(lr.labels.to_numpy(), y, unseeded)
    print(f"update chain accuracy {acc:.4f}")
    assert acc >= 0.95


def test_scikit_learn_and_numpy_write_identical_files(
    knn: ModuleType, blobs: tuple[Path, Path, np.ndarray], tmp_path: Path
) -> None:
    pytest.importorskip("sklearn")
    features, labels_file, _ = blobs
    for method in ("sklearn", "numpy"):
        args = [str(features), str(tmp_path / f"{method}.txt"), "--labels", str(labels_file)]
        assert knn.main([*args, "--method", method, "--arrival-batches", "4"]) == 0
    for suffix in (".txt", ".dgb", ".row_capacity.txt", ".true_labels.txt"):
        a = (tmp_path / f"sklearn{suffix}").read_bytes()
        assert a == (tmp_path / f"numpy{suffix}").read_bytes(), suffix


def test_negative_similarities_and_bad_inputs(knn: ModuleType, tmp_path: Path) -> None:
    x = np.array([[1.0, 0.0], [-1.0, 0.1], [0.9, 0.1], [-0.9, -0.2]])
    np.savetxt(tmp_path / "x.csv", x, delimiter=",")
    for rule in ("drop", "shift"):
        out = tmp_path / f"{rule}.txt"
        assert knn.main([str(tmp_path / "x.csv"), str(out), "-k", "3", "--negative", rule]) == 0
        w = np.array([float(line.split()[2]) for line in out.read_text().splitlines()])
        assert ((w > 0) & (w <= 1)).all()
        if rule == "drop":
            assert len(w) == 2  # only the two pairs of the same side
        else:
            assert len(w) == 6  # every pair, shifted into (0, 1]
    np.save(tmp_path / "z.npy", np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(SystemExit, match="norm 0"):
        knn.main([str(tmp_path / "z.npy"), str(tmp_path / "z.txt"), "-k", "1"])
    with pytest.raises(SystemExit, match="-k"):
        knn.main([str(tmp_path / "x.csv"), str(tmp_path / "k.txt"), "-k", "4"])
    with pytest.raises(SystemExit, match="--labels"):
        knn.main(
            [str(tmp_path / "x.csv"), str(tmp_path / "a.txt"), "-k", "1", "--arrival-batches", "2"]
        )


def test_method_sklearn_without_scikit_learn_is_a_clean_error(
    knn: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--method sklearn where scikit-learn is missing (stubbed as absent) ends in a message that
    names the fix, not a traceback; --method auto falls back to NumPy."""
    import sys

    for name in [m for m in sys.modules if m == "sklearn" or m.startswith("sklearn.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "sklearn", None)  # an import of it raises ImportError
    x = np.array([[1.0, 0.0], [-1.0, 0.1], [0.9, 0.1], [-0.9, -0.2]])
    np.savetxt(tmp_path / "x.csv", x, delimiter=",")
    with pytest.raises(SystemExit, match="needs scikit-learn.*--method numpy"):
        knn.main(
            [str(tmp_path / "x.csv"), str(tmp_path / "s.txt"), "-k", "1", "--method", "sklearn"]
        )
    assert knn.main([str(tmp_path / "x.csv"), str(tmp_path / "a.txt"), "-k", "1"]) == 0


def test_the_cli_equals_the_python_api(
    capsys: pytest.CaptureFixture[str],
    knn: ModuleType,
    blobs: tuple[Path, Path, np.ndarray],
    tmp_path: Path,
) -> None:
    features, labels_file, y = blobs
    out = tmp_path / "g.txt"
    args = [str(features), str(out), "--labels", str(labels_file), "--method", "numpy"]
    assert knn.main([*args, "--arrival-batches", "5"]) == 0
    seeds = np.arange(0, N, 97)
    (tmp_path / "seeds.txt").write_text("".join(f"{s} {int(y[s])}\n" for s in seeds))
    assert main([
        "label_propagation", "compute", "--graph", str(out), "--seeds", str(tmp_path / "seeds.txt"),
        "--tolerance", "1e-6", "--backend", "sequential", "--out", str(tmp_path / "c"), "--quiet",
    ]) == 0  # fmt: skip
    g = dyng.io.read_edge_list(
        out, weight_dtype="float32", ids="as_is", directed=False, resources=SEQ
    )
    r = dyng.label_propagation.compute(g, seeds, y[seeds], tolerance=1e-6)
    dyng.io.write_labels(tmp_path / "api.txt", r.labels)
    assert (tmp_path / "c" / "dynlp_labels.txt").read_bytes() == (tmp_path / "api.txt").read_bytes()
    meta = json.loads((tmp_path / "g.stream.json").read_text())
    assert main([
        "label_propagation", "update", "--batches", str(tmp_path / "g.dgb"), "--capacity",
        str(tmp_path / "g.row_capacity.txt"), "--true-labels", str(tmp_path / "g.true_labels.txt"),
        "--similarity-threshold", str(meta["similarity_threshold"]), "--update-schedule", "damped",
        "--backend", "sequential", "--out", str(tmp_path / "u"), "--quiet",
    ]) == 0  # fmt: skip
    lg = dyng.Graph.with_capacity(
        np.loadtxt(tmp_path / "g.row_capacity.txt", dtype=np.int64), resources=SEQ
    )
    lr = dyng.label_propagation.compute(
        lg, similarity_threshold=meta["similarity_threshold"], update_schedule="damped"
    )
    for b in dyng.io.read_binary_batches(tmp_path / "g.dgb"):
        assert isinstance(b, dyng.EdgeBatch)
        dyng.label_propagation.update(lg, b, lr)
    dyng.io.write_labels(tmp_path / "api_u.txt", lr.labels)
    got = (tmp_path / "u" / "dynlp_labels.txt").read_bytes()
    assert got == (tmp_path / "api_u.txt").read_bytes()
    capsys.readouterr()
