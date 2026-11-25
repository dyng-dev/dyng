#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Build the cosine-similarity kNN graph of a feature matrix for dyng.label_propagation.

    datasets/preprocess/knn_graph.py FEATURES OUT [--labels FILE] [-k 5] [--negative drop|shift]
        [--method auto|sklearn|numpy]
        [--arrival-batches B [--seed-fraction F] [--deleted-fraction F] [--seed S]]

The construction of the DynLP paper's non-graph datasets (the thesis' "experimental results":
pairwise cosine similarities sparsified by a k-nearest-neighbour rule with k = 5): every point
is a vertex; each vertex is joined to its k most similar other points; the graph is the union of
these pairs (symmetric: an edge {u, v} when u is among v's k nearest or v among u's), and an
edge's weight is the cosine similarity of its endpoints, in (0, 1]. This script is dynG's own
recipe (PLAN O10; ADR 0108, decided by the author on 2026-10-28 as AD18): it ships no dataset and
no third-party code. The paper's datasets (IMDB, ImageNet, Yelp, Amazon) are not redistributed;
get them from their publishers (for example https://ai.stanford.edu/~amaas/data/sentiment/,
https://www.image-net.org/, https://www.yelp.com/dataset,
https://cseweb.ucsd.edu/~jmcauley/datasets.html), compute their features (the paper used
ResNet-50 penultimate-layer features and TF-IDF vectors), save them as FEATURES, and run this
script.

Input:
    FEATURES    an n x d matrix: a .npy file, or a .csv / .txt file of n rows of d numbers
                (comma or whitespace separated; '#' comments). Every row must have a non-zero
                norm (the cosine of a zero vector is undefined).
    --labels    optional: n class labels 0 or 1 (.npy, or a text file with one integer per line
                or 'i class' lines), written as OUT's true labels and needed for the arrival
                batches' seeds.

Output (OUT is a file):
    OUT         the weighted edge list 'u v w', one line per undirected edge with u < v, sorted,
                0-based ids, w the float32 cosine similarity in its shortest round-trip form (the
                C++ writer's: dyng.io.write_edge_list), which dyng.io.read_edge_list(OUT,
                weight_dtype="float32", ids="as_is", directed=False) reads as an undirected
                float graph, and `dyng label_propagation compute --graph OUT --seeds SEEDS --out
                DIR` runs with seed labels SEEDS ('v class' lines, for example a share of
                OUT.true_labels.txt's lines: head -n 100 OUT.true_labels.txt > SEEDS).

With --arrival-batches B, also a vertex-arrival stream in the files `dyng generate` writes, next
to OUT (its stem followed by .dgb etc.): OUT.dgb (B batches of vertex operations and edges: the
vertices arrive in a seeded random order split into B nearly equal parts, a share --seed-fraction
of each part's arrivals with their true class; every edge in batch max(batch(u), batch(v)), as
LabelPropagation-CUDA's generator places them; from batch 1 on, --deleted-fraction of a batch's
arrivals in departures of earlier, still present vertices), OUT.row_capacity.txt (every vertex's
degree), OUT.true_labels.txt ('i class') and OUT.stream.json (the published threshold: the
average edge weight of the graph, as the original's computeAvgEdgeWeight). Run them with

    dyng label_propagation update --batches OUT.dgb --capacity OUT.row_capacity.txt \\
        --true-labels OUT.true_labels.txt --similarity-threshold <OUT.stream.json's> --out DIR

Rules:
    - Similarity: s(u, v) = x_u . x_v / (|x_u| |x_v|), in double precision, clipped to at most 1.
    - Neighbours: for each u, the k other points with the largest s, ties broken by the smaller
      index (a duplicate point is a neighbour like any other; u itself never is).
    - Negative similarities (--negative): `drop` (the default) keeps only pairs with s > 0, so an
      anti-similar neighbour adds no edge (features of the paper's kind, non-negative ReLU
      activations and TF-IDF vectors, have s >= 0); `shift` maps every weight to (1 + s) / 2
      (and drops the pairs at s = -1, whose weight would be 0).
    - --method: `sklearn` finds the candidate neighbours with scikit-learn's
      NearestNeighbors(metric="cosine", algorithm="brute") (BSD-3-Clause; an optional dependency
      imported only here, never vendored), `numpy` with a NumPy brute force over row blocks;
      `auto` (the default) uses scikit-learn when it is installed. Both rank their candidates by
      the rule above and weigh the edges with the same similarities, so on inputs without ties
      at the k-th neighbour they write identical files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_BLOCK = 1024  # rows of the NumPy brute force per block


def read_features(path: Path) -> np.ndarray:
    """An n x d float64 matrix from a .npy, .csv or text file."""
    if path.suffix == ".npy":
        x = np.load(path, allow_pickle=False)
    else:
        text = path.read_text().replace(",", " ")
        x = np.loadtxt(text.splitlines(), ndmin=2, comments="#")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise SystemExit(f"{path}: expected an n x d matrix with n >= 2, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise SystemExit(f"{path}: a feature is NaN or infinite")
    return x


def read_labels(path: Path, n: int) -> np.ndarray:
    """n class labels (0 or 1) from a .npy file, one integer per line, or 'i class' lines."""
    if path.suffix == ".npy":
        y = np.asarray(np.load(path, allow_pickle=False))
    else:
        rows = np.loadtxt(path, ndmin=2, comments="#", dtype=np.int64)
        y = rows[:, 0] if rows.shape[1] == 1 else rows[np.argsort(rows[:, 0]), 1]
    y = y.reshape(-1).astype(np.int64)
    if y.size != n or not np.isin(y, (0, 1)).all():
        raise SystemExit(f"{path}: expected {n} labels 0 or 1")
    return y.astype(np.int8)


def unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1)
    if (norms == 0).any():
        raise SystemExit(f"row {int(np.argmin(norms))} has norm 0: its cosine is undefined")
    out: np.ndarray = x / norms[:, None]
    return out


def _rank(u: int, cand: np.ndarray, sims: np.ndarray, k: int) -> np.ndarray:
    """The k best of the candidates of u: the largest similarity, then the smaller index."""
    keep = cand != u
    cand, sims = cand[keep], sims[keep]
    order = np.lexsort((cand, -sims))
    return cand[order[:k]]


def neighbours_numpy(unit: np.ndarray, k: int) -> np.ndarray:
    """n x k neighbour ids by a NumPy brute force over row blocks."""
    n = unit.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    ids = np.arange(n)
    for start in range(0, n, _BLOCK):
        s = np.minimum(unit[start : start + _BLOCK] @ unit.T, 1.0)
        for r in range(s.shape[0]):
            out[start + r] = _rank(start + r, ids, s[r], k)
    return out


def neighbours_sklearn(unit: np.ndarray, k: int) -> np.ndarray:
    """n x k neighbour ids: scikit-learn's brute-force cosine search finds k + 1 candidates per
    point, which the rule of the module description ranks with this script's similarities."""
    try:
        from sklearn.neighbors import NearestNeighbors  # optional, BSD-3-Clause
    except ImportError:
        raise SystemExit(
            "--method sklearn needs scikit-learn (pip install scikit-learn); "
            "use --method numpy or --method auto"
        ) from None

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(unit)
    _, cand = nn.kneighbors(unit)
    out = np.empty((unit.shape[0], k), dtype=np.int64)
    for u in range(unit.shape[0]):
        c = np.asarray(cand[u], dtype=np.int64)
        out[u] = _rank(u, c, np.minimum(unit[c] @ unit[u], 1.0), k)
    return out


def knn_edges(
    x: np.ndarray, k: int, negative: str, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """(u, v, w float32) of the symmetric kNN graph, u < v, sorted; and the method used."""
    n = x.shape[0]
    if not 1 <= k < n:
        raise SystemExit(f"-k must be in [1, {n - 1}], got {k}")
    unit = unit_rows(x)
    if method == "auto":
        try:
            import sklearn  # noqa: F401

            method = "sklearn"
        except ImportError:
            method = "numpy"
    nbr = neighbours_sklearn(unit, k) if method == "sklearn" else neighbours_numpy(unit, k)
    u = np.repeat(np.arange(n, dtype=np.int64), k)
    v = nbr.reshape(-1)
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)  # sorted, each edge once
    a, b = pairs[:, 0], pairs[:, 1]
    s = np.minimum(np.einsum("ij,ij->i", unit[a], unit[b]), 1.0)
    if negative == "shift":
        s = (1.0 + s) / 2.0
    w = s.astype(np.float32)
    keep = w > 0
    return a[keep].astype(np.int32), b[keep].astype(np.int32), w[keep], method


def average_weight(w: np.ndarray) -> float:
    """The original's computeAvgEdgeWeight: each undirected edge in both directions, a double sum,
    rounded to float (0.5 without edges)."""
    if w.size == 0:
        return 0.5
    total = 2.0 * float(np.asarray(w, dtype=np.float64).sum())
    return float(str(np.float32(total / (2 * w.size))))


def arrival_stream(
    n: int,
    a: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    labels: np.ndarray,
    batches: int,
    seed_fraction: float,
    deleted_fraction: float,
    seed: int,
) -> list[Any]:
    """The vertex-arrival batches (see the module description) as dyng.EdgeBatch objects."""
    import dyng

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    parts = np.array_split(order, batches)
    batch_of = np.empty(n, dtype=np.int64)
    for i, part in enumerate(parts):
        batch_of[part] = i
    edge_batch = np.maximum(batch_of[a], batch_of[b])
    present: list[int] = []
    out = []
    for i, part in enumerate(parts):
        arrivals = np.asarray(part, dtype=np.int32)
        vertex_labels = np.full(arrivals.size, -1, dtype=np.int8)
        num_seeds = max(1, round(seed_fraction * arrivals.size)) if arrivals.size else 0
        seeded = rng.choice(arrivals.size, size=num_seeds, replace=False)
        vertex_labels[seeded] = labels[arrivals[seeded]]
        departures = np.zeros(0, dtype=np.int32)
        if i > 0 and present:
            count = min(len(present), round(deleted_fraction * arrivals.size))
            pick = np.sort(rng.choice(len(present), size=count, replace=False))
            departures = np.asarray([present[j] for j in pick], dtype=np.int32)
            gone = set(departures.tolist())
            present = [v for v in present if v not in gone]
        present.extend(arrivals.tolist())
        mine = np.flatnonzero(edge_batch == i)
        out.append(
            dyng.EdgeBatch(
                insert=(a[mine], b[mine], w[mine]),
                insert_vertices=arrivals,
                vertex_labels=vertex_labels,
                delete_vertices=departures,
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module documentation (the file's header) for the rules.",
    )
    p.add_argument("features", type=Path, help="the n x d features (.npy, .csv or text)")
    p.add_argument("out", type=Path, help="the weighted edge list to write")
    p.add_argument("--labels", type=Path, default=None, help="n class labels (0 or 1)")
    p.add_argument("-k", type=int, default=5, help="neighbours per point (default 5)")
    p.add_argument("--negative", choices=("drop", "shift"), default="drop")
    p.add_argument("--method", choices=("auto", "sklearn", "numpy"), default="auto")
    p.add_argument("--arrival-batches", type=int, default=0, metavar="B")
    p.add_argument("--seed-fraction", type=float, default=0.01, metavar="F")
    p.add_argument("--deleted-fraction", type=float, default=0.09, metavar="F")
    p.add_argument("--seed", type=int, default=42, metavar="S")
    args = p.parse_args(argv)

    import dyng

    x = read_features(args.features)
    n = x.shape[0]
    labels = None if args.labels is None else read_labels(args.labels, n)
    a, b, w, method = knn_edges(x, args.k, args.negative, args.method)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dyng.io.write_edge_list(args.out, (a, b, w))
    print(f"knn_graph: n={n} d={x.shape[1]} k={args.k} edges={a.size} method={method}")
    if args.arrival_batches <= 0:
        return 0
    if labels is None:
        raise SystemExit("--arrival-batches needs --labels (the seeds' classes)")
    if not 0 < args.seed_fraction <= 1 or not 0 <= args.deleted_fraction <= 1:
        raise SystemExit("--seed-fraction must be in (0, 1] and --deleted-fraction in [0, 1]")
    stem = args.out.with_suffix("")
    batches = arrival_stream(
        n, a, b, w, labels, args.arrival_batches, args.seed_fraction, args.deleted_fraction,
        args.seed,
    )  # fmt: skip
    dyng.io.write_binary_batches(Path(f"{stem}.dgb"), batches)
    degree = np.bincount(np.concatenate([a, b]), minlength=n)
    np.savetxt(f"{stem}.row_capacity.txt", degree, fmt="%d")
    with open(f"{stem}.true_labels.txt", "w", encoding="utf-8", newline="") as f:
        f.writelines(f"{i} {int(c)}\n" for i, c in enumerate(labels.tolist()))
    meta = {
        "generator": "datasets/preprocess/knn_graph.py",
        "num_vertices": n,
        "num_edges": int(a.size),
        "num_batches": args.arrival_batches,
        "similarity_threshold": average_weight(w),
        "max_batch_vertices": max(int(bb.insert_vertices.size) for bb in batches),
        "max_batch_edges": max(bb.num_insertions for bb in batches),
        "options": {
            "k": args.k,
            "negative": args.negative,
            "seed": args.seed,
            "seed_fraction": args.seed_fraction,
            "deleted_fraction": args.deleted_fraction,
        },
    }
    Path(f"{stem}.stream.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"knn_graph: {args.arrival_batches} arrival batches -> {stem}.dgb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
