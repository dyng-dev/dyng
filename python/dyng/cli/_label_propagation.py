# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""``dyng label_propagation compute|update`` (LabelPropagation-CUDA's DynLP) and the generators
``dyng generate vertex_arrival_stream`` / ``dynlp_vertex_arrival_stream`` (0.4; ADR 0108).

The option flags are the kebab-case fields of :class:`dyng.label_propagation.Options`, one to
one (``--similarity-threshold`` takes ``published``, ``current`` or a number). The outputs are
the original's: ``dynlp_labels.txt`` (``i label``, 6 decimals: :func:`dyng.io.write_labels`)
and a ``results.csv``-style ``stats.csv`` (the original's columns ``batch, time_ms, iterations,
frontier_visits, components, accuracy, eval_ms``, then dynG's).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import dyng

from ._common import (
    CliError,
    Stopwatch,
    UsageError,
    add_function_flags,
    add_option_flags,
    function_kwargs,
    info,
)

__all__ = ["add_label_propagation", "add_generate_vertex_arrival"]

_HELP = """\
Dynamic label propagation under vertex batches (dyng.label_propagation; LabelPropagation-CUDA's
DynLP). `compute` solves the labels of a float-weighted edge list from seed labels (IrLP);
`update` runs a vertex-arrival stream batch by batch: a generated one (--generate legacy: the
original's generator, byte for byte; --generate library: the library's) or the batches of a
.dgt/.dgb file (--batches). The published setup (the default) gives the graph the stream's
full degrees as row capacities and the stream's threshold; --layout slack with
--similarity-threshold current is the streaming mode. Outputs: DIR/dynlp_labels.txt (the
original's bytes) and, for update, DIR/stats.csv (the original's results.csv columns, then
dynG's counters).
"""

#: The original's results.csv columns, then dynG's (stats.csv).
STATS_COLUMNS = (
    "batch",
    "time_ms",
    "iterations",
    "frontier_visits",
    "components",
    "accuracy",
    "eval_ms",
    "affected",
    "initial_frontier",
    "sparsified_edges",
    "converged",
    "fallback_used",
    "inserted_vertices",
    "deleted_vertices",
    "inserted_edges",
    "dropped_insertions",
)


def _backend_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("backend")
    g.add_argument(
        "--backend",
        choices=("sequential", "cuda"),
        default=None,
        help="the backend (label_propagation has no OpenMP backend in 0.4; default: cuda when "
        "the active module has it and a device is visible, else sequential)",
    )
    g.add_argument("--device", type=int, default=0, metavar="D", help="CUDA device (cuda only)")


def _resources(args: argparse.Namespace) -> dyng.Resources:
    backend = args.backend
    if backend is None:
        backend = "cuda" if dyng.config()["backends"]["cuda"] else "sequential"
    if backend == "cuda":
        return dyng.Resources.cuda(device=args.device)
    return dyng.Resources.sequential()


def _option_flags(p: argparse.ArgumentParser) -> list[str]:
    """One flag per field of Options (``--similarity-threshold`` added by hand: it also takes
    the words ``published`` and ``current``)."""
    names = add_option_flags(
        p,
        dyng.label_propagation.Options,
        "label_propagation options",
        skip=("similarity_threshold",),
    )
    p.add_argument(
        "--similarity-threshold",
        dest="opt_similarity_threshold",
        default=None,
        metavar="TAU",
        help="Options.similarity_threshold: 'published' (the stream's threshold, the average "
        "edge weight of the full graph; with --batches the average weight of the files' edges), "
        "'current' (None: the average weight of the current graph, the streaming mode) or a "
        "number (default: published for update, current for compute)",
    )
    return ["similarity_threshold", *names]


def _tau(text: str | None, default: str) -> str | float:
    word = default if text is None else text
    if word in ("published", "current"):
        return word
    try:
        value = float(word)
    except ValueError:
        raise UsageError(
            f"--similarity-threshold: 'published', 'current' or a number, got {word!r}"
        ) from None
    if not math.isfinite(value):
        raise UsageError(f"--similarity-threshold must be finite, got {word!r}")
    return value


def _options(args: argparse.Namespace, tau: float | None) -> dyng.label_propagation.Options:
    kw: dict[str, Any] = {
        n: getattr(args, "opt_" + n)
        for n in args.option_names
        if n != "similarity_threshold" and getattr(args, "opt_" + n) is not None
    }
    return dyng.label_propagation.Options(similarity_threshold=tau, **kw)


def add_label_propagation(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("label_propagation", help="dynamic label propagation", description=_HELP)
    verbs = p.add_subparsers(dest="verb", metavar="VERB", required=True)

    c = verbs.add_parser(
        "compute",
        help="the labels of a weighted edge list from seed labels (IrLP)",
        description="Read an undirected edge list 'u v w' with float weights in (0, 1] (one line "
        "per edge; datasets/preprocess/knn_graph.py writes one), and seed labels 'v class' "
        "(class 0 or 1), and write DIR/dynlp_labels.txt. The weights must be positive; 0.4 does "
        "not check it (a weight <= 0 gives labels that are not the harmonic solution).",
    )
    c.add_argument("--graph", required=True, metavar="FILE", help="the weighted edge list")
    c.add_argument("--seeds", required=True, metavar="FILE", help="'v class' lines")
    c.add_argument(
        "--index-base",
        type=int,
        default=0,
        metavar="B",
        help="the first vertex id of both files (default 0)",
    )
    c.add_argument(
        "--num-vertices",
        type=int,
        default=None,
        metavar="N",
        help="the vertex count (default: the largest id + 1)",
    )
    names = _option_flags(c)
    c.add_argument("--out", required=True, metavar="DIR", help="output directory")
    c.add_argument("--quiet", action="store_true", help="no report line")
    _backend_flags(c)
    c.set_defaults(func=run_compute, option_names=names)

    u = verbs.add_parser(
        "update",
        help="run a vertex-arrival stream batch by batch (DynLP)",
        description="The published setup: a graph with the stream's row capacities "
        "(Graph.with_capacity), compute() on the empty graph, then one update() per batch. "
        "The stream is generated (--generate) or read from batch files (--batches, applied in "
        "order).",
    )
    src = u.add_argument_group("the stream")
    src.add_argument(
        "--generate",
        choices=("legacy", "library"),
        default=None,
        help="generate the stream: legacy (LabelPropagation-CUDA's generator, byte for byte; "
        "dyng.generators.legacy.dynlp_vertex_arrival_stream) or library "
        "(dyng.generators.vertex_arrival_stream)",
    )
    gen_names = add_function_flags(
        u, dyng.generators.vertex_arrival_stream, "generator", skip=("resources", "edge_dtype")
    )
    src.add_argument(
        "--batches",
        nargs="+",
        default=None,
        metavar="FILE",
        help="read the stream from .dgt or .dgb files instead (float weights, vertex "
        "operations), applied in order",
    )
    src.add_argument(
        "--capacity",
        default="auto",
        metavar="auto|FILE",
        help="--batches: the row capacities, 'auto' (each vertex's edges over all batches) or a "
        "file of one integer per vertex (row_capacity.txt of `dyng generate`)",
    )
    src.add_argument(
        "--true-labels",
        default=None,
        metavar="FILE",
        help="--batches: the true classes ('i class' lines, true_labels.txt of `dyng "
        "generate`) for the accuracy column",
    )
    u.add_argument(
        "--layout",
        choices=("slotted", "slack"),
        default="slotted",
        help="the rows: slotted (the published setup) or slack (the streaming mode)",
    )
    names = _option_flags(u)
    u.add_argument("--out", required=True, metavar="DIR", help="output directory")
    u.add_argument(
        "--per-batch-labels",
        action="store_true",
        help="also write the labels after every batch (DIR/dynlp_labels_b<i>.txt)",
    )
    u.add_argument("--quiet", action="store_true", help="no per-batch report lines")
    _backend_flags(u)
    u.set_defaults(func=run_update, option_names=names, generator_names=gen_names)


# -------------------------------------------------------------------------------------------------
# compute
# -------------------------------------------------------------------------------------------------


def _read_seeds(path: str, base: int) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[int] = []
    labels: list[int] = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            parts = text.replace(",", " ").split()
            if len(parts) != 2:
                raise CliError(f"{path}:{number}: expected 'v class', got {line.strip()!r}")
            try:
                v, c = int(parts[0]) - base, int(parts[1])
            except ValueError:
                raise CliError(
                    f"{path}:{number}: expected integers, got {line.strip()!r}"
                ) from None
            if c not in (0, 1):
                raise CliError(f"{path}:{number}: the class must be 0 or 1, got {c}")
            vertices.append(v)
            labels.append(c)
    return np.asarray(vertices, dtype=np.int32), np.asarray(labels, dtype=np.int8)


def run_compute(args: argparse.Namespace) -> int:
    res = _resources(args)
    tau = _tau(args.opt_similarity_threshold, "current")
    if tau == "published":
        raise UsageError("compute: --similarity-threshold is 'current' or a number")
    g = dyng.io.read_edge_list(
        args.graph,
        weight_dtype="float32",
        ids="as_is",
        index_base=args.index_base,
        directed=False,
        resources=res,
    )
    seeds_v, seeds_c = _read_seeds(args.seeds, args.index_base)
    if args.num_vertices is not None and args.num_vertices != g.num_vertices:
        if args.num_vertices < g.num_vertices:
            raise UsageError(
                f"--num-vertices {args.num_vertices} is below the graph's {g.num_vertices}"
            )
        src, dst, w = g.edges()
        g = dyng.Graph.from_edges(
            src, dst, w, num_vertices=args.num_vertices, directed=False, resources=res
        )
    t = Stopwatch()
    r = dyng.label_propagation.compute(
        g, seeds_v, seeds_c, options=_options(args, None if tau == "current" else float(tau))
    )
    ms = t.ms()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dyng.io.write_labels(out / "dynlp_labels.txt", r.labels)
    info(
        args,
        f"compute: n={g.num_vertices} edges={g.num_edges // 2} seeds={seeds_v.size} "
        f"time_ms={ms:.3f}",
    )
    return 0


# -------------------------------------------------------------------------------------------------
# update
# -------------------------------------------------------------------------------------------------


def _read_batches(paths: list[str]) -> list[dyng.EdgeBatch]:
    out: list[dyng.EdgeBatch] = []
    for path in paths:
        with open(path, "rb") as f:
            binary = f.read(8) == b"DYNGBAT1"
        if binary:
            for b in dyng.io.read_binary_batches(path):
                if not isinstance(b, dyng.EdgeBatch):
                    raise CliError(f"{path}: holds hypergraph batches, not graph batches")
                out.append(b)
        else:
            out.extend(dyng.io.read_batches(path, weight_dtype="float32"))
    return out


def _auto_capacity(batches: list[dyng.EdgeBatch]) -> np.ndarray:
    """Each vertex's edges over every batch (a row's appends), for the vertices the batches
    name."""
    n = 0
    for b in batches:
        for a in (b.insert_src, b.insert_dst, b.insert_vertices, b.delete_vertices):
            if a is not None and a.size:
                n = max(n, int(a.max()) + 1)
    cap = np.zeros(n, dtype=np.int64)
    for b in batches:
        if b.num_insertions:
            np.add.at(cap, b.insert_src.astype(np.int64), 1)
            np.add.at(cap, b.insert_dst.astype(np.int64), 1)
    return cap


def _average_weight(batches: list[dyng.EdgeBatch]) -> float:
    """The average weight of the batches' edges, as the original's computeAvgEdgeWeight of the
    full graph (each undirected edge in both directions, a double sum, rounded to float; 0.5
    without edges)."""
    total = 0.0
    count = 0
    for b in batches:
        w = b.insert_weights
        if w is not None and w.size:
            total += 2.0 * float(np.asarray(w, dtype=np.float64).sum())
            count += 2 * int(w.shape[0])
    return 0.5 if count == 0 else float(np.float32(total / count))


def _read_true_labels(path: str, n: int) -> np.ndarray:
    out = np.full(n, -1, dtype=np.int8)
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            try:
                v, c = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                raise CliError(
                    f"{path}:{number}: expected 'i class', got {line.strip()!r}"
                ) from None
            if not 0 <= v < n:
                raise CliError(f"{path}:{number}: vertex {v} is out of range [0, {n})")
            out[v] = c
    return out


def _accuracy(
    labels: np.ndarray, truth: np.ndarray, status: np.ndarray, seeds: np.ndarray
) -> float:
    """The original's computeAccuracy: the active vertices without a seed label (its
    GROUND_TRUTH state is 'active with a seed label'), predicted class (F >= 0.5) against the
    true class; 1 when there is none."""
    mask = (status == int(dyng.VertexStatus.active)) & (seeds < 0) & (truth >= 0)
    total = int(mask.sum())
    if total == 0:
        return 1.0
    correct = int(((labels[mask] >= np.float32(0.5)) == (truth[mask] >= 1)).sum())
    return float(np.float32(correct) / np.float32(total))


def run_update(args: argparse.Namespace) -> int:
    res = _resources(args)
    stream: dyng.generators.VertexArrivalStream | None = None
    if (args.generate is None) == (args.batches is None):
        raise UsageError("update: give --generate legacy|library or --batches FILE (one of them)")
    if args.generate is not None:
        kw = function_kwargs(args, args.generator_names)
        make = (
            dyng.generators.legacy.dynlp_vertex_arrival_stream
            if args.generate == "legacy"
            else dyng.generators.vertex_arrival_stream
        )
        stream = make(resources=dyng.Resources.sequential(), **kw)
        batches = stream.batches
        capacity: np.ndarray = stream.row_capacity
        truth: np.ndarray | None = stream.true_labels
    else:
        if any(getattr(args, "arg_" + n) is not None for n in args.generator_names):
            raise UsageError("update: the generator flags go with --generate, not --batches")
        batches = _read_batches(args.batches)
        if args.capacity == "auto":
            capacity = _auto_capacity(batches)
        else:
            capacity = np.loadtxt(args.capacity, dtype=np.int64, ndmin=1)
        truth = None
        if args.true_labels is not None:
            truth = _read_true_labels(args.true_labels, int(capacity.size))
    tau_word = _tau(args.opt_similarity_threshold, "published")
    if tau_word == "published":
        tau: float | None = (
            stream.similarity_threshold if stream is not None else _average_weight(batches)
        )
    elif tau_word == "current":
        tau = None
    else:
        tau = float(tau_word)
    opt = _options(args, tau)
    g = dyng.Graph.with_capacity(capacity, layout=args.layout, resources=res)
    r = dyng.label_propagation.compute(g, options=opt)
    if stream is not None:
        r.reserve(stream.max_batch_vertices, stream.max_batch_edges)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, b in enumerate(batches):
        t = Stopwatch()
        st = dyng.label_propagation.update(g, b, r)
        ms = t.ms()
        e = Stopwatch()
        labels = r.labels.to_numpy()
        accuracy = ""
        if truth is not None:
            acc = _accuracy(labels, truth, g.vertex_status(), r.seed_labels.to_numpy())
            accuracy = f"{acc:.6f}"
        eval_ms = e.ms()
        if args.per_batch_labels:
            dyng.io.write_labels(out / f"dynlp_labels_b{i}.txt", labels)
        del labels  # no export outlives the batch: the next update writes in place
        rows.append(
            [
                str(i),
                f"{ms:.3f}",
                str(st.iterations),
                str(st.frontier_visits),
                str(st.components),
                accuracy,
                f"{eval_ms:.3f}",
                str(st.affected),
                str(st.initial_frontier),
                str(st.sparsified_edges),
                str(int(st.converged)),
                str(int(st.fallback_used)),
                str(st.batch.inserted_vertices),
                str(st.batch.deleted_vertices),
                str(st.batch.inserted_edges),
                str(st.batch.dropped_insertions),
            ]
        )
        info(
            args,
            f"batch {i}: {ms:.3f} ms, {st.iterations} iterations, {st.components} components"
            + (f", accuracy {float(accuracy) * 100:.2f}%" if accuracy else ""),
        )
    dyng.io.write_labels(out / "dynlp_labels.txt", r.labels)
    with open(out / "stats.csv", "w", encoding="utf-8", newline="") as f:
        f.write(",".join(STATS_COLUMNS) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    return 0


# -------------------------------------------------------------------------------------------------
# dyng generate vertex_arrival_stream | dynlp_vertex_arrival_stream
# -------------------------------------------------------------------------------------------------


def add_generate_vertex_arrival(gens: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The two vertex-arrival generators of ``dyng generate`` (AD12, E1: the Python names)."""
    for name, fn, what in (
        (
            "vertex_arrival_stream",
            dyng.generators.vertex_arrival_stream,
            "the library's vertex-arrival generator (dyng.generators.vertex_arrival_stream: "
            "library-owned draws, bit-exact on every platform)",
        ),
        (
            "dynlp_vertex_arrival_stream",
            dyng.generators.legacy.dynlp_vertex_arrival_stream,
            "LabelPropagation-CUDA's generator (dyng.generators.legacy."
            "dynlp_vertex_arrival_stream: the original's stream, byte for byte)",
        ),
    ):
        g = gens.add_parser(
            name,
            help=what.split(" (")[0] + " (-> batches, capacities, tau, true labels)",
            description=f"Generate {what}. Writes DIR/batches.dgb (or .dgt), "
            "DIR/row_capacity.txt (one integer per vertex: the full degrees), "
            "DIR/true_labels.txt ('i class'), DIR/stream.json (the published threshold "
            "similarity_threshold and the sizes) and, with --graph-files, the original's "
            "DIR/graph/csr{RowPtr,ColInd,Values,TrueLabels}.txt.",
        )
        names = add_function_flags(g, fn, "generator", skip=("resources", "edge_dtype"))
        # Python's edge_dtype (any NumPy dtype-like) is --edge-type int32|int64 on the command
        # line, as the 0.3 commands spell it (the author's decision of 2026-11-14).
        g._action_groups[-1].add_argument(
            "--edge-type",
            dest="arg_edge_dtype",
            choices=("int32", "int64"),
            default=None,
            help="the edge offset type of the stream's graph (default: int32)",
        )
        names.append("edge_dtype")
        g.add_argument("--out", required=True, metavar="DIR", help="output directory")
        g.add_argument(
            "--format",
            choices=("dgb", "dgt"),
            default="dgb",
            help="the batch file: dgb (binary, the default) or dgt (text)",
        )
        g.add_argument(
            "--graph-files",
            action="store_true",
            help="also write the original's graph text files into DIR/graph/",
        )
        g.add_argument("--quiet", action="store_true", help="no report line")
        g.set_defaults(func=run_generate_vertex_arrival, generator_names=names, generator=name)


def run_generate_vertex_arrival(args: argparse.Namespace) -> int:
    fn = (
        dyng.generators.vertex_arrival_stream
        if args.generator == "vertex_arrival_stream"
        else dyng.generators.legacy.dynlp_vertex_arrival_stream
    )
    kw = function_kwargs(args, args.generator_names)
    s = fn(resources=dyng.Resources.sequential(), **kw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    batch_file = out / f"batches.{args.format}"
    if args.format == "dgb":
        dyng.io.write_binary_batches(batch_file, s.batches)
    else:
        dyng.io.write_batches(batch_file, s.batches)
    np.savetxt(out / "row_capacity.txt", s.row_capacity, fmt="%d")
    with open(out / "true_labels.txt", "w", encoding="utf-8", newline="") as f:
        f.writelines(f"{i} {int(c)}\n" for i, c in enumerate(s.true_labels.tolist()))
    meta: dict[str, Any] = {
        "generator": args.generator,
        "num_vertices": s.num_vertices,
        "num_edges": int(s.col_ind.size) // 2,
        "num_batches": s.num_batches,
        "similarity_threshold": s.similarity_threshold,
        "max_batch_vertices": s.max_batch_vertices,
        "max_batch_edges": s.max_batch_edges,
        "options": {k: v for k, v in kw.items()},
    }
    (out / "stream.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    if args.graph_files:
        s.write_graph_files(out / "graph")
    info(
        args,
        f"{args.generator}: n={s.num_vertices} edges={meta['num_edges']} "
        f"batches={s.num_batches} similarity_threshold={s.similarity_threshold!r} -> {out}",
    )
    return 0
