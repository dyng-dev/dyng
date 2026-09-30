# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""``dyng sssp compute|update`` and ``dyng cycle_count compute|update``.

The outputs are the originals' files: MOSP's ``<out>/obj<k>/distances*.txt`` and
``SSSPTree*.txt`` (sssp), CycleEnumeration-GPU's histogram CSV (cycle_count), byte for byte.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

import dyng

from ._common import (
    Stopwatch,
    UsageError,
    add_backend_flags,
    add_function_flags,
    add_graph_flags,
    add_option_flags,
    fmt_ms,
    function_kwargs,
    histogram_text,
    info,
    load_graph,
    make_resources,
    note,
    options_from,
    write_text,
    write_tree,
)

# -------------------------------------------------------------------------------------------------
# Batches
# -------------------------------------------------------------------------------------------------


def add_mosp_batch_flags(p: argparse.ArgumentParser, *, required: bool) -> None:
    """``--changes DIR`` or ``--insert FILE --delete FILE`` (MOSP's batch files)."""
    g = p.add_argument_group("batch (MOSP's insert.txt / delete.txt)")
    g.add_argument(
        "--changes",
        metavar="DIR",
        default=None,
        help="the directory of insert.txt ('u v w1 .. wK' per line) and delete.txt ('u v')",
    )
    g.add_argument("--insert", metavar="FILE", default=None, help="the insertion file")
    g.add_argument("--delete", metavar="FILE", default=None, help="the deletion file")
    p.set_defaults(_batch_required=required)


def mosp_batch_paths(args: argparse.Namespace) -> tuple[Path, Path] | None:
    """The insert and delete files of the batch flags (None if none was given)."""
    if args.changes is not None:
        if args.insert is not None or args.delete is not None:
            raise UsageError("give either --changes or --insert/--delete")
        return Path(args.changes) / "insert.txt", Path(args.changes) / "delete.txt"
    if args.insert is None and args.delete is None:
        if args._batch_required:
            raise UsageError("a batch is required: --changes DIR or --insert FILE --delete FILE")
        return None
    if args.insert is None or args.delete is None:
        raise UsageError("--insert and --delete go together")
    return Path(args.insert), Path(args.delete)


def read_mosp_batch(args: argparse.Namespace, g: dyng.Graph) -> dyng.EdgeBatch | None:
    """The MOSP batch of the flags, read with MOSP's accept/reject rules (`mosp`'s)."""
    paths = mosp_batch_paths(args)
    if paths is None:
        return None
    batch = dyng.io.read_legacy_batch(
        paths[0],
        paths[1],
        num_weights=g.num_weights if g.weighted else 0,
        num_vertices=g.num_vertices,
        mosp_lenient=True,
        vertex_dtype=g.vertex_dtype,
    )
    if not g.weighted:  # "u v" insertion lines: the batch of an unweighted graph
        batch = dyng.EdgeBatch(
            insert=(batch.insert_src, batch.insert_dst), delete=(batch.delete_src, batch.delete_dst)
        )
    return batch


def batch_text(batch: dyng.EdgeBatch) -> str:
    """A batch as the text of CycleEnumeration-GPU's generator (``- u v`` per deletion, then
    ``+ u v`` per insertion)."""
    lines = [
        f"- {u} {v}\n"
        for u, v in zip(batch.delete_src.tolist(), batch.delete_dst.tolist(), strict=True)
    ]
    lines += [
        f"+ {u} {v}\n"
        for u, v in zip(batch.insert_src.tolist(), batch.insert_dst.tolist(), strict=True)
    ]
    return "".join(lines)


def read_text_batch(path: str, g: dyng.Graph) -> dyng.EdgeBatch:
    """Read the text batch of :func:`batch_text` (``+ u v [w1 .. wK]`` / ``- u v``, ``#``
    comments). Insertions without weights into a weighted graph get the weight 1."""
    ins: list[list[int]] = []
    dele: list[tuple[int, int]] = []
    k = g.num_weights if g.weighted else 0
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            op, rest = fields[0], fields[1:]
            try:
                values = [int(x) for x in rest]
            except ValueError:
                raise dyng.FileFormatError(
                    f"{path}:{number}: not an integer: {line.strip()!r}"
                ) from None
            if op == "-" and len(values) == 2:
                dele.append((values[0], values[1]))
            elif op == "+" and len(values) == 2:
                ins.append(values + [1] * k)
            elif op == "+" and k and len(values) == 2 + k:
                ins.append(values)
            else:
                raise dyng.FileFormatError(
                    f"{path}:{number}: expected '+ u v' or '- u v', got {line.strip()!r}"
                )
    # int64 arrays: the batch converts them to the graph's id type with a range check.
    ia = np.array(ins, dtype=np.int64).reshape(-1, 2 + k)
    da = np.array(dele, dtype=np.int64).reshape(-1, 2)
    insert: tuple[np.ndarray, ...] = (ia[:, 0].copy(), ia[:, 1].copy())
    if k:
        insert += (ia[:, 2:].copy(),)
    return dyng.EdgeBatch(insert=insert, delete=(da[:, 0].copy(), da[:, 1].copy()))


# -------------------------------------------------------------------------------------------------
# sssp
# -------------------------------------------------------------------------------------------------

_SSSP_HELP = """\
Dynamic single-source shortest paths (dyng.sssp; the SOSP update of DynaMOSP). The trees are
written in MOSP's layout, <out>/obj<k>/distances<Suffix>.txt and SSSPTree<Suffix>.txt, byte for
byte as MOSP's `mospPrep init` (compute) and `mosp` (update) write them. Without --objective
every weight column (objective) of the graph gets its own tree.
"""


def _objectives(g: dyng.Graph, objective: int | None) -> list[int]:
    if objective is not None:
        return [objective]
    return list(range(max(g.num_weights, 1)))


def _add_sssp_common(p: argparse.ArgumentParser) -> list[str]:
    add_graph_flags(p, default_properties="mosp_compatible")
    p.add_argument("--source", type=int, default=0, metavar="S", help="source vertex (default 0)")
    names = add_option_flags(p, dyng.sssp.Options, "sssp options")
    p.add_argument("--out", required=True, metavar="DIR", help="output directory")
    p.add_argument("--quiet", action="store_true", help="no report lines on standard output")
    add_backend_flags(p)
    return names


def add_sssp(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("sssp", help="dynamic single-source shortest paths", description=_SSSP_HELP)
    verbs = p.add_subparsers(dest="verb", metavar="VERB", required=True)
    c = verbs.add_parser(
        "compute",
        help="the shortest-path trees (= mospPrep init)",
        description="Compute the canonical shortest-path tree (ties to the lowest parent id) of "
        "each objective and write <out>/obj<k>/distancesOriginal.txt and SSSPTreeOriginal.txt, "
        "byte-identical to `mospPrep init`.",
    )
    names = _add_sssp_common(c)
    c.set_defaults(func=run_sssp_compute, option_names=names)
    u = verbs.add_parser(
        "update",
        help="apply a batch and update the trees (= mosp)",
        description="Read (--init) or compute the trees, apply the batch once to the graph and "
        "update every tree with it (dyng.update), then write <out>/obj<k>/distancesUpdated.txt "
        "and SSSPTreeUpdated.txt, byte-identical to MOSP's `mosp` driver.",
    )
    names = _add_sssp_common(u)
    add_mosp_batch_flags(u, required=True)
    u.add_argument(
        "--init",
        metavar="DIR",
        default=None,
        help="the initial trees <init>/obj<k>/distancesOriginal.txt and SSSPTreeOriginal.txt "
        "(default: computed)",
    )
    u.add_argument(
        "--canonicalize",
        action="store_true",
        help="--init: apply the lowest-id tie rule to the initial trees (MOSP's canonicalizeTree)",
    )
    u.add_argument(
        "--write-graph",
        metavar="PREFIX",
        default=None,
        help="also write the updated graph as <PREFIX>{RowPtr,ColInd,Values}.txt",
    )
    u.set_defaults(func=run_sssp_update, option_names=names)


def _sssp_options(args: argparse.Namespace) -> dyng.sssp.Options:
    return options_from(args, dyng.sssp.Options, args.option_names)


def run_sssp_compute(args: argparse.Namespace) -> int:
    res = make_resources(args)
    g = load_graph(args, res)
    opt = _sssp_options(args)
    for k in _objectives(g, args.opt_objective):
        t = Stopwatch()
        tree = dyng.sssp.compute(g, args.source, options=dataclasses.replace(opt, objective=k))
        ms = t.ms()
        write_tree(args.out, k, "Original", tree)
        reachable = int(np.count_nonzero(tree.distances.to_numpy() < dyng.sssp.INFINITE_DISTANCE))
        info(args, f"obj{k}: compute {fmt_ms(ms)} ms (reachable {reachable} of {g.num_vertices})")
    return 0


def run_sssp_update(args: argparse.Namespace) -> int:
    res = make_resources(args)
    g = load_graph(args, res)
    batch = read_mosp_batch(args, g)
    assert batch is not None
    opt = _sssp_options(args)
    objectives = _objectives(g, args.opt_objective)
    trees = []
    for k in objectives:
        ok = dataclasses.replace(opt, objective=k)
        if args.init is None:
            trees.append(dyng.sssp.compute(g, args.source, options=ok))
            continue
        d = Path(args.init) / f"obj{k}"
        dist = dyng.io.read_distances(d / "distancesOriginal.txt", g.num_vertices)
        parents = dyng.io.read_parents(
            d / "SSSPTreeOriginal.txt", g.num_vertices, vertex_dtype=g.vertex_dtype
        )
        trees.append(
            dyng.sssp.Result.from_arrays(
                g, args.source, dist, parents, canonicalize=args.canonicalize, options=ok
            )
        )
    t = Stopwatch()
    stats = dyng.update(g, batch, *trees)
    ms = t.ms()
    for k, tree, st in zip(objectives, trees, stats, strict=True):
        write_tree(args.out, k, "Updated", tree)
        info(
            args,
            f"obj{k}: invalidated {st.invalidated}, affected {st.affected}, iterations "
            f"{st.iterations}, engine {st.engine_used}",
        )
    b = stats[0].batch
    info(
        args,
        f"update {fmt_ms(ms)} ms ({len(trees)} tree(s); batch: inserted {b.inserted_edges}, "
        f"updated {b.updated_edges}, deleted {b.deleted_edges}, ignored deletions "
        f"{b.ignored_deletions})",
    )
    if args.write_graph is not None:
        dyng.io.write_csr_triplet(args.write_graph, g)
    return 0


# -------------------------------------------------------------------------------------------------
# cycle_count
# -------------------------------------------------------------------------------------------------

_CYCLE_HELP = """\
Exact directed simple-cycle histograms (dyng.cycle_count; CycleEnumeration-GPU). The histogram
is printed as CycleEnumeration-GPU's CSV ("# cycle_size, num_of_cycles", "len, count" per
non-zero length, "Total, N"), byte for byte as its `cycle-enum` prints it.
"""


def _add_cycle_common(p: argparse.ArgumentParser) -> list[str]:
    add_graph_flags(p, default_properties="cycle_enum_compatible", id_types=False, auto_mtx=False)
    names = add_option_flags(p, dyng.cycle_count.Options, "cycle_count options")
    p.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="write the histogram CSV to FILE (default: standard output)",
    )
    add_backend_flags(p)
    return names


def add_cycle_count(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "cycle_count", help="directed simple-cycle histograms", description=_CYCLE_HELP
    )
    verbs = p.add_subparsers(dest="verb", metavar="VERB", required=True)
    c = verbs.add_parser(
        "compute",
        help="the histogram of a graph (= cycle-enum --task count)",
        description="Count the directed simple cycles of the graph by length.",
    )
    names = _add_cycle_common(c)
    c.set_defaults(func=run_cycle_compute, option_names=names)
    u = verbs.add_parser(
        "update",
        help="apply a batch and update the histogram (= cycle-enum --task update)",
        description="Compute the histogram, apply a batch (read with --batch or --changes, or "
        "generated as CycleEnumeration-GPU's generate_batch with the flags of `dyng generate "
        "cycle_enum_batch`) and update the histogram; print the updated histogram. Standard "
        "error gets update_seconds= (and with --compare-recompute recompute_seconds= and "
        "match=yes|no), as the original prints them.",
    )
    names = _add_cycle_common(u)
    g = u.add_argument_group("batch")
    g.add_argument(
        "--batch",
        metavar="FILE",
        default=None,
        help="a text batch: '- u v' per deletion, '+ u v' per insertion (the output of `dyng "
        "generate cycle_enum_batch`)",
    )
    add_mosp_batch_flags(u, required=False)
    gen = add_function_flags(
        u, dyng.generators.legacy.cycle_enum_batch, "generated batch (when no file is given)"
    )
    u.add_argument(
        "--write-batch",
        metavar="FILE",
        default=None,
        help="write the batch (text format) before the update",
    )
    u.add_argument(
        "--compare-recompute",
        action="store_true",
        help="recompute the histogram of the updated graph and compare (exit status 1 if the "
        "two differ)",
    )
    u.set_defaults(func=run_cycle_update, option_names=names, generator_names=gen)


def _cycle_options(args: argparse.Namespace) -> dyng.cycle_count.Options:
    return options_from(args, dyng.cycle_count.Options, args.option_names)


def run_cycle_compute(args: argparse.Namespace) -> int:
    res = make_resources(args)
    g = load_graph(args, res)
    hist = dyng.cycle_count.compute(g, options=_cycle_options(args))
    write_text(args.output, histogram_text(hist))
    return 0


def run_cycle_update(args: argparse.Namespace) -> int:
    res = make_resources(args)
    g = load_graph(args, res)
    gen = function_kwargs(args, args.generator_names)
    sources = [args.batch is not None, mosp_batch_paths(args) is not None, bool(gen)]
    if sum(sources) > 1:
        raise UsageError("give one batch: --batch, --changes/--insert/--delete or generator flags")
    opt = _cycle_options(args)
    hist = dyng.cycle_count.compute(g, options=opt)
    batch: dyng.EdgeBatch
    if args.batch is not None:
        batch = read_text_batch(args.batch, g)
    elif sources[1]:
        mosp_batch = read_mosp_batch(args, g)
        assert mosp_batch is not None
        batch = mosp_batch
    else:
        batch = dyng.generators.legacy.cycle_enum_batch(g, **gen)
    if args.write_batch is not None:
        write_text(args.write_batch, batch_text(batch))
    t = Stopwatch()
    dyng.cycle_count.update(g, batch, hist)
    note(f"update_seconds={t.ms() / 1e3:.6g}")
    status = 0
    if args.compare_recompute:
        t = Stopwatch()
        again = dyng.cycle_count.compute(g, options=opt)
        note(f"recompute_seconds={t.ms() / 1e3:.6g}")
        match = again.counts.tolist() == hist.counts.tolist()
        note(f"match={'yes' if match else 'no'}")
        status = 0 if match else 1
    write_text(args.output, histogram_text(hist))
    return status
