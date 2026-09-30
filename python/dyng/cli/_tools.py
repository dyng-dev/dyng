# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""``dyng convert`` (between the 0.1 graph formats) and ``dyng generate`` (the legacy
generators of :mod:`dyng.generators.legacy`)."""

from __future__ import annotations

import argparse
from pathlib import Path

import dyng

from ._algorithms import batch_text
from ._common import (
    UsageError,
    add_function_flags,
    add_graph_flags,
    add_reader_flags,
    function_kwargs,
    graph_format,
    info,
    load_graph,
    read_edges,
    write_text,
)

# -------------------------------------------------------------------------------------------------
# convert
# -------------------------------------------------------------------------------------------------

_CONVERT_HELP = """\
Convert a graph between the formats of dynG 0.1: MOSP's text CSR (csr: a path prefix of
<prefix>RowPtr.txt, ColInd.txt, Values.txt), Matrix Market (mtx) and edge lists (edges: one
'src dst [w1 .. wK]' line per edge). mtx -> csr with --random-weights MIN,MAX,SEED and
--num-weights K is `mospPrep mtx2csr` (the same bytes). The edges keep the reader's order (csr:
MOSP's rows as stored; mtx: sorted and deduplicated; edges: by source, in file order).
"""


def add_convert(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("convert", help="convert a graph between formats", description=_CONVERT_HELP)
    p.add_argument("input", metavar="INPUT", help="the input graph (see --format)")
    p.add_argument("output", metavar="OUTPUT", help="the output (a prefix for csr)")
    add_reader_flags(p, default_properties=None)
    p.add_argument(
        "--to",
        choices=("auto", "csr", "mtx", "edges"),
        default="auto",
        help="output format (auto: mtx for *.mtx, else edges; csr must be asked for)",
    )
    p.add_argument(
        "--weight-column",
        type=int,
        default=0,
        metavar="K",
        help="mtx output: the weight column written as the value (default 0)",
    )
    p.set_defaults(func=run_convert)


def run_convert(args: argparse.Namespace) -> int:
    fmt = graph_format(args.input, args.format)
    to = args.to
    if to == "auto":
        to = "mtx" if args.output.lower().endswith(".mtx") else "edges"
    res = dyng.Resources.sequential()
    edges: dyng.Graph | dyng.io.EdgeList
    if fmt == "csr":
        edges = load_graph(args, res, path=args.input, properties="mosp_compatible")
    else:
        edges = read_edges(args, args.input, fmt)
    if to == "csr":
        if isinstance(edges, dyng.io.EdgeList):
            if edges.weights is None:
                raise UsageError(
                    "csr output needs weights: give --num-weights (edges) or --random-weights (mtx)"
                )
            # mosp_compatible keeps the reader's order and every edge.
            edges = edges.to_graph(
                properties="mosp_compatible", edge_dtype=getattr(args, "edge_type", None)
            )
        dyng.io.write_csr_triplet(args.output, edges)
    elif to == "mtx":
        dyng.io.write_matrix_market(args.output, edges, weight_column=args.weight_column)
    else:
        dyng.io.write_edge_list(args.output, edges)
    n = edges.num_vertices
    m = edges.num_edges
    print(f"convert: {fmt} -> {to}: n={n} edges={m}")
    return 0


# -------------------------------------------------------------------------------------------------
# generate
# -------------------------------------------------------------------------------------------------

_GENERATE_HELP = """\
The seeded generators of dyng.generators.legacy, bit-exact with the original tools: MOSP's change
generator (mosp_changes: insert.txt and delete.txt, as `mospPrep changes` writes them) and
CycleEnumeration-GPU's batch generator (cycle_enum_batch: '- u v' per deletion, then '+ u v' per
insertion, in graph ids). The flags are the generators' keyword arguments.
"""


def add_generate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("generate", help="seeded batch generators", description=_GENERATE_HELP)
    gens = p.add_subparsers(dest="generator", metavar="GENERATOR", required=True)

    g = gens.add_parser(
        "mosp_changes",
        help="MOSP's change generator (-> insert.txt, delete.txt)",
        description="MOSP's generateChangeBatch (dyng.generators.legacy.mosp_changes). Writes "
        "<out>/insert.txt and <out>/delete.txt and prints the report of `mospPrep changes`.",
    )
    add_graph_flags(g, default_properties="mosp_compatible")
    names = add_function_flags(g, dyng.generators.legacy.mosp_changes, "generator")
    g.add_argument("--out", required=True, metavar="DIR", help="output directory")
    g.add_argument("--quiet", action="store_true", help="no report line")
    g.set_defaults(func=run_mosp_changes, generator_names=names)

    g = gens.add_parser(
        "cycle_enum_batch",
        help="CycleEnumeration-GPU's batch generator (-> '- u v' / '+ u v' text)",
        description="CycleEnumeration-GPU's generate_batch "
        "(dyng.generators.legacy.cycle_enum_batch): the batch in graph ids, deletions first.",
    )
    add_graph_flags(g, default_properties="cycle_enum_compatible", id_types=False, auto_mtx=False)
    names = add_function_flags(g, dyng.generators.legacy.cycle_enum_batch, "generator")
    g.add_argument(
        "--out", default=None, metavar="FILE", help="output file (default: standard output)"
    )
    g.set_defaults(func=run_cycle_enum_batch, generator_names=names)


def run_mosp_changes(args: argparse.Namespace) -> int:
    g = load_graph(args, dyng.Resources.sequential())
    kw = function_kwargs(args, args.generator_names)
    kw.setdefault("num_changes", 0)
    batch, report = dyng.generators.legacy.mosp_changes(g, **kw)
    out = Path(args.out)
    dyng.io.write_legacy_batch(out / "insert.txt", out / "delete.txt", batch)
    info(args, f"changes: {report.summary}")
    return 0


def run_cycle_enum_batch(args: argparse.Namespace) -> int:
    g = load_graph(args, dyng.Resources.sequential())
    batch = dyng.generators.legacy.cycle_enum_batch(
        g, **function_kwargs(args, args.generator_names)
    )
    write_text(args.out, batch_text(batch))
    return 0
