# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""``dyng triad_count compute|update``: hypergraph triad counts (ESCHER-GPU).

The hypergraph is a ``.hg`` file read with the flags of :func:`dyng.io.read_hypergraph`; the
batches of ``update`` are ``.dgt`` or ``.dgb`` files applied in order. Two output formats:
``--format json`` (the default) writes the triad-bins JSON of :func:`dyng.io.write_triad_bins`
(the counts after the last batch and, for ``update``, every batch's deltas), and
``--format escher`` prints the lines ESCHER-GPU@abcf9b2's ``build/main``, ``type1``, ``type2``
and ``type3`` print (the ``Hypergraph:`` line, the baseline count, and per batch the ``Batch``
line, when there is more than one batch, and the delta count), byte for byte.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

import dyng

from ._common import (
    CliError,
    Stopwatch,
    UsageError,
    add_backend_flags,
    add_function_flags,
    add_option_flags,
    function_kwargs,
    make_resources,
    note,
    options_from,
    write_text,
)

_HELP = """\
Exact triad counts of a dynamic hypergraph (dyng.triad_count; ESCHER / ESCHER+). The hypergraph
is a .hg file (one hyperedge per line; ESCHER-GPU's loader by default, --read-options), the
batches of `update` .dgt or .dgb files. --format json writes the triad-bins JSON (the counts
and every batch's deltas); --format escher prints ESCHER-GPU's lines ("Hypergraph: ...", "Motif
counts (30 bins): ...", "Motif delta counts (30 bins): ...", "[GPU] TypeN ..."), byte for byte
as its build/main, type1, type2 and type3 print them.
"""

# The read flags come from dyng.io.read_hypergraph()'s keywords, but for these: the option set
# is --read-options (the keyword `options` would read as the algorithm's options), the weights
# are ignored by triad_count, and the properties are given field by field.
_READ_SKIP = ("options", "properties", "resources", "random_weights")

# ESCHER-GPU's lines (src/main.cu MotifCounter::print, src/type1.cu, type2.cu, type3.cu).
_BASELINE = {
    "stathyper_type1": "[GPU] Type1 baseline motif count (sum_e C(|e|, 3)): ",
    "stathyper_type2": "[GPU] Type2 baseline motif count: ",
    "stathyper_type3": "[GPU] Type3 baseline motif count: ",
}
_DELTA = {
    "stathyper_type1": "[GPU] Type1 motif delta count: ",
    "stathyper_type2": "[GPU] Type2 motif delta count: ",
    "stathyper_type3": "[GPU] Type3 motif delta count: ",
}


def escher_count_line(kind: str, values: Any, *, delta: bool) -> str:
    """One count line as ESCHER-GPU prints it (with the newline)."""
    v = [int(x) for x in np.asarray(values).reshape(-1)]
    if kind == "h_motif":
        head = "Motif delta counts (30 bins): " if delta else "Motif counts (30 bins): "
        return head + " ".join(str(x) for x in v) + "\n"
    return (_DELTA if delta else _BASELINE)[kind] + str(v[0]) + "\n"


def escher_hypergraph_line(h: dyng.Hypergraph) -> str:
    """ESCHER-GPU's ``Hypergraph:`` line (src/driver.cu): the hyperedges, the largest vertex id
    of its 1-based ids (dynG's vertex count) and the pins."""
    pins = int(h.h2v().col_ind.size)
    return (
        f"Hypergraph: {h.num_hyperedges} hyperedges, max vertex id {h.num_vertices}, {pins} pins\n"
    )


def escher_batch_line(index: int, batch: dyng.HyperedgeBatch) -> str:
    """ESCHER-GPU's ``Batch`` line (printed when a run has more than one batch)."""
    return f"Batch {index} ({batch.num_deletions} deletions, {batch.num_insertions} insertions)\n"


def _add_common(p: argparse.ArgumentParser) -> list[str]:
    g = p.add_argument_group("hypergraph input")
    g.add_argument(
        "--input",
        "--hypergraph",
        dest="input",
        required=True,
        metavar="PATH",
        help="the hypergraph: a .hg file (one hyperedge per line; docs/api/file_formats.md)",
    )
    g.add_argument(
        "--read-options",
        choices=("escher_gpu", "hsosp"),
        default=None,
        help="the loader's option set (dyng.io.read_hypergraph's options=; default: escher_gpu, "
        "ESCHER-GPU's loader); the flags below override its fields",
    )
    p.set_defaults(
        read_names=add_function_flags(
            p,
            dyng.io.read_hypergraph,
            "hypergraph read flags",
            skip=_READ_SKIP,
            see=dyng.io.HypergraphReadOptions,
            defaults={
                n: "the option set's" for n in ("index_base", "max_cardinality", "num_vertices")
            },
        )
    )
    p.add_argument(
        "--min-initial-payload",
        type=int,
        default=None,
        metavar="N",
        help="with --capacity escher_gpu: CapacityPolicy.escher_gpu(N), ESCHER-GPU's "
        "--payload-capacity (default 0)",
    )
    names = add_option_flags(p, dyng.triad_count.Options, "triad_count options")
    p.add_argument(
        "--format",
        choices=("json", "escher"),
        default="json",
        help="json: the triad-bins JSON (dyng.io.write_triad_bins); escher: ESCHER-GPU's count "
        "lines (default: json)",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="write the output to FILE (default: standard output)",
    )
    add_backend_flags(p)
    return names


def add_triad_count(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("triad_count", help="hypergraph triad counts", description=_HELP)
    verbs = p.add_subparsers(dest="verb", metavar="VERB", required=True)
    c = verbs.add_parser(
        "compute",
        help="the triad counts of a hypergraph (= ESCHER-GPU's --batches 0)",
        description="Count the closed triads of the hypergraph by --kind.",
    )
    names = _add_common(c)
    c.set_defaults(func=run_triad_compute, option_names=names)
    u = verbs.add_parser(
        "update",
        help="apply batches and update the counts (= ESCHER-GPU's batches)",
        description="Compute the counts, then apply the batches of the --batches files in order, "
        "updating the counts after each. With .dgb files that carry the expected ids, every "
        "batch's inserted ids must be them (exit status 1 otherwise: the ids diverged, and later "
        "deletions would name other hyperedges). Standard error gets update_seconds= (and with "
        "--compare-recompute recompute_seconds= and match=yes|no).",
    )
    names = _add_common(u)
    u.add_argument(
        "--batches",
        nargs="+",
        required=True,
        metavar="FILE",
        help="the batch files (.dgt hyperedge batches or .dgb binary batches), applied in order",
    )
    u.add_argument(
        "--compare-recompute",
        action="store_true",
        help="recompute the counts of the updated hypergraph and compare (exit status 1 if the "
        "two differ)",
    )
    u.set_defaults(func=run_triad_update, option_names=names)


def _load(args: argparse.Namespace, res: dyng.Resources) -> dyng.Hypergraph:
    kw = function_kwargs(args, args.read_names)
    if args.min_initial_payload is not None:
        if kw.get("capacity") != "escher_gpu":
            raise UsageError("--min-initial-payload needs --capacity escher_gpu")
        kw["capacity"] = dyng.CapacityPolicy.escher_gpu(args.min_initial_payload)
    return dyng.io.read_hypergraph(args.input, options=args.read_options, resources=res, **kw)


def _options(args: argparse.Namespace) -> dyng.triad_count.Options:
    return options_from(args, dyng.triad_count.Options, args.option_names)


def _batches(args: argparse.Namespace, h: dyng.Hypergraph) -> list[tuple[Any, np.ndarray | None]]:
    """Every batch of the files, in order, with the expected ids of a .dgb record (or None)."""
    out: list[tuple[Any, np.ndarray | None]] = []
    for path in args.batches:
        with open(path, "rb") as f:
            binary = f.read(8) == b"DYNGBAT1"
        if binary:
            for rec in dyng.io.BatchReader(path, num_vertices=h.num_vertices):
                if not isinstance(rec.batch, dyng.HyperedgeBatch):
                    raise CliError(f"{path}: holds graph batches, not hypergraph batches")
                out.append((rec.batch, rec.expected_ids))
        else:
            for b in dyng.io.read_hyperedge_batches(
                path, num_vertices=h.num_vertices, weighted=h.weighted
            ):
                out.append((b, None))
    return out


def _write(args: argparse.Namespace, text: str) -> None:
    write_text(args.output, text)


def run_triad_compute(args: argparse.Namespace) -> int:
    res = make_resources(args)
    h = _load(args, res)
    r = dyng.triad_count.compute(h, options=_options(args))
    if args.format == "escher":
        kind = str(r.kind)
        _write(args, escher_hypergraph_line(h) + escher_count_line(kind, r.counts, delta=False))
    else:
        _write(args, dyng.io.triad_bins_json(r))
    return 0


def run_triad_update(args: argparse.Namespace) -> int:
    res = make_resources(args)
    h = _load(args, res)
    opt = _options(args)
    r = dyng.triad_count.compute(h, options=opt)
    kind = str(r.kind)
    batches = _batches(args, h)
    lines = []
    if args.format == "escher":
        lines += [escher_hypergraph_line(h), escher_count_line(kind, r.counts, delta=False)]
    deltas: list[np.ndarray] = []
    seconds = 0.0
    for index, (batch, expected) in enumerate(batches, 1):
        t = Stopwatch()
        st = dyng.triad_count.update(h, batch, r)
        seconds += t.ms() / 1e3
        ids = st.batch.inserted_ids
        if expected is not None and not np.array_equal(ids, expected):
            raise CliError(
                f"batch {index}: the inserted ids {ids.tolist()[:8]} differ from the file's "
                f"expected ids {np.asarray(expected).tolist()[:8]} (build the hypergraph as the "
                "file's run did: --id-reuse, --capacity, --min-initial-payload, --num-vertices)"
            )
        deltas.append(st.delta)
        if args.format == "escher":
            if len(batches) > 1:
                lines.append(escher_batch_line(index, batch))
            lines.append(escher_count_line(kind, st.delta, delta=True))
    note(f"update_seconds={seconds:.6g}")
    status = 0
    if args.compare_recompute:
        t = Stopwatch()
        again = dyng.triad_count.compute(h, options=opt)
        note(f"recompute_seconds={t.ms() / 1e3:.6g}")
        match = again.counts.tolist() == r.counts.tolist()
        note(f"match={'yes' if match else 'no'}")
        status = 0 if match else 1
    if args.format == "escher":
        _write(args, "".join(lines))
    else:
        _write(args, dyng.io.triad_bins_json(r, deltas=deltas))
    return status
