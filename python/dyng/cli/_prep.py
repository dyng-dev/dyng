# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
# Derived from MOSP-OpenMP@c352151:src/mospPrep.cpp (the subcommands, their arguments, checks
# and report lines) and src/csrGraph.cpp (saveCsrGraphBinary, csrSourceIdentity)
"""``dyng prep``: MOSP's ``mospPrep`` subcommands on dynG, with the same arguments and outputs.

``mtx2csr``, ``widen``, ``cache``, ``changes``, ``init`` and ``expected`` take the positional
arguments and flags of ``mospPrep`` (MOSP-OpenMP c352151, MOSP-CUDA e220ee2: the same tool) and
write the same files byte for byte (the binary cache of ``cache`` included); the report lines on
standard output are the original's (the times differ). A usage error exits with status 2, a
failure with status 1 (after the original's ``<command> done in <ms> ms (rc=1)`` line).
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

import dyng
from dyng._backend import native

from ._common import (
    CliError,
    Stopwatch,
    UsageError,
    add_backend_flags,
    csr_objectives,
    fmt_ms,
    make_resources,
    write_tree,
)

_INT_MAX = 2**31 - 1

#: MOSP's binary cache format version 2 (csrGraph.cpp, kBinaryMagic).
CACHE_MAGIC = b"MOSPCSR2"

#: libstdc++'s std::filesystem::file_time_type counts nanoseconds from 2174-01-01 (its
#: __file_clock::_S_epoch_diff, 6437664000 s after the Unix epoch).
_FILE_CLOCK_EPOCH_NS = 6_437_664_000 * 1_000_000_000

_DESCRIPTION = """\
MOSP's input preparation (mospPrep of MOSP-OpenMP c352151 / MOSP-CUDA e220ee2) on dynG: the same
arguments, the same output files byte for byte, the same report lines.
"""


def _int_arg(text: str, name: str, lo: int, hi: int) -> int:
    """mospPrep's parseIntArg(): a whole decimal integer in [lo, hi]."""
    try:
        value = int(text, 10)
    except ValueError:
        value = None
    if value is None or not lo <= value <= hi:
        raise UsageError(f"{name} must be an integer in [{lo}, {hi}], got '{text}'")
    return value


def _seed(text: str) -> int:
    """A seed as ``static_cast<unsigned>(atoll(text))`` / ``strtoul`` reads it (modulo 2^32)."""
    try:
        return int(text, 10) & 0xFFFFFFFF
    except ValueError:
        raise UsageError(f"seed: expected an integer, got {text!r}") from None


def _weights(args: argparse.Namespace) -> tuple[int, int, int]:
    k = _int_arg(args.K, "K", 1, 32)
    wmin = _int_arg(args.wmin, "wmin", 1, _INT_MAX)
    wmax = _int_arg(args.wmax, "wmax", wmin, _INT_MAX)
    return k, wmin, wmax


def _read_csr(prefix: str, k: int, res: dyng.Resources | None = None) -> dyng.Graph:
    """readCsrGraph() + resolveObjectives(): MOSP's rows byte for byte."""
    return dyng.io.read_csr_triplet(
        prefix,
        num_weights=csr_objectives(prefix, k),
        properties="mosp_compatible",
        resources=res or dyng.Resources.sequential(),
    )


def _write_csr(prefix: str, row_ptr: np.ndarray, col_ind: np.ndarray, w: np.ndarray) -> None:
    """writeCsrGraph() of (row_ptr, col_ind, w) with w of shape (m, K)."""
    k = int(w.shape[1])
    w_km = np.ascontiguousarray(w.T, dtype=np.int32).reshape(-1)
    native.write_csr_triplet(os.fspath(prefix), row_ptr, col_ind, w_km, k)


# -------------------------------------------------------------------------------------------------
# The subcommands
# -------------------------------------------------------------------------------------------------


def _mtx2csr(args: argparse.Namespace) -> int:
    k, wmin, wmax = _weights(args)
    seed = _seed(args.seed)
    with open(args.input, "rb") as f:
        banner = f.readline()
    symmetric = b"symmetric" in banner  # the original's strstr() on the first line
    edges = dyng.io.read_matrix_market_arrays(
        args.input, num_weights=k, random_weights=(wmin, wmax, seed)
    )
    g = edges.to_graph(properties="mosp_compatible", resources=dyng.Resources.sequential())
    dyng.io.write_csr_triplet(args.out_prefix, g)
    print(f"mtx2csr: n={g.num_vertices} directed edges={g.num_edges} symmetric={int(symmetric)}")
    return 0


def _widen(args: argparse.Namespace) -> int:
    k, wmin, wmax = _weights(args)
    seed = _seed(args.seed)
    g = _read_csr(args.in_prefix, k)
    csr = g.to_csr()
    m = int(csr.col_ind.size)
    if m > 0 and k < g.num_weights:
        raise CliError(f"graph already has {g.num_weights} objectives")
    old = (
        np.asarray(csr.weights, dtype=np.int32)
        if csr.weights is not None and m > 0
        else np.zeros((m, 0), dtype=np.int32)
    )
    extra = k - int(old.shape[1])
    draws = native.legacy_mosp_weights(m * extra, wmin, wmax, seed).reshape(m, extra)
    _write_csr(args.out_prefix, csr.row_ptr, csr.col_ind, np.hstack([old, draws]))
    return 0


def cache_identity(prefix: str) -> bytes:
    """csrSourceIdentity(): the canonical path, size and modification time (libstdc++'s file
    clock, nanoseconds) of the three text files, one per line (b"" if one is missing)."""
    out = []
    for suffix in ("RowPtr.txt", "ColInd.txt", "Values.txt"):
        path = os.path.realpath(prefix + suffix)
        try:
            st = os.stat(path)
        except OSError:
            return b""
        out.append(f"{path}\n{st.st_size}\n{st.st_mtime_ns - _FILE_CLOCK_EPOCH_NS}\n")
    return "".join(out).encode()


def write_cache(path: str, g: dyng.Graph, source_prefix: str) -> None:
    """saveCsrGraphBinary(): MOSP's binary cache of ``g`` (the graph read from
    ``source_prefix``)."""
    csr = g.to_csr()
    k = g.num_weights if g.weighted else 0
    m = int(csr.col_ind.size)
    identity = cache_identity(source_prefix) if source_prefix else b""
    w = (
        np.zeros(0, dtype="<i4")
        if csr.weights is None or m == 0
        else np.ascontiguousarray(csr.weights, dtype="<i4").reshape(-1)  # edge-major, as MOSP
    )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(CACHE_MAGIC)
        f.write(struct.pack("<I", len(identity)))
        f.write(identity)
        f.write(struct.pack("<iiq", g.num_vertices, k, m))
        f.write(np.ascontiguousarray(csr.row_ptr, dtype="<i4").tobytes())
        f.write(np.ascontiguousarray(csr.col_ind, dtype="<i4").tobytes())
        f.write(w.tobytes())


def _cache(args: argparse.Namespace) -> int:
    g = _read_csr(args.csr_prefix, 0)
    write_cache(args.binary_path, g, args.csr_prefix)
    return 0


def _changes(args: argparse.Namespace) -> int:
    k = 0 if args.k is None else _int_arg(args.k, "-k", 1, 32)
    for flag, value in (("--wmin", args.wmin), ("--wmax", args.wmax)):
        if value is not None:
            _int_arg(value, flag, 1, _INT_MAX)
    g = _read_csr(args.csr_prefix, k)
    batch, report = dyng.generators.legacy.mosp_changes(
        g,
        num_changes=args.changes,
        insertion_percentage=args.ins,
        mode=args.mode,
        weight_min=1 if args.wmin is None else int(args.wmin),
        weight_max=100 if args.wmax is None else int(args.wmax),
        seed=_seed(args.seed),
        local_hops=args.local,
        safe_deletions=args.safe,
        source=args.source,
    )
    out = Path(args.out_dir)
    dyng.io.write_legacy_batch(out / "insert.txt", out / "delete.txt", batch)
    print(f"changes: {report.summary}")
    return 0


def _dijkstra_all(g: dyng.Graph, source: int, out: str, suffix: str) -> None:
    """dijkstraAll(): one tree per objective (dynG's compute: the same canonical tree)."""
    for k in range(max(g.num_weights, 1)):
        t = Stopwatch()
        tree = dyng.sssp.compute(g, source, objective=k)
        write_tree(out, k, suffix, tree)
        print(f"dijkstra obj{k}: {fmt_ms(t.ms())} ms")


def _tree_k(args: argparse.Namespace) -> int:
    return 0 if args.k is None else _int_arg(args.k, "-k", 1, 32)


def _init(args: argparse.Namespace) -> int:
    k = _tree_k(args)
    g = _read_csr(args.csr_prefix, k, make_resources(args))
    _dijkstra_all(g, args.source, args.out_dir, "Original")
    return 0


def _expected(args: argparse.Namespace) -> int:
    k = _tree_k(args)
    g = _read_csr(args.csr_prefix, k, make_resources(args))
    changes = Path(args.changes_dir)
    batch = dyng.io.read_legacy_batch(
        changes / "insert.txt",
        changes / "delete.txt",
        num_weights=g.num_weights,
        num_vertices=g.num_vertices,
        mosp_lenient=True,
        vertex_dtype=g.vertex_dtype,
    )
    g.apply(batch)  # applyChangeBatch() (graph_properties::mosp_compatible)
    _dijkstra_all(g, args.source, args.out_dir, "Updated")
    return 0


# -------------------------------------------------------------------------------------------------
# The parser
# -------------------------------------------------------------------------------------------------


def _tree_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", type=int, default=0, metavar="s", help="source vertex (default 0)")
    p.add_argument(
        "-k",
        default=None,
        metavar="K",
        help="objectives of a graph without edges (its empty Values file cannot tell)",
    )
    add_backend_flags(p)


def add_prep(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("prep", help="MOSP's input preparation (mospPrep)", description=_DESCRIPTION)
    cmds = p.add_subparsers(dest="prep_command", metavar="COMMAND", required=True)

    c = cmds.add_parser(
        "mtx2csr",
        help="Matrix Market -> MOSP CSR with K seeded random weights",
        description="SuiteSparse Matrix Market file -> CSR text. Symmetric matrices get both "
        "edge directions; self-loops and duplicate edges are dropped; every edge gets K uniform "
        "random weights in [wmin, wmax] (1 <= wmin <= wmax <= 2^31-1, 1 <= K <= 32).",
    )
    c.add_argument("input", metavar="in.mtx")
    c.add_argument("out_prefix", metavar="outPrefix")
    for name in ("K", "wmin", "wmax", "seed"):
        c.add_argument(name)
    c.set_defaults(func=_mtx2csr)

    c = cmds.add_parser(
        "widen",
        help="append seeded random objectives until the graph has K",
        description="Copy a graph and append random objectives until it has K (the existing "
        "objectives are kept unchanged).",
    )
    c.add_argument("in_prefix", metavar="inPrefix")
    c.add_argument("out_prefix", metavar="outPrefix")
    for name in ("K", "wmin", "wmax", "seed"):
        c.add_argument(name)
    c.set_defaults(func=_widen)

    c = cmds.add_parser(
        "cache",
        help="write MOSP's binary cache (mosp --cache)",
        description="Write the binary cache read by `mosp --cache` (format MOSPCSR2, with the "
        "identity of the three text files).",
    )
    c.add_argument("csr_prefix", metavar="csrPrefix")
    c.add_argument("binary_path", metavar="binaryPath")
    c.set_defaults(func=_cache)

    c = cmds.add_parser(
        "changes",
        help="MOSP's seeded change batch (insert.txt, delete.txt)",
        description="Generate <outDir>/insert.txt and <outDir>/delete.txt with MOSP's change "
        "generator (dyng.generators.legacy.mosp_changes; the modes of changeGenerator.h).",
    )
    c.add_argument("csr_prefix", metavar="csrPrefix")
    c.add_argument("out_dir", metavar="outDir")
    c.add_argument("--changes", type=int, default=0, metavar="N", help="number of changes")
    c.add_argument("--ins", type=float, default=50.0, metavar="PCT", help="share of insertions")
    c.add_argument(
        "--mode", choices=("uniform", "targeted", "reweight", "increase"), default="uniform"
    )
    c.add_argument("--local", type=int, default=0, metavar="HOPS", help="a local batch")
    c.add_argument("--safe", action="store_true", help="connectivity-safe deletions")
    c.add_argument("--seed", default="1", metavar="S")
    c.add_argument("--source", type=int, default=0, metavar="s")
    c.add_argument("--wmin", default=None, metavar="a")
    c.add_argument("--wmax", default=None, metavar="b")
    c.add_argument("-k", default=None, metavar="K", help="objectives of a graph without edges")
    c.set_defaults(func=_changes)

    c = cmds.add_parser(
        "init",
        help="the initial trees (Dijkstra per objective)",
        description="Initial SOSP trees: one per objective -> <outDir>/obj<k>/"
        "distancesOriginal.txt, SSSPTreeOriginal.txt.",
    )
    c.add_argument("csr_prefix", metavar="csrPrefix")
    c.add_argument("out_dir", metavar="outDir")
    _tree_flags(c)
    c.set_defaults(func=_init)

    c = cmds.add_parser(
        "expected",
        help="the ground truth after a batch",
        description="Apply the batch and compute the trees of the updated graph -> "
        "<outDir>/obj<k>/distancesUpdated.txt, SSSPTreeUpdated.txt.",
    )
    c.add_argument("csr_prefix", metavar="csrPrefix")
    c.add_argument("changes_dir", metavar="changesDir")
    c.add_argument("out_dir", metavar="outDir")
    _tree_flags(c)
    c.set_defaults(func=_expected)

    for command in cmds.choices.values():
        command.set_defaults(prep=True)


def run_prep(args: argparse.Namespace) -> int:
    """Run a prep subcommand with mospPrep's final report line."""
    t = Stopwatch()
    rc = 0
    try:
        rc = args.func(args)
    except UsageError:
        raise
    except (dyng.Error, OSError, CliError) as e:
        print(f"dyng: error: {e}", file=sys.stderr)
        rc = 1
    print(f"{args.prep_command} done in {fmt_ms(t.ms())} ms (rc={rc})")
    return rc
