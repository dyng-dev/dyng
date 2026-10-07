# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the ``dyng`` command line: errors, resources, option flags, graph loading.

Option flags are generated from the ``Options`` dataclasses (and the generators' keyword
arguments), so every flag is the kebab-case spelling of one field (PLAN Section 5.6): a field
added to ``dyng.sssp.Options`` appears as a flag without a change here.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import os
import re
import sys
import time
import types
import typing
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import dyng

#: The graph formats the commands read (``auto`` picks one from the path).
GRAPH_FORMATS = ("auto", "csr", "mtx", "edges")
PRESETS = ("default", "mosp_compatible", "cycle_enum_compatible")


class CliError(Exception):
    """A failure reported as ``dyng: error: <message>`` with exit status 1."""


class UsageError(CliError):
    """A command-line mistake: exit status 2 (as argparse and ``mospPrep`` use)."""


def fail(message: str) -> typing.NoReturn:
    """Raise :class:`CliError`."""
    raise CliError(message)


def info(args: argparse.Namespace, text: str) -> None:
    """A report line on standard output (suppressed by ``--quiet``)."""
    if not getattr(args, "quiet", False):
        print(text, flush=True)


def note(text: str) -> None:
    """A diagnostic line on standard error."""
    print(text, file=sys.stderr, flush=True)


class Stopwatch:
    """Milliseconds since construction (``time.perf_counter``)."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def ms(self) -> float:
        """Elapsed milliseconds."""
        return (time.perf_counter() - self._start) * 1e3


def fmt_ms(ms: float) -> str:
    """A duration as C++'s default ``operator<<`` of a double prints it (6 significant digits)."""
    return f"{ms:.6g}"


# -------------------------------------------------------------------------------------------------
# Resources
# -------------------------------------------------------------------------------------------------


def add_backend_flags(p: argparse.ArgumentParser) -> None:
    """``--backend``, ``--threads``, ``--device``."""
    g = p.add_argument_group("backend")
    g.add_argument(
        "--backend",
        choices=("sequential", "openmp", "cuda"),
        default=None,
        help="the backend (default: openmp if this build has it, else sequential; cuda needs "
        'a CUDA plugin wheel, from 0.2.0: pip install "dyng[cu13]" or "dyng[cu12]")',
    )
    g.add_argument(
        "--threads",
        type=int,
        default=0,
        metavar="T",
        help="OpenMP threads (0: the OpenMP default, which honours OMP_NUM_THREADS)",
    )
    g.add_argument("--device", type=int, default=0, metavar="D", help="CUDA device (cuda only)")


def make_resources(args: argparse.Namespace) -> dyng.Resources:
    """The resources the backend flags ask for."""
    backend = args.backend
    if backend is None:
        backend = "openmp" if dyng.config()["backends"]["openmp"] else "sequential"
    if args.threads < 0:
        raise UsageError(f"--threads must be >= 0, got {args.threads}")
    if backend == "sequential":
        return dyng.Resources.sequential()
    if backend == "openmp":
        return dyng.Resources.openmp(args.threads)
    return dyng.Resources.cuda(device=args.device, host_threads=args.threads)


# -------------------------------------------------------------------------------------------------
# Flags generated from option fields
# -------------------------------------------------------------------------------------------------


def flag_of(name: str) -> str:
    """The flag of a field: ``max_length`` -> ``--max-length``."""
    return "--" + name.replace("_", "-")


def _clean(text: str) -> str:
    text = re.sub(r":\w+:`~?([^`]+)`", r"\1", text)
    text = text.replace("``", "")
    return " ".join(text.split()).replace("%", "%%")


def doc_section(obj: Any, section: str) -> dict[str, str]:
    """The entries of a Google-style docstring section (``Attributes:`` or ``Args:``)."""
    doc = inspect.getdoc(obj) or ""
    out: dict[str, str] = {}
    lines = doc.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == f"{section}:")
    except StopIteration:
        return out
    indent: int | None = None
    current: str | None = None
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        width = len(line) - len(line.lstrip())
        if indent is None:
            indent = width
        if width < indent:
            break
        m = re.match(r"(\w+)(?: \([^)]*\))?: ?(.*)", line.strip())
        if width == indent and m:
            current = m.group(1)
            out[current] = m.group(2)
        elif current is not None:
            out[current] += " " + line.strip()
    return {k: _clean(v) for k, v in out.items()}


def int_list(text: str) -> list[int]:
    """``"4,1,4"`` -> ``[4, 1, 4]`` (a list flag such as ``--preferences``, MOSP's ``--pref``
    format)."""
    try:
        return [int(x) for x in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers (e.g. 4,1,4), got {text!r}"
        ) from None


def _add_typed_flag(
    g: argparse._ArgumentGroup, name: str, hint: Any, default: Any, help_text: str, dest: str
) -> None:
    flag = flag_of(name)
    origin = typing.get_origin(hint)
    if origin is list:
        g.add_argument(
            flag,
            dest=dest,
            type=int_list,
            default=None,
            metavar="N1,..,NK",
            help=f"{help_text} (default: {default})".strip(),
        )
        return
    if hint is bool:
        g.add_argument(
            flag, dest=dest, action=argparse.BooleanOptionalAction, default=None, help=help_text
        )
        return
    if origin is Literal:
        g.add_argument(flag, dest=dest, choices=typing.get_args(hint), default=None, help=help_text)
        return
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in typing.get_args(hint) if a is not type(None)]
        hint = inner[0] if len(inner) == 1 else str
    kind: Callable[[str], Any] = int if hint is int else float if hint is float else str
    metavar = "N" if kind is int else "X" if kind is float else None
    if re.match(r"--[\w-]+", help_text):  # a generator argument documented by its original flag
        help_text = f"the original's {help_text}"
    g.add_argument(
        flag,
        dest=dest,
        type=kind,
        default=None,
        metavar=metavar,
        help=f"{help_text} (default: {default})".strip(),
    )


def add_option_flags(
    p: argparse.ArgumentParser,
    options_cls: type,
    title: str,
    *,
    skip: Iterable[str] = (),
    defaults: dict[str, str] | None = None,
) -> list[str]:
    """One flag per field of an ``Options`` dataclass; returns the field names.

    ``defaults`` replaces the "(default: ...)" text of fields whose unset flag does something
    other than the dataclass default (``--objective``: every weight column).
    """
    hints = typing.get_type_hints(options_cls)
    helps = doc_section(options_cls, "Attributes")
    g = p.add_argument_group(title, f"the fields of {options_cls.__module__}.Options")
    names = []
    for f in dataclasses.fields(options_cls):
        if f.name in skip:
            continue
        default = f.default if f.default is not dataclasses.MISSING else None
        if defaults and f.name in defaults:
            default = defaults[f.name]
        _add_typed_flag(g, f.name, hints[f.name], default, helps.get(f.name, ""), "opt_" + f.name)
        names.append(f.name)
    return names


def options_from[T](args: argparse.Namespace, options_cls: type[T], names: Iterable[str]) -> T:
    """The ``Options`` object of the given flags (unset flags keep the field's default)."""
    kw = {n: getattr(args, "opt_" + n) for n in names if getattr(args, "opt_" + n) is not None}
    return options_cls(**kw)


def add_function_flags(
    p: argparse.ArgumentParser, fn: Callable[..., Any], title: str, *, skip: Iterable[str] = ()
) -> list[str]:
    """One flag per keyword-only argument of ``fn`` (a generator); returns the names."""
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    helps = doc_section(fn, "Args")
    g = p.add_argument_group(title, f"the arguments of {fn.__module__}.{fn.__name__}()")
    names = []
    for name, param in sig.parameters.items():
        if param.kind is not inspect.Parameter.KEYWORD_ONLY or name in skip:
            continue
        default = None if param.default is inspect.Parameter.empty else param.default
        _add_typed_flag(g, name, hints[name], default, helps.get(name, ""), "arg_" + name)
        names.append(name)
    return names


def function_kwargs(args: argparse.Namespace, names: Iterable[str]) -> dict[str, Any]:
    """The keyword arguments of the flags that were given."""
    return {n: getattr(args, "arg_" + n) for n in names if getattr(args, "arg_" + n) is not None}


# -------------------------------------------------------------------------------------------------
# Graphs
# -------------------------------------------------------------------------------------------------


def add_graph_flags(
    p: argparse.ArgumentParser,
    *,
    default_properties: str,
    required: bool = True,
    positional: bool = False,
    id_types: bool = True,
    auto_mtx: bool = True,
) -> None:
    """``--graph`` and the reader flags of the three graph formats (``auto_mtx=False``: ``auto``
    reads ``*.mtx`` with the edge-list reader, CycleEnumeration-GPU's parser, which reads Matrix
    Market files as edge lists)."""
    p.set_defaults(_auto_mtx=auto_mtx)
    if positional:
        p.add_argument("graph", metavar="GRAPH", help="the graph (see --format)")
    else:
        p.add_argument(
            "--graph",
            "--input",
            dest="graph",
            required=required,
            metavar="PATH",
            help="the graph: a MOSP CSR prefix (<prefix>RowPtr.txt, ColInd.txt, Values.txt), a "
            "Matrix Market file or an edge list (see --format)",
        )
    add_reader_flags(p, default_properties=default_properties, id_types=id_types)


def add_reader_flags(
    p: argparse.ArgumentParser,
    *,
    default_properties: str | None,
    id_types: bool = True,
    prefix: str = "",
) -> None:
    """The reader flags (``--format``, ``--num-weights``, ...)."""
    g = p.add_argument_group("graph input")
    g.add_argument(
        f"--{prefix}format",
        dest="format",
        choices=GRAPH_FORMATS,
        default="auto",
        help="csr: MOSP's text CSR prefix; mtx: Matrix Market (dyng.io.read_matrix_market); "
        "edges: an edge list (src dst [w..] [ts], TUDataset *_A.txt, SNAP; a file with a "
        "%%%%MatrixMarket banner is read as one). auto (default): csr if <path>RowPtr.txt "
        "exists, else mtx for *.mtx (sssp, convert), else edges",
    )
    g.add_argument(
        "--num-weights",
        type=int,
        default=None,
        metavar="K",
        help="csr: the objectives of a graph without edges (as mospPrep -k); mtx: K random "
        "weight columns (default 1); edges: weight columns after src dst (default 0)",
    )
    g.add_argument(
        "--random-weights",
        default=None,
        metavar="MIN,MAX,SEED",
        help="mtx: seeded random weights, bit-exact with `mospPrep mtx2csr` (e.g. 1,100,12345)",
    )
    g.add_argument(
        "--ids",
        choices=("compact", "as_is"),
        default="compact",
        help="edges: renumber ids 0, 1, ... in ascending order (compact, the CycleEnumeration-GPU "
        "parser) or keep them (as_is)",
    )
    g.add_argument(
        "--index-base",
        type=int,
        default=0,
        metavar="B",
        help="edges, --ids as_is: subtracted from every id (1 for 1-based files)",
    )
    g.add_argument(
        "--symmetrize", action="store_true", help="edges: also add (dst, src) for every line"
    )
    g.add_argument(
        "--keep-self-loops",
        action="store_true",
        help="edges and mtx: keep self-loops (dropped by default, as the originals do)",
    )
    g.add_argument(
        "--duplicates",
        choices=("merge", "keep"),
        default="merge",
        help="edges: merge repeated (u, v) lines (default) or keep them",
    )
    if id_types:
        g.add_argument(
            "--vertex-type", choices=("int32", "int64"), default="int32", help="vertex id type"
        )
        g.add_argument(
            "--edge-type",
            choices=("int32", "int64"),
            default=None,
            help="edge offset type (default: int32 when it fits; int64 with int64 ids)",
        )
    if default_properties is not None:
        g.add_argument(
            "--properties",
            choices=PRESETS,
            default=default_properties,
            help=f"graph properties preset (default: {default_properties})",
        )


def parse_random_weights(text: str | None) -> tuple[int, int, int] | None:
    """``MIN,MAX,SEED`` -> a tuple."""
    if text is None:
        return None
    parts = text.split(",")
    try:
        lo, hi, seed = (int(x) for x in parts)
    except ValueError:
        raise UsageError(f"--random-weights: expected MIN,MAX,SEED, got {text!r}") from None
    return lo, hi, seed


def csr_files(prefix: str | os.PathLike[str]) -> list[Path]:
    """The three files of a MOSP CSR prefix."""
    return [Path(f"{os.fspath(prefix)}{s}.txt") for s in ("RowPtr", "ColInd", "Values")]


def is_csr_prefix(path: str | os.PathLike[str]) -> bool:
    """True if ``<path>RowPtr.txt`` exists."""
    return csr_files(path)[0].is_file()


def graph_format(path: str, fmt: str, auto_mtx: bool = True) -> str:
    """The format of ``path`` (``auto`` resolved)."""
    if fmt != "auto":
        return fmt
    if is_csr_prefix(path):
        return "csr"
    if auto_mtx and path.lower().endswith(".mtx"):
        return "mtx"
    return "edges"


def csr_objectives(prefix: str | os.PathLike[str], k: int | None) -> int:
    """mospPrep's resolveObjectives(): -k counts only for a graph without edges (an empty Values
    file); otherwise K comes from the file (0 = infer)."""
    if not k:
        return 0
    values = csr_files(prefix)[2]
    try:
        empty = not values.read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return 0  # the reader reports the missing file
    return k if empty else 0


def read_edges(args: argparse.Namespace, path: str, fmt: str) -> dyng.io.EdgeList:
    """An edge list (edges or mtx) with the reader flags."""
    vertex = getattr(args, "vertex_type", "int32")
    if fmt == "mtx":
        rw = parse_random_weights(args.random_weights)
        return dyng.io.read_matrix_market_arrays(
            path,
            num_weights=1 if args.num_weights is None else args.num_weights,
            random_weights=rw,
            drop_self_loops=not args.keep_self_loops,
            vertex_dtype=vertex,
        )
    if args.random_weights is not None:
        raise UsageError("--random-weights applies to Matrix Market input only")
    return dyng.io.read_edge_list_arrays(
        path,
        num_weights=args.num_weights or 0,
        ids=args.ids,
        index_base=args.index_base,
        symmetrize=args.symmetrize,
        drop_self_loops=not args.keep_self_loops,
        duplicates=args.duplicates,
        vertex_dtype=vertex,
    )


def load_graph(
    args: argparse.Namespace,
    resources: dyng.Resources,
    *,
    path: str | None = None,
    properties: str | None = None,
) -> dyng.Graph:
    """The graph of ``--graph`` (or ``path``) with the reader flags."""
    path = args.graph if path is None else path
    props = properties or getattr(args, "properties", None) or "default"
    fmt = graph_format(path, args.format, getattr(args, "_auto_mtx", True))
    vertex = getattr(args, "vertex_type", "int32")
    edge = getattr(args, "edge_type", None)
    if fmt == "csr":
        if args.random_weights is not None:
            raise UsageError("--random-weights applies to Matrix Market input only")
        return dyng.io.read_csr_triplet(
            path,
            num_weights=csr_objectives(path, args.num_weights),
            vertex_dtype=vertex,
            edge_dtype=edge or ("int64" if vertex == "int64" else "int32"),
            properties=props,
            resources=resources,
        )
    return read_edges(args, path, fmt).to_graph(
        properties=props, edge_dtype=edge, resources=resources
    )


# -------------------------------------------------------------------------------------------------
# Result files
# -------------------------------------------------------------------------------------------------


def objective_dir(out: str | os.PathLike[str], k: int) -> Path:
    """``<out>/obj<k>`` (MOSP's layout), created."""
    d = Path(out) / f"obj{k}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_tree(out: str | os.PathLike[str], k: int, suffix: str, tree: dyng.sssp.Result) -> None:
    """``<out>/obj<k>/distances<suffix>.txt`` and ``SSSPTree<suffix>.txt`` (MOSP's files)."""
    d = objective_dir(out, k)
    dyng.io.write_distances(d / f"distances{suffix}.txt", tree.distances)
    dyng.io.write_parents(d / f"SSSPTree{suffix}.txt", tree.parents)


def histogram_text(counts: Any) -> str:
    """CycleEnumeration-GPU's histogram CSV."""
    return dyng.io.histogram_csv(counts)


def write_text(path: str | None, text: str) -> None:
    """``text`` to ``path`` (standard output for None or ``-``)."""
    if path is None or path == "-":
        sys.stdout.write(text)
        sys.stdout.flush()
        return
    p = Path(path)
    if p.parent != Path():
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)
