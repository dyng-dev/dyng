# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""dyng.io: readers and writers of the 0.1 file formats (C++ ``dyng::io``).

Graphs: edge lists (``src dst [w1..wK] [ts]``, TUDataset ``*_A.txt``, SNAP), Matrix Market
coordinate files, and MOSP's text CSR (``<prefix>{RowPtr,ColInd,Values}.txt``). Batches: MOSP's
``insert.txt`` / ``delete.txt``. Results: MOSP's distance, tree and path-cost files and
CycleEnumeration-GPU's histogram CSV. The writers produce the originals' bytes. Every reader
validates its input and raises :class:`~dyng.FileFormatError` with ``.path`` and ``.line``.
The formats are specified in the documentation, "File formats".

Readers of graphs return a :class:`dyng.Graph` (the ``*_arrays`` variants return the arrays);
the vertex id type is int32 unless ``vertex_dtype="int64"`` is given.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import IO, Any, Literal

import numpy as np

from . import _dtypes
from ._backend import native
from ._convert import as_bool, as_int, enum_member
from .array import Array
from .batch import EdgeBatch
from .errors import InvalidArgumentError
from .graph import BatchSemantics, Graph, GraphProperties, make_properties
from .resources import Resources, resolve

__all__ = [
    "EdgeList",
    "EdgeListInfo",
    "read_edge_list",
    "read_edge_list_arrays",
    "write_edge_list",
    "read_matrix_market",
    "read_matrix_market_arrays",
    "write_matrix_market",
    "read_csr_triplet",
    "write_csr_triplet",
    "read_batches",
    "write_batches",
    "read_legacy_batch",
    "write_legacy_batch",
    "read_distances",
    "read_parents",
    "write_distances",
    "write_parents",
    "write_path_costs",
    "histogram_csv",
    "write_histogram_csv",
]

PathLike = str | os.PathLike[str]


@dataclass(frozen=True)
class EdgeListInfo:
    """What :func:`read_edge_list` found besides the edges (C++ ``io::edge_list_info``).

    Attributes:
        external_ids: With ``ids="compact"``, the id in the file of graph vertex i.
        timestamps: With ``duplicates="keep"``, the timestamp of every edge.
        matrix_market: The file had a ``%%MatrixMarket`` banner.
        symmetric: Its symmetry was symmetric, skew-symmetric or hermitian.
    """

    external_ids: np.ndarray
    timestamps: np.ndarray
    matrix_market: bool
    symmetric: bool


@dataclass(frozen=True)
class EdgeList:
    """An edge list read from a file (the arrays own their memory).

    Attributes:
        num_vertices: Ids lie in [0, num_vertices).
        src: Source of each edge.
        dst: Destination of each edge.
        weights: Shape (m, K) int32, or None for an unweighted list.
        info: Only from :func:`read_edge_list_arrays`.
    """

    num_vertices: int
    src: np.ndarray
    dst: np.ndarray
    weights: np.ndarray | None
    info: EdgeListInfo | None = None

    @property
    def num_edges(self) -> int:
        """The number of edges."""
        return int(self.src.size)

    def to_graph(
        self,
        *,
        properties: GraphProperties | str | None = None,
        directed: bool | None = None,
        semantics: BatchSemantics | str | None = None,
        edge_dtype: Any = None,
        resources: Resources | None = None,
        **kwargs: Any,
    ) -> Graph:
        """A graph of these edges (see :meth:`dyng.Graph.from_edges`)."""
        return Graph.from_edges(
            self.src,
            self.dst,
            self.weights,
            num_vertices=self.num_vertices,
            properties=properties,  # type: ignore[arg-type]
            directed=directed,
            semantics=semantics,  # type: ignore[arg-type]
            vertex_dtype=self.src.dtype,
            edge_dtype=edge_dtype,
            resources=resources,
            **kwargs,
        )


def _path(p: PathLike) -> str:
    return os.fspath(p)


def _vcode(vertex_dtype: Any) -> str:
    return "i32" if _dtypes.id_dtype(vertex_dtype, "vertex_dtype") == _dtypes.INT32 else "i64"


def _edge_list(parts: tuple[Any, ...]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray | None]:
    n, src, dst, w, _k = parts
    return int(n), src, dst, w


# -------------------------------------------------------------------------------------------------
# Edge lists
# -------------------------------------------------------------------------------------------------


def read_edge_list_arrays(
    path: PathLike,
    *,
    num_weights: int = 0,
    weighted: bool | None = None,
    ids: Literal["compact", "as_is"] = "compact",
    index_base: int = 0,
    symmetrize: bool = False,
    drop_self_loops: bool = True,
    duplicates: Literal["merge", "keep"] = "merge",
    threads: int = 0,
    vertex_dtype: Any = "int32",
) -> EdgeList:
    """Read a directed edge list as arrays (see :func:`read_edge_list` for the arguments)."""
    if weighted is True and num_weights == 0:
        num_weights = 1
    if weighted is False and num_weights != 0:
        raise InvalidArgumentError("read_edge_list: weighted=False with num_weights > 0")
    opt = native.EdgeListOptions()
    opt.num_weights = as_int(num_weights, "read_edge_list: num_weights")
    opt.ids = enum_member(native.VertexIds, ids, "read_edge_list: ids")
    opt.index_base = as_int(index_base, "read_edge_list: index_base")
    opt.symmetrize = as_bool(symmetrize, "read_edge_list: symmetrize")
    opt.drop_self_loops = as_bool(drop_self_loops, "read_edge_list: drop_self_loops")
    opt.duplicates = enum_member(native.DuplicateEdges, duplicates, "read_edge_list: duplicates")
    opt.threads = as_int(threads, "read_edge_list: threads")
    wcode = "i32" if num_weights > 0 else "u"
    reader = getattr(native, f"read_edge_list_{_vcode(vertex_dtype)}_{wcode}")
    parts, info = reader(_path(path), opt)
    n, src, dst, w = _edge_list(parts)
    return EdgeList(
        n,
        src,
        dst,
        w,
        EdgeListInfo(
            info["external_ids"],
            info["timestamps"],
            bool(info["matrix_market"]),
            bool(info["symmetric"]),
        ),
    )


def read_edge_list(
    path: PathLike,
    *,
    num_weights: int = 0,
    weighted: bool | None = None,
    ids: Literal["compact", "as_is"] = "compact",
    index_base: int = 0,
    symmetrize: bool = False,
    drop_self_loops: bool = True,
    duplicates: Literal["merge", "keep"] = "merge",
    threads: int = 0,
    vertex_dtype: Any = "int32",
    properties: GraphProperties | str | None = None,
    directed: bool | None = None,
    semantics: BatchSemantics | str | None = None,
    edge_dtype: Any = None,
    resources: Resources | None = None,
) -> Graph:
    """Read a directed edge list into a graph (C++ ``io::read_edge_list``).

    One edge per line, ``src dst [w1..wK] [ts]``, separated by spaces, tabs or commas; ``#`` and
    ``%`` comment lines. A file starting with ``%%MatrixMarket`` is read as Matrix Market. With the
    defaults the reader reproduces CycleEnumeration-GPU's parser exactly (self-loops dropped, ids
    numbered 0, 1, ... in ascending order, repeated pairs merged).

    Args:
        path: The file.
        num_weights: Weight columns after ``src dst`` (0: a third column is a timestamp).
        weighted: False: no weights (the plan's spelling of ``num_weights=0``); True with
            ``num_weights=0``: one weight column.
        ids: ``"compact"`` (renumbered; the mapping is in :func:`read_edge_list_arrays`'s info) or
            ``"as_is"`` (the file's id minus ``index_base``).
        index_base: ``ids="as_is"`` only: subtracted from every id (1 for 1-based files).
        symmetrize: Also add (dst, src) for every row.
        drop_self_loops: Skip rows with src == dst.
        duplicates: ``"merge"`` (one edge per pair) or ``"keep"``.
        threads: Parser threads (0: the hardware concurrency); the result does not depend on it.
        vertex_dtype: ``"int32"`` (default) or ``"int64"``.
        properties: Graph properties (a preset name or :class:`dyng.GraphProperties`), e.g.
            ``"cycle_enum_compatible"``.
        directed: Overrides ``properties.directed``.
        semantics: Overrides ``properties.semantics``.
        edge_dtype: The edge offset type (default: int32 when it fits).
        resources: The resources of the graph (default: the default resources).

    Raises:
        FileFormatError: the file cannot be read or is malformed (``.path``, ``.line``).
    """
    edges = read_edge_list_arrays(
        path,
        num_weights=num_weights,
        weighted=weighted,
        ids=ids,
        index_base=index_base,
        symmetrize=symmetrize,
        drop_self_loops=drop_self_loops,
        duplicates=duplicates,
        threads=threads,
        vertex_dtype=vertex_dtype,
    )
    return edges.to_graph(
        properties=properties,
        directed=directed,
        semantics=semantics,
        edge_dtype=edge_dtype,
        resources=resources,
    )


def _edges_of(
    edges: Graph | EdgeList | tuple[Any, ...],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray | None]:
    if isinstance(edges, Graph):
        s, d, w = edges.edges()
        return edges.num_vertices, s, d, w
    if isinstance(edges, EdgeList):
        return edges.num_vertices, edges.src, edges.dst, edges.weights
    if isinstance(edges, tuple) and len(edges) in (2, 3):
        s = _dtypes.to_numpy(edges[0], "src")
        d = _dtypes.to_numpy(edges[1], "dst")
        w = (
            None
            if len(edges) == 2 or edges[2] is None
            else _dtypes.weights_matrix(edges[2], s.size, "weights")
        )
        n = int(max(s.max(initial=-1), d.max(initial=-1))) + 1
        return n, s, d, w
    raise TypeError("expected a dyng.Graph, a dyng.io.EdgeList or a (src, dst[, weights]) tuple")


def _prepared(
    edges: Graph | EdgeList | tuple[Any, ...],
) -> tuple[str, int, np.ndarray, np.ndarray, np.ndarray | None, int]:
    n, s, d, w = _edges_of(edges)
    vertex = _dtypes.infer_id_dtype([s, d], None, "write")
    s = _dtypes.checked_cast(s, vertex, "src")
    d = _dtypes.checked_cast(d, vertex, "dst")
    k = 0 if w is None else int(w.shape[1])
    flat = None if w is None else np.ascontiguousarray(w, dtype=np.int32).reshape(-1)
    return ("i32" if vertex == _dtypes.INT32 else "i64"), n, s, d, flat, k


def write_edge_list(path: PathLike, edges: Graph | EdgeList | tuple[Any, ...]) -> None:
    """Write edges as text, one line ``src dst [w1..wK]`` per edge (0-based, single spaces).

    Args:
        path: The file (its parent directory is created).
        edges: A graph (its stored edges in row order), an :class:`EdgeList` or a
            ``(src, dst[, weights])`` tuple.
    """
    vcode, n, s, d, w, k = _prepared(edges)
    writer = getattr(native, f"write_edge_list_{vcode}_{'i32' if w is not None else 'u'}")
    writer(_path(path), n, s, d, w, k)


# -------------------------------------------------------------------------------------------------
# Matrix Market
# -------------------------------------------------------------------------------------------------


def read_matrix_market_arrays(
    path: PathLike,
    *,
    weights: Literal["automatic", "none", "from_file", "random"] | None = None,
    num_weights: int = 1,
    random_weights: tuple[int, int, int] | None = None,
    drop_self_loops: bool = True,
    sort_and_dedupe: bool = True,
    vertex_dtype: Any = "int32",
) -> EdgeList:
    """Read a Matrix Market coordinate file as arrays (see :func:`read_matrix_market`)."""
    opt = native.MatrixMarketOptions()
    if weights is None:
        weights = "random" if random_weights is not None else "automatic"
    opt.weights = enum_member(native.MatrixMarketWeights, weights, "read_matrix_market: weights")
    lo, hi, seed = random_weights if random_weights is not None else (1, 100, 12345)
    what = "read_matrix_market"
    opt.set_random(
        as_int(num_weights, f"{what}: num_weights"),
        as_int(lo, f"{what}: random_weights[0]"),
        as_int(hi, f"{what}: random_weights[1]"),
        as_int(seed, f"{what}: random_weights[2]"),
    )
    opt.drop_self_loops = as_bool(drop_self_loops, f"{what}: drop_self_loops")
    opt.sort_and_dedupe = as_bool(sort_and_dedupe, f"{what}: sort_and_dedupe")
    reader = getattr(native, f"read_matrix_market_{_vcode(vertex_dtype)}_i32")
    n, src, dst, w = _edge_list(reader(_path(path), opt))
    if w is not None and w.shape[1] == 0:
        w = None
    return EdgeList(n, src, dst, w)


def read_matrix_market(
    path: PathLike,
    *,
    weights: Literal["automatic", "none", "from_file", "random"] | None = None,
    num_weights: int = 1,
    random_weights: tuple[int, int, int] | None = None,
    drop_self_loops: bool = True,
    sort_and_dedupe: bool = True,
    vertex_dtype: Any = "int32",
    properties: GraphProperties | str | None = None,
    directed: bool | None = None,
    semantics: BatchSemantics | str | None = None,
    edge_dtype: Any = None,
    resources: Resources | None = None,
) -> Graph:
    """Read a Matrix Market ``coordinate`` file into a graph (0-based ids).

    With ``random_weights=(1, 100, 12345)`` (min, max, seed) and ``num_weights=K`` the result is
    bit-exact with ``mospPrep mtx2csr <in.mtx> <prefix> K 1 100 12345`` of MOSP.

    Args:
        path: The file.
        weights: ``"automatic"`` (an ``integer`` file's values; none for ``pattern``),
            ``"none"``, ``"from_file"`` or ``"random"`` (default: ``"random"`` when
            ``random_weights`` is given, else ``"automatic"``).
        num_weights: K for random weights.
        random_weights: ``(min, max, seed)`` of the seeded weights.
        drop_self_loops: Skip diagonal entries (mospPrep).
        sort_and_dedupe: Sort edges by (source, destination), drop duplicates (mospPrep).
        vertex_dtype: ``"int32"`` (default) or ``"int64"``.
        properties: Graph properties (a preset name or :class:`dyng.GraphProperties`).
        directed: Overrides ``properties.directed``.
        semantics: Overrides ``properties.semantics``.
        edge_dtype: The edge offset type.
        resources: The resources of the graph.

    Raises:
        FileFormatError: the file cannot be read or is malformed.
    """
    return read_matrix_market_arrays(
        path,
        weights=weights,
        num_weights=num_weights,
        random_weights=random_weights,
        drop_self_loops=drop_self_loops,
        sort_and_dedupe=sort_and_dedupe,
        vertex_dtype=vertex_dtype,
    ).to_graph(
        properties=properties,
        directed=directed,
        semantics=semantics,
        edge_dtype=edge_dtype,
        resources=resources,
    )


def write_matrix_market(
    path: PathLike, edges: Graph | EdgeList | tuple[Any, ...], *, weight_column: int = 0
) -> None:
    """Write a ``general`` coordinate Matrix Market file (1-based ids; field ``pattern`` without
    weights, else ``integer`` with column ``weight_column``)."""
    vcode, n, s, d, w, k = _prepared(edges)
    getattr(native, f"write_matrix_market_{vcode}_i32")(
        _path(path), n, s, d, w, k, as_int(weight_column, "write_matrix_market: weight_column")
    )


# -------------------------------------------------------------------------------------------------
# MOSP text CSR and batches
# -------------------------------------------------------------------------------------------------


def read_csr_triplet(
    prefix: PathLike,
    *,
    num_weights: int = 0,
    vertex_dtype: Any = "int32",
    edge_dtype: Any = "int32",
    properties: GraphProperties | str | None = None,
    directed: bool | None = None,
    semantics: BatchSemantics | str | None = None,
    resources: Resources | None = None,
) -> Graph:
    """Read MOSP's text CSR ``<prefix>RowPtr.txt``, ``<prefix>ColInd.txt``, ``<prefix>Values.txt``.

    Args:
        prefix: The common path prefix of the three files.
        num_weights: Weight columns (0: from the first line of Values.txt; a graph without edges
            needs the value).
        vertex_dtype: ``"int32"`` or ``"int64"``.
        edge_dtype: ``"int32"`` or ``"int64"`` (int64 ids need int64 offsets).
        properties: Graph properties; ``"mosp_compatible"`` keeps MOSP's rows byte for byte.
        directed: Overrides ``properties.directed``.
        semantics: Overrides ``properties.semantics``.
        resources: The resources of the graph.
    """
    vertex = _dtypes.id_dtype(vertex_dtype, "read_csr_triplet: vertex_dtype")
    edge = _dtypes.id_dtype(edge_dtype, "read_csr_triplet: edge_dtype")
    if vertex == _dtypes.INT64:
        edge = _dtypes.INT64
    gtype = _dtypes.find_graph_type(vertex, edge, True)
    row_ptr, col_ind, w_km, k = getattr(native, f"read_csr_triplet_{gtype.code}")(
        _path(prefix), as_int(num_weights, "read_csr_triplet: num_weights")
    )
    props = make_properties(properties, directed=directed, semantics=semantics)  # type: ignore[arg-type]
    return Graph._from_csr_objective_major(
        row_ptr, col_ind, w_km, int(k), True, props, vertex, edge, col_ind, resolve(resources)
    )


def write_csr_triplet(prefix: PathLike, graph: Graph) -> None:
    """Write a graph's out-edges in MOSP's text CSR (byte-identical to MOSP's writeCsrGraph())."""
    if not graph.weighted:
        raise InvalidArgumentError("write_csr_triplet: MOSP's CSR has weights; the graph has none")
    csr = graph.to_csr()
    w_km = np.ascontiguousarray(csr.weights.T).reshape(-1)  # type: ignore[union-attr]
    native.write_csr_triplet(_path(prefix), csr.row_ptr, csr.col_ind, w_km, graph.num_weights)


def read_legacy_batch(
    insert_path: PathLike,
    delete_path: PathLike,
    *,
    num_weights: int = 1,
    num_vertices: int = -1,
    mosp_lenient: bool = False,
    vertex_dtype: Any = "int32",
) -> EdgeBatch:
    """Read a MOSP batch: ``insert.txt`` (``u v w1 .. wK`` per line) and ``delete.txt``
    (``u v``).

    Args:
        insert_path: The insertion file.
        delete_path: The deletion file.
        num_weights: K, the weights per insertion line.
        num_vertices: Ids must be < num_vertices (-1: only ids >= 0 are checked).
        mosp_lenient: MOSP's accept/reject rules instead of dynG's stricter ones.
        vertex_dtype: ``"int32"`` or ``"int64"``.
    """
    reader = getattr(native, f"read_legacy_batch_{_vcode(vertex_dtype)}_i32")
    parts = reader(
        _path(insert_path),
        _path(delete_path),
        as_int(num_weights, "read_legacy_batch: num_weights"),
        as_int(num_vertices, "read_legacy_batch: num_vertices"),
        as_bool(mosp_lenient, "read_legacy_batch: mosp_lenient"),
    )
    return EdgeBatch._from_arrays(parts)


def read_batches(
    path: PathLike,
    *,
    num_weights: int | None = None,
    num_vertices: int = -1,
    resources: Resources | None = None,
) -> list[EdgeBatch]:
    """Read a ``.dgt`` batch text file: its batches, in file order (PLAN Section 5.7).

    One operation per line: ``+e u v [w1 .. wK]`` (an edge insertion) or ``-e u v`` (an edge
    deletion); ``%batch <id>`` starts a batch, an optional first line ``%dgt 1`` names the
    format version, and lines starting with ``#`` are comments. The format is specified in
    :doc:`/api/file_formats`.

    Args:
        path: The file.
        num_weights: K, the weights of every insertion; None: taken from the first insertion.
        num_vertices: Ids must be < num_vertices (-1: only ids >= 0 are checked).
        resources: Accepted for symmetry with the graph readers; batches are host objects that
            are converted to the graph's types and memory space when they are used.

    Returns:
        One :class:`dyng.EdgeBatch` per batch (int64 ids, int32 weights; converted with range
        checks to the types of the graph each is applied to).

    Raises:
        FileFormatError: the file is missing or malformed (with ``.path`` and ``.line``).

    Example:
        >>> import dyng, tempfile, os
        >>> path = os.path.join(tempfile.mkdtemp(), "b.dgt")
        >>> dyng.io.write_batches(path, [dyng.EdgeBatch(insert=([0], [2], [5]), delete=([0], [1]))])
        >>> [(b.num_insertions, b.num_deletions) for b in dyng.io.read_batches(path)]
        [(1, 1)]
    """
    if resources is not None:
        resolve(resources)  # type check only
    k = -1 if num_weights is None else as_int(num_weights, "read_batches: num_weights")
    parts = native.read_batches_i64_i32(
        _path(path), k, as_int(num_vertices, "read_batches: num_vertices")
    )
    out = []
    for ins_src, ins_dst, w, del_src, del_dst, kk in parts:
        weights = None if kk == 0 else w
        out.append(EdgeBatch._from_arrays((ins_src, ins_dst, weights, del_src, del_dst, kk)))
    return out


def write_batches(path: PathLike, batches: Iterable[EdgeBatch]) -> None:
    """Write batches as a ``.dgt`` batch text file (read back by :func:`read_batches`).

    Writes ``%dgt 1``, then per batch ``%batch <i>``, its deletions (``-e u v``) and its
    insertions (``+e u v w1 .. wK``). Every batch with insertions must have the same K.

    Raises:
        InvalidArgumentError: a batch has vertex operations (0.3) or a different K.
    """
    gtype = _dtypes.GraphType(_dtypes.INT64, _dtypes.INT64, True)
    native_batches = []
    for i, b in enumerate(batches):
        if not isinstance(b, EdgeBatch):
            raise TypeError(f"write_batches: item {i} is not a dyng.EdgeBatch")
        native_batches.append(b._native_for_type(gtype, b.num_weights))
    native.write_batches_i64_i32(_path(path), native_batches)


def write_legacy_batch(insert_path: PathLike, delete_path: PathLike, batch: EdgeBatch) -> None:
    """Write a batch as MOSP ``insert.txt`` / ``delete.txt`` (byte-identical to MOSP's
    writeChangeBatch())."""
    vertex = _dtypes.infer_id_dtype(
        [batch.insert_src, batch.insert_dst, batch.delete_src, batch.delete_dst],
        None,
        "write_legacy_batch",
    )
    vcode = "i32" if vertex == _dtypes.INT32 else "i64"
    gtype = _dtypes.GraphType(
        vertex, _dtypes.INT64 if vertex == _dtypes.INT64 else _dtypes.INT32, True
    )
    nb = batch._native_for_type(gtype, batch.num_weights)
    getattr(native, f"write_legacy_batch_{vcode}_i32")(_path(insert_path), _path(delete_path), nb)


# -------------------------------------------------------------------------------------------------
# Results
# -------------------------------------------------------------------------------------------------


def read_distances(path: PathLike, num_vertices: int) -> np.ndarray:
    """Read a MOSP distance file (``v d`` or ``v INF`` per vertex) as int64 (INF reads as
    :data:`dyng.sssp.INFINITE_DISTANCE`)."""
    out: np.ndarray = native.read_distances(
        _path(path), as_int(num_vertices, "read_distances: num_vertices")
    )
    return out


def read_parents(path: PathLike, num_vertices: int, *, vertex_dtype: Any = "int32") -> np.ndarray:
    """Read a MOSP SSSP-tree file (``v p`` per vertex, -1 for none)."""
    reader = getattr(native, f"read_parents_{_vcode(vertex_dtype)}")
    out: np.ndarray = reader(_path(path), as_int(num_vertices, "read_parents: num_vertices"))
    return out


def _host_array(a: Any, what: str) -> np.ndarray:
    if isinstance(a, Array):
        return a.to_numpy(copy=None)
    return _dtypes.to_numpy(a, what)


def write_distances(path: PathLike, distances: Any) -> None:
    """Write distances, one line ``v d`` per vertex (``INF`` for unreachable), byte-identical to
    MOSP's writeDistances()."""
    d = _dtypes.checked_cast(_host_array(distances, "distances"), _dtypes.INT64, "write_distances")
    native.write_distances(_path(path), d)


def write_parents(path: PathLike, parents: Any) -> None:
    """Write a parent array, one line ``v p`` per vertex, byte-identical to MOSP's
    writeParents()."""
    p = _host_array(parents, "parents")
    vertex = _dtypes.infer_id_dtype([p], None, "write_parents")
    native.write_parents(_path(path), _dtypes.checked_cast(p, vertex, "write_parents"))


def write_path_costs(path: PathLike, costs: Any) -> None:
    """Write path costs, one line ``v c1 .. cK`` per vertex (``INF`` for unreachable),
    byte-identical to MOSP's mospCosts.txt (the ``writeCosts()`` of its ``mosp`` driver).

    Args:
        path: The file (parent directories are created).
        costs: An (n, K) integer array (``dyng.mosp.Result.path_costs``, or
            :func:`dyng.testing.mosp_path_costs`).
    """
    c = costs.to_numpy(copy=None) if isinstance(costs, Array) else np.asarray(costs)
    if c.dtype.kind not in "iu":
        raise InvalidArgumentError(f"write_path_costs: costs must be integers, got {c.dtype}")
    if c.ndim != 2 or c.shape[1] < 1:
        raise InvalidArgumentError(
            f"write_path_costs: costs must be an (n, K) array with K >= 1, got shape {c.shape}"
        )
    k = int(c.shape[1])
    flat = _dtypes.checked_cast(
        np.ascontiguousarray(c).reshape(-1), _dtypes.INT64, "write_path_costs"
    )
    native.write_path_costs(_path(path), flat, k)


def histogram_csv(counts: Any, *, include_total: bool = True) -> str:
    """A cycle histogram as CycleEnumeration-GPU's CSV (``# cycle_size, num_of_cycles``, one
    ``len, count`` line per non-zero length, ``Total, N``)."""
    if hasattr(counts, "counts") and not isinstance(counts, (np.ndarray, Array)):
        counts = counts.counts  # a cycle_count.Result
    c = _host_array(counts, "counts")
    c = (
        np.ascontiguousarray(c, dtype=np.uint64)
        if c.dtype != np.uint64
        else np.ascontiguousarray(c)
    )
    return str(native.histogram_csv(c, as_bool(include_total, "histogram_csv: include_total")))


def write_histogram_csv(
    file: PathLike | IO[str], counts: Any, *, include_total: bool = True
) -> None:
    """Write :func:`histogram_csv` to a path or an open text file."""
    text = histogram_csv(counts, include_total=include_total)
    if hasattr(file, "write"):
        file.write(text)
        return
    path = _path(file)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
