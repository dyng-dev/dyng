# File formats

Formats read and written by `dyng::io` in this release. Every reader validates its input and
throws `dyng::io_error` with the path, the 1-based line and (where it applies) the column; readers
never print. A token is a run of characters other than space, tab, carriage return and newline;
integer tokens must be complete decimal integers (optional leading `-`, no `+`, nothing after the
digits). The edge-list reader follows CycleEnumeration-GPU instead: commas separate tokens as well,
and a leading `+` is accepted.

## MOSP text CSR (`read_csr_triplet` / `write_csr_triplet`)

Three files with a common prefix:

| File | Content |
|---|---|
| `<prefix>RowPtr.txt` | n + 1 offsets, one per line, starting at 0, non-decreasing (n >= 1) |
| `<prefix>ColInd.txt` | one 0-based neighbour in [0, n) per edge, in row order |
| `<prefix>Values.txt` | one line per edge with its K weights separated by single spaces; each weight in [1, max(weight_t)] |

K is inferred from the first non-blank line of `Values.txt`; a graph without edges needs
`csr_triplet_options::num_weights`. The writer produces the same bytes as MOSP's
`writeCsrGraph()`. In memory the weights are objective-major (`csr::weights[k * m + e]`).

## dynG batch text (`.dgt`: `read_batches` / `write_batches`)

The library's batch format, for tests, examples and hand edits: a sequence of
batches in one ASCII text file, one operation per line.

- Lines are separated by `\n` (a trailing `\r` is ignored). Blank lines and lines whose first
  token starts with `#` are skipped; `#` later on a line is not a comment.
- An optional first line `%dgt 1` names the format version (only version 1 exists).
- `%batch <id>` starts a batch; the ids are non-negative decimal integers, strictly increasing
  (the writer numbers the batches 0, 1, ...). A file without `%batch` lines holds one batch (none
  if it holds no operation); in a file with `%batch` lines every operation follows one.
- `+e u v [w1 .. wK]` inserts the edge (u, v) with exactly K weights; `-e u v` deletes it. Ids
  are 0-based, `>= 0` and, when the reader is given a vertex count, below it; weights must fit
  the weight type. K is the reader's `num_weights`, or (the default, `-1` / `None`) the count of
  the first insertion of the file, which every other insertion must then match.
- Within a batch the insertions and the deletions each keep file order. Which of the two applies
  first is the graph's batch semantics (`BatchSemantics.deletions_first`), not the file's.
- Reserved and rejected by 0.1 with an `io_error` / `FileFormatError`: the vertex operations
  `+v u [label]`, `-v u` (label_propagation, 0.4) and the hypergraph operations `+h v1 v2 ...
  [; w]`, `-h id`, `+i h v`, `-i h v` (0.3).
- Every token is a whole decimal integer; errors report the path, line and column.

The writer writes `%dgt 1`, then per batch `%batch <i>`, its deletions and its insertions in batch
order, and requires one K for every batch with insertions. C++: `dyng::io::read_batches<V, W>()`
and `write_batches<V, W>()` in `<dyng/io/batch_io.hpp>` (`W` may be `dyng::unweighted`);
Python: `dyng.io.read_batches()` (int64 ids, converted with range checks to the graph's types
when a batch is applied) and `dyng.io.write_batches()`; the command line reads a one-batch
`.dgt` file with `dyng cycle_count update --batch FILE.dgt`.

```text
%dgt 1
%batch 0
-e 0 1
+e 0 3 7
%batch 1
+e 2 0 1
```

## MOSP batches (`read_legacy_batch` / `write_legacy_batch`)

`insert.txt` holds one insertion per line, `u v w1 .. wK`; `delete.txt` one deletion per line,
`u v`. Ids are 0-based; blank lines are skipped. dynG rejects lines that MOSP silently accepted
or ignored: extra tokens, and deletion lines with fewer than two integers. The writer produces
the same bytes as MOSP's `writeChangeBatch()`. How a batch changes a graph is decided by the
graph's `batch_semantics` (ADR 0010).

## Distances and SSSP trees (`read_distances`, `read_parents`, `write_distances`, `write_parents`)

One line `v d` per vertex (`v INF` for unreachable vertices; values >= `infinite_distance() / 2`
are written as `INF`), and one line `v p` per vertex (`p = -1` for none). Every vertex is listed
exactly once, in any order. The writers produce the same bytes as MOSP's `writeDistances()` and
`writeParents()`.

## Matrix Market (`read_matrix_market` / `write_matrix_market`)

`%%MatrixMarket matrix coordinate <field> <symmetry>` with field `real`, `double`, `integer`,
`complex` or `pattern` and symmetry `general`, `symmetric`, `skew-symmetric` or `hermitian`;
comment lines start with `%`; the matrix must be square; the file must hold exactly the announced
number of entries with 1-based indices. Non-general files give both directions of every
off-diagonal entry. By default self-loops are dropped and the edges are sorted and deduplicated
(the last entry of a duplicate wins).

With `matrix_market_weights::random`, K weights per edge are drawn in edge order from
`std::mt19937(seed)` with the libstdc++ `uniform_int_distribution` algorithm, reproduced by the
library. With the defaults (K weights in [1, 100], seed 12345) the result is bit-exact with
`mospPrep mtx2csr <in.mtx> <prefix> K 1 100 12345` of MOSP-OpenMP and MOSP-CUDA, on any C++
standard library. (mospPrep reads `hermitian` files as general; dynG mirrors them.)

The writer writes a `general` file (field `pattern` without weights, else `integer` with one
chosen weight column).

## Edge lists (`read_edge_list` / `write_edge_list`)

One edge per line: `src dst [w1..wK] [ts]` (`num_weights` = K, default 0), integers separated by
spaces, tabs or commas, an optional `+` or `-` sign; blank lines and lines whose first non-blank
character is `#` or `%` are skipped. With K = 0 a third column is a timestamp and a fourth is an
error. TUDataset `*_A.txt` files (`  u,  v`) and SNAP edge lists load directly. A file whose first
line starts with `%%MatrixMarket` is read as a Matrix Market `coordinate` file (dimensions line
skipped, value columns ignored; symmetric, skew-symmetric and hermitian entries give both
directions; `array` and unknown symmetries are rejected).

With the defaults the reader reproduces CycleEnumeration-GPU@0a976ad's parser exactly: self-loops
are dropped before the ids are numbered, the ids that remain are numbered 0, 1, ... in ascending
order of their value (`vertex_ids::compact`; any 64-bit ids; the mapping is returned in
`edge_list_info::external_ids`), and the rows are sorted by (source, destination) with repeated
pairs merged (`duplicate_edges::keep` keeps every row and returns its timestamp). Options:
`vertex_ids::as_is` with `index_base` (ids kept, e.g. 1-based files with base 1), `symmetrize`,
`drop_self_loops = false`, `threads` (the file is parsed in line-aligned parts on std::thread
workers; the result does not depend on the count). Errors are `io_error` with the path and the
line of the first malformed row in file order.

The writer writes `src dst [w1..wK]` per edge, 0-based.

## Text batches of the command line (`dyng cycle_count update --batch`, `dyng generate cycle_enum_batch`)

One change per line: `- u v` for a deletion, `+ u v [w1 .. wK]` for an insertion (without
weights into a weighted graph, the weight 1 in every column), in graph ids; blank lines and lines
starting with `#` are skipped. `dyng generate cycle_enum_batch` writes the deletions first, then
the insertions, each sorted by (source, destination): the text of CycleEnumeration-GPU's batch
generator (the goldens of `parity/cycle_count_goldens.py` and `dyng-compat-cycle-enum
--write-batch`). This is a command-line format of 0.1, read by the Python package only; the
versioned batch format of the library is `.dgt` (above), which `--batch` also reads when the file
name ends in `.dgt`.

## MOSP's binary graph cache (`dyng prep cache`)

The file `mosp --cache` reads (MOSP's `saveCsrGraphBinary()`, format version 2), little-endian:
the magic `MOSPCSR2`; a uint32 length L and L bytes of source identity (for each of
`<prefix>RowPtr.txt`, `ColInd.txt`, `Values.txt`: its canonical path, its size and its
modification time as libstdc++'s `file_time_type` counts it, nanoseconds since 2174-01-01, each
followed by a newline); int32 n and K; int64 m; then n + 1 int32 row offsets, m int32 column
indices and m * K int32 weights, edge-major. `dyng prep cache` writes the same bytes as
`mospPrep cache`. A versioned cache format of dynG's own (`.dgc`) is planned, not in 0.1 or 0.2.
