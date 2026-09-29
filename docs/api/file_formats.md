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
