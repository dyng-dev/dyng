# File formats

Formats read and written by `dyng::io` in this release. Every reader validates its input and
throws `dyng::io_error` with the path, the 1-based line and (where it applies) the column; readers
never print. A token is a run of characters other than space, tab, carriage return and newline;
integer tokens must be complete decimal integers (optional leading `-`, no `+`, nothing after the
digits).

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
