# File formats

Formats read and written by `dyng::io` in this release. Every reader validates its input and
throws `dyng::io_error` with the path, the 1-based line and (where it applies) the column; readers
never print. A token is a run of characters other than space, tab, carriage return and newline;
integer tokens must be complete decimal integers (optional leading `-`, no `+`, nothing after the
digits). The edge-list reader follows CycleEnumeration-GPU instead: commas separate tokens as well,
and a leading `+` is accepted; the `.hg` reader follows ESCHER-GPU's separators (commas too). The
binary batch files (`.dgb`) report the record's index and byte offset instead of a line. The
formats of dynG's own (`.dgt`, `.dgb`, `.hg`) carry their own format versions and are a stable
tier (PLAN 5.9): version 1 of each is the format of 0.3 (that `.hg` and `.dgb` are stable from
0.3.0 awaits the author's confirmation at the release).

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

## dynG batch text (`.dgt`: `read_batches` / `write_batches`, `read_hyperedge_batches` / `write_hyperedge_batches`)

The library's batch format, for tests, examples and hand edits: a sequence of
batches in one ASCII text file, one operation per line. Format version 1.

- Lines are separated by `\n` (a trailing `\r` is ignored). Blank lines and lines whose first
  token starts with `#` are skipped; `#` later on a line is not a comment.
- An optional first line `%dgt 1 [graph|hypergraph]` names the format version (only version 1
  exists) and the kind of container the batches are for. **A file holds the batches of one
  container kind**: graph operations and hyperedge operations do not mix in a file, and a file
  whose first line names a kind is read only by that kind's reader (`%dgt 1` alone, the line of
  0.1 and 0.2, names none). A reader meets an operation or a header of the other kind with an
  `io_error` that names the other reader.
- `%batch <id>` starts a batch; the ids are non-negative decimal integers, strictly increasing
  (the writers number the batches 0, 1, ...). A file without `%batch` lines holds one batch (none
  if it holds no operation); in a file with `%batch` lines every operation follows one.
- **Graph operations** (`read_batches`): `+e u v [w1 .. wK]` inserts the edge (u, v) with exactly
  K weights; `-e u v` deletes it. Ids are 0-based, `>= 0` and, when the reader is given a vertex
  count, below it; weights must fit the weight type. K is the reader's `num_weights`, or (the
  default, `-1` / `None`) the count of the first insertion of the file, which every other
  insertion must then match. Within a batch the insertions and the deletions each keep file
  order. Which of the two applies first is the graph's batch semantics
  (`BatchSemantics.deletions_first`), not the file's.
- **Hyperedge operations** (`read_hyperedge_batches`, 0.3): `+h v1 v2 .. [; w]` inserts a
  hyperedge with the listed members (at least one; repeated members are kept in the file and
  dropped, and counted, by `hypergraph::apply()`), with its weight in [1, max] after `;` for a
  weighted hypergraph (`;` may stand alone or touch its neighbours: `+h 0 1 ; 5`, `+h 0 1;5`);
  `-h id` deletes the hyperedge with that id (`>= 0`; dead or out-of-range ids are kept and
  skipped, and counted, by `apply()`). Insertions and deletions each keep file order; which
  applies first is the hypergraph's `id_reuse` property, not the file's. Ids of inserted
  hyperedges are the hypergraph's choice (`hyper_apply_summary::inserted_ids`): a `-h` in a later
  batch names an id of the state that batch applies to.
- Reserved and rejected with an `io_error` / `FileFormatError` naming 0.4: the vertex operations
  `+v u [label]`, `-v u` (label_propagation) and the incidence edits `+i h v`, `-i h v`
  (hyper_sssp).
- Every token is a whole decimal integer; errors report the path, line and column.

The graph writer writes `%dgt 1`, then per batch `%batch <i>`, its deletions and its insertions in
batch order, and requires one K for every batch with insertions. The hypergraph writer writes
`%dgt 1 hypergraph`, then per batch `%batch <i>`, its deletions (`-h id`) and its insertions
(`+h v1 .. vk`, with ` ; w` when weighted). Both refuse a batch that would not read back (a
negative id, an empty hyperedge, a weight below 1) with an `invalid_argument_error`, and incidence
edits with `not_supported_error`. C++: `dyng::io::read_batches<V, W>()`, `write_batches<V, W>()`,
`read_hyperedge_batches<V, W>()` and `write_hyperedge_batches<V, W>()` in
`<dyng/io/batch_io.hpp>` (`W` may be `dyng::unweighted`); Python: `dyng.io.read_batches()` (int64
ids, converted with range checks to the graph's types when a batch is applied) and
`dyng.io.write_batches()`; the command line reads a one-batch `.dgt` file with `dyng cycle_count
update --batch FILE.dgt`.

```text
%dgt 1
%batch 0
-e 0 1
+e 0 3 7
%batch 1
+e 2 0 1
```

```text
%dgt 1 hypergraph
%batch 0
-h 4
+h 0 2 7 ; 3
%batch 1
+h 1 5 ; 9
```

## dynG binary batches (`.dgb`: `batch_reader`, `batch_writer`, `read_binary_*` / `write_binary_*`)

The binary form of the same batches, for benchmarks and replays: one file holds a sequence of
graph batches or of hypergraph batches, read one batch at a time in bounded memory. **Format
version 1** (0.3); every number is little-endian.

**Header** (64 bytes):

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `DYNGBAT1` |
| 8 | 4 | format version (u32): 1 |
| 12 | 4 | header size in bytes (u32): 64 |
| 16 | 1 | container kind: 0 graph, 1 hypergraph |
| 17 | 1 | vertex id type: 1 int32, 2 int64 |
| 18 | 1 | weight type: 0 none, 1 int32, 2 int64 |
| 19 | 1 | hyperedge id type: 0 for graph files, 2 (int64) for hypergraph files |
| 20 | 4 | K (u32): weights per insertion; 0 or 1 for hypergraph files (one weight per hyperedge); at most 64 |
| 24 | 8 | flags (u64), below |
| 32 | 8 | batch count (u64) |
| 40 | 20 | reserved, zero |
| 60 | 4 | CRC32 of bytes 0-59 |

**Flags** (hypergraph files; graph files have none): bit 0 `deleted_members` (every deletion
also carries the member set of the hyperedge it deleted when the batch was generated: replay by
content), bit 1 `expected_ids` (every insertion carries the id the generating run gave it: replay
by key, checked against `inserted_ids`), bit 2 `insert_first` (those ids hold under
`id_reuse::insert_first`; without it, `erase_first`). A reader rejects a file with any other bit.

**Records**, one per batch, back to back after the header: the payload length (u64, a multiple
of 8), the CRC32 of the payload (u32), 4 zero bytes, then the payload. The payload starts with
u64 counts and continues with arrays, each starting at a multiple of 8 bytes from the payload's
start (zero padding after an array of 4-byte values):

| Kind | Counts | Arrays, in this order |
|---|---|---|
| graph | batch id, deletions d, insertions i | deletion sources [d], deletion destinations [d], insertion sources [i], insertion destinations [i] (vertex type), weights [i x K] (weight type, insertion-major; absent for K = 0) |
| hypergraph | batch id, deletions d, insertions i, members m, deleted members q (0 without `deleted_members`) | deleted ids [d] (int64), insertion offsets [i + 1] (int64, from 0, strictly increasing, ending at m), members [m] (vertex type), weights [i] (weight type; absent for unweighted); with `deleted_members`: deleted-member offsets [d + 1] (int64) and deleted members [q] (vertex type); with `expected_ids`: expected ids [i] (int64) |

The payload length equals what its counts imply. The batch ids are non-negative and strictly
increasing (the writers number 0, 1, ...); the number of records equals the header's count, and
nothing follows the last record. Ids are `>= 0` (and below the reader's vertex count, when
given); hyperedge weights lie in [1, max]. The CRC32 is the IEEE one (zlib's `crc32`).

The reader checks every size field against the rest of the file before it allocates anything
(the batch count against what the file can hold, a record's length against the bytes left, each
count against the bytes of its record), so a file of N bytes makes it allocate O(N) at most; then
the CRC32 and the contents. A truncated, size-lying, damaged, newer-version or unknown-flag file,
trailing bytes and a wrong record count are `io_error`s with the path and the record's index and
byte offset; so is a file of the other container kind given to `read_binary_batches` /
`read_binary_hyperedge_batches` (the message names the right reader, as for `.dgt`), checked
from the header before any record is read. Asking for other types than the header's is an
`invalid_argument_error` (the whole-file readers check it from the header too). After an
`io_error` of `next()` the `batch_reader` stops: every later `next()` and batch view throws an
`io_error` naming the record where it failed (reopen the file to read it again), so a damaged
record is never skipped. With CUDA
resources `batch_reader` stages every record through pinned buffers and copies it to the device
on the resources' stream: its views are device arrays, valid until the next `next()`.

C++ (`<dyng/io/batch_io.hpp>`): `batch_writer` (header, `write()` per batch with the annotations
its flags need, `close()` writes the count), `batch_reader` (`header()`, `next()`,
`edge_batch<V, W>()`, `hyperedge_batch<V, W>()`, `annotations<V>()`), and
`write_binary_batches` / `read_binary_batches` / `write_binary_hyperedge_batches` /
`read_binary_hyperedge_batches` for whole files. `.dgt` and `.dgb` hold the same batches:
converting either way and back gives the same batches. `tools/compat/escher`
(`dyng-compat-escher to-dgb`) writes ESCHER-GPU's batch dumps as `.hg` + `.dgb` files with
`deleted_members` and `expected_ids`. Python (`dyng.io`): `write_binary_batches(path, batches,
*, deleted_members, expected_ids, insert_first)` (graph or hypergraph batches, the flags from the
arguments), `read_binary_batches(path)` and the streaming `BatchReader(path)` (records with
`batch_id`, `batch`, `deleted_members` and `expected_ids`; host batches).

## Hypergraphs (`.hg`: `read_hypergraph` / `write_hypergraph`)

One hyperedge per line: its vertex ids, decimal integers separated by spaces, tabs, carriage
returns or commas, and for a weighted file ` ; w` at the end (w in [1, max]; `;` may touch its
neighbours). A line whose **first** character is `#` or `%` is a comment; a line without ids is
skipped. Each line's ids are sorted and deduplicated; hyperedge i of the result is the i-th kept
line (the id `hypergraph::from_hyperedges()` gives it). Tokens are whole integers with an
optional `-` (no `+`). Format version 1 (0.3).

The reader's options, applied in file order: `max_cardinality` drops hyperedges with more
distinct vertices; `dedupe_hyperedges` keeps the first hyperedge of every distinct vertex set;
`relabel` numbers the vertices: `none` (the id minus `index_base`, default 1; the vertex count is
the largest id + 1 unless `num_vertices` is given), `dense_sorted` (the distinct ids in ascending
order, any 64-bit integers) or `first_appearance` (in the order in which they first appear,
within a line in ascending id order; `hypergraph_file_info::external_ids` returns the mapping of
both); `random_weights` draws one weight per kept hyperedge, in file order, uniformly from
[lo, hi] with one `std::mt19937_64(seed)` (the algorithm of libstdc++'s
`uniform_int_distribution`, reproduced by the library).

The originals' preprocessing are two option sets:

| Option set | Original | What it gives |
|---|---|---|
| `hypergraph_read_options::escher_gpu()` | ESCHER-GPU@abcf9b2 `loadHypergraph` | every line a hyperedge, ids kept (vertex v of the file is v - 1; the vertex count is the largest id), no weights: on coauth-DBLP, ESCHER-GPU's 2,467,389 hyperedges, hyperedge i is its key i + 1 |
| `hypergraph_read_options::hsosp(seed = 20260725, max_cardinality = 25)` | MOSP_ESCHER@4b86159 `loadHypergraphFile` (`hsospBench --hg`) | hyperedges of at most 25 vertices, vertices numbered by first appearance, weights U[1, 100] drawn after the original's draw of its virtual target vertex: on coauth-DBLP, H-SOSP's 2,466,792 hyperedges with its numbering and weights bit for bit. The virtual source and target hyperedges are not in the result: `hyper_sssp` adds them (0.4) |

Errors are `io_error` with the path, line and column (a token that is not an integer, an id
outside the range, a weight missing, misplaced or out of range, more vertices than the vertex
type holds). The writer writes every hyperedge's members in list order plus `index_base`
(default 1), separated by single spaces, and ` ; w` when weighted; it refuses empty hyperedges,
members outside [0, n) and weights below 1. `datasets/preprocess/benson_to_hg.py` converts the
timestamped simplicial datasets of Benson's collection (coauth-DBLP, ...) to `.hg` with either
original's preprocessing (`--variant escher` or `hsosp`; on coauth-DBLP-full both write the bytes
`datasets/registry.toml` pins). Python: `dyng.io.read_hypergraph(path, options="escher_gpu" |
"hsosp", ...)` (a `dyng.Hypergraph`; the option keywords and the hypergraph's property keywords),
`read_hypergraph_arrays()` (the arrays and what the reader dropped) and `write_hypergraph()`.

```text
% three hyperedges on vertices 1..6 (1-based)
1 2 3
2,4
6 5 2
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
