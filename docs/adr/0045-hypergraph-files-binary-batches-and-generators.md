# ADR 0045: Hypergraph files, hypergraph batch files, the binary batch format `.dgb` and the hypergraph generators

- **Status:** Accepted under delegation (2026-10-14; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out M8b step 4 (PLAN 5.7, 5.8, 5.9, 6.4.1, 6.4.6, 8.3; the M8b acceptance
  criteria 13-16) and records where it differs from the plan's sketches. It changes no rule the
  author approved: the parity rules (byte and id identity with the originals) are checked as
  they are, no gate or measurement protocol changes, the names PLAN fixes stay (`.hg`, `.dgb`,
  the magic `DYNGBAT1`, `read_hypergraph`, `write_hypergraph`, `batch_reader`,
  `dedupe_hyperedges`, `max_cardinality`, `relabel`, `random_weights`). **Pending the author:**
  file formats are a stable tier with their own format version (PLAN 5.9); `.dgb` is written
  here at version 1, `.hg` has no version field (amended by the M8b review: section 1), and
  that both are stable from 0.3.0 awaits the author's confirmation at the release (the plan's
  default applies until then).
- **Date:** 2026-10-14
- **Deciders:** the AI assistant, on the author's behalf (M8b, step io-generators)
- **Builds on:** ADR 0034 (reader fuzzers), ADR 0036 (the CBST traces, whose batch records are the
  dumps), ADR 0041 (the container, its ids and `id_reuse`), ADR 0010 (strict readers).

## Context

PLAN 5.7 lists `read_hypergraph` / `write_hypergraph` for `.hg` files ("one hyperedge per line";
options `dedupe_hyperedges`, `max_cardinality`, `relabel = dense_sorted | first_appearance`), the
hypergraph operations of the batch text format `.dgt` (`+h v1 v2 .. [; w]`, `-h id`, `+i`, `-i`;
0.1 rejects them), and a binary batch format `.dgb` ("magic `DYNGBAT1`, versioned header
(container kind, dtypes, K, semantics flags), length-prefixed batch records, CRC32"; a streaming
`batch_reader` with pinned staging), which the M7 retrospective (deviation 3) moved to M8 to be
designed with the hypergraph batches. PLAN 5.8 asks for generators bit-exact per (seed, major
version) with distributions owned by the library, and `generators::legacy` reproductions of the
originals' streams (ESCHER `generateHypergraph`), checked at run time against libstdc++. PLAN
6.4.6 replays ESCHER-GPU's batches by content (its `deletedRows`) because A's generator reads the
state it deletes from.

The two originals load `.hg` files differently. ESCHER-GPU@abcf9b2 (`utils/utils.cpp`
`loadHypergraph`): a line is a comment only if its first character is `#` or `%`; separators are
space, tab, carriage return and comma; ids 1..2^31-2; each line sorted and deduplicated; empty
lines skipped; repeated hyperedges kept; the vertex count is the largest id. MOSP_ESCHER@4b86159
(`HypergraphGen.cpp` `loadHypergraphFile`): 64-bit signed ids (a sign allowed), no comments,
lines sorted and deduplicated, lines with more than `--maxcard` (25) vertices dropped, vertices
renumbered by first appearance, weights U[1, 100] from `std::mt19937_64(seed)` after one draw for
its virtual target vertex, and the virtual source and target hyperedges added around the rows.

## Decision

### 1. `.hg` files (`io/hypergraph_io.hpp`)

- `read_hypergraph<V, W>(path, hypergraph_read_options, hypergraph_file_info*)` returns a
  `hyperedge_list`; hyperedge i is the i-th kept line, so `hypergraph::from_hyperedges()` gives it
  id i. The line rules are ESCHER-GPU's (comments at the first character only, the four
  separators, blank lines skipped, ids sorted and deduplicated per line); the tokens are dynG's
  (whole decimal integers, an optional `-`, no `+`; MOSP_ESCHER also accepts `+` and reads `1-2`
  as two ids, dynG rejects both: ADR 0010's strict readers). Errors are `io_error` with path, line
  and column; nothing prints.
- Options: `index_base` (default 1, the convention of both originals' files), `dedupe_hyperedges`,
  `max_cardinality` (-1: none), `relabel` (`vertex_relabel::none`, `dense_sorted`,
  `first_appearance`), `random_weights` (`std::optional<weight_spec>{lo, hi, seed,
  hsosp_target_draw}`) and `num_vertices` (relabel none: the vertex count, -1 = largest id + 1).
  Filters run in file order: cardinality limit, then first occurrence, then the numbering.
- **The two originals are option sets**, static factories like the capacity presets:
  `hypergraph_read_options::escher_gpu()` (relabel none, base 1, nothing dropped, no weights:
  vertex v of the file is v - 1, the vertex count is the largest id, A's key k is id k - 1) and
  `hsosp(seed = 20260725, max_cardinality = 25)` (first appearance, U[1, 100] after the target
  draw). `escher_gpu()` accepts two inputs A's loader rejects (M8b review, recorded as
  deliberate): the id 2^31 - 1 (A's `kMaxVertexId` is 2^31 - 2 because its CBST keys are ids
  + 1; dynG's limit is the vertex type's: vertex 2^31 - 2, a vertex count of 2^31 - 1), and a
  file without hyperedges (A fails with "no hyperedges"; dynG returns an empty list, from which
  `from_hyperedges()` builds a hypergraph of 0 hyperedges, criterion 3). Neither occurs in A's
  datasets; both are tested. The virtual source and target hyperedges of H-SOSP are not part of the file's
  hypergraph: `hyper_sssp` adds them (M11); `weight_spec::hsosp_target_draw` reproduces only the
  engine draw the original makes for its target, so the weights stay bit-exact.
- **Weights in a file**: a weighted `.hg` line ends with `; w` (a token of its own or attached,
  `1 2;5`), w in [1, max(W)]; an unweighted read rejects them, a weighted read without
  `random_weights` needs them on every line. Neither original writes weights to `.hg` files, so
  the extension cannot conflict with their files.
- **No version field** (M8b review). This ADR said `.hg` is written at version 1, but a `.hg`
  file holds no version and the reader reads none; PLAN 5.9 lists `.dgt`, `.dgb` and `.dgc` as
  the formats with their own versions, not `.hg`. A version line (`%hg 1`) was considered and not
  adopted: ESCHER-GPU and dynG skip `%` lines, but MOSP_ESCHER's `loadHypergraphFile` rejects
  any non-numeric token, so the line would make dynG's unweighted files unreadable by one of the
  originals. `.hg` therefore stays unversioned: the 0.3 rules of `file_formats.md` (ESCHER-GPU's
  line format plus the `; w` extension) are its stable definition, and a later incompatible
  change needs a new format, not a new version of this one. Whether the author wants a version
  line instead is in the decisions for the 0.3.0 release (M8b retrospective).
- `write_hypergraph(path, list, index_base = 1)` writes members as listed plus `index_base`, then
  ` ; w` when weighted; it rejects what would not read back (an empty hyperedge, a member outside
  [0, n), a weight below 1). The vertex count is not written.
- `datasets/preprocess/benson_to_hg.py` is dynG's converter of Benson's datasets with both
  originals' preprocessing (`--variant escher` and `hsosp`); on coauth-DBLP-full both variants
  write the bytes the registry pins (one file for both originals).

### 2. Hypergraph batches in `.dgt` files

- `read_hyperedge_batches<V, W>` / `write_hyperedge_batches<V, W>` with `+h v1 v2 .. [; w]`
  (repeated members are kept: `apply()` drops and counts them) and `-h id` (id >= 0; dead and
  out-of-range ids are kept: `apply()` skips and counts them). `+i` / `-i` stay rejected with an
  `io_error` naming 0.4.
- **One container kind per file.** The first line may name the kind: `%dgt 1 graph` or `%dgt 1
  hypergraph` (`%dgt 1` alone stays valid, so every 0.1/0.2 file reads as before). A reader meets
  an operation of the other kind, or a header naming it, with an `io_error` that names the other
  reader; the hypergraph writer writes `%dgt 1 hypergraph`, the graph writer keeps `%dgt 1`
  (byte-identical files). A mixed file would need a container that takes both kinds of batch, and
  none does; splitting by kind keeps both readers strict.
- The writers reject what their reader would reject (`io/batch_checks.hpp`, shared with `.dgb`):
  negative ids, an empty insertion, a weight below 1. For graph batches this adds the negative-id
  check to `write_batches()` (such a file did not read back: a bug fix).

### 3. The binary batch format `.dgb` (format version 1)

Little-endian; the full layout is in docs/api/file_formats.md.

- **Header (64 bytes):** magic `DYNGBAT1`; format version (u32, 1); header size (u32, 64);
  container kind (u8: graph 0, hypergraph 1, `dyng::container_kind`'s values); dtypes (u8) of the
  vertex ids, the weights and the hyperedge ids (none 0, int32 1, int64 2); K (u32: weights per
  insertion, 0 or 1 for hypergraphs); flags (u64); the batch count (u64); 20 reserved zero bytes;
  CRC32 of the first 60 bytes.
- **Flags:** `deleted_members` (every hyperedge deletion carries the member set of the hyperedge
  it deleted: replay by content, PLAN 6.4.6), `expected_ids` (every insertion carries the id the
  generating run gave it: replay by key checked against `inserted_ids`), `insert_first` (those
  ids hold under `id_reuse::insert_first`). These are the "semantics flags" of PLAN 5.7; graph
  files have none. A reader rejects an unknown flag ("written by a newer dynG").
- **Records:** payload length (u64), CRC32 of the payload (u32), 4 zero bytes, then the payload:
  u64 counts (batch id, deletions, insertions[, members, deleted members]) followed by the arrays
  in a fixed order, each starting at a multiple of 8 bytes (zero padding), so a payload in an
  8-aligned buffer is used in place (no copy per array, and one host-to-device copy per record).
  The payload length must equal what its counts imply exactly.
- **Bounded reading (the M6b lesson: readers trusted a size field).** Every size field is checked
  against the rest of the file before anything is allocated: the batch count against what the
  file can hold, a record length against the bytes left, each count against the payload's bytes
  left before it is multiplied. A file of N bytes makes the reader allocate O(N) at most. Then
  the CRC32, then the contents (offsets from 0, strictly increasing to the member count, ids in
  range, weights >= 1, batch ids strictly increasing). Truncated, size-lying, damaged, wrong-kind,
  newer-version files, trailing bytes and a count that does not match the records are `io_error`s
  with the path and the record's index and byte offset; a request for the wrong types is an
  `invalid_argument_error`. (M8b review: the whole-file readers raised `invalid_argument_error`
  for a file of the other kind; they now raise `io_error` naming the right reader, as `.dgt`
  does, and check kind and types from the header before reading a record. After an `io_error`
  of `next()` the reader is fail-stop: the stream is past the damaged record, so every later
  `next()` and view throws instead of returning the following record as valid.)
- **`batch_reader`** streams one record at a time; its views are valid until the next `next()`.
  With CUDA resources each record is read into one of two pinned host buffers from the resources'
  staging memory resource and copied to a device buffer asynchronously on the resources' stream;
  before the host reuses a pinned buffer it waits for that buffer's copy only (a
  `cuda_stream_fence` per buffer), so reading batch k + 1 overlaps the work on batch k. Templated
  accessors (`edge_batch<V, W>()`, `hyperedge_batch<V, W>()`, `annotations<V>()`) check the
  header's types.
- **`batch_writer`** writes the header with a batch count of 0, appends records, and `close()`
  writes the count (an unclosed file is rejected). `write_binary_batches`,
  `write_binary_hyperedge_batches` and the two `read_binary_*` helpers wrap them. `.dgt` and
  `.dgb` hold the same batches (round trips both ways are tested).
- The seeds of its fuzzer are written by `cpp/fuzz/make_dgb_seeds.py` from the specification
  (Python's `struct` and `zlib.crc32`), an implementation independent of the library's writer,
  and read by a unit test.

### 4. The dumps on the parity side

- The "ESCHER dump" is the batch records of M8a's CBST traces (`escher_gpu.build`,
  `escher_gpu.update`). `cpp/tests/parity/escher/escher_dump.{hpp,cpp}` decodes them (the loaded
  rows, per batch the deleted ids and rows, the inserted rows and ids, all in dynG's 0-based ids),
  parses A's command line with A's rules, and holds the **by-content resolver**: each deleted
  member set (sorted, deduplicated) resolves to the smallest live id with that set not taken
  earlier in the batch. It is test-side code (no public API): the API review of step 5 decides
  whether a resolver belongs in `dyng::testing`. `dyng_hypergraph_ids` uses it for its by-content
  replay (refactored onto it; the replay records of steps 1-3 are unchanged), and
  `cbst_trace::trace_source` moved into the dynG-free trace library so that tools can open
  traces.
- `tools/compat/escher` (`dyng-compat-escher`) converts a dump to `<prefix>.hg` (A's loaded
  hypergraph in A's ids), `<prefix>.dgb` (A's batches with `deleted_members` and
  `expected_ids`) and `<prefix>.txt` (A's command line, vertex count and H2V payload capacity),
  and replays them by id and by content (M9's replay path).

### 5. Generators (`generators/hypergraphs.hpp`)

- `uniform_hypergraph<V, W>(options)` and `random_hyperedge_batch<V, W>(state, options)` (the
  state as h2v rows by id and weights, or a hypergraph), with the mixes `insert_only`,
  `delete_only`, `mixed`, `reuse` (delete-then-reuse within a batch), `heavy_delete`,
  `delete_all`, `reinsert`, `larger` (larger than the hypergraph), `duplicates`, `dead_ids`,
  `skewed` (a hyperedge of min(n, `skewed_cardinality` = 1,100) members), `limits` (ids at the
  index limits) and `empty`.
- **Library-owned distributions:** `std::mt19937_64(seed)` (specified by the standard), uniform
  integers by Lemire's multiply-and-reject on 128-bit products and Fisher-Yates passes, both the
  library's code; the streams are pinned by golden hashes in the tests (a change is a major-version
  change, PLAN 5.9).
- The conformance kit's hypergraph mixes and models now come from these generators (seeded from
  the kit's engine); the kit derives what the commit must report from its model. The graph suites
  and every test count are unchanged.

### 6. `generators::legacy` for ESCHER-GPU

- `escher_hypergraph<V, W>(std::mt19937_64&, escher_hypergraph_options)` is `hyperedge2vertex`;
  `escher_batch_generator(std::mt19937_64&, escher_batch_options).next(res, hypergraph)` is
  `BatchGenerator::next`, reading the live ids of the container in increasing order (A's live keys
  in key order) and their rows. Both share the caller's engine and consume it in A's order; the
  `<int>` and `<size_t>` distributions of A are `detail::legacy_uniform_int64` over the range's
  width (libstdc++'s 128-bit path for a 64-bit engine; the signed width computed as libstdc++
  does).
- `standard_library_matches_legacy_streams()` is the run-time check of PLAN 5.8: it draws from
  both the library's reproductions and the build's standard library and reports whether they
  agree (true under libstdc++ 11+, which built the originals).
- Driven against a container under {erase_first, escher_gpu()}, the generators reproduce A's
  hypergraphs and batches; from the second batch on that needs the container's ids to be A's, so
  the check is also an end-to-end proof of the id parity.

## Consequences

- Criterion 13: `read_hypergraph(coauth.hg, escher_gpu())` equals A's loaded rows (2,467,389
  hyperedges) and `hsosp(20260725, 25)` equals B's (2,466,792 hyperedges, in B's member order,
  its first-appearance numbering, vertex count and weights); `benson_to_hg.py` writes the pinned
  bytes in both variants (parity/results/M8b.md section 4).
- Criterion 16: the legacy generators give all 20 synthetic hypergraphs and all 180 batches of
  A's `make test`, the coauth50k batches (sample mode) and the syn1M seed 1-3 hypergraphs and
  batches.
- A `.dgb` file is bigger than the arrays it holds only by its 64-byte header, 16 bytes and the
  padding per record; `deleted_members` adds the deleted member sets (A's batches carry them
  anyway).
- The kit's batches changed (same mixes, other draws): the hypergraph suites were re-run green;
  the planted kit bug is still detected.
- Not done here: Python bindings of these functions (step 5), the API review and the freeze
  (step 5); a `.dgb` converter for MOSP_ESCHER's dumps (M11 needs vertex-kind batches first).
