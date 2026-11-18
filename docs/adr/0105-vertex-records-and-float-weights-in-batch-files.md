# ADR 0105: Vertex records and float weights in batch files (AD13, M10's section of the joint format decision)

- **Status:** Accepted (2026-11-11). The format decision is the author's: **AD13 (sheet D1),
  decided by the author on 2026-10-28 jointly with M11's HD10** (GOVERNANCE.md's approvals log,
  the row "The file formats of 0.4 (sheet D1-D2)"). The record layout, the messages, the tests and
  the fuzz coverage are accepted under delegation (GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. M10b step 2 (the M10b acceptance criterion 6; PLAN 5.7). M11's
  section of the same decision is ADR 0087 (incidence edits, bit 3, the per-kind masks).
- **Date:** 2026-11-11
- **Deciders:** S M Shovan (AD13 / HD10); the AI assistant, on the author's behalf (the rest)

## Context

label_propagation (0.4) applies vertex batches (arrivals with an optional seed label, the edges
they bring, departures) to float graphs. PLAN 5.7 lists `+v u [label]` / `-v u` among the `.dgt`
graph operations and leaves the `.dgb` layout to the release; 0.3 reserved both and refused them
loudly (`.dgt`: "reserved for the vertex batches of label_propagation"; `.dgb`: the weight code 3
and flag bit 4 rejected as unknown). Both formats are stable tier at version 1 (GOVERNANCE.md,
2026-10-22). AD13 decided how 0.4 adds to version 1:

- `.dgt` stays `%dgt 1`, with `+v u [label]` / `-v u` (graph files) and float weight tokens for
  float graphs only; M11's `+i h v` / `-i h v` are hypergraph operations;
- `.dgb` stays version 1, with `batch_dtype::float32 = 3` in a graph file's weight byte only and
  `batch_file_flags::vertex_records = 16U` (bit 4, graph files only), M11's bit 3
  `incidence_edits` for hypergraph files, the masks `known_graph` / `known_hypergraph` (M11a
  defined them; `known` their union), the header checked per kind, the flags fixed at open;
- old readers refuse every addition loudly, and a file using none of them is byte-identical to
  what 0.3 writes; `docs/api/file_formats.md` states once that such an addition is not a silent
  change of version 1 (M11a wrote that sentence; it is kept, not repeated).

## Decision

1. **`.dgt`.** `read_batches` reads `+v u [label]` (label `-1`, `0`, `1`, or none for -1) and
   `-v u` into `edge_batch::insert_vertex` / `delete_vertex` for every graph type (the batch
   semantics decide what they mean; only `net_effect()` graphs take them, ADR 0069), the ids by the
   edge rules. A hypergraph file refuses them (an `io_error` naming `read_batches()`), as a graph
   file refuses `+i` / `-i`. With `weight_t = float` a weight token is parsed by
   `std::from_chars<float>` (general format) and must be finite: NaN, infinities and values
   beyond the float range are an `io_error` at the token; integer graphs keep `parse_integer`
   and refuse such a token. `write_batches` writes, per batch, `+v` lines (` label` only for 0 or
   1), `-v` lines, then the 0.3 lines; float weights in their shortest round-trip form
   (`std::to_chars(float)`). A batch without vertex operations is written as 0.3 writes it. A
   batch that has vertex operations before its first insertion keeps them when the reader
   learns K (the rebuild of the K-less batches copies them).
2. **`.dgb`: code 3.** `dtype_of<float>()` is `float32` (4 bytes, the IEEE bits). `header_problem`
   accepts code 3 only in a graph file's weight byte: a hypergraph file with it, and code 3 as the
   vertex type, are rejected with messages naming the rule. The reader refuses a non-finite
   float32 weight (an `io_error` at the record), as the text readers do.
3. **`.dgb`: bit 4 and the 48-byte count block** (M10's layout proposal for the joint record
   rule, recorded first in `docs/api/file_formats.md`). With `vertex_records` every graph record's
   count block is 48 bytes: the batch id, d, i (0.3's 24 bytes), then n inserted vertices, l seed
   labels (0 or n) and x deleted vertices; after 0.3's arrays (deletions, insertions, weights)
   follow the inserted vertices [n] (vertex type), the labels [l] (int8 in {-1, 0, 1},
   zero-padded to 8 bytes) and the deleted vertices [x] (vertex type). Storing l separately keeps
   a view with no label array distinct from one whose labels are all -1 in `.dgb` (amended after
   M10b's review: `batch_reader`'s views give back l = 0 as written; the two mean the same, no
   seed, and the owning paths normalize them: `edge_batch` always holds one label per inserted
   vertex, so `read_binary_batches` gives -1 labels for l = 0, and `.dgt` writes no token for
   either; docs/api/file_formats.md). `known_graph = vertex_records` (an api-change line in
   `batch_io.hpp`; `known` stays the union). A hypergraph file with bit 4 is refused ("unknown
   flags 16 for a file of hypergraph batches", so that M11a's test of that refusal is unchanged),
   a graph file with bit 3 too ("flags 8 in a file of graph batches").
4. **Flags fixed at open.** `batch_writer` takes its header's flags at construction; a batch with
   vertex operations in a file opened without `vertex_records` raises `invalid_argument_error`
   before anything is written; a file opened with it takes edge-only batches (zero vertex counts).
   `write_binary_batches` sets the flag only when some batch has a vertex operation, so a file of
   edge batches (also float ones) has the 24-byte count block and no flag. The batch checks are one
   function for both formats (`check_edge_batch_arrays`, with a `vertex_records` switch): ids
   `>= 0`, 0 or one label per inserted vertex in {-1, 0, 1}, finite float weights.
5. **Float edge lists** (PLAN 6.4.7's weighted edge-list path, the kNN recipe's files):
   `read_edge_list<V, float>` parses the weight columns with the same rule, keeping the float bits
   in the parser's int64 weight array; `write_edge_list<V, float>` writes the shortest round-trip
   form and refuses a non-finite weight. The integer paths are unchanged; `read_matrix_market`,
   `read_csr_triplet` and `read_legacy_batch` stay integral (MOSP and CycleEnumeration-GPU
   formats).
6. **The readers of 0.3 refuse the additions loudly.** 0.3's `header_problem` rejects weight code
   3 ("weight type 3 (none = 0, int32 = 1 or int64 = 2)") and, in a graph file, any flag ("unknown
   flags 16"); 0.3's `.dgt` reader rejects `+v` / `-v` ("reserved for the vertex batches of
   label_propagation") and a float token ("expected a decimal integer"). The same code paths
   remain in this library for the codes and bits it does not know (a weight code 4, flag 32: the
   tests `VertexBatchBinary.ReadersRejectWhatTheyDoNotKnow`), and the 0.3-era seeds still read
   and are written back byte for byte (M11a's `TheFilesOf03AreWrittenByteForByteAsBefore`, the
   corpus replays).
7. **Fuzzing.** `fuzz_batch_text` gains the float graph type (option 2 = 6) and checks the vertex
   operations and their round trip; `fuzz_batch_binary` instantiates int32 / float32 and checks
   the vertex records and their round trip; `fuzz_edge_list` gains the float result type;
   `fuzz_result_io` gains `read_labels`. New seeds: `batch_text/vertex_ops_float`,
   `batch_binary/graph_float_vertex_records` (written by `make_dgb_seeds.py` from the
   specification), `edge_list/float_weights`, `result_io/labels`.

## Consequences

- The legacy generator's batches at 5k5 and 50k round-trip through `.dgt` and `.dgb` to identical
  batches (the float weights bit for bit; `VertexBatchFiles.LegacyGeneratorBatchesRoundTrip`).
- Python and the command line read and write these files in M10b step 3 (their readers refuse
  `vertex_records` and code 3 until then, as 0.3's did).
- A new weight code or flag remains a version question under the rule of the format page's
  introduction.

## Alternatives considered

- **Always n labels** (-1 filled in). Rejected: an `edge_batch_view` may carry no label array,
  and the round trip would not give back the same view.
- **A new format version 2.** Rejected by AD13: the additions are refused loudly by every
  earlier reader.
- **Accepting NaN / infinity in float tokens.** Rejected: label_propagation's similarity weights
  are finite by contract, and a non-finite weight would poison the sums silently.
