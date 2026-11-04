# ADR 0087: Incidence edits in the batch files (`.dgt` `+i` / `-i`, the `.dgb` flag `incidence_edits`)

- **Status:** Accepted. The format decisions are the author's: HD10 (the `.dgt` operations
  `+i h v` / `-i h v`; the `.dgb` flag bit 3 `incidence_edits`, format version 1 kept; the files
  of 0.3 written byte for byte as before; bit 4 and the weight code 3 reserved), **decided by the
  author on 2026-10-28** (GOVERNANCE.md, approvals log, the rows of 2026-10-28). The technical
  parts (the record layout, the checks, the tests, the converter) are accepted under delegation
  (GOVERNANCE.md, "Delegation of technical ADRs"). It carries out M11a step 2 (PLAN 5.7; the M11a
  acceptance criteria 9 and 11).
- **Date:** 2026-10-29
- **Deciders:** S M Shovan (HD10, 2026-10-28); the AI assistant, on the author's behalf, for the
  technical parts (M11a, step carried-host-incidence-slack)
- **Builds on:** ADR 0041 (section 8: the batch's four groups), ADR 0045 (the `.dgb` format and
  its stable tier), ADR 0085 (the edits' semantics).

## Context

0.3 made `.dgt`, `.dgb` and `.hg` stable-tier formats (version 1): a later release reads every
version-1 file as documented, and a change of rules is a new version. 0.3 reserved the `.dgt`
operations `+i` / `-i` (rejected, naming 0.4), and its `.dgb` readers reject any flag they do
not know. The writers refused batches with incidence edits (`not_supported_error`). With 0.4
applying the edits (ADR 0085), the batch files must carry them, MOSP_ESCHER's vertex-kind
sessions must convert to them, and no 0.3 file may change.

## Decision

1. **`.dgt`** (`%dgt 1`): in a hypergraph file, `-i h v` removes vertex v from the hyperedge with
   id h, `+i h v` adds it (h >= 0; v >= 0 and, with a vertex count, below it); removals and
   additions each keep file order; `apply()` takes them after the deletions and before the
   insertions whatever the line order. In a graph file they are hypergraph operations (an
   `io_error` naming `read_hyperedge_batches()`). The writer writes `-h`, `-i`, `+i`, `+h` per
   batch. The version stays 1: the operations were reserved by version 1 for 0.4, and a 0.3
   reader rejects them with its message.
2. **`.dgb`** (format version 1): `batch_file_flags::incidence_edits = 8` (hypergraph files): the
   record's count block has two more counts (removals r, additions a: 56 bytes instead of 40),
   and after every array of 0.3 come the removal ids [r] (int64), removal vertices [r] (vertex
   type), addition ids [a], addition vertices [a], each checked by the reader (ids >= 0,
   vertices in range). `known_graph` (no flags) and `known_hypergraph` are new constants and
   `known` is their union (an api-change line); `header_problem` checks the flags per kind. Bit 4
   (`vertex_records`, graph files) and weight code 3 (float32, graph files) are reserved in the
   page and rejected. The page states the rule both milestones rely on once, in general: an
   additive flag, weight code or operation that version 1 reserved and that every earlier reader
   refuses loudly (`io_error`) is not a silent change of version 1; bit 4 and code 3 are
   reserved for label_propagation in 0.4 (M10b declares them; the M11a review corrected "a later
   release" in the page and in `batch_file_flags::known_graph`'s comment, documentation only).
3. **Flags fixed at open.** The whole-file writer sets `incidence_edits` only when a batch has an
   edit, so a file of batches without edits is the file 0.3 wrote, byte for byte
   (`BinaryBatches.TheFilesOf03AreWrittenByteForByteAsBefore` rewrites the 0.3 seeds, which
   `make_dgb_seeds.py` writes from the specification, and the converted ESCHER-GPU dump);
   `batch_writer` refuses a batch with edits for a file opened without the flag
   (`invalid_argument_error`, replacing 0.3's `not_supported_error`: a documented-behaviour
   change of the stable `batch_io` surface, an api-change entry).
4. **The old reader refuses.** dynG 0.3.0 (tag v0.3.0, built from `git archive` in a scratch
   directory and removed after) reads 0.4's files without edits and rejects those with edits:
   `.dgb` "invalid header: unknown flags 8 (written by a newer dynG?)" (its
   `batch_binary.cpp:155`, `flags & ~known`), `.dgt` "the incidence operation '-i' is reserved
   for the incidence edits of hypergraph batches (0.4)"
   (`parity/results/M11a-old-reader-refusal.json`).
5. **Fuzzing and round trips.** The `.dgb` fuzzer's hypergraph copy carries the edits and its
   round trip compares them; a seed `hyper_incidence_edits` written from the specification; the
   `.dgt` fuzzer checks the edits' ranges and round trip, with two seeds. Text and binary round
   trips, `.dgt` <-> `.dgb`, the refusals (negative ids or vertices, a vertex out of range, an
   unknown flag) are unit tests.
6. **The Python layer** takes incidence edits with M11b: until then its readers raise
   `NotSupportedError` for a `.dgb` file with the flag or a `.dgt` batch with an edit instead of
   dropping them (a deviation, the retrospective).
7. **The converter** `dyng-compat-hsosp to-dgb` (`tools/compat/hsosp`, built after
   `dyng-compat-escher`) writes every session MOSP_ESCHER loaded as `.hg` (its virtual rows at
   their positions, weight 1), `.dgb` (expected ids = finalIds - 1, `insert_first`, the edits as
   (B id - 1, v); the operations on a virtual row filtered and counted) and `.txt` (the vertex
   count, the virtual rows' ids and vertices, the Caps and CSR knobs, per batch B's skippedOps and
   the filtered counts); `replay` checks the ids and skip counts on slack_csr storage. Converting
   twice gives the same bytes; the digests of the T and P50 conversions are in
   `parity/goldens.toml` (`[sets.hyper_sssp_mosp_escher.converted]`, written by
   `hsosp_goldens.py convert --write-toml`, kept by a new export).

## Consequences

- 0.3's files, readers and writers behave as before for every batch without edits; a 0.3 reader
  meets a 0.4 file with edits with an error, never a silent drop.
- The reserved bit and weight code leave room for the vertex records and float weights of graph
  files without a new format version.
