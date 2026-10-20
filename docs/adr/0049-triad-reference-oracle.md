# ADR 0049: The triad oracle `testing::triad_reference` and its independence

- **Status:** Accepted under delegation (2026-10-15; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out the oracle of M9a step 2 (PLAN 5.8, 8.2; the M9a
  acceptance criterion 4). No approved rule changes: PLAN 5.8 names `triad_reference` among the
  public oracles of `dyng::testing`, and the author's decision D3 (2026-10-15) makes it the
  independent oracle of the incremental host engines.
- **Amended by:** ADR 0057 (2026-10-20): the oracle computes its products and sums in unsigned
  128-bit integers and throws `std::overflow_error` past 2^64 - 1 (its 64-bit arithmetic wrapped
  with the engines at the limit).
- **Date:** 2026-10-15
- **Deciders:** the AI assistant, on the author's behalf (M9a, step
  oracle-harness-driver-prereads)
- **Builds on:** ADR 0037 (the CBST host model and its isolation check), ADR 0042 (the line-graph
  oracle `rebuild_line_graph` and its isolation check), ADR 0048 (the golden set).

## Context

`triad_count` keeps ESCHER-GPU@abcf9b2's (A's) four counters exact under hyperedge batches. PLAN
5.8 asks for an oracle that "shares no code with the algorithms it checks"; D3 makes it the
independent check of the sequential and OpenMP backends, which are now incremental host engines
(a transliteration of A's wedge engine) rather than recounts, so a recount can no longer serve as
their oracle. A's own oracle is its CPU reference (`utils/reference.cpp`, behind `--reference`),
which M9a step 1 cross-checked against A's GPU counters on 90 runs (ADR 0048 section 3). It
classifies a triangle with A's table `ESCHER_MOTIF_ID_TO_INDEX` (`include/motif_table.hpp`), the
same table A's kernels use (`kernel/motif_utils.cuh`): copying that table into the oracle would
make the oracle and the port share the one piece of data a wrong port would most likely get wrong.

## Decision

1. **API** (`cpp/include/dyng/testing/triad_reference.hpp`, tier `tracked`, host-only):
   `triad_pattern {h_motif, stathyper_type1, stathyper_type2, stathyper_type3}` (the oracle's own
   enumeration: it must not include `dyng/triad_count.hpp`), `triad_reference(offsets, members,
   num_vertices, pattern, {threads})` (30 values or 1), `triad_reference_all` (the requested
   patterns from one line graph), `triad_reference_removed(G, D)` and `triad_reference_added(G',
   I)` (per bin `counts(G) - counts(G \ ids)`, for the statistics of small test sets), and the
   class map: `triad_class_representatives()`, `triad_venn_permuted()`, `triad_venn_canonical()`,
   `triad_motif_class()`, `triad_motif_class_map()`, `triad_venn_id()`. The input is the layout of
   `hypergraph::h2v()` or of a `hyperedge_list` on the host (int64 offsets, int32 members; an
   empty row is a dead id; repeated members count once). Counts are `uint64` (A's reference uses
   `long long`; every count is non-negative).
2. **Algorithm: A's CPU reference, transliterated.** The line graph is rebuilt per hyperedge from
   the hyperedges of its members (`referenceLineGraph`; the vertex-to-hyperedge lists come from
   the incidences sorted by vertex, so no array is indexed by vertex id and ids up to 2^31 - 2
   work); every triangle i < j < k is enumerated by intersecting sorted neighbour rows
   (`forEachTriangle`, per-thread accumulators, OpenMP `schedule(dynamic, 64)`); h_motif
   classifies each triangle by its seven Venn regions (`referenceMotifCounts`), type 1 sums
   C(|e|, 3), type 2 sums C(a, 2) x b x c over adjacent pairs, type 3 sums x x y x z of the
   pairwise-only overlaps over triangles (`referenceType1/2/3`). Sequential (`threads = 1`) and
   OpenMP (`threads = 0` or N) give the same counts (integer sums).
3. **The class map is derived, not copied.** A class is an orbit of the 128 Venn ids under the
   six permutations of the three hyperedges (a permutation moves the "X only" bits and the "X and Y
   only" bits together); ids with fewer than two intersecting pairs (a pair intersects when its
   pairwise-only region or the triple region is non-empty) are in no class. The 30 orbits are
   numbered in ESCHER's (MoCHy's) order by the documented list of their canonical ids (the
   smallest id of each orbit): 64, 65, 67, 71, 72, 73, 76, 75, 77, 79, 88, 90, 89, 91, 93, 95,
   120, 121, 123, 127 (the triple region non-empty), 24, 26, 25, 27, 29, 31 (the open triads), 56,
   57, 59, 63 (the three pairwise-only regions, no triple region); the header's table names their
   regions. The list is the only ESCHER-specific datum, and it is a list of 30 orbit
   representatives, not the 128-entry table: a wrong entry is caught by the orbit test (each
   representative must be canonical and the classes must be the orbits) and by the comparison
   with A's table.
4. **Independence, checked.** The oracle is its own object library (`dyng_testing_triad_reference`,
   linked into `dyng::testing`); CTest `triad_reference_isolation` runs `check_model_isolation
   .cmake` on it (ADRs 0037 and 0042): the preprocessor's dependency list of its source names no
   project file but its own header (so nothing of `cpp/src/algorithms/triad_count`, of any engine
   or of the container is included), and its object file references no `dyng::` symbol (it links
   nothing of the library). Errors are `std::invalid_argument` / `std::bad_alloc` (it may not
   include `dyng/core/error.hpp`), as `rebuild_line_graph`'s (ADR 0047, finding 15).
5. **Tests** (`dyng_triad_reference_tests`, labels cpu testing triad_count): the derived map equals
   A's `ESCHER_MOTIF_ID_TO_INDEX` on all 128 ids, read from a **test-only fixture**
   (`cpp/tests/testing/escher_motif_table.hpp`, with A's provenance line; the oracle never
   includes it); the classes are the permutation orbits (32 ids in no class, 96 in 30 classes);
   one triple per class (one vertex per non-empty region) lands in its bin alone for the 24
   reachable classes, and in no bin for the 6 open triads; hand cases of the three StatHyper types
   and of duplicate hyperedges, dead rows and repeated members, a hyperedge of 1,024 members
   (C(1024, 3) in 64 bits), vertex ids at 2^31 - 2; an independent brute force over all triples
   of 40 random hypergraphs (classified with the fixture) on 1, 3 and the default thread count;
   `removed` / `added` against the brute force; malformed input refused.
6. **Against A itself** (`parity/triad_replay.py singles`, recorded in
   `parity/results/M9a-oracle-singles-escher_gpu.json`): the 30 single-triad `.hg` files run
   through A's **unpatched** `build/main --input F --batches 0` and through the oracle
   (`dyng_triad_replay --patterns h_motif`); A prints the same 30 bins as the oracle on all 30,
   a single 1 in bin k - 1 for the 24 reachable classes. The `.hg` files are not committed (the
   test regenerates the triples; the record keeps their lines and SHA-256).
7. **The planted mutation** `DYNG_MUTATION_TRIAD_REFERENCE_CLASS_MAP` swaps classes 2 and 3 of the
   representative list; CTest `testing.mutation.triad_reference_class_map` (a copy of the tests
   compiled with it) must fail, and `parity/mutate.py` lists it in its suite `triad_reference`.

## Consequences

- The incremental host engines (steps 3-4) are checked against an algorithm that shares neither
  code nor the class table with them; the CUDA engine's table (A's, ported) is checked against
  the derived map by the same test.
- The oracle is quadratic in the line-graph degrees: the go/no-go replays of step 2 run it on the
  syn1m grid (about 10 s per evaluation of all four patterns), and on coauth-DBLP only for
  type 1 (A's own h-motif reference took 143.5 s there).
- Python access (`dyng.testing.triad_reference`) is M9b's.

## Alternatives considered

- **A's table in the oracle.** Rejected: the table is exactly what a port copies; a shared table
  would hide a wrong entry from both.
- **Hard-coding the 30 classes as region lists in the code without the orbit construction.**
  Equivalent data, but the orbit construction lets a test prove that the list is a set of orbit
  representatives (and that exactly 96 of the 128 ids get a class), independently of A.
- **An oracle over the public `h2h()` view.** Rejected: the line graph would come from the
  container's own merge; the oracle rebuilds it from the rows.
