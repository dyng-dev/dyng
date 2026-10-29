# ADR 0067: The DynLP legacy generator, byte-identical to LabelPropagation-CUDA, and the reproduction of libstdc++ 12's draws

- **Status:** Accepted under delegation (2026-10-29; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out M10a step 2 (PLAN 5.8, 6.4.7, 8.3; the M10a
  acceptance criterion 8). The rule it implements is the author's: AD8 (the generator
  byte-identical to the original's; libstdc++-12's float draws and its 32-bit-engine shuffle
  reimplemented and checked against `std::` at run time instead of a toolchain pin; the large
  digests on GCC 12.2 only, the 5k and 50k digests also in hosted gcc-13 and clang-18 CI),
  **decided by the author on 2026-10-28** (GOVERNANCE.md, approvals log, the row "The parity rules
  of 0.4 (sheet C1-C8)"); the names are AD12's (the row "The new public names of 0.4").
- **Date:** 2026-10-29
- **Deciders:** the AI assistant, on the author's behalf (M10a, step oracles-generator-writers)

## Context

`label_propagation`'s parity runs replay the stream of LabelPropagation-CUDA@a276a3a's (O's)
generator: `generateSparseGraph` (an Erdős–Rényi graph over a random spanning tree, with
similarity weights by class), `generateBatches` (the arrivals, the ground truth, the edges of each
batch and the deletions) and the published threshold `computeAvgEdgeWeight`. PLAN 8.3 planned to
pin GCC / libstdc++ for the legacy generator goldens, because O's stream depends on three
implementation-defined pieces of libstdc++: `std::uniform_int_distribution<int>`, `std::shuffle`
with a 32-bit engine (`std::mt19937`), and `std::uniform_real_distribution<float>` (through
`std::generate_canonical<float, 24>`). dynG already reproduces the first (`legacy_uniform_int`,
`cpp/src/util/rng.hpp`) and the 64-bit-engine shuffle (`legacy_shuffle`), while
`generators::legacy::mosp_changes` calls the real `std::shuffle` and needs libstdc++
(`legacy.cpp:334`).

## Decision

### 1. The public surface (provisional until M10b's freeze)

`<dyng/generators/vertex_arrival.hpp>` (in `dyng/dyng.hpp`, an additive api-change):

- `generators::vertex_arrival_options {total_vertices, average_degree, num_batches, seed,
  ground_truth_fraction = 0.01f, deleted_fraction = 0.09f}`: O's DynLPConfig fields with O's
  defaults (the float fractions are O's float fields);
- `generators::vertex_arrival_batch<vertex_t>`: one batch as plain arrays: `insert_vertices` (O's
  insertedNodeIds order), `insert_vertex_labels` (-1, or the class by O's `< 0.5f` rule),
  `edge_sources` / `edge_destinations` / `edge_weights` (u < v, O's order), `delete_vertices`
  (O's order). M10a step 3 adds the conversion to `edge_batch` once it carries vertex arrivals;
- `generators::vertex_arrival_stream<vertex_t, edge_t>`: `graph` (the full CSR, both directions,
  sorted rows, one float weight column), `true_labels` (int8), `batches`, `row_capacity` (the full
  degrees: the slot capacities of `graph::with_capacity`), `similarity_threshold` (tau, bit-equal
  to computeAvgEdgeWeight), `max_batch_vertices`, `max_batch_edges`;
- `generators::legacy::dynlp_vertex_arrival_stream<vertex_t = int32, edge_t = int32>(res, opt)`,
  instantiated for `<int32, int32>` and `<int32, int64>`.

Inside the library, `detail::dynlp_legacy_generate` (`cpp/src/generators/dynlp_stream.hpp`) keeps
O's own shapes (CSRGraph with 32-bit offsets, float true labels, BatchData with the ground-truth
ids in O's shuffled order and their float labels): the parity tests compare those arrays with O's
recorded ones field for field, so nothing of O's output, not even an order the public stream does
not carry, is left unchecked. The library's own vertex-arrival generator (library-owned
distributions) is M10b's.

### 2. The reproduction of libstdc++ 12 (instead of a toolchain pin)

`cpp/src/util/rng.hpp` gains the twins of the three remaining draws, each written from the
headers of this machine's libstdc++ 12 (`bits/random.tcc`, `bits/stl_algo.h`,
`bits/uniform_int_dist.h`; the algorithms are the same in libstdc++ 13, 14 and 16, compared on the
conda package caches):

- `legacy_shuffle32` (std::shuffle with a 32-bit engine): when `(2^32 - 1) / n >= n` (n <= 65535),
  one single swap for an even count and then the swap positions of two successive elements from
  one `uniform_int_distribution` draw in `[0, (i + 1)(i + 2) - 1]` (`__gen_two_uniform_ints`);
  otherwise one draw in `[0, i]` per element; every draw is a 32-bit-engine draw, i.e.
  `legacy_uniform_int`;
- `legacy_generate_canonical_float` (`generate_canonical<float, 24>` with a 32-bit engine): one
  engine call, `float(x) * 1.0f / float(2^32)` in float, and the header's correction of a result
  >= 1 (x near 2^32 rounds to 2^32 in float) to `nextafter(1.0f, 0.0f)`;
- `legacy_uniform_real_float(a, b)` (`uniform_real_distribution<float>`): `(u * (b - a)) + a`, two
  separately rounded float operations.

The twins are exact, so the fallback of the step's plan (calling `std::shuffle` under
`standard_library_matches_legacy_streams()`, as `legacy.cpp:334` does) is not needed: the stream
does not depend on the standard library. The runtime comparison with `std::` runs on libstdc++
builds (`cpp/tests/generators/vertex_arrival_test.cpp`: shuffles of 0 to 100,001 elements across
the paired / single boundary at 65,535, 200,000 float draws per distribution and seed, the >= 1
correction with a fixed engine), and `standard_library_matches_legacy_streams()` now also compares
DynLP's draws (`detail::dynlp_draws_match_standard_library`, a documentation-only addition to the
function's description). Another standard library makes that check false and leaves the stream
unchanged.

### 3. No floating-point contraction

O's float arithmetic (the weights' `(b - a) * u + a`, the counts `int(size * fraction)` as a float
product) was compiled for x86-64 without FMA, so no product was fused. `legacy_dynlp.cpp` and the
test file that compares the twins with `std::` are compiled with `-ffp-contract=off` (GCC and
Clang; `cpp/CMakeLists.txt`, `cpp/tests/CMakeLists.txt`), so a build with `-march=native` (FMA on
this CPU) or on an architecture that contracts by default still computes O's values.

### 4. The draw order and O's decisions

Exactly O's: the graph from `std::mt19937(seed)` (the class shuffle, the permutation shuffle, the
spanning tree's weights, then the random edges), the batches from `std::mt19937(seed + 12345)`
(the two class shuffles, the remaining vertices, per batch one shuffle for its ground truth and one
for its deletions). O's open-addressing EdgeSet and its hash, the stop after 10 x target
consecutive failed draws (a 64-bit counter), the rejection of more than 2^31 - 1 directed entries
(O's 32-bit offsets; `invalid_argument_error`, whatever `edge_t` is), `max(2, int(n * f))` initial
and `max(1, int(arrivals * f))` per-batch ground truth, `int(arrivals * 0.09f)` deletions among the
active unlabelled vertices (a sorted vector, the iteration order of O's std::set). With one batch,
only b0 exists (the remaining vertices never arrive), as in O. Fractions outside [0, 1] are refused
(O does not check them; a documented difference that no corpus case reaches).

### 5. Threads

Parallel where O is parallel, with the host threads of the resources: the CSR's degrees and fill
with atomic offsets and the per-row insertion sort (the rows are sorted and their columns
distinct, so the result does not depend on the fill order), and the batch edge lists counted and
filled per chunk of rows at prefix offsets (O's 1024 chunks, which keep the sequential order). No
region allocates, so none needs `region_exceptions`.

### 6. The writers (internal)

`text_writer::put_fixed6` (O's putFixed6: `std::to_chars(double(x), fixed, 6)`) and
`detail::write_dynlp_graph_files` / `write_dynlp_labels` (`cpp/src/io/dynlp_text.{hpp,cpp}`) write
O's `graph/csr{RowPtr,ColInd,Values,TrueLabels}.txt` and `dynlp_labels.txt` bytes; the public
`io::write_labels` is M10b's (AD13 D2).

## Outcome (M10a step 2; parity/results/M10a.md section 9)

- `compare.py label_propagation generator` over the whole corpus G (14 cases: 5k5, 50k, 500k by
  bytes; 5m, 5m1 and 50m by the recorder's SHA-256; seeds 1-3 at 5k5 and 50k; average degree 3 and
  7 at 50k): every array of O's generator record and of every batch equal (raw byte for byte, or
  by digest), O's four text files byte-identical through the writers, and the same stream digest
  for every run: the GCC 12.2 builds with OpenMP (Release `cpu-only`) and with `-march=native` at
  1, 8 and 56 threads, and the `openmp=OFF` build.
- The published tau is bit-equal to O's on every case of P+ (the trace's `sparsify.tau`).
- The 5k5 and 50k digests are committed (`cpp/tests/data/goldens/label_propagation/
  generator_*.sha256`, written from O's goldens, never from dynG's output) and checked by the CPU
  suite (`dyng_lp_replay_tests`), so the hosted gcc-13 and clang-18 jobs of cpu.yml check them.
  The 500k, 5m, 5m1 and 50m digests are checked on GCC 12.2 only: no linkable local GCC 13 or
  Clang 18 with OpenMP exists (R030.md:83, :568).
- Generation time at 5m (seed 42, 56 threads, Release): 2.2 s, against O's `gen.graph` 1,260 ms +
  `gen.batches` 940 ms = 2.20 s (`--profile`, three runs); generation counts inside `end_to_end`.

## Consequences

- PLAN 8.3's toolchain pin is unnecessary for this generator: its stream is the library's own
  reproduction, checked against `std::` where `std::` is libstdc++; a later libstdc++ that changed
  an algorithm would make the runtime comparison fail (not the stream).
- M10a step 3 completes the batches' conversion into `edge_batch` with vertex arrivals.

## Alternatives considered

- **Calling std::shuffle under `standard_library_matches_legacy_streams()`** (the
  `legacy.cpp:334` precedent): not needed, the twin is exact; it would have made the stream depend
  on the standard library.
- **Pinning GCC 12 for the goldens (PLAN 8.3):** superseded by AD8.
- **A sequential CSR fill:** byte-identical as well, but 0.45 s slower than O at 5m.
