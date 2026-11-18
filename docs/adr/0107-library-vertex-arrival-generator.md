# ADR 0107: The library's vertex-arrival generator (`generators::generate_vertex_arrival_stream`)

- **Status:** Accepted under delegation (2026-11-11; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. The name is AD12's ("the vertex-arrival generators, the
  library's and the legacy one"; the row "The new public names of 0.4", decided by the author on
  2026-10-28); the CLI command `dyng generate vertex_arrival_stream` is E1's. M10b step 2 (the
  M10b acceptance criterion 5; PLAN 5.8).
- **Date:** 2026-11-11
- **Deciders:** the AI assistant, on the author's behalf

## Context

PLAN 5.8: "New generators use library-owned distributions (portable across libstdc++, libc++
and MSVC). `generators::legacy::*` reproduces the originals' streams". M10a ported
LabelPropagation-CUDA's generator as `generators::legacy::dynlp_vertex_arrival_stream()` (ADR
0067), byte-identical to the original for a seed; it reproduces libstdc++ 12's `std::shuffle`
with a 32-bit engine and `std::uniform_real_distribution<float>`, so its stream is defined by the
original's toolchain. PLAN 5.3's "bit-exact" comment on the generators means, for this family,
`generators::legacy` (the original's bytes); the library's own generator is bit-exact per (seed,
major version) on every platform instead.

## Decision

1. **One structure, two sets of draws.** `generate_graph` and `generate_batches` of
   `src/generators/legacy_dynlp.cpp` take their draws as a parameter: `legacy_draws`
   (`std::mt19937` with util/rng.hpp's libstdc++ twins; the legacy stream, unchanged: the M10a
   generator goldens and `LpReplay.GeneratorReproducesTheOriginal{5k5,50k}` still pass) and
   `library_draws`:
   - the engine is `std::mt19937_64` (specified by the C++ standard), the graph's seeded with
     `seed`, the batches' with `seed + 0x9E3779B97F4A7C15`;
   - a uniform integer in [lo, hi] is Lemire's multiply-and-reject on 128-bit products (the
     library's `legacy_uniform_int64`, also used by the hypergraph generators);
   - a shuffle is Fisher-Yates from the front, one such draw per position;
   - a uniform float in [lo, hi] is `float(double(lo) + double(hi - lo) * k / 2^24)` for the top
     24 bits k of one engine call: the product of two 24-bit significands is exact in double, so
     the value does not depend on whether a compiler contracts it into a fused multiply-add (the
     file is compiled with `-ffp-contract=off` anyway, for the legacy draws).
   Everything else (the graph model, the CSR and its sorted rows, the batch phases A-C, the
   published threshold, the chunked and thread-independent batch edge lists) is the legacy code,
   so the library's stream has the published setup's structure and both feed
   `vertex_arrival_stream`.
2. **The surface.** `template <vertex_t = int32, edge_t = int32> vertex_arrival_stream<vertex_t,
   edge_t> generators::generate_vertex_arrival_stream(const resources&, const
   vertex_arrival_options&)` in `<dyng/generators/vertex_arrival.hpp>` (already in
   `<dyng/dyng.hpp>`): the same options (`vertex_arrival_options::seed` documents both seedings)
   and the same stream type: the true labels; per batch the arrivals with their seed labels, the
   edges with float weights and the departures (`vertex_arrival_batch::view()`); `row_capacity`
   (the full degrees, which the published setup gives `graph::with_capacity()` and the streaming
   mode on `row_layout::slack` does not need); `similarity_threshold` (the published τ; the
   streaming mode passes `std::nullopt`, ADR 0104). No new type or option.
3. **Digests.** `LibraryVertexArrivalStream.CommittedDigests` pins four configurations (5k5 at
   seed 42, 5k5 at seed 1 with average degree 3, 50k x 10 at seed 42, 3001 x 4 at seed 7 with
   degree 7) by the SHA-256 of a canonical serialization (offsets as int64, floats as bits, every
   array with its length); it runs in every CPU test suite (the local GCC 12.2 build and the
   hosted gcc-13 and clang-18 jobs). `IndependentOfThreadsAndOffsetType` compares 1, 2, 8 and 13
   host threads and both offset types; `IsNotTheLegacyStream` keeps the two apart.

## Consequences

- A change of a library draw, of its order or of the structure changes the digests: a change of
  the stream for the library's major version (a CHANGELOG entry), never silent.
- The legacy stream is untouched (the refactor only passes its draws as a parameter).
- Python (`dyng.generators.vertex_arrival_stream`) and the CLI (`dyng generate
  vertex_arrival_stream`) expose it in M10b step 3.

## Alternatives considered

- **A separate copy of the generator** for the library draws. Rejected: two copies of the
  structure would drift; the template keeps one.
- **Counter-based draws (Philox)**. Rejected for now: `std::mt19937_64` with the library's
  algorithms is the convention of the hypergraph generators (0.2), already portable.
