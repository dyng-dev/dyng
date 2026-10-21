# ADR 0052: The triad_count API

- **Status:** The technical parts accepted under delegation (2026-10-15; GOVERNANCE.md,
  "Delegation of technical ADRs"); **the new public names await the author** (naming is not
  delegated; the author's decision D9 of 2026-10-15 takes them on the plan's defaults until the
  0.3.0 release). It carries out the API of M9a step 3 (PLAN 4.4.2, 4.4.3, 5.1, 5.5, 4.5.5 I2 /
  I7; the sketch `docs/design/sketches/triad_count.md`; the M9a acceptance criteria 5 and 6). The
  header is `tracked`, not frozen: the freeze is M9b's (D4).
- **Amended by:** ADR 0054 (2026-10-15: the host backends are incremental; `triads_removed` /
  `triads_added` are the counts of the two phases on every backend and `fallback_used` is always
  false, so item 3's sentences on a recounting backend are historical) and ADR 0057
  (2026-10-20: the 64-bit limits, a net change of a bin outside `stats::delta`'s signed 64-bit
  range is a capacity_error).
- **Date:** 2026-10-15
- **Deciders:** the AI assistant, on the author's behalf for the technical parts (M9a, step
  cuda-port-api-early-gate); the names: S M Shovan (pending)
- **Builds on:** ADR 0023 (the uniform contract, frozen 0.1), ADR 0041 / 0043 / 0047 (the
  hypergraph container, its update path, its freeze), ADR 0048 (the parity protocol), ADR 0053
  (the engine behind it).

## Context

The sketch fixed the shape (a pattern enumeration, options, stats derived from `update_stats`, an
opaque host-array result, `compute` / `update` over `hypergraph<V, W>`) and left two questions
open: whether the per-bin signed delta is worth a `stats` field (the paper's plots show per-batch
deltas; PLAN 5.5's Python example prints `st.delta`), and how the pattern is chosen (PLAN 5.5:
`kind="h_motif"`). The four patterns read different parts of the hypergraph: the h-motifs and
StatHyper's types 2 and 3 read the line graph and the vertex-to-hyperedge rows, type 1 only the
hyperedges' sizes. ESCHER-GPU's classifying merge (`venn3`) ends a row at `INT_MAX`, and the rows
hold `vertex + 1`.

## Decision

### 1. The header `cpp/include/dyng/triad_count.hpp`

Host-only, self-contained, included by `dyng/dyng.hpp`; no CBST key, `+1` or sentinel in it (the
source rules' public-header check passes).

| Entity | Form |
|---|---|
| `enum class pattern : std::uint8_t` | `h_motif`, `stathyper_type1`, `stathyper_type2`, `stathyper_type3`; `std::string_view to_string(pattern) noexcept` |
| `inline constexpr int max_bins = 30` | the bins of `h_motif`, the size of `stats::delta` |
| `struct options` | `pattern kind = pattern::h_motif` (fixed at `compute()`), `engine cuda_engine = engine::automatic` (a tunable); an aggregate whose fields are only appended |
| `struct stats : update_stats` | `hyper_apply_summary batch`; `std::int64_t deleted_frontier, inserted_frontier`; `std::uint64_t triads_removed, triads_added`; `std::array<std::int64_t, 30> delta` (0 at and above `num_bins()`) |
| `class result` | `kind()`, `num_bins()` (30 or 1), `counts()` (a host `array_view<const std::uint64_t>` of `num_bins()` values), `count(int bin)`, `total()`, `get_options()`, `set_options()` (tunables only), `graph_version()`, `space()` (`memory_space::host`), `clone(res)`; move-only, pimpl |
| `compute(res, hypergraph<V, W>, options = {})`, `update(res, hypergraph<V, W>&, hyperedge_batch_view<V, W>, result&)` | for `(int32, unweighted)` and `(int32, int32)`; other types stop at a `static_assert` (`DYNG_TRIAD_COUNT_TYPES_MESSAGE`) |
| `detail::triad_count_supported_v<V, W>`, `detail::make_triad_count_participant<V, W>`, `detail::update_traits<triad_count::result>` | `dyng::update(res, h, batch, r1, r2, ...)` with triad results (a non-hypergraph container stops at `DYNG_TRIAD_COUNT_CONTAINER_MESSAGE`) |

Doxygen: `@guarantee`, `@throws`, `@backends`, `@determinism`, `@paper` (`escher2026`,
`escherplus2026`) and `@sync` on `compute` and `update`; the group page states the bins (0-based;
ESCHER-GPU's open triads 21-26 are bins 20-25, always 0), the patterns, the template's stages and
the requirements.

### 2. The choices behind it

1. **`options::kind`**, PLAN 5.5's `kind=`: one result counts one pattern; several patterns are
   several results updated together by one `dyng::update()` (one commit; the replays do exactly
   that with all four).
2. **`stats::delta`** (the sketch's open question 1): yes, as a fixed-size
   `std::array<std::int64_t, max_bins>`, so the stats of every update allocate nothing (the reason
   the sketch hesitated was a `std::vector`); `delta == counts() after - counts() before` per bin,
   tested on every backend. `max_bins` names its size.
3. **`triads_removed` / `triads_added`** are the sums over the bins of the two phases (for the
   weighted types: the sums of the weights). *Historical (M9a step 3, superseded by ADR 0054):* on
   a backend that recounts (`fallback_used`: the host backends in M9a step 3), they are every
   triad before and after: a recount removes them all. They are deterministic on each backend;
   the kit's cross-backend counters are the frontier sizes and the deltas until step 4 makes the
   host backends incremental. *Since step 4 (ADR 0054)* every backend is incremental: they are
   the counts of the two phases on every backend, identical across backends, and
   `fallback_used` is always false.
4. **Requirements per pattern, checked in `compute()`** and naming the property: `h_motif` and
   types 2 and 3 need `hypergraph_properties::line_graph != none` and `store_v2h` (under
   `automatic`, `compute()` builds the line graph once); type 1 needs neither, and its problem's
   `reads_prepared_graph()` is false, so its updates keep no line graph. The manifest's
   `graph_requirements` list the default pattern's.
5. **The int32 limit.** `h_motif` and type 3 classify a triad by one merge over the three rows,
   which ends a row at `INT32_MAX`; the rows hold `vertex + 1`, so a hypergraph with
   2,147,483,647 vertices could hold the vertex 2^31 - 2 that the merge would read as the end of
   its row. `compute()` refuses those patterns on such a hypergraph with `invalid_argument_error`
   naming the limit (vertex ids up to 2^31 - 3), before it checks the line graph. Type 1 counts
   such a hypergraph (tested with a store without v2h, which keeps no per-vertex rows); type 2 does
   not merge three rows (only its line-graph requirement applies).
6. **The engine.** `cuda_engine` `automatic` and `fused` run the wedge engine (Tier B); `operators`
   throws `not_supported_error` on cuda in `compute()` and in `update()` before anything changes;
   the host backends ignore it. An invalid value is an `invalid_argument_error` on every backend.
7. **Guarantees.** `compute()`: strong. `update()`: strong for every error found before the commit
   (backend, engine, stale or poisoned result, a malformed batch); basic after it: a bin that would
   go negative (`internal_error`), an unclassifiable triad (the device error word, `internal_error`),
   a count past 2^64 - 1 (`capacity_error`), a CUDA or allocation failure leave the hypergraph at
   the new version and the result poisoned (`stale_result_error` on every later use).
8. **`count(bin)`** throws `invalid_argument_error` for a bin outside `[0, num_bins())`
   (cycle_count's `count(length)` returns 0 for any length: its lengths are open-ended, the bins
   are not); `total()` throws `capacity_error` past 2^64 - 1.

### 3. The new public names (for the author; D9)

`dyng::triad_count`, `pattern` (`h_motif`, `stathyper_type1`, `stathyper_type2`,
`stathyper_type3`), `to_string(pattern)`, `max_bins`, `options` (`kind`, `cuda_engine`), `stats`
(`batch`, `deleted_frontier`, `inserted_frontier`, `triads_removed`, `triads_added`, `delta`),
`result` (`kind`, `num_bins`, `counts`, `count`, `total`, `get_options`, `set_options`,
`graph_version`, `space`, `clone`), `compute`, `update`; the macros
`DYNG_TRIAD_COUNT_TYPES_MESSAGE`, `DYNG_TRIAD_COUNT_CONTAINER_MESSAGE`; in `dyng::detail` (listed
by the API snapshot's detail contract): `triad_count_supported_v`, `make_triad_count_participant`,
`update_traits<triad_count::result>`, `triad_count_compute`, `triad_count_update`. Differences from
the sketch: `max_bins` and `stats::delta` are new, `options::kind` replaces nothing (the sketch had
it), the rest is the sketch's.

## Consequences

- The header is `tracked` by `ci/api_snapshot.py` (an experimental algorithm's header) until M9b
  reviews and freezes it (`maturity = "stable"` then).
- The Python layer (M9b) can offer `st.delta` as the array as it is.
- The kit runs one suite per pattern (test-only tags with the manifest's registry entry), so the
  four patterns get C0-C12 on every backend.
