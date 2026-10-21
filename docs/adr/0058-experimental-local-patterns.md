# ADR 0058: experimental::count_local_patterns: one driver, one engine, and the Op concept

- **Status:** Technical parts accepted under delegation (2026-10-21; GOVERNANCE.md, "Delegation
  of technical ADRs"); **the new public names await the author** (naming is not delegated: D9,
  the Open decisions table; taken on the plan's defaults until the 0.3.0 release). The extension
  point stays `experimental` (D4: installed, tracked, never frozen). It carries out M9b step 1
  (local-patterns-experimental; the M9b acceptance criteria 1-4) under PLAN 4.1, 4.2, 5.9, 6.2 row
  7, 6.4.6, 9.1-9.3, 10.2 and Appendix B. It changes no rule the author approved: the parity is
  exact, no gate or protocol is touched, no frozen header (`hypergraph/*`, `update.hpp`,
  `core/*`, `graph/*`) is edited, and the four built-in counting kernels are SASS- and
  resource-identical to M9a's build.
- **Date:** 2026-10-21
- **Deciders:** the AI assistant, on the author's behalf for the technical parts (M9b, step
  local-patterns-experimental); the names: S M Shovan (pending)
- **Builds on:** ADR 0052 (the triad_count API), ADR 0053 (the wedge engine's port and its Op form,
  c6: GO), ADR 0054 (the incremental host engine), ADR 0055 (`operators::count_delta`), ADR 0057
  (the 64-bit limits), ADR 0033 (the tutorial algorithms and the quoted tutorials), ADR 0023 / 0047
  (the API snapshot's tiers and the frozen headers).

## Context

PLAN 6.4.6 and Appendix B expose ESCHER-GPU's wedge engine (`countTriangles<Op>`, `countPairs`,
`countKeys`) as `experimental::count_local_patterns<op_t>` in `experimental/local_patterns.cuh`,
the one public header allowed to need nvcc (PLAN 4.1), with an "add an Op" tutorial. The sketch's
open question 2 asked how the Op concept relates to `triad_count::pattern`. M9a left the engine in
its Op form (c6): host-and-device operations that see 0-based ids, a rows handle and a bins handle,
with the frontier filters and the signs inside the engine, and a host engine that runs the same
operations as loops. Three constraints shaped the design: the four built-in kernels must keep M9a's
machine code (their gates were read on it); `triad_count.hpp` must stay host-compilable; and a
user's Op must run on every backend through the same update path as triad_count, so its result
composes with `dyng::update`.

## Decision

### 1. One driver, one engine

- **The driver** is triad_count's problem, generalized: `local_pattern_problem<V, W, flavor_t>`
  (`cpp/src/algorithms/triad_count/problem.hpp`, `triad_count.cpp`). Its hooks (Step 0's live
  deletions, the old and new row views, the frontiers of the deleted and the ASSIGNED inserted ids,
  the workspace reserved before the commit, the commit, the delta with its sign checks, the stale
  and poison rules) are the same for every pattern. What depends on the pattern is a table,
  `detail::local_patterns::kernels` (`num_bins`, `shape`, whether it is built in, the merge limit,
  `launch`: the Op's CUDA kernel launch, and `source`: the host work of one source), and the Op's
  value (its bytes), both kept in the result's state (`local_pattern_core`). The flavor names the
  stages and the stats: `triad_flavor` (`triad_count.*`, `triad_count::stats`; the alias
  `triad_count_problem<V, W>`) and `local_patterns_flavor` (`count_local_patterns.*`,
  `experimental::local_patterns_stats`). Both flavors are instantiated in libdyng for
  `(int32, unweighted)` and `(int32, int32)`; users never instantiate the driver.
- **The engine** is the M9a code moved, unchanged, into installed headers: the per-Op kernels
  (`triangle_kernel`, `pair_kernel`, `key_kernel`), the block bins and their flush, the
  warp-cooperative wedge loop and the exactly-once predicates (with their recorded mutation hooks)
  in `dyng/experimental/detail/local_patterns_kernels.cuh` and `.../local_patterns_engine.hpp`; the
  set primitives, the rows handle and the member span in `dyng/experimental/local_patterns.hpp`.
  The driver keeps what does not depend on the Op: the wedge count of the sources, its CUB scan and
  the one read of its total (`engine/wedge.cuh`, `count_shape()`), then calls the table's `launch`.
  On the host backends `operators::count_owned` (ADR 0055) partitions the sources and calls the
  table's `source` through a type-erased item (`max_pattern_bins` bins per thread; the Op's own
  bins handle bounds them to its `num_bins`).
- **The four patterns are built-in Ops** of that engine: `motif_op` (h_motif), `choose3_op`
  (type 1), `type2_op`, `type3_op` (`engine/*_ops.hpp`, each with a `shape`), whose tables are built
  in `builtin_ops.cpp` (host work) and `counters.cu` (kernels), pre-instantiated in libdyng, so
  `triad_count.hpp` needs no CUDA. `local_patterns::builtin_op<op_t>` marks them: their device bins
  handle tests the sign of a bin only and adds unchecked when the Op adds 1 (M9a's handle), their
  messages and requirements stay triad_count's. **A new pattern is a new Op, not a new enumerator**
  (the sketch's open question 2).

### 2. The public surface (experimental)

`<dyng/experimental/local_patterns.hpp>` (host-compilable; the group `experimental`):

| Entity | Form |
|---|---|
| `enum class shape : std::uint8_t` | `triangles`, `pairs`, `keys`; `to_string(shape)` |
| `max_pattern_bins` | 1024 (8 KiB of shared memory per block) |
| `member_code` | `std::int32_t`, documented as opaque and ascending |
| `class member_span` | `begin()`, `end()`, `size()`, `empty()`, `operator[]` (DYNG_HD) |
| `class hyperedge_rows` | `row(id)`, `size(id)`, `static decode(code)` (DYNG_HD; constructed by the engine only) |
| `struct venn_counts`, `intersect_size()`, `venn3()`, `intersects3()` | the set helpers on spans (one merge of the codes) |
| `class row_access`, `access_rows(h)` | the rows the engine reads of a state (`hyperedges()`, `id_bound()`, `has_line_graph()`, `space()`, `device()`); compiled into libdyng for both instantiations |
| `struct local_patterns_stats : update_stats` | `batch`, `deleted_frontier`, `inserted_frontier`, `removed`, `added`, `delta` (a vector of num_bins) |
| macros | `DYNG_HD`, `DYNG_HD_INLINE`, `DYNG_LOCAL_PATTERNS_TYPES_MESSAGE` |

`<dyng/experimental/local_patterns.cuh>` (nvcc only; `#error` with a clear message otherwise):

| Entity | Form |
|---|---|
| `template <class op_t> struct count_local_patterns` | `valid`, `result`, `stats`, `static compute(res, h, op)`, `static compute(res, h)` (op_t{}), `static update(res, h, batch, result&)` |
| `template <class op_t> class local_patterns_result` | `num_bins()`, `counts()`, `count(bin)`, `total()`, `op()`, `graph_version()`, `space()`, `clone(res)`; move-only |
| `detail::update_traits<local_patterns_result<op_t>>` | `dyng::update(res, h, batch, r1, r2, ...)` composes it with triad_count and every hypergraph result |

The shape of the API: `count_local_patterns<op_t>` is a class template with static `compute` and
`update` (PLAN's name; ESCHER-GPU's `countTriangles<Op>`); **the Op value is fixed at
`compute()`** and kept by the result (`op()`), as `options::kind` is: an Op's parameters (a
threshold, a vertex) cannot change between updates, which keeps "update == recompute" true.
`update()` therefore takes no Op. The result is a template so that a result of one Op cannot be
updated with another (`dyng::update` specializes its traits per Op type).

### 3. The Op concept

- `static constexpr int num_bins` in [1, `max_pattern_bins`]; `static constexpr
  dyng::experimental::shape shape`;
- the callbacks of the shape: triangles `begin_wedge(rows, a, b)` and `template <class bins_t>
  triangle(rows, c, bins)`; pairs `pair(rows, a, b, bins)`; keys `key(rows, a, bins)`; and
  `flush(bins)`; all DYNG_HD;
- trivially copyable and default-constructible: the Op value given to `compute()` is the initial
  state of every worker (a GPU thread, a host source), which flushes once.

`detail::local_patterns::check_op<op_t>()` fires one `static_assert`, with its own message, per
missing or ill-formed member (no cascade: each later check is skipped when its premise fails).
The callbacks are detected with the Op's real host bins handle (`host_bins<num_bins>`), not a
probe type, because nvcc warns about (and with `-Werror` refuses) a callback instantiated for a
type no code calls it with. Compile-failure tests run each case through the host compiler
(hosted CPU CI) and, in CUDA builds, through nvcc and `count_local_patterns<op_t>::compute()`
itself (`cpp/tests/compile_fail/local_patterns_op.cpp`). Whether a callback is DYNG_HD cannot be
tested by a `static_assert`; nvcc reports a host-only or device-only callback at the call.

**What the callbacks see**: 0-based live hyperedge ids and member spans of the store's codes
(`member + 1`, ascending); `decode()` gives the vertex. No CBST key (id + 1), sentinel (the
merge's `INT_MAX`) or flag reaches them: the keys of the line graph, its rows, the frontier and
the flags stay in the engine (tested by probe Ops on every backend, whose checksums equal a brute
force over the hypergraph's own member lists). For the keys shape the engine skips the empty
row of a dead id below the bound for an Op of a user (`key()` is called once per live hyperedge;
the built-in type 1 adds C(0, 3) = 0 for them as in M9a). The Op never sees the frontier, the
sign or the ownership rule, and holds no view, so the subtraction cannot read G_{t+1} (I1 by
construction).

**The bins of an Op of a user** are bounded and checked on every backend: a bin outside
`[0, num_bins)` is not written and raises `invalid_argument_error` (on cuda through the device
error word's internal bit, which the driver names for a user's Op; for a built-in Op it stays
triad_count's `internal_error` for a triad of no class); every add is carry-tested.

### 4. Requirements and contract

The contract is triad_count's: strong before the commit, basic after it (a poisoned result),
`stale_result_error` for a stale or poisoned result, `invalid_argument_error` for a moved-from
one, `not_supported_error` for an unavailable backend, and also when the header's hand-off
version (`local_patterns::abi_version`) differs from the library's. The shapes triangles and
pairs need the line graph and the vertex-to-hyperedge rows (built under `automatic`); keys needs
neither. Every Op of a user is refused on a hypergraph with 2,147,483,647 vertices (the public
`venn3()` ends a row at `INT_MAX` as the built-in merge does). The device phases keep M9a's bin
layout (a stride of max(num_bins, 30)), so the built-ins' device memory and copies are unchanged.

### 5. Packaging

- `cmake/install.cmake` installs `*.cuh` with the headers (also from a CPU-only build: an nvcc
  program then counts on the host backends). The `.cuh` headers compile alone under nvcc
  (`dyng_cuh_self_contained`) and without nvcc stop at their `#error`; the `.hpp` headers are in
  the existing host self-containment test.
- Doxygen reads `*.cuh` (`EXTENSION_MAPPING`, the CUDA qualifiers predefined away); the group
  `experimental` (PLAN 9.1); `ci/doxygen_coverage.py` holds `dyng/experimental/` to it, to the
  freeze's documentation rules and to the banner "EXPERIMENTAL PUBLIC; may change in any minor
  release" in every `@file` brief.
- `ci/api_snapshot.py` gains the tier `experimental` (PLAN 5.9): `dyng/experimental/*` is listed
  in the baseline, and a difference is reported and never fails the check; `--against` never
  treats it as frozen.
- `ci/cmake_consumer.sh` builds a host-only consumer of the experimental host header (CXX only)
  and, from a CUDA build, a user Op compiled by nvcc against the installed package
  (`find_package(dyng)`, no flag of dynG's build: the headers need no `--expt-relaxed-constexpr`),
  run on the GPU and the host backends; `ci/gpu_local.sh` (step `experimental`) runs it with
  memcheck, racecheck and synccheck over the tutorial's and the API tests' CUDA runs.
- `dyng_triad_replay --counter local_patterns` (CUDA builds): the built-in Ops through
  `count_local_patterns` beside triad_count's results in one commit per batch, bit for bit, and
  compared with ESCHER-GPU's goldens.

### 6. The new public names (for the author; D9)

`dyng::experimental`, `count_local_patterns` (`valid`, `result`, `stats`, `compute`, `update`),
`local_patterns_result` (`op_type`, `num_bins`, `counts`, `count`, `total`, `op`,
`graph_version`, `space`, `clone`), `local_patterns_stats` (`batch`, `deleted_frontier`,
`inserted_frontier`, `removed`, `added`, `delta`), `shape` (`triangles`, `pairs`, `keys`),
`to_string(shape)`, `max_pattern_bins`, `member_code`, `member_span` (`begin`, `end`, `size`,
`empty`, `operator[]`), `hyperedge_rows` (`row`, `size`, `decode`), `venn_counts` (`ab`, `bc`,
`ca`, `abc`), `intersect_size`, `venn3`, `intersects3`, `row_access` (`hyperedges`, `id_bound`,
`has_line_graph`, `space`, `device`), `access_rows`; the macros `DYNG_HD`, `DYNG_HD_INLINE`,
`DYNG_LOCAL_PATTERNS_TYPES_MESSAGE`; the profiler stages `count_local_patterns.*`; in
`dyng::detail` (the API snapshot's detail contract): `local_patterns_supported_v`,
`update_traits<local_patterns_result<op_t>>`, and the hand-off (`local_patterns::kernels`,
`device_pass`, `host_pass`, `host_bins`, `builtin_op`, `check_op`, `make_kernels`,
`local_patterns_compute` / `_update` / `_participant` and the result's accessors).

### 7. What may change (experimental)

Any of section 6 in a minor release, with a CHANGELOG entry: the bins handle (a typed handle per
shape; a `bins.add()` without the carry test for an Op that declares `unit_counts`), more shapes
(open wedges, for the open triads; incidence-based patterns), Ops that see the hyperedges'
weights, a device-only Op on cuda alone, a sparse or larger bin space, and the promotion of the
engine to `cpp/src/operators/` as the operator `local_patterns` of PLAN 4.5.3 (with a second
in-tree user).

## Consequences

- One engine: the built-in kernels are the installed templates instantiated in libdyng
  (`parity/results/M9b.md` section 1: the five counting kernels' machine code, registers, stack,
  shared and local memory identical to M9a's build, by `parity/triad_sass.py compare`), so M9a's
  gate readings stand without a re-read; the host engine runs the same per-source work through
  `count_owned` (no performance gate on the host backends, D3).
- `count_owned` keeps its two users (ADR 0055: triangle_delta and triad_count's host engine).
- The tutorial "Count a new hypergraph pattern (an Op)" quotes tested code
  (`examples/tutorial_algorithms/local_patterns/`), pinned by `ci/tutorial_quotes.py`, which also
  pins the quotes of the existing tutorials.
- Python has no counterpart by design (`docs/developer/python_gaps.md`).
