# ADR 0084: `escher::flatten_rows` leaves the library for the tests (carried item (c) of 0.4)

- **Status:** Accepted under delegation (2026-10-29; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. The item is the carried item (c) of 0.4 (the author's recorded
  instruction of 2026-10-23: "fix with their own gate readings"); its placement as the second
  commit of M11a step 2 is the orchestrator's delegated reconciliation of 2026-10-27. Its reading
  follows HD16, **decided by the author on 2026-10-28** (static evidence plus the escher-path host
  reading of ADR 0024).
- **Date:** 2026-10-29
- **Deciders:** the AI assistant, on the author's behalf (M11a, step
  carried-host-incidence-slack)
- **Amends:** ADR 0038 (the store port's file list: `flatten.cpp` is no longer a library file).
- **Builds on:** ADR 0061 (D13; its table listed `flatten_rows` as a region that cannot throw),
  PLAN 4.7.4 (every OpenMP region takes the thread count of the resources).

## Context

`dyng::detail::escher::flatten_rows` (`cpp/src/hypergraph/escher/flatten.cpp`) is the port of
ESCHER-GPU@abcf9b2's `flatten2DVector` (`utils/flatten.cpp`), the CBST construct layout of a list
of rows, made in M8a's transplant (ADR 0038). Its row copy ran in `#pragma omp parallel for
schedule(dynamic, 1024)` without `num_threads`: the only OpenMP region of `cpp/src` that did not
take the thread count of the resources (PLAN 4.7.4), so on a `resources::openmp(k)` or
`sequential()` handle it would run on the OpenMP default team (every hardware thread). It never
ran in the library: no library code calls it (the host store and the device store lay out their
rows themselves); its only caller is the unit test `cpp/tests/hypergraph/escher/cbst_host_test.cpp`
(`cpp/tests/parity/escher/cbst_graph_test.cpp` names it in a comment and computes the layout
itself). The two alternatives of the step: give it the resources' thread count (if `slack_csr`
adopted it), or move it out of the library. The `slack_csr` store of 0.4 (ADR 0086) does not use
the CBST construct layout.

## Decision

1. **`flatten_rows` moves to `cpp/tests/support/escher_flatten.hpp`** (`dyng::test::flatten_rows`,
   header-only, with the provenance lines of `utils/flatten.cpp` and `include/utils.hpp`), with
   the same layout, the same 64-bit total, the same `capacity_error` beyond `max_slots` (at most
   `INT_MAX`). The rows are written in order: the layout does not depend on the order, and a test
   helper needs no region. `cbst_host_test.cpp` keeps both of its tests (the layout; the limit),
   now on the helper. `cpp/src/hypergraph/escher/flatten.{hpp,cpp}` are removed and the module's
   source list no longer names `flatten.cpp`; the escher README's padding rule 1 names the helper.
2. The transplant check (`parity/escher_port/transplant.py --check --rev <c1>`) compares the
   generated files with M8a's transplant commit and is not affected; `rename_map.toml` keeps its
   `flatten.cpp` entry, which describes that commit.

## The evidence (HD16)

| Check | Before (`e84d338`) | After |
|---|---|---|
| `nm -C` of `libdyng.so` (dev, Debug; cpu-only, Release; dev-cuda) | `flatten_rows` defined (`T`, and its `._omp_fn.0` / `.cold` clones) | no `flatten_rows` symbol |
| Calls of `flatten_rows` in `libdyng.so` (`objdump -d`, every `call` naming it) | 0 in each of the three builds | (no symbol) |
| `nm -C` of `dyng-compat-cycle-enum`, `dyng-compat-escher`, `dyng-compat-mosp` (dev) | no `flatten_rows` symbol (defined or undefined) | no `flatten_rows` symbol |
| `#pragma omp parallel` without `num_threads` in `cpp/src` (every line, continuation lines joined) | one: `hypergraph/escher/flatten.cpp:55` | **none** |
| `dyng_cbst_host_tests` | 7 tests pass | 7 tests pass |

Nothing in the library called the function, so no behaviour of the library changes. The removed
object changes the code placement of `libdyng.so` (the R030 lesson: on this JCC-erratum CPU a
neutral change moves readings by 2-8 %): its timing reading is part of the ADR 0024 reading of
the escher path in M11a step 5 (`hypergraph.build` and `.apply` on escher storage on sequential
and openmp-8, `hypergraph.apply` and `triad_count.update` on cuda; the bar 1.02 of HD16, a region
above it located and listed for the author, the D14 path).

## Consequences

- Every OpenMP region of `cpp/src` takes the thread count of the resources (PLAN 4.7.4).
- `libdyng` no longer exports a function that nothing calls.
