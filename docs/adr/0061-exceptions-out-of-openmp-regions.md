# ADR 0061: Exceptions out of OpenMP regions (D13) and the GoogleTest rows of the golden-mutation record

- **Status:** Accepted under delegation (2026-10-22; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It fixes a defect against the documented exception guarantees of
  the frozen hypergraph container and of `triad_count` (PLAN 4.7.3; the headers are unchanged)
  and decides a technical question of the release certificate's mutation check. It changes no
  rule the author approved: the gates and their protocol, the parity rules, the stability policy
  and the frozen headers are untouched; the dynG-against-dynG reading follows ADR 0024.
- **Date:** 2026-10-22
- **Deciders:** the AI assistant, on the author's behalf (R030, step d13-and-records; the fix
  itself was classified as technical by the orchestrator's note of 2026-10-22)
- **Builds on:** ADR 0024 (reading a refactor against an earlier dynG build), ADR 0041 and 0047
  (the container and its guarantees), ADR 0055 (`count_owned` carries an exception out of its
  region), ADR 0060 finding 8 (the defect, found by its allocation-failure test), the M9b review
  (its scope: a region whose `if` clause is false is still a region).

## Context

An exception must not leave an OpenMP parallel region, nor an iteration of a worksharing loop
(OpenMP 5.2, sections 10.1 and 11.5): the thread that threw must catch it inside the same
iteration. With GCC's libgomp one that leaves ends the process (`terminate called after throwing
an instance of 'std::bad_alloc'`), also when the region's `if` clause is false and it runs on the
calling thread alone. The M9b review found two such regions in the hypergraph container whose
work allocates (D13):

1. `incidence_model::v2h_append` (the host store's commit of `hypergraph::apply`, reached by
   `dyng::update()`, `triad_count::update()` and `count_local_patterns::update()`): a sorted
   insert into each vertex's v2h row;
2. the gather region of the line graph's row build (`flat_hyperedge_adjacency`,
   `line_graph/flat_adjacency.cpp`): the per-row buffer and the per-thread arena. It runs at
   construction with a line graph (`line_graph_policy::sorted_merge`), at the first `h2h()` or
   `compute()` under `automatic`, in `check_integrity()`, and on the host for the CUDA store too
   (`escher_store.cu`: its construction, its restore from a snapshot, its first line graph under
   `automatic`).

The headers document `out_of_memory_error` for these calls, with the hypergraph poisoned when the
failure is inside the store phase (`hypergraph::apply`'s `@guarantee`) and, for
`triad_count::update()`, the three outcomes of its `@guarantee`. The process ended instead, in
every build with OpenMP (the default builds and the wheels), on the sequential backend as on
openmp, with any thread count. 0.3.0 would have shipped it as a "Known issue" (M9b); the
orchestrator's note of 2026-10-22 asked for the fix in the release preparation.

## Decision

### 1. The rule

Work inside an OpenMP region of the library that can throw runs through
`dyng::detail::region_exceptions` (`cpp/src/util/region_exceptions.hpp`): `run(work)` catches every
exception of one iteration (or of one thread's work), keeps the first one to be caught and makes
the remaining calls return at once (one relaxed atomic load each); `rethrow()`, after the region,
throws the kept exception on the calling thread. The exception is the original one (a
`std::bad_alloc`, or a `std::length_error`), so the entry points' handlers translate it
(`DYNG_TRANSLATE_ALLOCATION_FAILURE`: `out_of_memory_error` with the call's context) and the
container's store phase poisons the hypergraph, exactly as for an allocation outside a region.
"First" is the first to be caught: with several threads, which one depends on timing; the regions
it guards throw only allocation failures, reported alike. `operators::count_owned` (triad_count's
host engine) already does the same with its own `exception_ptr` (ADR 0055), and the other regions
below cannot throw.

### 2. What changed

| Region | Can its work throw? | Change |
|---|---|---|
| `incidence_model::v2h_append` (`hypergraph/incidence_model.cpp`) | yes: `std::vector::insert` | each group through `region_exceptions` |
| `incidence_model::v2h_remove` | no: `lower_bound` and `erase` of `int64_t` values do not allocate or throw | none |
| the gather region of `build_flat_rows` (`line_graph/flat_adjacency.cpp`) | yes: the row buffer, the arena | each row through `region_exceptions` |
| the copy region of `build_flat_rows` | no: copies into the preallocated payload | none |
| the merge region of `merge_line_graph` (`line_graph/sorted_merge_host.cpp`) | no: every array is allocated before it; a broken row is counted and raised after the region (`DYNG_FAIL`) | none |
| `escher::flatten_rows` (`hypergraph/escher/flatten.cpp`) | no: copies into the preallocated payload | none |
| `operators::count_owned` / `count_delta` (triad_count's host engine, ADR 0055) | yes (an Op of a user may throw) | none: it carries the exception out since M9a |
| `testing::triad_reference`: its line graph's rows | yes: each row | its own `try` / `catch` per row, the first exception kept under a named `critical` and rethrown after the region: the oracle shares no code with the library (its isolation check), so it does not use the helper |
| `testing::triad_reference`: the triangles' accumulators | yes: `parts.push_back` inside the `critical` | `parts` is reserved for every thread before the region, so nothing in it allocates |
| `testing::triad_reference`: the type-2 sum | no | none |

The headers do not change: they already document the behaviour that now holds. No signature
changes; nothing changes when nothing throws (the same work in the same order).

### 3. The tests

`cpp/tests/support/failing_allocator.hpp` replaces the global `operator new` in a test executable
and fails the k-th allocation of the calling thread or of any thread (the OpenMP workers
included); `fail_each_allocation<expected>` fails every allocation of a call in turn and checks
the state after each failure. Every test below ended the process before the fix
(`$DYNG_SCRATCH/runs/r030-d13/before-fix.txt`: exit status 134 with `terminate called after
throwing an instance of 'std::bad_alloc'`, but the clone sweep, which reaches no region):

- `dyng_hypergraph_allocation_failure_tests` (new): `hypergraph::apply()` on sequential and on
  openmp with 1 and 4 threads, under every `line_graph_policy`: an `out_of_memory_error`, and the
  hypergraph either unchanged (version, rows; the batch still applies) or poisoned (every later
  call refuses it), **both reached**; construction with the line graph and the first `h2h()` under
  `automatic` (strong: after the failure the hypergraph is not poisoned and the next `h2h()` is
  the line graph); each region directly with its loop run in parallel (v2h_append's 1,200 groups,
  above its threshold of 1,024, on 4 threads; the row build on 4 threads), every thread's
  allocations failed in turn: the `std::bad_alloc` comes out on the calling thread.
- `dyng_triad_count_allocation_failure_tests` (ADR 0060's, extended): `update()` of every pattern
  on sequential and openmp (1 and 4 threads), with `sorted_merge` and `automatic`: each failure is
  one of the three documented outcomes (before the commit: `h` and `r` unchanged and usable; inside
  it: `h` poisoned, `r` unchanged and not poisoned; after it: `r` poisoned), the first two
  reached for every pattern; `compute()` under `automatic` (the line graph built inside it); the
  oracle's regions on 4 threads (its documented `std::bad_alloc`).

The sanitizer presets skip these executables (the sanitizers replace `operator new`), as ADR
0060's did.

### 4. The GoogleTest rows of the golden-mutation record

`parity/certify.py` reads the golden-mutation record (`parity/mutate.py run`) with the library
scope only, and since `922e3a6` the rows of a GoogleTest suite of `mutate.py` (cbst since M8a;
hypergraph, triad_count, kit and the triad oracle since) are exempt from the golden-manifest check,
since they replay no golden corpus. Their mutations are detected by the C++ tests, so a record
taken on a commit whose tests differ from the release's says nothing about the release's tests.
The `tests` scope's own record (`mutation-ctests`, `ctest -L mutation`) does not cover them: it
runs the CTests `*.mutation.*` of its build, and a cpu-only build has none of the cbst device
mutations.

**Decided:** a golden-mutation record with rows of a GoogleTest suite must also have the release's
`cpp/tests` (`GTEST_MUTATION_PATHS`, with `paths_diff()`; the library scope stays as it was, with
its metadata-equivalence exception). Not the whole `tests` scope: `README.md`, `examples/` and
`cpp/fuzz` are not part of those executables, so a final release that changes only them (the
status block of 0.3.0) keeps its record, where the sanitizer checks of the `tests` scope run again
(`docs/developer/release.md`). Tested by `parity/tests/test_release_cert.py::
test_the_gtest_mutation_rows_need_the_release_tests` (a record whose tests differ fails with the
suites named; one without GoogleTest rows does not need the tests; the real diff of the last
commit that changed `cpp/tests`).

### 5. The stability records and the deliberate change

The author confirmed on 2026-10-22 that the hypergraph surface and `triad_count` are stable from
0.3.0 (GOVERNANCE.md). The checks already treat both as frozen (`ci/api_snapshot.py`:
`dyng/hypergraph/*` in `CORE_FROZEN`, `dyng/triad_count.hpp` through its manifest's maturity), so
R030 changed records only. The deliberate-change demonstration is recorded under "Outcome".

## The reading (ADR 0024)

The fix (`e6e4cce`, side B) against `main` (`e6f87ed`, side A), both `dyng-compat-escher` from
the `parity` preset (Release, `-O3`, no CUDA) built from `git archive` copies, read with
`parity/experiments/hypergraph_host_ab.py` (new): the golden cases `coauth_b50k_s1` (coauth-DBLP,
3 x 50K) and `syn1m_b50k_s1` (syn1M, 3 x 50K), sequential and openmp with 8 threads, mode `apply`
(`--no-count`: `hypergraph::apply()` alone) and mode `update` (triad_count's update: type 1 on
sequential and on coauth-DBLP, whose h-motif full count takes minutes on the host; h_motif on
syn1M with 8 threads); A/B rounds over 17 heap layouts, **51 rounds per side** (ADR 0024 rule
1), the exclusive perf lock, the contamination monitor (one round repeated in all), the stdout
guard; read with `parity/ab_modes.py --bar 1.02` (rules 2 and 3). 2026-10-22 01:40-04:10 UTC.
Record: `parity/results/R030-d13-reading.json` (`$DYNG_SCRATCH/runs/r030-d13/ab`).

| Case / backend / mode | `hypergraph.apply` (3 batches) | `triad_count.update` | `.apply.insert` | `hypergraph.build` (before the batches) |
|---|---|---|---|---|
| coauth sequential, apply | 4,552.1 -> 4,551.9 ms **1.000** | | 108.6 -> 110.1 ms 1.014 | 6,568.8 -> 7,122.3 ms 1.084 |
| coauth sequential, update (type 1) | 4,525.4 -> 4,525.3 ms **1.000** | 4,535.9 -> 4,535.7 ms **1.000** | 1.013 | 1.085 |
| coauth openmp 8, apply | 2,660.5 -> 2,672.1 ms **1.004** | | 1.005 | 1,783.3 -> 1,831.3 ms 1.027 |
| coauth openmp 8, update (type 1) | 2,686.3 -> 2,676.3 ms **0.996** | 2,693.4 -> 2,683.3 ms **0.996** | 0.999 | 1.035 |
| syn1M sequential, apply | 2,613.2 -> 2,616.5 ms **1.001** | | 94.7 -> 98.8 ms 1.043 | 2,959.1 -> 3,138.1 ms 1.060 |
| syn1M sequential, update (type 1) | 2,603.0 -> 2,596.3 ms **0.997** | 2,615.1 -> 2,608.4 ms **0.997** | 1.043 | 1.056 |
| syn1M openmp 8, apply | 1,857.5 -> 1,857.6 ms **1.000** | | 0.998 | 930.4 -> 949.4 ms 1.020 |
| syn1M openmp 8, update (h_motif) | 1,866.2 -> 1,864.0 ms **0.999** | 5,032.1 -> 5,025.5 ms **0.999** | 1.005 | 1.019 |

(Medians of the 51 rounds; the bootstrap intervals and the per-batch rows are in the record.)

**The apply and update regions are unchanged:** every `hypergraph.apply`, `apply_ms`,
`triad_count.update` and `update_ms` region reads 0.996-1.004, unimodal, within the 1.02 bar,
on both cases and both backends.

**Two sub-regions read above the bar**, and both were located (rule 3):

- `hypergraph.apply.insert` (the store's insert step, 2-4 % of an apply) on sequential: 1.013
  (coauth) and 1.043 (syn1M), +1.5 to +4.1 ms over three batches, 0.1 % of the apply; on 8
  threads 0.998-1.005. The per-group `try` and check of `v2h_append`; with the JCC-erratum
  mitigation flags (below) coauth reads 1.022, so a small real cost of the guard, kept.
- `hypergraph.build` (the construction before the batches, not an apply or update region): 1.084
  / 1.085 on coauth sequential, 1.056-1.060 on syn1M sequential, 1.019-1.035 on 8 threads. Its
  line-graph row build is the changed region. Located as **code placement on this CPU**, not the
  cost of the code: a series of builds of variants (`diagnostics` in the record; medians of 6-8
  interleaved rounds) shows the try block per row costs 6-10 % of the row build in the release
  build (the check alone 0 %, the lambda alone 0.5 %), and **0.3 % when both sides are built with
  the jump-conditional-code (JCC) erratum mitigation** (`-falign-functions=64 -falign-loops=32
  -Wa,-mbranches-within-32B-boundaries`, R020's flag set FB). The reading of both FB builds (17
  rounds over 17 layouts) confirms it: `hypergraph.build` 1.007 on sequential and on 8 threads,
  `hypergraph.apply` 1.004 and 0.999. A restructured guard (`parallel_chunks`: chunks claimed by
  hand, one try block per thread) did not help in the release build (+2 to +7 %) and was slower
  with the mitigation (+1.8 to +2.6 %), so the simple per-row form stays. The Xeon Gold 6258R has
  the erratum (docs/roadmap.md, the deferred alignment option); the release builds keep the
  one-layout protocol of the author's decision of 2026-10-08.

What it means for the gates: the host construction is not a gated region of the host backends
(D3: no host gate); on cuda, the triad_count end-to-end gate contains the host construction of the
line graph, run with the handle's host threads (56 by default), inside a build of 0.76 s on coauth-DBLP
(ESCHER-GPU's: 1.41 s), where an effect of the size seen on 8 threads (+2-3 % of the build,
some 20 ms) is a few tenths of a percent of the 8.8 s end to end (0.940 of the 1.10 bar at
M9b). The release certificate reads every gate
again on the final library (R030 step 2), so this is checked, not assumed.

## Outcome

- **The defect is fixed** (`e6e4cce`), with the tests of section 3 (`7c2865d`), each of which
  ended the process before it.
- **The deliberate change** (2026-10-22, a throwaway clone of the branch at `6e2c497`, base
  `e6f87ed` = `main` as pushed; log `$DYNG_SCRATCH/runs/r030-d13/demo.log`): unchanged,
  `ci/api_check.sh e6f87ed` passes both halves. C++: the default of
  `hypergraph::from_hyperedges(res, hyperedges, props = {})` removed: `ci/docs.sh --doxygen-only`
  exits 1 (`api-snapshot: the public C++ API differs from the baseline`, the `- ... props={}` /
  `+ ... props` lines), and with the baseline regenerated (`ci/docs.sh --update-api`, as a pull
  request would) `ci/api_check.sh e6f87ed` exits 1 (`breaking change of the frozen API:
  [dyng/hypergraph/hypergraph.hpp] ...`). Python: `dyng.Hypergraph.apply(batch, ...)` with `batch`
  renamed: `ci/api_check.sh e6f87ed` exits 1 (`Parameter was removed`, `Parameter was added as
  required`). Reverted: passes. (triad_count's own demonstration is ADR 0060's.)
- **The certificate's mutation check** binds the GoogleTest rows to `cpp/tests` (`7cca49e`).
- **The reading** above: the apply and update regions unchanged (0.996-1.004); the two
  sub-regions above the bar located.
- **Parity**: the replays the change can reach (the host backends) were run again on the fixed
  library; R030's parity record (`parity/results/R030.md` section 1) has them.

## Consequences

- The CHANGELOG's "Known issues" entry of M9b is gone; the fix is a "Fixed" entry of 0.3.0rc1,
  and the triad_count and hypergraph pages say that host memory exhaustion is the documented
  error with its guarantee.
- A later OpenMP region that allocates or calls user code uses `region_exceptions` (or carries its
  exception out as `count_owned` does); the allocation-failure tests are the pattern for its test.
- Found in passing and not changed (outside D13's scope, recorded for the author and the
  orchestrator): `sssp`'s OpenMP backend pushes into thread-local lists inside its regions
  (`algorithms/sssp/openmp.cpp`), which can allocate when a list grows, and
  `operators::openmp_exec::for_each` has no catch (its users' functors, the tutorial algorithms,
  do not allocate); `escher::flatten_rows` runs with the global OpenMP thread count rather than
  the handle's (PLAN 4.7.4).
