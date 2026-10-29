# ADR 0083: `operators::openmp_exec::for_each` carries an exception out of its region (carried item (b) of 0.4)

- **Status:** Accepted under delegation (2026-10-29; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. The fix is the carried item (b) of 0.4 (the three items the
  author asked on 2026-10-23 to be fixed "with their own gate readings"); its placement as the
  first commit of M11a step 2, with M11 as its single owner, is the orchestrator's delegated
  reconciliation of 2026-10-27. Its reading follows HD16, **decided by the author on 2026-10-28**
  (sheet D2: object identity plus an informational timing of the tutorial tests, no new harness).
- **Date:** 2026-10-29
- **Deciders:** the AI assistant, on the author's behalf (M11a, step
  carried-host-incidence-slack)
- **Builds on:** ADR 0061 (D13: an exception must not leave an OpenMP region;
  `region_exceptions`), ADR 0055 (the executors and the rule of two), ADR 0024 (dynG-against-dynG
  readings).

## Context

`operators::openmp_exec::for_each(n, f)` (`cpp/src/operators/execution.hpp`) ran `f(i)` directly
in a `#pragma omp parallel for`. An exception thrown by `f(i)` left the worksharing loop, which
ends the process (`terminate called after throwing ...`), the D13 defect class of ADR 0061: the
review of ADR 0061 listed the regions of the hypergraph container and of the triad oracle, and
the executors' `for_each` was left as the carried item (b), because none of its functors in the
library throws today (the functors of dynamic_bfs and triangle_delta write preallocated arrays).
A functor written for the executors by a user of the operators (the tutorial's pattern) may
throw, and the documented behaviour of an error is an exception, not the end of the process.

The host callers of `openmp_exec::for_each` are dynamic_bfs (`algorithms/dynamic_bfs/
engine.hpp`) and triangle_delta (`algorithms/triangle_delta/engine.hpp`), the two tutorial
algorithms. `operators::count_delta` (triangle_delta and triad_count's host engine) calls
`for_each` only on its device path (`cuda_exec`); on the host it runs its own region, which
carries exceptions since M9a (ADR 0055). sssp, mosp, triad_count and the hypergraph store do not
instantiate it.

## Decision

1. **`openmp_exec::for_each`** runs every `f(i)` through a `region_exceptions` carrier
   (`errors.run([&f, i] { f(i); })`): the first exception caught is kept, the carrier's stop
   flag makes the remaining calls return at once (one relaxed load each), and `errors.rethrow()`
   after the region throws it on the calling thread. Which elements ran before the stop depends
   on the schedule (with one thread: exactly the elements up to the throwing one). The build
   without OpenMP and `sequential_exec::for_each` are unchanged (a plain loop, which stops at the
   first exception by itself). The documentation of `for_each` says so (`@throws`).
2. **Tests** (`cpp/tests/operators/execution_test.cpp`, its own executable
   `dyng_operators_execution_tests`, since it replaces `operator new`): a throwing functor at the
   first, a middle and the last of 10,000 elements and with every third element throwing, at 1
   and 8 threads (the exception's message, no element run twice, with one thread exactly the
   elements up to the throw, with 8 threads and a throw at element 0 fewer than all elements);
   no throw: every element once (also n = 0); an allocation inside the work failed in turn for
   every one of 2,000 allocations of any thread (`failing_allocator.hpp`): a `std::bad_alloc`
   on the calling thread each time; `sequential_exec` the plain loop. On the parent library
   (`d0d63f6`) the executable ends with `terminate called after throwing an instance of
   'std::runtime_error'`, exit status 134 (`$DYNG_SCRATCH/runs/m11a-step2/for_each-parent/
   before-fix.txt`).
3. **A test knob** (`cpp/tests/support/test_threads.hpp`): the host tests of dynamic_bfs and
   triangle_delta take their openmp thread count from `DYNG_TEST_OPENMP_THREADS` when it is set
   (default 4, as before), so that the timing below can run them at 1 and 8 threads with the
   tests unchanged.

## The reading (HD16)

**Object identity.** The `cpu-only` preset (Release, GCC 12.2) built at the parent `d0d63f6`
and with the fix: of the 28 library objects of `dyng_sssp`, `dyng_mosp`, `dyng_triad_count`,
`dyng_hypergraph` (every store translation unit), `dyng_dynamic_bfs` and `dyng_triangle_delta`,
**26 are byte-identical** (SHA-256); the two that differ are `dynamic_bfs/openmp.cpp.o` and
`triangle_delta/openmp.cpp.o`, the only instantiations of `openmp_exec::for_each`
(`$DYNG_SCRATCH/runs/m11a-step2/for_each-objects-{parent,fix}.sha256`). No stable gated path
(sssp, mosp, triad_count, the hypergraph container) is touched, so no ADR 0024 reading of them is
needed for this commit.

**Informational timing, no bar** (HD16: no harness times the tutorial algorithms;
`perf_ab.py run` drives `dyng-compat-mosp` only; the full 17-layout x 51-round reading of a new
`tutorial_ab.py` was the alternative the author did not choose). Both sides built from
`git archive` copies with the `cpu-only` preset (the parent with the test knob of decision 3
applied, so the tests are the same), the host test executables of dynamic_bfs and
triangle_delta (their sequential and openmp cases, each process repeating its tests 20 times),
21 alternating rounds per thread count, wall time per process, one exclusive-lock window
(`tools/locklog.py run EXCL M11a carried-host-incidence-slack for_each-informational-timing`,
2026-10-29 05:32:46Z-05:32:49Z, rc=0). Medians (record `parity/results/M11a-for-each-timing.json`):

| Tests | Threads | Parent (ms) | Fix (ms) | Fix / parent |
|---|---|---|---|---|
| dynamic_bfs | 1 | 16.82 | 17.23 | 1.024 |
| dynamic_bfs | 8 | 27.21 | 27.49 | 1.010 |
| triangle_delta | 1 | 13.57 | 13.35 | 0.984 |
| triangle_delta | 8 | 15.00 | 15.12 | 1.008 |

The processes are short (process start dominates); the readings are recorded with no bar, as
HD16 decides.

## Consequences

- An exception of a functor run by `openmp_exec::for_each` is an exception of the call, on every
  build; no process ends. The rule of ADR 0061 now holds for the executors' region too.
