# Retrospective: M9b (count_local_patterns, Python, the CLI, the docs and the triad_count freeze)

Status: **in progress** on the branch `m9b-local-patterns`. M9b is the second half of PLAN 11.3's
M9 row (the author's decision D2 of 2026-10-15): `experimental::count_local_patterns<op_t>` over
M9a's wedge engine with its packaging and the Op tutorial; Python (`dyng.triad_count`,
`dyng.update` over hypergraphs), the CLI, `result_io` and the wheels; the docs and examples; and
the `triad_count` API review and freeze. It runs in three steps, and each step adds its section
here. Release 0.3.0 = M8a + M8b + M9a + M9b (PLAN Appendix F).

**The author's decisions in force (2026-10-15; GOVERNANCE.md).** D1: the Nsight Systems peak of
live allocations gated (M9a). D2: the M9a / M9b split. D3: incremental host engines. D5: the
1.25 MB test-data budget. Taken on the plan's defaults, pending the author's confirmation at the
0.3.0 release: D4 (`triad_count`, `dyng.triad_count` and `dyng triad_count` frozen at the end of
M9b and `stable`; `count_local_patterns` stays experimental: installed, tracked, never frozen),
D9 (the new public names, listed in the ADRs and the API-review ADR), D10 (the "paper vs fixed
code" wording).

## Step 1: count_local_patterns and its packaging (local-patterns-experimental)

**Done** (M9b acceptance criteria 1-4). `experimental::count_local_patterns<op_t>` is the public,
nvcc-only extension point over the driver and the engine of `triad_count`: one driver
(triad_count's problem generalized over a pattern's table and the Op's value) and one engine (M9a's
kernels and host work, moved unchanged into installed headers). The four patterns of
`triad_count` are built-in Ops of it, and their five counting kernels are SASS- and
resource-identical to M9a's build, so no M9a gate needs a re-read. ADR 0058 records the design and
the new public names (D9, for the author).

### What was built

| Commit | What |
|---|---|
| `b21cb47` | the engine moved into `dyng/experimental/{local_patterns.hpp, detail/local_patterns_engine.hpp, detail/local_patterns_kernels.cuh}`; `local_pattern_problem<V, W, flavor_t>` (the triad and the experimental flavors); the built-in tables (`builtin_ops.cpp`, `counters.cu`); the host engine through `count_owned` and a type-erased source; `access_rows()` and `to_string(shape)` (`local_patterns.cpp`) |
| `c69f311` | `local_patterns.cuh` (`count_local_patterns`, `local_patterns_result`, `update_traits`); the API tests, the compile-failure tests (g++ and nvcc), the nvcc self-containment test; the tutorial's code (`examples/tutorial_algorithms/local_patterns/`) |
| `67d6f3a` | `dyng_triad_replay --counter local_patterns` (`parity/compare.py triad_count --counter local_patterns`) |
| `7dc6711` | packaging: `*.cuh` installed; Doxygen and `doxygen_coverage` (the group `experimental`, the banner); the API snapshot's experimental tier; `ci/cmake_consumer.sh`'s host-only and nvcc consumers; `ci/gpu_local.sh` step `experimental` |
| `888fe07` | the tutorial page, `ci/tutorial_quotes.py` (the drift check), the API page, `python_gaps.md`, the labels, ADR 0058, GOVERNANCE, CHANGELOG, the sketch's open question 2 |
| (the records commit) | `parity/results/M9b.md` section 1 and its records, this section |

### Deviations from the plan and the step's text (each with the same intent)

1. **The branch** `m9b-local-patterns` was created by the orchestrator from `m9a-triad-count` at
   `6716eec` (the accepted M9a; PR #16 open): `origin/main` (`9cde147`, M8b) does not contain M9a.
2. **The shape of the API** (within the PLAN name): `count_local_patterns<op_t>` is a class
   template with static `compute(res, h[, op])` and `update(res, h, batch, result)`; the Op value
   is fixed at `compute()` and kept by the result (`op()`), as `options::kind` is, so `update()`
   takes no Op (an Op's parameters cannot change between updates, which keeps "update ==
   recompute" true). The result is the template `local_patterns_result<op_t>` (a nested class
   could not carry a deducible `update_traits` specialization).
3. **The host engine changed its plumbing, not its loops**: M9a's per-shape items moved into the
   installed `host_source<op_t>` unchanged, and `count_owned` now runs `max_pattern_bins` bins per
   thread with one type-erased call per source (the source's own handle bounds the bins to the
   Op's `num_bins`, as M9a's `host_count_bins<num_bins>` did). The host backends have no
   performance gate (D3); their parity was taken again (`parity/results/M9b.md` 1.2-1.3).
4. **The device phases keep a bin stride of max(num_bins, 30)**, M9a's layout, so the built-ins'
   memsets and read-backs are the same bytes; a user's Op of more than 30 bins gets its own stride.
5. **The keys shape skips the empty row of a dead id for an Op of a user** (in the full count the
   sources are every id below the bound): `key()` is called once per live hyperedge, so an Op that
   counts hyperedges is right. The built-in type 1 keeps M9a's behaviour (C(0, 3) = 0 for them).
6. **The bins of a user's Op are bounded and carry-checked** on every backend; a bin outside
   `[0, num_bins)` is an `invalid_argument_error` (on cuda through the error word's internal bit,
   named for a user's Op). M9a's built-in handle (sign only, unchecked unit adds) is kept for the
   built-ins through `local_patterns::builtin_op`.
7. **The Op concept's callbacks are detected with the Op's real host bins handle**, not a probe
   type: nvcc warned (error #177 under `-Werror`) about a callback instantiated for a type no code
   calls it with, in an Op of an unnamed namespace, as a user's would be. A device-only or
   host-only callback cannot be told by a `static_assert`; nvcc reports it at the call (ADR 0058
   section 3). The overload `compute(res, h)` without an Op is separate from `compute(res, h, op)`
   (a default argument `op_t{}` was evaluated before the concept's messages for an Op without a
   default constructor).
8. **"The drift check fails when a quoted line moves"** is `ci/tutorial_quotes.py`: every quote of
   a tutorial page (marked ranges, `start-at` / `end-at`, line ranges) carries a `% quote:` pin
   with the digest of the quoted text, and `ci/docs.sh` fails when the text changes, also when the
   markers stay; the two existing tutorials are pinned too (16 pins, no prose change).
9. **"The tutorial code is a test ... in hosted CPU CI where the header allows"**: the extension
   header needs nvcc, so the engine's runs of the tutorial (`tutorial_test.cu`) build in CUDA
   builds only (gpu_local; the hosted cuda-build compiles them); the host half
   (`tutorial_host_test.cpp`: the Ops' callbacks driven by hand on `access_rows()`, against the
   brute force and triad_count) runs in the CPU jobs.
10. **Doxygen**: `MACRO_EXPANSION` with `EXPAND_ONLY_PREDEF` and the CUDA qualifiers and
    `__CUDACC__` predefined, so the `.cuh` headers are read as plain C++; the listing of every
    other header is unchanged (the API baseline differs only by the experimental sections and the
    header comment).
11. **Stage names**: the experimental flavor's stages are `count_local_patterns.*`; triad_count's
    stay `triad_count.*` (the benchmark suites use them).
12. **The compile-failure tests through nvcc carry the label `gpu`** (they need nvcc, no device),
    so `ctest -L gpu` in `ci/gpu_local.sh` runs them; the host-compiler cases are `cpu`.
13. **`DYNG_HD` is defined by the installed header** too (identically to the private
    `operators/execution.hpp`, a benign redefinition), with `DYNG_HD_INLINE` for the inlined set
    helpers (the built-ins' `__host__ __device__ __forceinline__`).

### Measurements and checks

(`parity/results/M9b.md` section 1.)

- **The built-in kernels**: the five counting kernels of the parity-cuda `counters.cu.o` are
  identical to M9a's merge build (`6716eec`) by `parity/triad_sass.py compare` (normalized SASS
  digest, instruction counts, every loop, registers 14 / 39 / 40 / 37 / 22, no stack, no local
  memory, the same shared memory); the `-res-usage` lines equal. No M9a gate is read again.
- **count_local_patterns == triad_count, bit for bit**: `compare.py triad_count --counter
  local_patterns --mode both`: sequential 120/120 (make-test and the syn100k grid), openmp at 8
  threads 120/120, cuda 120/120, every batch's counts, deltas, removed / added and frontiers equal
  to triad_count's and the counts equal to A's goldens.
- **CUDA parity unchanged**: `--counter algorithm --mode both` on make-test, the syn1m s1 grid and
  coauth_b50k_s1: 84/84.
- Tests: `ctest -L 'triad_count|operators|testing'` 334 / 334 (the triad mutations detected on
  both engines), `ctest -L experimental` 57 / 57; `ci/gpu_local.sh` steps `triad`, `experimental`
  and `gpu` (GPU 1); `ci/check.sh` (after the harness fix); Clang 18 syntax 419 files; the
  fortified cpu-only build; the frozen headers against `65a62c9`: empty.

### Machine time

The replays: 26 minutes (sequential 17, openmp 5, cuda 1, the cuda algorithm replays 3), then
again from the final tree (30 minutes, under the CPU gate's load); `ci/check.sh` about 45 minutes;
the gpu_local steps `triad` (37 minutes, the racecheck replays) and `experimental` (1 minute); the
SASS builds a few seconds each. Everything on GPU 1 under the shared lock; no timed reading.

### Notes for step 2 (Python, the CLI, the wheels, the docs)

- `count_local_patterns` has no Python counterpart by design (`python_gaps.md`); `dyng.update()`
  over hypergraphs needs `triad_count` results only.
- The hosted cuda-build compiles four more CUDA sources (`local_patterns_test.cu`,
  `tutorial_test.cu`, `triad_replay_local_patterns.cu`, the two self-containment units) and the
  nvcc compile-failure cases are not compiled by it (they are tests). Record the ci-cuda13 compile
  time with them (criterion 7).
- The wheels do not install the `.cuh` headers (no C++ headers in the wheels); nothing to do.
