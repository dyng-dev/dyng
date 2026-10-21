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

## Step 2: Python, the command line, the triad bins and the wheels (python-cli-wheels)

**Done** (M9b acceptance criteria 5-7). `dyng.triad_count` and `dyng.update()` over hypergraphs in
Python, the triad bins as JSON (C++ and Python), `dyng.testing.triad_reference`, `dyng triad_count
compute|update` with `--format json|escher`, and the three wheels. ADR 0059 records the design and
the new public names (D9, for the author). No library code of the algorithm, the engine or the
operators changed (only the manifest's `python` key), so the built-in kernels are those of step 1
and no M9a gate needs a reading.

### What was built

| Commit | What |
|---|---|
| `6e98222` | `io::triad_bins`, `io::triad_bins_size()`, `io::write_triad_bins_json()`, `io::read_triad_bins_json()` (`cpp/src/io/triad_bins.cpp`: a strict RFC 8259 reader), the format in `file_formats.md`, `dyng_io_tests` `TriadBins.*`, the fuzzer `fuzz_triad_bins` and its corpus, the API baseline (tracked additions) |
| `86e9224` | `python/bindings/triad_count.cpp`, the hypergraph branch of `python/bindings/update.cpp`, the bindings of the JSON and of the oracle; `python/dyng/triad_count.py`, `_update.py`, `io.py` (`TriadBins` ...), `testing.triad_reference`; the manifest, `_algorithms.py` and the stubs regenerated; `test_triad_count.py`, `test_triad_count_cuda.py`, `triad_fixture.py`, the API surface and the PLAN 5.5 example; `ci/plugin_smoke.py`; `python_gaps.md` |
| `db25f77` | `python/dyng/cli/_triad_count.py`, the generated flags' unions (`cli/_common.py`), `docs/api/cli.md`, `test_triad_count_cli.py` |
| (the records commit) | ADR 0059, GOVERNANCE, CHANGELOG, `parity/results/M9b.md` section 2, this section; a test fix of the CUDA tests (below) |

### Deviations from the plan and the step's text (each with the same intent)

1. **The container parameter is named `hypergraph`** in `dyng.triad_count.compute` and `update`
   (PLAN 5.4's table writes `graph`): the uniform-contract test reads the expected name from the
   registry's `container`. `dyng.update(graph, batch, *results)` keeps its frozen parameter name
   and takes either container.
2. **`Pattern` is an `enum.StrEnum`**, the first public enumeration class of the typed layer (PLAN
   5.5's `kind=str|enum`); every function still takes the names (PLAN 5.4 rule 5), and a member is
   its name. `PatternName` (the `Literal`) and `MAX_BINS` are new names beside it.
3. **`Result.counts` is a read-only NumPy copy** (the step's text), not a `dyng.Array` as the other
   results' arrays; `Stats.delta` has the result's `num_bins` values (C++: 30). Both in
   `python_gaps.md`.
4. **`--format escher` prints the whole of A's output for a file input**, not only the count lines:
   the `Hypergraph:` line and the `Batch` lines are reproduced too, so the diff against A's stdout
   is of every line but the five its generator prints for a generated input (the test removes that
   block). The `[GPU]` prefix is A's text on every backend.
5. **The read flags and two flags that are not fields.** The `.hg` read flags are generated from
   `dyng.io.read_hypergraph`'s keywords; its `options=` is `--read-options` (a flag `--options`
   would read as the algorithm's options), and `--min-initial-payload N` builds
   `CapacityPolicy.escher_gpu(N)` (A's `--payload-capacity`, needed for A's ids). The flag
   generator now unwraps `X | None` first and gives a union with one `Literal` its choices; no
   existing flag changed (none had such a hint; the CLI tests and the documented-flags test pass).
6. **A `.dgb` recording's expected ids are checked** by `dyng triad_count update` (a mismatch exits
   with status 1): without it a hypergraph built with other properties than the recording's would
   delete other hyperedges silently.
7. **The triad bins are a C++ writer and reader** (tracked, `result_io.hpp`) with a JSON parser of
   their own (no third-party code) and, as PLAN 5.7 asks of every reader, a libFuzzer target; the
   ninth target lowers the fuzz workflow's dispatch limit from 750 s to 660 s per target.
8. **`dyng.testing.triad_reference`** binds ADR 0049's oracle, so the Python tests check against
   it and against a pure-Python brute force (which classifies triads by the documented table of
   canonical Venn ids, independent of both).
9. **The diff against A's stdout needs the golden set** (367 MB, `$DYNG_SCRATCH`): the test skips
   without it; hosted CI runs the fixture variant (the four synthetic groups regenerated, recorded
   as `.hg` / `.dgb` files and replayed through the CLI against the committed counts).
10. **The wheels were written to `$DYNG_SCRATCH/wheels/m9b-step2-plugins`** (`DYNG_PLUGIN_OUT`), so
    M8b's records in `0.3.0.dev0-plugins` stay as they were for the growth comparison.
11. **The first install test found a test bug**: `test_results_on_cuda` built its "host" hypergraph
    without resources, which under a plugin module are the cuda defaults; it now names
    `Resources.sequential()`. The wheels were not rebuilt (the bug was in `python/tests`, which the
    install tests read from the tree); `DYNG_PLUGIN_TEST_ONLY=1` reran every install test.

### Measurements and checks

(`parity/results/M9b.md` section 2.)

- **`--format escher` == A's unpatched stdout**: all 60 stdout files of make-test (15 groups x 4
  executables), empty diff; the expected ids of every `.dgb` record matched.
- **Python == A**: the committed make-test counts on sequential, openmp and cuda (both plugins);
  both oracles after every batch; Hypothesis "update chain == compute".
- **Wheels**: `dyng` 2.10 MB (+5.6 % since M8b), `dyng-cu12` 8.67 MB (+4.8 %), `dyng-cu13`
  8.58 MB (+4.8 %), all far below 90 MB; `pytest -m gpu` 69 passed with each plugin and Python;
  the smoke test runs triad_count compute and a two-result update on GPU 1.
- **ci-cuda13**: 216 s wall, 94.2 CPU-minutes (+1.6 against M9a's merge measured the same way);
  the hosted job took 10.5-12.7 minutes at M9a's head, against its 120-minute timeout.
- Tests: `ci/python.sh` 715 passed; the CPU wheel in a fresh venv `pytest -k triad` 78 passed;
  `ci/gpu_local.sh` step `plugin`; Clang 18 syntax 444 files; the fortified build; the frozen
  headers against `65a62c9` and `origin/main`: empty; `ci/api_check.sh` green.

### Machine time

The wheels 4.5 minutes (the core wheel and both plugins, up to the first install test's failure),
then the install tests of both plugins on both Pythons again after the test fix (5 minutes);
the two ci-cuda13 builds 4 minutes each; the portability builds 10 minutes; `ci/python.sh` 1
minute; the golden diff 35 s. Everything under the shared lock, the GPU work on GPU 1; no timed
reading.

### Notes for step 3 (the docs, the examples, the API review and the freeze)

- The API review covers `dyng.triad_count` as built here (ADR 0059 section 5 lists its names) and
  `dyng triad_count`'s flags; freezing them means: the manifest's maturity `stable` (with
  `origin`), the API snapshot section of `triad_count.hpp` frozen, griffe's baseline the reviewed
  names, and `docs/api/cli.md`'s table the frozen flags.
- `examples/python/triad_count.py` can follow `test_plan_examples.py::test_triad_count_example`;
  the tutorial "Hypergraph triads" can quote `dyng.triad_count`'s docstring example and the CLI's
  `--format escher` (the make-test fixture's groups give real output without the golden set).
- The triad-bins JSON is tracked (io); whether its format version 1 is stable from 0.3.0 belongs
  with the `.hg` / `.dgb` line of the Open decisions.
