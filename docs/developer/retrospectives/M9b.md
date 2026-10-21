# Retrospective: M9b (count_local_patterns, Python, the CLI, the docs and the triad_count freeze)

Status: **done** on the branch `m9b-local-patterns` (ready for the pull request; the author merges it with a merge commit). M9b is the second half of PLAN 11.3's
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
   `6716eec` (the accepted M9a; PR #16 open): `origin/main` (`9cde147`, M8b) did not contain M9a
   then. The author merged #16 as `4cc5c3f` (2026-10-21, with the review fix `511f8ff`), and
   `origin/main` was merged into this branch at the M9b review (`12afe75`).
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
| `7012270` | ADR 0059, GOVERNANCE, CHANGELOG, `parity/results/M9b.md` section 2, this section; a test fix of the CUDA tests (below) |
| `759303d` | the two bins tests that need triad_count moved from `dyng_io_tests` to `dyng_triad_count_bins_tests` (the scaffold check builds without the algorithm); clang-format of the new C++ files |

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
12. **The first `ci/check.sh` failed `format` and `scaffold`**: four new C++ files were not
    clang-formatted (the formatter of the environment, 22.1.8, differs from what was typed), and
    the scaffold check builds the library without triad_count (`DYNG_ALGORITHMS`), where two tests
    in `dyng_io_tests` (the pattern names against `triad_count::to_string()`, a result's round
    trip) did not link: they are in `dyng_triad_count_bins_tests` now, built with the algorithm.
    The io module itself never depended on triad_count (its own list of names, checked by that
    test). The second run passed every step.

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
- Tests: `ci/check.sh` all steps (the second run; cpu-only 1,273, dev 1,306, python 715 passed);
  the CPU wheel in a fresh venv `pytest -k triad` 78 passed; `ci/gpu_local.sh` step `plugin`;
  Clang 18 syntax 445 files; the fortified build; the frozen headers against `65a62c9` and
  `origin/main`: empty; `ci/api_check.sh` green.

### Machine time

The wheels 4.5 minutes (the core wheel and both plugins, up to the first install test's failure),
then the install tests of both plugins on both Pythons again after the test fix (5 minutes);
the two ci-cuda13 builds 4 minutes each; the portability builds 10 minutes (twice); `ci/check.sh`
30 minutes (twice); `ci/python.sh` 1 minute; the golden diff 35 s. Everything under the shared lock, the GPU work on GPU 1; no timed
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

## Step 3: the docs, the freeze and the release checks (docs-freeze-release)

**Done** (M9b acceptance criteria 8-10, and the final check of all ten). The algorithm page with
every PLAN 9.5 section, the tutorial "Hypergraph triads" in C++ and Python, the two examples, the
ESCHER-GPU history page, and the triad_count API review and freeze (ADR 0060). The review found
one library defect (host allocation failures of `compute()`, `update()` and `clone()` left the
library as `std::bad_alloc`) and fixed it in `b732aa9`; with that, and with steps 1 and 2, the
library was no longer the one M9a's gates measured, so **every M9a timing and memory gate was
read again on M9b's library** with M9a step 5's protocol: both suites passed, the replays are
exact on the whole coverage table, and the dry 0.3 certificate passes over the new records.

### What was built

| Commit | What |
|---|---|
| `046a6cf` | the header's documentation amendments (the stats of types 1-3, `compute()`'s shape per pattern, `max_bins`) |
| `7b8234f` | the manifest's maturity `stable` (pending D4), the regenerated tables and registry, the API baseline's `[dyng/triad_count.hpp] frozen`, `ci/doxygen_coverage.py`'s freeze rules for the header |
| `b732aa9` | `DYNG_TRANSLATE_ALLOCATION_FAILURE` on triad_count's and count_local_patterns' compute, update and clone; `dyng_triad_count_allocation_failure_tests` |
| `ef28d64` | `examples/cpp/triad_count_update.cpp`, `examples/python/triad_count.py`, `examples/data/triad_count.{hg,dgt}` and the expected output; the CTests, `test_examples.py`, the installed-package consumer in `ci/cmake_consumer.sh`; `compare_output.cmake` with arguments and the skip code |
| `825ec85` | `docs/algorithms/triad_count.md`, `docs/tutorials/hypergraph_triads.md`, `docs/history/escher_gpu.md` and the indexes; `dyng_triad_count_docs_tests` (the 30-bin table against the code); `ci/tutorial_quotes.py` reads `# [tutorial: ...]` markers |
| `2c9b490` | ADR 0060, the freeze tests of both API checks, the Python docstrings, the sketch frozen |
| `155f4ef`, `7c3781d` | CHANGELOG, GOVERNANCE, plan.md, the roadmap; ADR 0060's demonstrations |
| `877dc00`, `29a9ae3`, the records commit | `parity/results/M9b-gates/` (the suites, the replays, the dry certificate), `M9b-sass-final.json`, `parity/results/M9b.md` section 3, this section |

### Deviations from the plan and the step's text (each with the same intent)

1. **The gates were read again, not carried over.** The step allowed the carry-over only if the
   measured library scope was otherwise unchanged (certify's library-scope diff); it was not
   (step 1's driver and engine move, step 2's triad-bins reader, this step's fix), and
   `certify.py` refuses M9a's records at this tree ("measured commit 2eda830 does not have the
   release's code"; the replays likewise). Every gate of both suites was read on `b732aa9` from a
   clean clone with M9a step 5's protocol (`parity/results/M9b.md` 3.1-3.3): 45 / 45 and 11 / 11.
2. **A first timing execution was stopped** after its first case when the review found the
   allocation defect (a library change); its partial records were moved aside, unused.
3. **The review changed the library** (`b732aa9`, finding 7), where ADRs 0035 and 0047 changed only
   documentation: the header promised `out_of_memory_error`, the code let `std::bad_alloc` out.
   No signature changed; the counting kernels are SASS-identical; the gates and replays were read
   on this library.
4. **A container defect is reported, not fixed** (ADR 0060 finding 8, D13): a host allocation
   failure inside `cpp/src/hypergraph`'s OpenMP regions ends the process. A fix is M8b's code,
   outside this step and behind an ADR 0024 reading; the allocation test leaves `update()` out
   with the reason in its source.
5. **The Python half of the freeze has no baseline file**: `ci/api_check.sh` compares with the
   base branch, so `dyng.triad_count` is checked by griffe from its merge into `main`; until then
   `ci/tests/test_api_check.py` pins three breaking changes, and the demonstration used the freeze
   commit `155f4ef` as the base. A renamed keyword-only parameter of `compute()` is not a breaking
   change for griffe (absorbed by `**kwargs`), so the test renames `update()`'s positional
   parameter.
6. **The examples share their output**: both print the same eight lines, so one expected file
   (`examples/data/triad_count_update.expected.txt`, equal to a brute force) checks the C++
   program in the tree and against the installed package and the Python program in pytest. The
   example data live in `examples/data/` (not `cpp/tests/data`, whose sets are fixtures of an
   original in `fixtures.toml`; 1.1 KB).
7. **"Built and run against the installed package in CI"**: the in-tree examples link the build's
   `dyng::dyng` like the others; `ci/cmake_consumer.sh` (the `consumer` step of `ci/check.sh`,
   and of `ci/gpu_local.sh` on cuda) builds `triad_count_update.cpp` with `find_package(dyng)`
   against `cmake --install` and compares its output on every backend of the build.
8. **The 30-bin table is tested from the page**: `TriadCountDocs.*` parses the page's table and
   checks it against the engine's Venn table and the oracle's representatives (the "generated
   from or tested against" of the step: tested).
9. **The "Paper vs fixed code" quotes ESCHER-GPU's `CHANGES.md`** (E-C1, E-C3, E-C7 are recorded
   there; the README states the closed-triad scope, quoted in section 1 of the page). Marked for
   the author's review (D10).
10. **`compare_output.cmake` takes its arguments with `|`**: a `;` inside a `-D` argument of
    `add_test` splits it.

### Measurements and checks

(`parity/results/M9b.md` section 3; the checks in section 4 there.)

- Gates on `b732aa9`: timing 45 / 45 (update per batch 0.861-1.031, full count 0.999-1.003, end
  to end 0.920-0.991); memory 11 / 11 (store 1.0000 at every gated point, the Nsight Systems peak
  1.0015-1.0021); no audit flag; 8 rounds repeated by the monitor (foreign load).
- Replays on `b732aa9`, both modes: cuda 300/300, sequential 120/120, openmp 142/142 at 8 and 1
  threads. The dry 0.3 certificate PASSED over them.
- SASS of the five counting kernels: identical to M9a's merge build.
- `ci/gpu_local.sh`: all 21 steps passed (`ctest -L gpu` 861 / 861, the sanitizers, the triad and
  count_local_patterns steps, the cu13 plugin wheel, clang-tidy); the three wheels: `dyng`
  2.10 MB, `dyng-cu12` 8.67 MB, `dyng-cu13` 8.58 MB (+5.7 / +4.8 / +4.8 % since M8b), their
  install tests on both Pythons (`pytest -m gpu` 69 passed, `DYNG_CPU_ONLY=1` 719 passed).
- `ci/check.sh`: all checks passed (cpu-only 1,279, dev 1,312, harness 349, python 719 passed);
  `ci/docs.sh` -W, the link, snippet and quote checks; `ci/api_check.sh` no breaking change against
  `origin/main`; Clang 18 syntax 429 files, 0 failures; a fresh clone: cpu-only with
  `-D_FORTIFY_SOURCE=3` and no nvcc, no warning, 1,279 / 1,279; parity-cuda, `ctest -L
  'triad_count|experimental|examples'` 397 / 397 on GPU 1, and the consumers (the triad example
  on sequential, openmp and cuda; the user Op by nvcc).

### Machine time

| What | Time |
|---|---|
| the parity-cuda build of the measurement clone (twice: `7b8234f`, then incremental at `b732aa9`) | 15 min, 2 min |
| the timing suite (stopped once after its first case, then in full) and the memory suite | 06:42-07:23 and 07:32-07:39 CDT (exclusive lock on GPU 0; the memory readings on GPU 1) |
| the replays (cuda 3 jobs, sequential 6, openmp 8 threads 3, 1 thread 10; concurrent, the shared lock) | 41, 15, 12 and 31 min |
| `ci/check.sh`; `ci/gpu_local.sh` (concurrent) | about 40 min; 4 h 0 min (07:44-11:44 CDT) |
| the fresh clone (cpu-only fortified; parity-cuda and its tests) | 10 and 15 min |
| the three wheels and their install tests (`ci/plugin_wheels.sh`, cu12 and cu13) | 9 min |

## The milestone

### The acceptance criteria of M9b, checked at the end

| # | Criterion | Where | State |
|---|---|---|---|
| 1 | the experimental header (banner, group, `#error` without nvcc), `count_local_patterns<op_t>` over the triad driver on every backend, the contract, the Op concept with its messages and compile-fail tests, 0-based ids and opaque codes (no key, `+1` or sentinel), the accessor in libdyng, the frozen headers byte-identical | step 1; ADR 0058; results 1 | met |
| 2 | packaging: the `.cuh` installed and self-contained, Doxygen and coverage of the group, the snapshot's experimental tier, the nvcc consumer and the host-only consumer | step 1; results 1.4; ADR 0060 (the experimental change reported, not failed) | met |
| 3 | the four built-in kernels SASS- and resource-identical to M9a's merge build (step 1 and the final build); the built-in Ops equal triad_count bit for bit on make-test and syn100k on every backend; CUDA parity on make-test, syn1m s1 and coauth_b50k_s1 unchanged. M9a's records could not be carried over (the library scope changed), so every M9a gate was read again: passed | results 1.1-1.3, 3.1-3.3 | met |
| 4 | the Op tutorial: a pattern not in the enum, the 1-bin triangles Op against `total()`, the brute-force oracle and "baseline + deltas == recount" on every backend, quotes pinned by the drift check, `good-first-issue:op`, `python_gaps.md` | step 1 | met |
| 5 | Python: `dyng.triad_count` (compute, update, Options, Result, Stats), `dyng.update` over hypergraphs, `dyng.citation`, stubs and griffe, pytest on CPU and with the cu13 plugin on GPU, the PLAN 5.5 example | step 2; results 2.2; the plugin step of `ci/gpu_local.sh` | met |
| 6 | the CLI with flags from the fields, triad bins as JSON in a round trip, `--format escher` byte for byte against ESCHER-GPU's unpatched stdout of make-test | step 2; results 2.1 | met |
| 7 | the wheels <= 90 MB with their growth, the plugin smoke test with triad_count on GPU, the ci-cuda13 compile time with its margin | step 2; results 2.3, 4 | met |
| 8 | the docs: the algorithm page with every PLAN 9.5 section (the tested 30-bin table, requirements per pattern, the template mapping, the paper differences, "Paper vs fixed code", the mapping, the gate table), the tutorial "Hypergraph triads" in C++ and Python, tested, the two examples (<= 60 lines, built and run against the installed package), the README catalogue, the CHANGELOG, `docs/history/escher_gpu.md`, the sketch frozen, the docs with -W and the link and snippet checks | step 3 | met |
| 9 | the API review ADR (0060) with every new public name; the maturity `stable` with origin pending the author (a GOVERNANCE row); the baselines frozen and a deliberate change failing both checks, an experimental change reported; the frozen paths' diff against `65a62c9` and `main` empty | step 3; ADR 0060 | met |
| 10 | `ci/check.sh` and `ci/gpu_local.sh` green; fresh clones CPU-only and CUDA; the dry 0.3 certificate over the ESCHER records; the records; merge commits only | step 3; results 3-4 | met |

### Deviations of M9b from the plan (the milestone)

- **PLAN 6.4.6 / 11.3**: `count_local_patterns` ships in 0.3 (the plan's 0.5 for the extension
  point), as experimental, on triad_count's own driver and engine (ADR 0058); triad_count is
  frozen at M9b's end, not at M9a's (D2, D4).
- **PLAN 5.5**: `kind=` takes a string or a `Pattern` `StrEnum` (the typed layer's first public
  enumeration class); `Result.counts` is a NumPy copy; the container parameter is `hypergraph`
  (ADR 0059).
- **PLAN 8.6 / criterion 3**: the gates were read again rather than carried over (the library scope
  changed), with the same verdicts as M9a's.
- The review's library fix and the container defect it exposed (ADR 0060, findings 7 and 8).

### Effort

M9b's three steps ran on 2026-10-21 (CDT), one day of wall-clock time, most of it machine time
(the replays twice in step 1, the wheels in step 2, the gate readings, the replays and the full
GPU gate in step 3).

### The consolidated decisions for the author (0.3.0)

Every item below is taken on the plan's default and awaits the author's decision at the 0.3.0
release (GOVERNANCE.md, open decisions):

1. **Stable hypergraph headers and file formats** (M8b, ADR 0047): the hypergraph headers frozen
   at M8b are stable from 0.3.0; `.dgb` at format version 1, `.hg` without a version field
   (ADR 0045 section 1; a `%hg 1` line is the alternative); and, with them, the triad-bins JSON
   at format version 1 (ADR 0059).
2. **`triad_count` stable (D4)**: `<dyng/triad_count.hpp>`, `dyng.triad_count` and `dyng
   triad_count` frozen at the end of M9b and stable from 0.3.0 (ADR 0060; the certificate carries
   the full gate record, read on M9b's library); `count_local_patterns` stays experimental.
3. **The new public names (D9)**: every name M9 introduces is listed in ADR 0060 ("The new public
   names"), with ADRs 0049, 0051, 0052, 0058 and 0059: triad_count's (frozen), Python's and the
   command line's (frozen), the experimental template, its members and the Op member names, the
   row accessors, the oracle, the triad bins, the generators' host model, `dyng-compat-escher run`.
4. **The test-data budget (D5)**: decided (1.25 MB; 1,002,671 bytes used under `cpp/tests/data`,
   plus 1.1 KB of example data in `examples/data`).
5. **The host-engine scope change (D3)**: decided (the incremental host engines instead of PLAN
   4.6's recounts; untuned, no performance gate; the OpenMP tuning stays on the roadmap).
6. **The per-backend golden coverage (D6)**: cuda every golden case; sequential make-test and the
   syn100k grid; openmp at 1 and 8 threads the same plus the syn1m s1 grid and coauth_b50k_s1
   types 1-2 (the 0.3 certificate requires exactly this table).
7. **The nsys-peak reading (D1, decided) and its neighbours**: the gated peak of live allocations
   1.0015-1.0021; the footprint between batches 1.63-1.80x (D11, reported, not gated); the host
   model in the gated update (D12: `update_with_host_model` up to 1.072 on synthetic rows, reported).
   With them the gate rows, cases and rounds (D7) and the two suites (D8).
8. **The "paper vs fixed code" wording (D10)**: section 8 of `docs/algorithms/triad_count.md`
   (E-C1, E-C3, E-C7 in ESCHER-GPU's words), marked for the author's review.
9. **The alignment-flag ADR** (R030, the 0.3 release preparation; the decision of 2026-10-08):
   the flags are a 0.3 performance option with their own ADR; if the release builds change their
   flags, the ESCHER suites (and every other suite of the certificate) must be measured again.
10. **New: the container's OpenMP allocation-failure defect (D13)**: a host allocation failure in
    `cpp/src/hypergraph`'s OpenMP regions ends the process; fix in 0.3.0 (with its ADR 0024
    reading) or in a 0.3.x release.

Also the author's: the merge of M9b's pull request (a merge commit, with the `api-change` label).
M9a's (#16) is merged (`4cc5c3f`).

### Open items

- **Hosted CI has not run on this branch** (nothing is pushed); the pull request is the
  orchestrator's.
- D13 (above), and the names and maturity pending at 0.3.0.

### Ready-to-paste pull request summary

```markdown
## M9b: count_local_patterns, triad_count in Python and on the command line, the docs and the freeze (0.3)

Follows M9a (#16, merged as `4cc5c3f`; `main` is merged into this branch). Completes M9 (PLAN 11.3; the author's decision D2): with M8a, M8b and M9a it
is the content of 0.3.0.

**`experimental::count_local_patterns<op_t>`** (`<dyng/experimental/local_patterns.cuh>`, nvcc
only; EXPERIMENTAL PUBLIC, may change in any minor release; ADR 0058): count a hypergraph pattern
of your own, an Op over the triangles, pairs or hyperedges of the line graph, with triad_count's
wedge engine, exact under hyperedge batches on cuda, sequential and openmp. The four patterns of
triad_count are built-in Ops of the same engine; their counting kernels are SASS- and
resource-identical to M9a's. Tutorial: "Count a new hypergraph pattern (an Op)".

**Python and the command line** (ADR 0059): `dyng.triad_count` (compute, update, Options, Result,
Stats; `kind` as a string or `Pattern`), `dyng.update()` over hypergraphs (several results, one
commit), `dyng.io` triad bins as JSON, `dyng.testing.triad_reference`; `dyng triad_count
compute|update` with `--format escher`, byte for byte against ESCHER-GPU's unpatched output on all
60 make-test cases. Wheels: dyng 2.1 MB, dyng-cu12 8.7 MB, dyng-cu13 8.6 MB.

**The freeze** (ADR 0060, the API review): `<dyng/triad_count.hpp>`, `dyng.triad_count` and `dyng
triad_count` frozen, maturity `stable` from 0.3.0 (the plan's default D4, pending the author's
confirmation at the release). **api-change**: the header moves from tracked to frozen; no
signature changed. The review fixed one defect (host allocation failures of compute, update and
clone are `out_of_memory_error`, as documented) and reports one in the container (D13).

**Docs**: `docs/algorithms/triad_count.md` (the 30-bin table tested against the code, requirements
per pattern, differences from the paper, "Paper vs fixed code" for the author's review: D10, the
gate record), the tutorial "Hypergraph triads" (C++ and Python), `examples/cpp/triad_count_update.cpp`
and `examples/python/triad_count.py` (run in CI, the C++ one against the installed package),
`docs/history/escher_gpu.md`.

**Gates, read again on M9b's library** (the library scope changed since M9a's readings; ESCHER-GPU
re-measured in the same session, GPU 0, base clock lock, M9a's protocol):

| | coauth-DBLP 50K s1-s3 (11 rounds) | syn1M / syn2M 50K s1-s3 (21 rounds) | bar |
|---|---|---|---|
| update, each of 3 batches | 0.952-0.973 | 0.861-1.031 | 1.05 |
| full count | 0.999 | 1.002-1.003 | 1.05 |
| end to end | 0.920-0.925 | 0.957-0.991 | 1.10 |
| store accounting (every gated point) | 1.0000 | 1.0000 | 1.05 |
| Nsight Systems peak of live allocations (D1) | 1.0015 | 1.0021 / 1.0016 | 1.05 |

Parity (exact, by key and by content, on M9b's library): cuda 300/300, sequential 120/120, openmp
142/142 at 1 and 8 threads; the dry 0.3 certificate of triad_count passes
(`parity/results/M9b-gates/`).

**Checks**: `ci/check.sh` all passed; `ci/gpu_local.sh` all 21 steps passed on GPU 1 (`ctest -L gpu` 861 / 861; memcheck, synccheck, racecheck and initcheck clean, the triad and count_local_patterns steps included); a fresh clone
(CPU-only fortified without nvcc: 1,279 / 1,279; parity-cuda: the triad, experimental and example
tests 397 / 397, the installed-package consumers on every backend); Clang 18 syntax; `ci/docs.sh`
-W; the API checks and their deliberate-change demonstrations; frozen headers unchanged
(`git diff 65a62c9 -- cpp/include/dyng/{hypergraph,update.hpp,core,graph}` empty, also against
`main`).

**For the author** (GOVERNANCE.md, open decisions; the list in
`docs/developer/retrospectives/M9b.md`): D4 (triad_count stable), D9 (the new public names, ADR
0060), D10 (the paper-vs-fixed wording), the hypergraph and file-format stability of M8b, D6-D8,
D11, D12, and the new D13. ADRs 0058-0060. Records: `parity/results/M9b.md`. Merge with a merge
commit.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```
