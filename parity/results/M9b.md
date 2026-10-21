# M9b parity record: count_local_patterns, Python, the CLI, the docs and the triad_count freeze

Date: 2026-10-21 onwards. M9b is the second half of PLAN 11.3's M9 row (the author's decision
D2): `experimental::count_local_patterns<op_t>` over M9a's wedge engine, Python, the CLI, the docs
and the `triad_count` freeze. Its gated code paths are M9a's: the four built-in counting kernels
must stay SASS- and resource-identical to M9a's build, otherwise every M9a gate is read again.
Each step adds its section.

| Item | Value |
|---|---|
| Branch | `m9b-local-patterns`, created by the orchestrator from `m9a-triad-count` at `6716eec` (the accepted M9a; PR #16 open, not merged: `origin/main` is `9cde147`, M8b) |
| Reference | ESCHER-GPU `abcf9b2bd42f896ec3103a6835edf713ed5e416c` (A) |
| Golden set | `triad_count_escher_gpu`, manifest `4fb7dbcf...` (M9a's, unchanged) |
| Toolchain | GCC 12.2.0, CUDA 13.1 (`/usr/local/cuda-13.1/bin/nvcc`), sm_86 |
| Host | Xeon Gold 6258R (28 cores, 56 threads), 124 GB, 2x RTX A5000 (sm_86); every run of step 1 on GPU 1 under the shared lock (no timed reading) |

## 1. Step local-patterns-experimental: one driver, one engine (criteria 1-4)

`experimental::count_local_patterns<op_t>` (ADR 0058) runs on the driver of `triad_count`
(`local_pattern_problem<V, W, flavor_t>`) and on M9a's engine, whose kernels and host work moved
unchanged into the installed headers `dyng/experimental/detail/local_patterns_{kernels.cuh,
engine.hpp}`; triad_count's four patterns are built-in Ops of it. Commits `b21cb47` (the engine
and the driver), `c69f311` (the API and its tests), `67d6f3a` (the replay counter), `7dc6711`
(packaging), `888fe07` (docs), and the records commit.

### 1.1 The built-in kernels: SASS and resources (criterion 3; `M9b-sass-m9a-merge-build.json`, `M9b-sass-step1.json`)

`parity/triad_sass.py report` on `engine/counters.cu.o` of the reduced Release build of the
parity-cuda preset (`cmake --preset parity-cuda -DDYNG_ALGORITHMS=triad_count`), built from
M9a's tree `6716eec` (the merge build: the same library sources as M9a's review readings,
`2eda830`; the five digests equal `M9a-review-sass-escher_gpu.json`'s `head`) and from this
step's final tree, then `triad_sass.py compare` (the normalized SASS digest, the instruction
count, every backward-branch loop, `cuobjdump -res-usage`):

| Kernel | REG | stack, local | shared (B) | instructions | largest loops | M9a / M9b step 1 |
|---|---:|---|---:|---:|---|---|
| count_wedges | 14 | 0, 0 | 0 | 48 | 1 | **identical** |
| triangles<motif> | 39 | 0, 0 | 240 | 560 | 479, 462, 459 | **identical** |
| triangles<type3> | 40 | 0, 0 | 8 | 576 | 467, 464, 457 | **identical** |
| pairs<type2> | 37 | 0, 0 | 8 | 512 | 420, 416, 228 | **identical** |
| keys<choose3> | 22 | 0, 0 | 8 | 288 | 96, 95, 21 | **identical** |

The digests compare the machine code with the functions' own names removed (the kernels' mangled
names changed: they are `dyng::detail::local_patterns::triangle_kernel<motif_op>` etc. now, with
`line_rows` and `experimental::hyperedge_rows` parameters of the same layout as M9a's
`escher::row_view` and `hyperedge_rows`). The `-res-usage` lines of the object's eight functions
are equal line for line. **No M9a gate needs a re-read**: the gated kernels are M9a's, and the
host side of an update makes the same CUDA calls in the same order (the memsets and read-backs of
the same 61 words: the phases keep M9a's stride of 30 bins), with the launch through the Op's
table one indirect call per phase.

### 1.2 count_local_patterns equals triad_count, bit for bit (criterion 3)

`parity/compare.py triad_count --counter local_patterns --mode both` (the counter of `67d6f3a`:
the four built-in Ops through `count_local_patterns`, instantiated by nvcc in the replay as a
user's Op is, updated beside triad_count's own four results in one commit per batch; after every
batch the counts, the per-bin deltas, `removed` / `added`, both frontier sizes, `affected` and the
inserted ids must be equal bit for bit, the counts must equal a fresh `count_local_patterns`
compute(), and the counts reported, count_local_patterns', are compared with A's goldens by key and
by content with integrity after every batch), from the parity-cuda build at the commit named in
each record (`M9b-parity-local_patterns-{sequential,openmp8,cuda}-escher_gpu.json`):

| Backend | Cases | Result | Wall |
|---|---|---|---|
| sequential | make-test (60) + the syn100k grid (60) | **120/120 exact**, both modes | 1,031 s (4,622 s of group time; 6 jobs) |
| openmp, 8 threads | make-test + the syn100k grid | **120/120 exact**, both modes | 279 s (708 s; 3 jobs) |
| cuda (GPU 1) | make-test + the syn100k grid | **120/120 exact**, both modes | 66 s (190 s; 3 jobs) |

The API tests (`dyng_local_patterns_tests`, `cpp/tests/experimental`) check the same equality on
random hypergraphs and batches on the three backends and both instantiations ((int32,
unweighted), (int32, int32)), the probes that see no key, `+1` or sentinel, and the contract; the
tutorial's test runs its four Ops on every backend against a brute force and triad_count.

### 1.3 CUDA parity unchanged (criterion 3)

`parity/compare.py triad_count --backend cuda --counter algorithm --mode both --cases
make-test,syn1m_s1,coauth_b50k_s1` (triad_count on the new driver; `M9b-parity-cuda-escher_gpu.json`):
**84/84 exact** (382 s, 731 s of group time, 3 jobs, GPU 1): make-test (60), the syn1m s1 grid (20) and
coauth_b50k_s1 (4), the full counts, every net delta and count after equal to the goldens', by key
and by content, a fresh compute() after every batch. The host backends of triad_count run through
the new per-source plumbing (ADR 0058 section 1); the sequential and OpenMP replays of 1.2 run
triad_count on them beside the Ops, equal bit for bit, so they also cover triad_count's host
parity on make-test and the syn100k grid.

A first run of all four replays (from `888fe07`'s tree but a replay tool built at `67d6f3a`, before
two `noexcept` qualifiers of the experimental header's row accessors) gave the same verdicts
(120/120, 120/120, 120/120, 84/84); the records are the runs with the tool built from the final
tree.

### 1.4 Tests and checks of the step

| Check | Result |
|---|---|
| `ctest -L 'triad_count\|operators\|testing'` (dev-cuda, GPU 1) | 334 / 334 (the mutation CTests `triad_count.mutation.*`: every mutant detected on both engines) |
| `ctest -L experimental` (dev-cuda, GPU 1) | 57 / 57: the API tests and the tutorial's on cuda, sequential and openmp, both instantiations; 13 compile-failure cases through g++ and 12 through nvcc, each matching its own message; the `.cuh` without nvcc; the nvcc self-containment objects |
| `ci/gpu_local.sh` step `experimental` (GPU 1) | passed: memcheck, racecheck (0 hazards) and synccheck on the tutorial's CUDA runs, memcheck on the API tests' CUDA runs, `ci/cmake_consumer.sh build/dev-cuda` (the documented consumer, the host-only consumer, a user Op compiled by nvcc against the installed package on sequential, openmp and cuda) |
| `ci/gpu_local.sh` step `gpu` (`ctest -L gpu`, dev-cuda, GPU 1) | **860 / 860** |
| `ci/gpu_local.sh` step `triad` (GPU 1) | passed: synccheck and initcheck on `dyng_triad_count_cuda_tests`, memcheck / synccheck / initcheck on one coauth_b50k_s1 batch of `dyng-compat-escher run`, racecheck on the kit's C1, C11 and C12 and on the make-test replays (`--counter algorithm`, both modes: 60/60) |
| `ci/check.sh` | every step passed but `harness` on the first run: `parity/tests/test_release_cert.py::test_mutation_points_match_the_code_once` looks for the triad mutation hooks in `cpp/src/algorithms/triad_count`, and the two frontier-filter hooks moved with the predicates into the installed engine header; the test now reads that header too, and `DYNG_CHECK_ONLY=harness ci/check.sh` **passed**. The run: clang-format, cpu-only (1,263 / 1,263 cpu tests) and dev (1,296 / 1,296), the consumers (the documented one and the host-only consumer of the experimental header against the installed cpu-only build), clang-tidy, REUSE, provenance, the data budget (1,002,671 of 1,250,000 bytes), regen, python, the API check (frozen C++ API and griffe against `origin/main`), the scaffold, docs, pre-commit |
| Clang 18 `-fsyntax-only` (`clang_syntax.py`, a dev database) | 419 files, **0 failures** |
| cpu-only with `-D_FORTIFY_SOURCE=3`, no nvcc on PATH | built without warnings; `ctest -L experimental` passed (the host half of the tutorial, the g++ compile-failure cases) |
| frozen headers against `65a62c9` (`hypergraph/*`, `update.hpp`, `core/*`, `graph/*`) | empty |
| `docs` (`ci/docs.sh`, -W) | passed: Doxygen and its coverage (the group `experimental` and its banner), the API snapshot (the experimental tier), the tutorial quotes (25 pins), Sphinx, the links |

The other steps of `ci/gpu_local.sh` (the whole-suite memcheck / synccheck / racecheck, the CBST and
hypergraph sanitizers and replays, the plugin wheel, clang-tidy over the CUDA branches) were not
run in this step; the release-readiness run of the milestone (criterion 10) runs them all.

## 2. Step python-cli-wheels: Python, the CLI, the triad bins, the wheels (criteria 5-7)

`dyng.triad_count` and `dyng.update()` over hypergraphs (ADR 0059), the triad-bins JSON
(`io::read_triad_bins_json`, `dyng.io.read_triad_bins`), `dyng.testing.triad_reference`, and
`dyng triad_count compute|update` with `--format json|escher`. Commits `6e98222` (the JSON),
`86e9224` (Python), `db25f77` (the CLI), `7012270` (the records), `759303d` (the bins tests
without triad_count in `dyng_io_tests`, clang-format). No library code of the
algorithm, the engine or the operators changed in this step (`git diff 2771348 -- cpp/src/algorithms
cpp/src/engine cpp/src/operators cpp/include/dyng/experimental cpp/include/dyng/triad_count.hpp`:
only the manifest's `python` key): the built-in kernels are those of section 1.1, and no M9a gate
needs a reading.

### 2.1 `--format escher` against A's unpatched stdout (criterion 6)

`python/tests/test_triad_count_cli.py::test_escher_format_equals_escher_gpus_stdout`: for every
make-test group of the golden set (15 groups; the input `.hg` of A's loaded rows and the group's
`.dgb` batches, decompressed from `$DYNG_SCRATCH/goldens/triad_count/escher_gpu`), `dyng
triad_count update --format escher` (or `compute` for the groups without batches) with A's
vertex count, `--id-reuse erase_first --capacity escher_gpu --min-initial-payload <A's payload
capacity>` and each pattern, against `<exe>.stdout` (A's unpatched `build/main`, `type1`, `type2`,
`type3`) without A's generator block (the five lines its generator prints for a generated input):

| Groups | Executables | Diff |
|---|---|---|
| 01-smoke ... 15-medium (15) | main, type1, type2, type3 (60 stdout files) | **empty for all 60**: the `Hypergraph:` line, the baseline, every `Batch` line and every delta line byte for byte (openmp, 4 threads; 35 s, 24 s of them the h-motif count of 14-skewed's 1,200-hyperedge clique) |

Every `.dgb` record carries the expected ids; the CLI checked them after every batch (they
matched). In hosted CI (no golden set) `test_escher_format_equals_the_make_test_fixture`
regenerates the four synthetic groups (01, 03, 07, 12) with the legacy generators, records them as
`.hg` and `.dgb` files, and compares the CLI's count and batch lines with A's counts in the
committed fixture (`escher_gpu_make_test.txt`): equal on sequential and openmp.

### 2.2 Python against A's counts and the oracles (criterion 5)

| Test (`python/tests/test_triad_count.py`) | Result |
|---|---|
| A's make-test counts (the committed fixture, D5): the four synthetic groups regenerated, all four patterns on 01 and 12, types 1 and 2 on the syn100k groups 03 and 07, every batch's (deletions, insertions, delta) through `dyng.update()` | equal on sequential and openmp (and on cuda with the cu12 and cu13 plugins, `test_triad_count_cuda.py`) |
| compute / update against `dyng.testing.triad_reference` (ESCHER-GPU's CPU reference) and a pure-Python brute force (every triple, its Venn regions, the documented table of canonical ids), every pattern, unweighted and weighted, both host backends | equal after every batch |
| Hypothesis "update chain == compute" (`dst.hypergraphs()`, `dst.hyperedge_batches(model)`, both id orders, 1-4 patterns updated by one `dyng.update()`, 40 examples x 3 batches) | passed; every chain also equal to the brute force and its deltas to the counts' differences |
| the PLAN 5.5 example (adapted: `inserted_ids` is already an int64 array) | passed (`test_plan_examples.py`) |

### 2.3 Wheels (criterion 7)

`ci/plugin_wheels.sh` with `DYNG_PLUGINS="cu12 cu13"` (CUDA 12.9 and 13.1, the glibc 2.28
toolchain, the release architectures), records in `$DYNG_SCRATCH/wheels/m9b-step2-plugins`:

| Distribution | M8b (`0.3.0.dev0-plugins`, 2026-10-14) | M9b step 2 | Growth | Budget |
|---|---:|---:|---:|---:|
| `dyng` (CPU) | 1,991,049 B | 2,102,460 B (2.10 MB) | +111,411 B (+5.6 %) | 90 MB |
| `dyng` sdist | 1,422,705 B | 1,588,201 B | +165,496 B | - |
| `dyng-cu12` | 8,268,291 B | 8,665,367 B (8.67 MB) | +397,076 B (+4.8 %) | 90 MB |
| `dyng-cu13` | 8,188,078 B | 8,579,930 B (8.58 MB) | +391,852 B (+4.8 %) | 90 MB |

The growth is triad_count (M9a: the wedge engine's kernels, the host engine, the oracle) and this
step's bindings and JSON reader; `ci/wheel_check.py`, `twine check --strict` and the code-object
check (SASS sm_75-sm_120 and PTX sm_120 per toolkit) passed. Install tests in fresh venvs (Python
3.12 and 3.13, each plugin): the smoke test (`ci/plugin_smoke.py`, now with triad_count compute
and a `dyng.update()` of two results on GPU 1, equal to the sequential backend), the fallbacks
(no device, no driver), `pytest -m gpu` **69 passed** (61 before; the 8 triad_count CUDA tests),
the suite with `DYNG_CPU_ONLY=1` **715 passed**. The CPU wheel alone in a fresh venv:
`pytest -k triad` **78 passed** (the golden diff included). `ci/gpu_local.sh` step `plugin`
(`DYNG_GPU_PLUGIN_WHEELS`, cu13, GPU 1): passed.

**The ci-cuda13 compile time.** The compile-only preset `ci-cuda13` (CUDA 13.1, release
architectures, every instantiation, test and mutant), from `git archive` copies, `-j 28` on the
shared machine one after the other (load 9-22; CPU-minutes are the sums of `.ninja_log` step
times):

| | M9a merge `6716eec` | M9b step 2 (`db25f77`) |
|---|---:|---:|
| wall | 214 s | 216 s |
| steps | 1,032 | 1,065 |
| CPU-minutes | 92.5 | 94.2 (+1.6: `local_patterns_test.cu` 28 s, `tutorial_test.cu` 22 s, `triad_replay_local_patterns.cu` 13 s, the two `.cuh` self-containment units 6 s each, the host units) |
| `libdyng.so` (unstripped) | 41,659,800 B | 41,801,272 B (+0.3 %) |

The hosted `cuda-build` jobs of M9a's head (`6716eec`, run 39555385200) took 10.5 min (CUDA 13.1.1),
12.0 min (13.4.1) and 12.7 min (ci-cuda12 with 12.9.2) of their `timeout-minutes: 120`; M9b adds
1.8 % of the preset's compile work, so the margin stays about 107 minutes.

### 2.4 Tests and checks of the step

| Check | Result |
|---|---|
| `ci/python.sh` (editable install, stubs, mypy strict, pytest) | 715 passed, 80 skipped (the CUDA tests without a plugin, pandas) |
| `dyng_io_tests` `TriadBins.*` (dev) | 8 / 8: the writer's text, round trips (extremes, a real result), any JSON layout, the reader's errors with line and column |
| `fuzz.replay.triad_bins`, `fuzz.selftest.triad_bins` | passed (the planted writer bug reported) |
| `ci/api_check.sh` against `origin/main` | no breaking change: the frozen C++ API (an include added to `dyng.hpp` by M9a) and griffe (additions only) |
| frozen headers against `65a62c9` and `origin/main` | empty |
| `ci/check.sh` (all steps) | the first run (`7012270`): every step passed but `format` (clang-format of four new C++ files) and `scaffold` (the scaffold's build has no triad_count, and two tests of `dyng_io_tests` used it); after `759303d` (the two tests moved to `dyng_triad_count_bins_tests`, the files formatted) **all checks passed**: clang-format, cpu-only 1,273 / 1,273 and dev 1,306 / 1,306 (`ctest -L cpu`), the consumers, clang-tidy, REUSE, provenance, the data budget, regen, the harness (344 passed), python (715 passed, 80 skipped), the API check, the scaffold (80 + 3 + 54 + 41), docs (-W), pre-commit |
| Clang 18 `-fsyntax-only` (a dev database with the Python bindings) | 445 files at `759303d`, **0 failures** |
| cpu-only with `-D_FORTIFY_SOURCE=3` (with the bindings) | built without warnings at `759303d` |

The wheels and the install tests ran on the tree of `db25f77` (plus the test fix of the records
commit); the later commits change test files, formatting and docs only (`git diff db25f77
759303d -- cpp/src cpp/include python/dyng python/bindings`: the whitespace of `result_io.hpp`,
`triad_bins.cpp` and two binding files, and two docstrings), so the wheels' sizes and their tests
stand.

## 3. Step docs-freeze-release: the freeze, the gates on M9b's library, the release checks (criteria 8-10)

The algorithm page, the tutorial "Hypergraph triads", the examples, ESCHER-GPU's history page and
the triad_count API review and freeze (ADR 0060). Commits `046a6cf` (the header's documentation
amendments), `7b8234f` (maturity `stable`, the header frozen), `b732aa9` (the review's one library
fix: host allocation failures of compute, update and clone are `out_of_memory_error`), `ef28d64`
(the examples), `825ec85` (the page, the tutorial, the history page), `2c9b490` (the review's
tests), `155f4ef` (CHANGELOG, GOVERNANCE, plan, roadmap), the ADR's demonstrations and the records.

### 3.1 Carry-over or re-read: re-read (criterion 3)

Step 1 recorded the four built-in kernels as SASS- and resource-identical to M9a's merge build,
but the measured library scope of `certify.py` (`git diff <measured> <release> -- cpp/include
cpp/src cpp/CMakeLists.txt tools cmake CMakeLists.txt CMakePresets.json docs/references.bib`) is
not otherwise unchanged since M9a's readings (`2eda830`): step 1 moved the engine and generalized
the driver, step 2 added the triad-bins reader, and this step's review fixed the allocation
failures (`b732aa9`). So M9a's records cannot certify 0.3.0, and **every M9a timing and memory gate
was read again** on `b732aa9`, with M9a step 5's protocol exactly: a clean clone at `b732aa9`
(`$DYNG_SCRATCH/m9b-gates/dyng`, no `+dirty`), its parity-cuda build, GPU 0 under the exclusive
lock taken by the harness, the base clock lock (SM 1170 / memory 7601 MHz) for every gated and
recorded-at-lock reading, `CUDA_MODULE_LOADING=EAGER`, no `OMP_*`, one heap layout, A/B
alternation, the per-round guard, the contamination monitor and audit, never
`--keep-contaminated`; the memory readings on GPU 1 under the shared lock with no foreign process.
`git diff b732aa9 HEAD` over the library scope is empty at the end of the step. The counting
kernels of the measured build are identical to M9a's merge build
(`parity/triad_sass.py compare`: digests, instruction counts 48 / 560 / 576 / 512 / 288, every
loop, registers 14 / 39 / 40 / 37 / 22, no stack or local memory; `M9b-sass-final.json`).

A first execution of the timing suite was started on `7b8234f` and stopped after its first case,
when the review found the allocation-failure defect (a library change); its partial records were
moved aside (`$DYNG_SCRATCH/runs/bench/m9b-aborted-7b8234f`) and are not used.

### 3.2 The timing gates (`ipdps26_escher.yaml`, `M9b-gates/ipdps26_escher.json`)

`bench_suite.py run benchmarks/paper/ipdps26_escher.yaml --version m9b` (2026-10-21 06:42-07:23
CDT): **45 / 45 gated rows within their gates, ratios 0.861-1.031; the suite PASSED.** Medians
(ms), A / port, ratio:

| Case (rounds) | update[b1] | update[b2] | update[b3] | full_count | end_to_end |
|---|---|---|---|---|---|
| coauth_b50k_s1 (11) | 510.5 / 486.8 **0.954** | 549.5 / 531.5 **0.967** | 550.7 / 534.3 **0.970** | 5,294.1 / 5,286.9 **0.999** | 9,563.9 / 8,844.3 **0.925** |
| coauth_b50k_s2 (11) | 540.8 / 515.9 **0.954** | 539.6 / 525.1 **0.973** | 546.1 / 520.0 **0.952** | 5,293.3 / 5,286.1 **0.999** | 9,587.1 / 8,826.8 **0.921** |
| coauth_b50k_s3 (11) | 524.7 / 501.3 **0.956** | 538.0 / 519.7 **0.966** | 533.9 / 516.3 **0.967** | 5,293.1 / 5,286.9 **0.999** | 9,577.1 / 8,810.0 **0.920** |
| syn1m_b50k_s1 (21) | 164.2 / 143.5 **0.874** | 144.2 / 145.7 **1.011** | 155.1 / 147.3 **0.949** | 635.3 / 636.7 **1.002** | 2,482.1 / 2,460.5 **0.991** |
| syn1m_b50k_s2 (21) | 164.3 / 143.9 **0.876** | 155.4 / 147.5 **0.949** | 144.7 / 145.9 **1.008** | 640.3 / 641.7 **1.002** | 2,494.2 / 2,463.6 **0.988** |
| syn1m_b50k_s3 (21) | 148.5 / 142.5 **0.960** | 161.6 / 147.1 **0.910** | 156.0 / 148.0 **0.949** | 639.4 / 641.1 **1.003** | 2,491.2 / 2,469.0 **0.991** |
| syn2m_b50k_s1 (21) | 184.2 / 158.5 **0.861** | 154.2 / 158.5 **1.028** | 169.1 / 161.3 **0.954** | 1,273.5 / 1,276.4 **1.002** | 4,440.5 / 4,249.1 **0.957** |
| syn2m_b50k_s2 (21) | 157.3 / 153.3 **0.975** | 181.8 / 162.6 **0.895** | 155.6 / 159.7 **1.026** | 1,279.1 / 1,282.2 **1.002** | 4,431.0 / 4,255.0 **0.960** |
| syn2m_b50k_s3 (21) | 158.0 / 154.6 **0.978** | 181.3 / 163.4 **0.901** | 154.4 / 159.2 **1.031** | 1,280.3 / 1,283.5 **1.003** | 4,424.5 / 4,269.3 **0.965** |
| gate | <= 1.05 | <= 1.05 | <= 1.05 | <= 1.05 | <= 1.10 |

The same picture as M9a's readings (step 5: 0.858-1.028; the review: 0.853-1.019): the highest
synthetic rows (1.008-1.031) are the batches where A's maintain is at its low, 0.019-0.042 below
the bar. The audit: no reading flagged; the monitor rejected and repeated 8 attempts (rounds 10
and 11 of syn1m_b50k_s2 and round 13 of syn1m_b50k_s3: foreign CPU load 2.0-9.7 cores and, once,
foreign `gst-plugin-scanner` processes on GPU 0, the machine's other users). Host synchronizations
per batch: the port 22 (coauth) and 19-22, A 90-96, as before.

**Recorded, not gated** (ratios):

| Reading | Case | update b1 / b2 / b3 | full_count | end_to_end |
|---|---|---|---|---|
| default clocks (11 rounds) | coauth_b50k_s1 | 0.956 / 0.970 / 0.964 | 1.004 | 0.935 |
| | syn1m_b50k_s1 | 0.888 / 1.011 / 0.990 | 1.002 | 1.011 |
| | syn2m_b50k_s1 | 0.852 / 1.031 / 0.956 | 1.000 | 0.961 |
| batch sizes (base, 5 rounds) | coauth_b100k_s1 | 0.987 / 0.993 / 0.976 | 0.999 | 0.934 |
| | coauth_b200k_s1 | 0.990 / 1.019 / 1.018 | 0.999 | 0.956 |
| StatHyper types (base, 5 rounds; `tkde26_escher_plus.yaml`) | coauth_b50k_s1 type1 | 0.741 / 0.774 / 0.753 | 0.143 (3.8 / 0.5 ms) | 0.810 |
| | type2 | 0.748 / 0.797 / 0.769 | 0.828 | 0.813 |
| | type3 | 0.955 / 0.956 / 0.969 | 0.992 | 0.938 |

The update with the driver's host model counted (`update_with_host_model[b<N>]`, D12, reported
beside the gate): coauth 0.963-0.990; synthetic up to 1.052-1.072 on four rows at the base lock
(M9a's review: up to 1.061) and 1.080 on syn2m_b50k_s1 b2 at default clocks; the asymmetry and the
choice stay the author's (D12).

### 3.3 The memory gates (`tkde26_escher_plus.yaml`, `M9b-gates/tkde26_escher_plus.json`)

`bench_suite.py run benchmarks/paper/tkde26_escher_plus.yaml --version m9b` (07:32-07:39 CDT; GPU
1, the shared lock, no foreign process before or after any case; the count guard on all four processes of each
case): **11 / 11 gated rows within 1.05, ratios 1.000-1.002; PASSED.**

| Case | store_memory (every gated point) | device_memory: A peak | port peak (range) | port / A | triad MiB during the apply | footprint at the peak; between batches |
|---|---|---|---|---:|---:|---|
| coauth_b50k_s1 | build, b1, b2, b3 equal in bytes: **1.0000** (A's print 1,833.7 / 1,835.0 MB agrees) | 3,018.9 MiB | 3,023.3 MiB (`hypergraph.apply.line_graph`) | **1.0015** | 4.2 | 1.028; 1.63-1.68 (flagged, not gated: D11) |
| syn1m_b50k_s1 | build, b3: **1.0000** | 1,131.4 MiB | 1,133.7 MiB | **1.0021** | 2.1 | 1.018; 1.64-1.69 (flagged) |
| syn2m_b50k_s1 | build, b3: **1.0000** | 2,211.7 MiB | 2,215.4 MiB | **1.0016** | 3.6 | 0.998; 1.63 (flagged) |

Recorded: coauth_b100k_s1 and b200k_s1, store 1.0000, nsys peak 1.0018 and 1.0023, triad MiB
during the apply 5.0 and 6.6, between batches 1.66-1.80 (flagged). The same as M9a's review
reading to the MiB (the peaks 3,023.3 / 1,133.7 / 2,215.4 MiB there too).
