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
`86e9224` (Python), `db25f77` (the CLI), and the records commit. No library code of the
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
| Clang 18 `-fsyntax-only` (a dev database with the Python bindings) | 444 files, **0 failures** |
| cpu-only with `-D_FORTIFY_SOURCE=3` (with the bindings) | built without warnings |
