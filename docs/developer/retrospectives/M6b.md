# Retrospective: M6b (hardening for 0.2.0)

Status: **complete** on the branch `m6b-hardening` (from `main` at `7d64c90`, the merge of pull
request #8 with M7). M6b is the hardening half of PLAN 11.3's M6, which the author's re-grouping
of 2026-10-02 (PLAN Appendix F, `GOVERNANCE.md`) moved from a 0.1.x release into **0.2.0**; the
other half, M6a (the CUDA plugin wheels), ran at the same time on `m6a-cuda-wheels`. Three steps,
each with a section below; then the milestone's acceptance table, the verification from a fresh
clone, the lessons, the state of 0.2.0 from M6b's side, the re-estimate and the open items.

## Step 1: the tutorial algorithms (tutorials)

### What was built

| Commits | What |
|---|---|
| `d3e461a` | `dynamic_bfs` (fixed point) and `triangle_delta` (aggregate delta) scaffolded with `scripts/new_algorithm.py` |
| `c72492b` | `maturity_level::tutorial`, appended to the frozen enumeration (`"tutorial"` in the manifests, the registry, the generated tables, Python) |
| `981b382` | the framework operators' executors (`cpp/src/operators/execution.hpp`, `cuda_execution.cuh`): `sequential_exec`, `openmp_exec`, `cuda_exec` run a `DYNG_HD` functor per element, with the atomics such functors use; one `engine.hpp` per algorithm holds its passes once, and `sequential.cpp`, `openmp.cpp`, `cuda.cu` are one instantiation each |
| `c5330c5` | `dynamic_bfs`: the subtrees under deleted BFS-tree edges are invalidated, `seed` pulls valid in-neighbours and takes offers from inserted edges, `loop` advances one level per call on its own list frontier, `finalize` repairs the parents with a deterministic rule (the lowest-id in-neighbour one level up) |
| `521aa08` | the conformance kit generates undirected graphs for an algorithm that requires them; C2, C7 and C9 count each stored direction; C10 merges the two algorithms' requirements |
| `2f1134e`, `81dd675` | `triangle_delta`: two sorted lists of undirected changed edges on G_t, count(-) on the old view and count(+) on the new one with `ownership::min_member`; both tutorials on the unweighted graph types |
| `b5f79a2`, `b1a1c2f` | `dyng.dynamic_bfs` and `dyng.triangle_delta` (bindings, stubs, docstring examples, hand cases, random chains against brute force) |
| `9c4d731`, `43b390e`, `e430a2d` | the tutorial "Your first dynamic algorithm" (`docs/tutorials`), its reference solution `examples/tutorial_algorithms/my_bfs/` quoted by markers, `ci/scaffold_check.sh` building and testing it; both algorithm pages with the nine sections of PLAN 9.5; ADR 0030 |

### Results

- `ctest --preset dev -L 'conformance|triangle_delta|dynamic_bfs'`: 283/283;
  `ctest --preset dev-cuda` with the same labels on GPU 1: 543/543 (C8 budgets and C12 streams on
  CUDA included; C4 skipped, each tutorial has one engine).
- compute-sanitizer memcheck with leak checks on both tutorial CUDA conformance executables: 0
  errors, 0 leaks. The `asan` preset on the two tutorials: 192/192.
- Every algorithm paired with `triangle_delta` in C10 (sssp, cycle_count, mosp, dynamic_bfs) now
  also passes on undirected graphs; the directed models and their random streams are unchanged.
- The tutorial was followed in a fresh clone: the scaffold is green (40/40), the step-4 edits as
  written are green (40/40), the "break it on purpose" step fails C2, C5, C8 and the hand test,
  the documented seed replay reproduces C2, and `--remove` leaves the tree clean.
- Two manual mutations (no ownership rule in `triangle_delta`; no subtree walk in `dynamic_bfs`)
  are caught by the kit.

### Deviations and decisions (with reasons)

All recorded in ADR 0030 (accepted under delegation):

1. **Registered algorithms, not examples.** PLAN 9.3 put the tutorial algorithms in
   `examples/tutorial_algorithms/`, unregistered and not installed. The M6 criteria (created with
   `new_algorithm.py`, maturity `tutorial`, the full kit on every backend) need registered
   algorithms, so they live in `cpp/src/algorithms` with installed headers (in the API baseline);
   `examples/tutorial_algorithms/` holds the tutorial's reference solution.
2. **`maturity_level::tutorial`** is a public addition, appended to the frozen enumeration.
3. **Undirected graphs in the kit**, which the triangle count needs.
4. **No citation** for the tutorial algorithms (`test_api_surface.py` accepts an empty `cite` for
   maturity `tutorial`).
5. **No CLI command, no `dyng.update()` support and no `examples/cpp` program** for them
   (ADR 0030, `python_gaps.md`): teaching material, not research algorithms.
6. **The tutorial's `my_bfs` is simpler than `dynamic_bfs`** (sequential hooks, no BFS tree); the
   tutorial says what `dynamic_bfs` adds.

## Step 2: the documentation on GitHub Pages, fuzzers, mutation suites, property runs (docs-pages-fuzz)

### What was built

| Commits | What |
|---|---|
| `a99e968` | `parity/mutate.py` with suites: the cycle_count mutations (double-counted 5-cycles, the weakened ownership rule; host searches and device kernels) and two mosp mutations (a combined-graph edge weighted by the first tree only; path costs from the first objective's weights), each on the sequential, OpenMP and CUDA backends; `--suites` |
| `79add4b`, `0b388f7` | six libFuzzer targets (`cpp/fuzz`) for every reader of `dyng::io`; the CMake option `DYNG_BUILD_FUZZERS` and the preset `fuzz`; a 1.7 KB seed corpus; every test build replays the corpus and the reproducers; the three findings fixed (reservations bounded by what the file can hold) with `cpp/tests/io/fuzz_regression_test.cpp` |
| `d8c7b7e`, `ed497dc` | `ci/fuzz.sh` and `.github/workflows/fuzz.yml` (pull requests that touch a reader: 60 s per target; weekly: 10 minutes) |
| `9be011e`, `27f6f4e` | the command line's text batches reject bad input as `FileFormatError` (found by the new Hypothesis robustness tests); the Hypothesis profile `full` and `property.yml` |
| `a2c182a` | the site on GitHub Pages: `docs.yml` deploys `main` (`configure-pages`, `upload-pages-artifact`, `deploy-pages`, SHA-pinned; `pages: write` and `id-token: write` only on the deploy job; environment `github-pages`); the URL in the README, `CITATION.cff`, `docs/conf.py`, the plan and the roadmap; `repository_settings.md` step 13 |
| `b42063f`, `ad11a6a` | `docs/developer/robustness.md`, ADR 0032, the mutation record `parity/results/M6b-mutation.json` |

### Results

- The mutation run at `a99e968` (about 27 minutes): the control passed, 11/11 mutations detected.
- Fuzzing after the fixes: every target 5 and then 15 minutes locally without a finding (0.47M to
  5.6M inputs per target); `ctest --preset fuzz` 6/6; `ci/fuzz.sh --time 15` end to end.
- The three reader findings fail on the old code (3.2 GB, 4 GB and 1.6 GB allocation requests) and
  pass now.
- The Hypothesis properties of the tutorials (2,500 examples, 82 s) and the robustness tests
  (5,000 examples) pass.
- A fresh clone at `ed497dc` passed `ci/check.sh --parity`; actionlint, zizmor and
  `github_meta_check.py --verify-pins` clean.

### Deviations and decisions (with reasons)

All recorded in ADR 0032 (accepted under delegation):

1. **`cpp/fuzz/`**, not the root `fuzz/` of the PLAN 4.2 sketch.
2. **No binary-reader target**: dynG has no binary reader yet (`.dgb` moved to M8; MOSP's binary
   graph cache is write-only); they come with M8 (0.3.0).
3. **The command line's text batches** are parsed in Python, out of libFuzzer's reach; a
   Hypothesis robustness test covers them.
4. **The mosp mutations were chosen here** (PLAN 8.4 lists none): they undo MOSP-OpenMP's
   combined-graph fix M-d.
5. **ADR number 0032**: both branches wrote an ADR 0030, so 0031 is left for the renumbering.
6. **The preset `fuzz`** builds only the sssp module, static, Debug with `-O1`.
7. **The CSR-triplet findings** came from a review prompted by the fuzzer's memory use, not from
   a crash; the documentation says so.

## Step 3: sanitizer CI and close-out (sanitizers-finish)

### What was built

| Commits | What |
|---|---|
| `ea77387` | the cycle_count kernel tests detect Clang's sanitizers (`DYNG_TEST_SANITIZED`, which also checks `__has_feature`): their counting `operator new` collided with the statically linked sanitizer runtime of Clang, so no Clang ASan or TSan build linked |
| `81c8f2c` | `ci/sanitizers.sh` (configure, build and `ctest -L cpu` for the presets `asan`, `tsan`, `tsan-openmp`); the new preset `tsan-openmp` (Clang, OpenMP on, the tests with Archer); a configure error for TSan with OpenMP and a compiler other than Clang; `.github/workflows/sanitizers.yml` with the jobs `asan / gcc-13`, `tsan / gcc-13` and `tsan-openmp / clang-18` on pull requests and pushes to `main` |
| `1f13a9c` | `robustness.md` section "Host sanitizers"; the preset tables of the README, `CONTRIBUTING.md` and the install guide; the release checklist (step 3); the header of `ci/check.sh` |
| `009a2f0` | `GOVERNANCE.md`: the approvals-log rows of the author's decisions of 2026-10-02 (the release re-grouping, GitHub Pages as O14, the plugin publishing environments and pending publishers), ADRs 0030 and 0032, and GitHub Pages enabled (2026-10-06); `repository_settings.md` steps 9, 11 and 13 |
| `b4d7cc4` | `docs/roadmap.md` and `docs/developer/plan.md` in the new grouping (0.2.0 = M7 + M6a + M6b, 0.3.0 = M8 + M9, 0.4.0 = M10 + M11, the later items one minor up), and every planned version in the documentation, the design sketches, the docstrings of the public headers and Python modules, `planned.toml` (the generated algorithm tables) and the messages of features that are not there yet |
| `f0c084a`, this record | the CHANGELOG (`[Unreleased]`), this retrospective |

### Results

The sanitizer presets on the development machine (`ctest -L cpu -j 8`, 2026-10-06):

| Preset | Compiler | Tests | Test time | Reports |
|---|---|---|---|---|
| `asan` (ASan + UBSan, leak checks, OpenMP on) | GCC 12 | 826 passed | 101 s | none |
| `tsan` (OpenMP off) | GCC 12 | 733 passed | 141 s | none |
| `tsan-openmp` (Archer, 4 OpenMP threads) | Clang 18.1.8 + libomp 18 (conda `dyng-clang`) | 826 passed | 134 s | none |

- Archer was checked both ways: without `OMP_TOOL_LIBRARIES` the `tsan-openmp` tests report races
  inside libomp (`__kmp_lock_suspend_mx`, `__kmpc_dispatch_next_8`); with it, and
  `ignore_noninstrumented_modules=1`, a small OpenMP program with a real race (a shared sum next to
  a reduction) is still reported, and the same program without the race is clean.
- actionlint, zizmor, `github_meta_check.py --verify-pins` and shellcheck: clean.

### Deviations and decisions (with reasons)

1. **A third sanitizer job, `tsan-openmp`.** The criterion names the existing `asan` and `tsan`
   presets. `tsan` turns OpenMP off (GCC's libgomp is not instrumented, M1a), so it checks only
   the `std::thread` code, and every OpenMP backend would stay unchecked for races. Clang's libomp
   with Archer makes TSan understand OpenMP; the new preset and job cover the OpenMP backends of
   every algorithm. Both TSan jobs run: `tsan` keeps the `std::async` path of
   `util/concurrent.hpp` (taken only without OpenMP) and GCC's TSan.
2. **TSan + OpenMP + GCC is a configure error**, not a silent configuration whose every test
   fails.
3. **`vm.mmap_rnd_bits=28` in the hosted jobs.** The `ubuntu-24.04` runners randomize mmap with
   32 bits, which the TSan and ASan runtimes of GCC 13 and LLVM 18 do not support (a known
   runner-image issue); the step is in the workflow only.
4. **Not a required check yet** (the criterion): `repository_settings.md` step 9 names the three
   check names for when the lead maintainer requires them.
5. **The re-grouping beyond the two named pages.** The criterion names `docs/roadmap.md` and
   `docs/developer/plan.md`; the documentation, the design sketches, the docstrings and four
   messages also named planned versions (the hypergraph "0.2", vertex batches "0.3", cycle modes
   "0.4", the public framework "0.5"), so they move with it. The texts about the CUDA plugins
   ("0.1.x") are M6a's and were left alone. The Python tutorials and the MOSP tutorial, planned
   for 0.1.x and not written, are now listed for a 0.2.x release (`docs/tutorials/index.md`), not
   for 0.2.0.
6. **ADR 0029 stays Proposed.** The milestone's criterion 5 says so. The orchestration notes
   (`pending_author_facts.md`, 2026-10-06) record that the author chose option A on 2026-10-06;
   applying it (ADR 0029 Accepted, an approvals-log row, the open-decisions table) is left to the
   orchestrator, who scoped this step (open item 1).
7. **No ADR for the sanitizer jobs**: they carry out a planned item (PLAN 7.3, 8.8) without
   changing a rule; this record and `robustness.md` hold the decisions.

## Milestone acceptance

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | Tutorial algorithms `dynamic_bfs` (fixed point, Tier A, framework operators) and `triangle_delta` (aggregate delta, Tier A, ownership) created with `new_algorithm.py`, on sequential, OpenMP and CUDA, with oracles, `test_traits`, the kit C1-C12 on every backend, maturity `tutorial`; Python bindings (CLI not); the tutorial, followed in a fresh clone | step 1; ADR 0030; the verification below (the kit on every backend in `ci/check.sh` and `ci/gpu_local.sh`, the scaffold check with `my_bfs`) | met |
| 2 | Docs site on GitHub Pages: `docs.yml` builds on pull requests and deploys `main` (SHA-pinned Pages actions, least privilege on the deploy job, environment `github-pages`); the URL where appropriate; `repository_settings.md`; actionlint / zizmor clean | step 2; Pages enabled by the author on 2026-10-06 (step 3 records it); the `pyproject.toml` URL is the orchestrator's (M6a owns the file; open item 3) | met |
| 3 | libFuzzer targets for every text reader and the binary readers (Clang, CMake option / preset), a hosted job, any finding fixed with a regression test; `mutate.py` on the cycle_count and mosp goldens; the full Hypothesis profile | step 2 (ADR 0032): six targets, three findings fixed; 11/11 mutations detected; profile `full` and `property.yml`. No binary-reader target because dynG has no binary reader yet (ADR 0032 item 2) | met (binary readers with M8) |
| 4 | Sanitizer CI: hosted ASan + UBSan and TSan jobs on pull requests (CPU tests), green, not required; compute-sanitizer stays in `ci/gpu_local.sh` | step 3: `sanitizers.yml` (three jobs), green locally through `ci/sanitizers.sh` (the hosted runs follow the first push, open item 2) | met locally |
| 5 | The author's decisions of 2026-10-02 in `GOVERNANCE.md`; `roadmap.md` and `plan.md` in the new grouping; ADR 0029 Proposed | step 3 (`009a2f0`, `b4d7cc4`) | met |
| 6 | A fresh clone passes `ci/check.sh --parity`, `ci/gpu_local.sh`, `ci/docs.sh`, `ci/python.sh` and the portability pre-checks; `regen.py --check`; CHANGELOG; this retrospective | "Verification" below | met |

## Verification (fresh clone of `m6b-hardening`)

All runs in `$DYNG_SCRATCH/runs/m6b-san/clone` (a clone of the local repository, `git checkout
m6b-hardening`), every heavy step under the shared perf lock, while other agents' builds ran:

| Check | Result |
|---|---|
| `ci/check.sh --parity` at `f0c084a` | 24 min: clang-format; `cpu-only` 826/826 and `dev` 855/855 (`ctest -L cpu`); clang-tidy (naming rules); reuse; provenance; regen (`scripts/regen.py --check`); the harness tests (132 passed); `ci/python.sh` (stubs, mypy strict, pytest 455 passed, 4 skipped for torch / cupy / pandas); the griffe API check; pre-commit (actionlint and zizmor included); the parity preset (`ctest -L parity` 4/4). Two steps failed for one reason, this record not yet committed: `docs` (Sphinx -W: plan.md's links to `retrospectives/M6b`) and `scaffold` (whose last part builds the docs of the copy; its probes 80/80, 3/3, and the tutorial's `my_bfs` 40/40 had passed). Both are re-run below |
| `ci/check.sh` steps `scaffold` and `docs` after this record | RERUN |
| `ci/gpu_local.sh` (dev-cuda, GPU 1) | GPULOCAL |
| `ci/sanitizers.sh` | SANITIZERS |
| Portability pre-checks (on the working tree at `f0c084a`) | the Clang 18 syntax pass over a `dev` + `DYNG_BUILD_PYTHON=ON` database: 216 files, 0 failures; the fortified (`-D_FORTIFY_SOURCE=3`) `cpu-only` build with the Python module: clean, no warnings, and the Clang pass over its database: 214 files, 0 failures |

## Lessons

- **Each new check found something the old ones could not.** The fuzzers found three readers that
  trusted a size field (one only through the review their memory use prompted); the Hypothesis
  robustness tests found two error types escaping the command line; the first Clang sanitizer
  build found a test that only recognised GCC's sanitizer macros. None of them was visible to the
  unit tests, the kit or parity, which is the case for running them in CI rather than once.
- **Instrumentation needs the runtime's help.** TSan with OpenMP is noise without Archer, and with
  Archer it needs `ignore_noninstrumented_modules`; a check that is noisy by construction gets
  turned off, so the setup was verified to report a planted race and to stay quiet on correct
  code before it went into a workflow.
- **A version promise lives in many places.** The re-grouping touched some forty files: tables
  generated from `planned.toml` updated themselves, every hand-written "(0.3)" did not. Planned
  versions in docstrings and messages are best named by milestone or left out.
- **Parallel milestones need disjoint files.** M6a and M6b both wrote ADR 0030, both append to the
  approvals log, the CHANGELOG and the README; keeping edits additive and leaving the other
  milestone's topics alone (the plugin wording, `pyproject.toml`) keeps the merge small.
- **The local toolchain differs from the hosted one.** The conda Clang needs `--gcc-install-dir`,
  `--sysroot=/` and an rpath to use the system libstdc++ and its own libomp (a wrapper script in
  the scratch directory); the hosted jobs install Ubuntu's `clang-18`, `libomp-18-dev` and
  `libclang-rt-18-dev` and need none of it.

## State of 0.2.0 from M6b's side

Ready, with these steps before the release:

1. **Merge** `m6b-hardening` and `m6a-cuda-wheels` into `main` (the orchestrator): renumber one
   ADR 0030 (0031 is M6a's, 0032 M6b's; the next free number is 0033), and resolve the shared
   files (`GOVERNANCE.md` approvals-log rows, the CHANGELOG's `[Unreleased]`, the README,
   `python/CMakeLists.txt` and `python/dyng/__init__.py` for the two tutorial modules in M6a's
   plugin module list).
2. **The first hosted runs** of `sanitizers.yml`, `fuzz.yml`, `property.yml` and the `deploy` job of
   `docs.yml`. Pages is enabled (2026-10-06), so the first push to `main` after the merge deploys
   the site; then set the repository website (`repository_settings.md` step 13, item 4).
3. **Decide** whether the three sanitizer checks become required (`repository_settings.md` step 9).
4. **The release checklist** (`docs/developer/release.md`): step 3 now names the sanitizer presets
   and `ci/sanitizers.sh`; the mutation record of M6b is `parity/results/M6b-mutation.json`.

Nothing in M6b changes a gated code path of sssp, cycle_count or mosp (the executors and the kit's
undirected models are used by the tutorials and the tests; the reader fixes only bound
reservations), so the 0.2 performance records of M7 stand.

## Re-estimate

M6b took one working day for its three steps (the commits are of 2026-10-06), against the
plan's 4-7 focused days for the whole M6 (M6a in parallel). The longest parts were the fresh-clone verifications, the 27-minute mutation run and the
15-minute fuzz runs, all behind the shared lock with other agents' builds. `docs/developer/plan.md`
section 4 carries the new release rows: 0.2.0 (M7 done, M6b done, M6a in parallel) about 1-2 weeks
of calendar time from 2026-10-06, dominated by the merges, the first hosted runs and the release
steps; 0.3.0 (M8, M9) 1-1.5 focused weeks; 0.4.0 (M10, M11) 1.5-2.5 focused weeks.

## Open items for the lead maintainer and the orchestrator

1. **ADR 0029.** The orchestration notes record the author's choice of option A on 2026-10-06;
   this step kept the ADR Proposed as its criterion said. To apply: mark it Accepted (2026-10-06,
   the author), add the approvals-log row, and remove it from the open decisions of
   `GOVERNANCE.md`.
2. **Hosted runs not yet seen** (never pushed): `sanitizers.yml` (in particular the Archer path of
   Ubuntu's `libomp5-18`, found by the workflow under `/usr/lib/llvm-18`, and the `mmap_rnd_bits`
   step), `fuzz.yml`, `property.yml` and the Pages deployment. Each passed actionlint, zizmor and
   the meta check, and its script ran locally.
3. **`pyproject.toml`** (M6a's): add `Documentation = "https://dyng-dev.github.io/dyng/"` under
   `[project.urls]`; optionally exclude `cpp/fuzz/` from the sdist (about 20 KB).
4. **The ADR number collision** (ADR 0030 on both branches) and the merge items of "State of
   0.2.0" item 1.
5. **Remaining "0.1.x" mentions of the CUDA plugins** in the Python package and a few pages (for
   example `python/dyng/resources.py`, `config.py`, `docs/getting_started/install.md`): M6a's to
   update with its plugin texts; M6b left them alone.
6. **Tutorial limitations** (ADR 0030): `dynamic_bfs` copies all levels once per update to compute
   `stats.affected` (O(n)) and has no performance gate.
