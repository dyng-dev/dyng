# Retrospective: M6b (hardening for 0.2.0)

Status: **complete** on the branch `m6b-hardening` (from `main` at `7d64c90`, the merge of pull
request #8 with M7). M6b is the hardening half of PLAN 11.3's M6, which the author's re-grouping
of 2026-10-02 (PLAN Appendix F, `GOVERNANCE.md`) moved from a 0.1.x release into **0.2.0**; the
other half, M6a (the CUDA plugin wheels), ran at the same time on `m6a-cuda-wheels`. Three steps,
each with a section below; then the review and its fixes, the milestone's acceptance table, the
verification from a fresh clone, the lessons, the state of 0.2.0 from M6b's side, the re-estimate
and the open items. The milestone's two ADRs were written as 0030 and 0032 and renumbered 0033 and
0034 in the review (M6a uses 0030-0032); the sections below use the new numbers.

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
| `9c4d731`, `43b390e`, `e430a2d` | the tutorial "Your first dynamic algorithm" (`docs/tutorials`), its reference solution `examples/tutorial_algorithms/my_bfs/` quoted by markers, `ci/scaffold_check.sh` building and testing it; both algorithm pages with the nine sections of PLAN 9.5; ADR 0033 |

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

All recorded in ADR 0033 (accepted under delegation):

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
   (ADR 0033, `python_gaps.md`): teaching material, not research algorithms.
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
| `b42063f`, `ad11a6a` | `docs/developer/robustness.md`, ADR 0034, the mutation record `parity/results/M6b-mutation.json` |

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

All recorded in ADR 0034 (accepted under delegation):

1. **`cpp/fuzz/`**, not the root `fuzz/` of the PLAN 4.2 sketch.
2. **No binary-reader target**: dynG has no binary reader yet (`.dgb` moved to M8; MOSP's binary
   graph cache is write-only); they come with M8 (0.3.0).
3. **The command line's text batches** are parsed in Python, out of libFuzzer's reach; a
   Hypothesis robustness test covers them.
4. **The mosp mutations were chosen here** (PLAN 8.4 lists none): they undo MOSP-OpenMP's
   combined-graph fix M-d.
5. **ADR number**: written as 0032, because both branches had written an ADR 0030 and 0031 was
   left for the renumbering; M6a then wrote its own 0032 too, so the review renumbered M6b's
   ADRs to 0033 (the tutorial algorithms) and 0034 (this one).
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
| `009a2f0` | `GOVERNANCE.md`: the approvals-log rows of the author's decisions of 2026-10-02 (the release re-grouping, GitHub Pages as O14, the plugin publishing environments and pending publishers), ADRs 0030 and 0032 (now 0033 and 0034), and GitHub Pages enabled (2026-10-06); `repository_settings.md` steps 9, 11 and 13 |
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
6. **ADR 0029 stays Proposed** (superseded in the review, which applied the author's decision of
   2026-10-06: `6fab3d9`). The milestone's criterion 5 says so. The orchestration notes
   (`pending_author_facts.md`, 2026-10-06) record that the author chose option A on 2026-10-06;
   applying it (ADR 0029 Accepted, an approvals-log row, the open-decisions table) is left to the
   orchestrator, who scoped this step (open item 1).
7. **No ADR for the sanitizer jobs**: they carry out a planned item (PLAN 7.3, 8.8) without
   changing a rule; this record and `robustness.md` hold the decisions.

## Review and fixes

Independent reviewers (lenses: the tutorials for a newcomer, robustness, the docs site) found 22
problems, each confirmed by a second reviewer; all are fixed on this branch (two pairs reported
the same problem under two lenses and share a row):

| # | Finding (severity) | Fix | Commits |
|---|---|---|---|
| 1 | `dynamic_bfs` (every backend) and the tutorial's `my_bfs` gave levels that were too low with `deletions_first = false`: a batch that inserts and deletes u -> v leaves no edge, but `applied.delta` lists the requested insertion and the seed offered `level[u] + 1` along it (graph 0 -> 1 -> 2: {0, 1, 1} instead of {0, 1, 2}) (blocker) | the seed offers only along an edge of G_{t+1} (it looks for v in u's out-row, as `find_roots` does for deletions), in `engine.hpp` and in `my_bfs`; hand cases on every backend (the hand tests of both tutorial algorithms now also build a CUDA executable, label gpu) and in the reference solution; the tutorial and the algorithm page explain the check. With the fix reverted, the new hand case fails on sequential, OpenMP and CUDA | `2da6215`, `a8a60ad` |
| 2 | The kit never applied the insertions first and never generated a batch that both inserts and deletes one edge; `applied_batch::delta` was documented as "the effective changes" (high) | a third property preset (`deletions_first = false`) in every check that loops over the presets, the batch mix `cancel` (mixed, plus absent edges inserted and deleted and existing edges deleted and re-inserted; its effect on the model follows the semantics) in C2, C3, C4 and C9, and C10 over the three semantics with `cancel` as its second batch; `apply_delta` / `applied_batch::delta` documented as the requested changes, pointing to `compute_structural_change` and `normalized` for the net change. With finding 1 reverted, C2 fails for `dynamic_bfs` on every backend and graph type; every other suite (sssp, cycle_count, mosp, triangle_delta) is green under the new preset and mix | `1e0a5eb`, `b512d38`, `37fb44a` |
| 3 | Tutorial step 4.2 added a second `reads_prepared_graph()` to a class that has one: following the prose did not compile; `scaffold_check.sh` copied the finished files, so CI could not see it (medium) | the step says to replace the scaffold's definition; `ci/tutorial_edits.py` makes every edit of section 4 in the scaffold's own files where the prose puts them (the quoted ranges from the reference's markers, the other edits as the prose words them; an anchor that is missing or repeated fails), and the scaffold check builds and tests that instead of copying files | `a8a60ad`, `62220e2` |
| 4 | `dynamic_bfs` and `triangle_delta` kept the scaffold line, so `new_algorithm.py <name> --remove` deleted them without `--force` (medium) | the line is gone; `regen.py --check` rejects it in a manifest whose maturity is not `experimental` | `dff1f4d` |
| 5 | The reference solution kept the placeholder's comments ("so update() recomputes", a TODO) (low) | the comments describe the incremental algorithm; section 4.7 tells the reader which doc comments to reword | `a8a60ad` |
| 6 | The tutorial presented C3 as a check of the OpenMP backend, and did not explain the skipped tests (low) | 4.9: C3 is trivially green until `openmp.cpp` has its own path (`dynamic_bfs` is the example); step 3: why C4 and C12 report Skipped | `a8a60ad` |
| 7, 17 | After the tutorial's clean-up the build tree kept `DYNG_ALGORITHMS=my_bfs`, so later builds silently covered sssp only (low, medium) | section 6 configures again with `-DDYNG_ALGORITHMS=all`; `new_algorithm.py --remove` says so; a configure whose `DYNG_ALGORITHMS` names an algorithm that does not exist now fails with the list of algorithms | `a8a60ad`, `dff1f4d`, `0ebbc7e` |
| 8, 19 | The tutorial implied that the conda environment provides a compiler and gave other minimum versions than the install page (low) | step 1 says a system compiler is needed with or without conda and points to the install page's table; the install page no longer says the compiler comes "from the dyng-dev environment" | `a8a60ad`, `70f2219` |
| 9 | The fuzzers' round trip ran inside `read_or_reject()`, so a copy that its own reader rejected counted as a quietly rejected input (medium) | `read_or_reject()` wraps the first read only and returns its result; `round_trip()` turns any exception in the writer or the second read into a failure; `fuzz.selftest.<target>` (every test build) replays the corpus with a planted writer bug (`DYNG_FUZZ_CORRUPT_WRITES`) and passes only if the failure is reported | `f9323cd` |
| 10 | `fuzz.yml` lacked the `vm.mmap_rnd_bits=28` step that the project says LLVM 18's ASan needs on the runners (medium) | the step, as in `sanitizers.yml`; `robustness.md` names both workflows | `83b0a38` |
| 11 | The third reader finding had no reproducer, and libFuzzer's default 2 GB limit could not catch it (low) | `cpp/fuzz/regressions/csr_triplet/oom_weight_columns` (400 MB before the fix) and `-malloc_limit_mb=256` for `fuzz.corpus.<target>`; with the reader fixes of `0b388f7` reverted the libFuzzer replay reports all three (3 GB, 4 GB, 400 MB) | `1b3d151` |
| 12 | `workflow_dispatch` accepted any duration, but the job times out after 120 minutes and a cancelled job skipped the upload of the reproducers (low) | at most 1000 s per target; the upload also runs when the job is cancelled | `83b0a38` |
| 13 | The Pages site (built from `main`) said it documents 0.1.0 and that the other algorithms are planned (high) | a banner on every page while `VERSION` is a development version (`docs/conf.py`, the latest release read from `CHANGELOG.md`); the status blocks of the README and the landing page name 0.1.0 as the latest release and what `main` adds; the release checklist names the deploy check | `c2d728e` |
| 14 | ADR numbers collided with M6a's twice, and the merge note was wrong (high) | M6b's ADRs are 0033 and 0034, with every reference; the merge note below is corrected | `6fab3d9` (the renames and the index and log rows, committed there by mistake), `f3f27bf` |
| 15 | The roadmap promised Python tutorials (and a DOI) in 0.2.0 that no milestone delivers (medium) | moved to 0.2.x, as the tutorials index says; the DOI depends on checkpoint A4 | `cdd6116` |
| 16 | `docs/api/cli.md` (and M6a's CLI help) still sent users to "the CUDA plugins of 0.1.x" (medium) | `cli.md` names the plugin wheels of 0.2.0; the CLI help text is M6a's file (merge item 1) | `22ab6ba` |
| 18 | ADR 0029 was shown as Proposed although the author decided on 2026-10-06 (medium) | Accepted (2026-10-06, by the author, option A): the ADR, the index, an approvals-log row (and out of the open decisions), the pages, header and test that called it proposed | `6fab3d9` |
| 20 | `new_algorithm.py` defaulted `--since` to 0.1, so a newcomer's algorithm appeared as released in 0.1 (low) | the default comes from `VERSION` (0.2 for 0.2.0.dev0; the next minor after a final release) | `dff1f4d` |
| 21 | User pages named milestones (M1a, M7) and plan sections, and sssp's page still said the MOSP totals "are gated when mosp is ported" (low) | the maturity lines name releases ("new in 0.2.0, not released yet"); the sentence points to mosp's gate section; the plan references are gone from the algorithm, API and tutorial pages; `documentation.md` states the rule | `22ab6ba`, `a8a60ad` |
| 22 | The release checklist named Read the Docs and no Pages check; `documentation.md` did not say where the site is published (low) | `release.md`: the deploy check after the final-release merge and the status texts; `documentation.md`: the Pages deployment | `c2d728e` |

Not fully done, with the reason:

- **Plan references outside the algorithm, API and tutorial pages.** The contributor how-tos
  (`how_to/add_an_algorithm.md`, `how_to/port_research_code.md`) and the design sketches
  (`docs/design/sketches/`) still cite PLAN sections: they are written for contributors, who are
  pointed to the plan's summary in `developer/plan.md`; rewording them is left to the next
  documentation pass.
- **`docs/getting_started/install.md`, `docs/getting_started/index.md` and
  `docs/api/python/index.md`** still say the CUDA plugins come in "0.1.x" on this branch: M6a
  rewrites those passages, so M6b leaves them to the merge (item 1 below) instead of creating a
  conflict.
- **`ci/tutorial_edits.py`** encodes the prose of section 4; a change of that prose needs the
  matching change of the script (its docstring says so), so the check proves the prose builds as
  the script reads it, not that a human reads it the same way. The tutorial was also followed by
  hand-written edits in a fresh clone (below).

### Results

- The minimal case of finding 1 fails before the fix on every backend (sequential, OpenMP, CUDA:
  `DynamicBfs.AnEdgeInsertedThenDeletedOffersNoLevel`) and passes after it.
- `ctest -R 'C[0-9]+_|DynamicBfs|TriangleDelta'` in `dev-cuda` (GPU 1): 541/541 with the new
  preset and mix (every algorithm, host and CUDA executables).
- `ci/fuzz.sh --time 30` (Clang 18): `ctest --preset fuzz` 6/6, every target 30 s without a
  finding; `ctest -L fuzz` in `dev`: 14/14 (six replays, five self-tests, three regressions).
- `ci/scaffold_check.sh`: the probes 80/80 and the tutorial's `my_bfs` made by
  `ci/tutorial_edits.py` 41/41.
- The tutorial followed in a fresh clone at `ef594cd` (step 3: 40/40 with C4 and C12 skipped per
  graph type; section 4's edits: 41/41; "break it on purpose": C2, C5, C8 and the hand test fail,
  with the documented seed line; the clean-up: a configure that still names `my_bfs` fails with
  the list of algorithms, `-DDYNG_ALGORITHMS=all` configures every algorithm, the tree is clean).

## Milestone acceptance

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | Tutorial algorithms `dynamic_bfs` (fixed point, Tier A, framework operators) and `triangle_delta` (aggregate delta, Tier A, ownership) created with `new_algorithm.py`, on sequential, OpenMP and CUDA, with oracles, `test_traits`, the kit C1-C12 on every backend, maturity `tutorial`; Python bindings (CLI not); the tutorial, followed in a fresh clone | step 1; ADR 0033; the verification below (the kit on every backend in `ci/check.sh` and `ci/gpu_local.sh`, the scaffold check with `my_bfs`) | met |
| 2 | Docs site on GitHub Pages: `docs.yml` builds on pull requests and deploys `main` (SHA-pinned Pages actions, least privilege on the deploy job, environment `github-pages`); the URL where appropriate; `repository_settings.md`; actionlint / zizmor clean | step 2; Pages enabled by the author on 2026-10-06 (step 3 records it); the `pyproject.toml` URL is the orchestrator's (M6a owns the file; open item 3) | met |
| 3 | libFuzzer targets for every text reader and the binary readers (Clang, CMake option / preset), a hosted job, any finding fixed with a regression test; `mutate.py` on the cycle_count and mosp goldens; the full Hypothesis profile | step 2 (ADR 0034): six targets, three findings fixed; 11/11 mutations detected; profile `full` and `property.yml`. No binary-reader target because dynG has no binary reader yet (ADR 0034 item 2) | met (binary readers with M8) |
| 4 | Sanitizer CI: hosted ASan + UBSan and TSan jobs on pull requests (CPU tests), green, not required; compute-sanitizer stays in `ci/gpu_local.sh` | step 3: `sanitizers.yml` (three jobs), green locally through `ci/sanitizers.sh` (the hosted runs follow the first push, open item 2) | met locally |
| 5 | The author's decisions of 2026-10-02 in `GOVERNANCE.md`; `roadmap.md` and `plan.md` in the new grouping; ADR 0029 Proposed | step 3 (`009a2f0`, `b4d7cc4`); ADR 0029 then accepted in the review, applying the author's later decision of 2026-10-06 (`6fab3d9`) | met |
| 6 | A fresh clone passes `ci/check.sh --parity`, `ci/gpu_local.sh`, `ci/docs.sh`, `ci/python.sh` and the portability pre-checks; `regen.py --check`; CHANGELOG; this retrospective | "Verification" below | met |

## Verification after the review fixes (fresh clones of `m6b-hardening`)

Two clones of the local repository at `ef594cd` (`git checkout m6b-hardening`) in
`$DYNG_SCRATCH/review`, run at the same time, every heavy step under the shared perf lock:

| Check | Result |
|---|---|
| `ci/check.sh --parity` | all checks passed in 26 min: clang-format; `cpu-only` 834/834 and `dev` 863/863 (`ctest -L cpu`, the new preset, the `cancel` mix and the fuzz self-tests included); clang-tidy (naming); reuse; provenance; regen; the harness and CI-script tests 139 passed; `ci/python.sh` (stubs, mypy strict, pytest 455 passed, 4 skipped for torch / cupy / pandas); griffe; the docs (Sphinx -W, the link check); pre-commit (actionlint, zizmor); the scaffold check (probes 80/80 and 3/3, `my_bfs` 41/41); the parity preset 4/4 |
| `ci/gpu_local.sh` (dev-cuda, GPU 1, RTX A5000, driver 590.48.01, CUDA 13.1) | all steps passed in 66 min: `ctest -L gpu` 464/464 (the CUDA hand cases of both tutorial algorithms included); `ctest -L cpu` in the CUDA build 863/863; the sssp golden corpus ALL EQUAL (495 cases x 2 configurations); the cycle_count CUDA corpus ALL EQUAL (48 replays); compute-sanitizer memcheck, synccheck (sssp, mosp, cycle_count), racecheck; clang-tidy on the CUDA branches |
| After `ef594cd` | `62220e2` (the scaffold check through `ci/tutorial_edits.py`) and this record: `ci/scaffold_check.sh` 80/80, 3/3 and 41/41, the CI-script tests, `ci/docs.sh` and `regen.py --check` on the final commit (below) |
| `ci/sanitizers.sh` (working tree, `ctest -L cpu`) | `asan` (GCC 12) 834/834, `tsan` (GCC 12) 741/741, no sanitizer report; `tsan-openmp` not re-run (the review changed no OpenMP code; it needs the local Clang wrapper of step 3) |
| Portability pre-checks (working tree after the C++ changes) | the Clang 18 syntax pass over a `dev` + `DYNG_BUILD_PYTHON=ON` database: 226 files, 0 failures; the fortified (`-D_FORTIFY_SOURCE=3`) `cpu-only` build with the Python module: clean |
| `ci/fuzz.sh --time 30` (Clang 18) | `ctest --preset fuzz` 6/6 (256 MB limit), every target 30 s without a finding |

## Verification before the review (fresh clone of `m6b-hardening`)

All runs in `$DYNG_SCRATCH/runs/m6b-san/clone` (a clone of the local repository, `git checkout
m6b-hardening`), every heavy step under the shared perf lock, while other agents' builds ran:

| Check | Result |
|---|---|
| `ci/check.sh --parity` at `f0c084a` | 24 min: clang-format; `cpu-only` 826/826 and `dev` 855/855 (`ctest -L cpu`); clang-tidy (naming rules); reuse; provenance; regen (`scripts/regen.py --check`); the harness tests (132 passed); `ci/python.sh` (stubs, mypy strict, pytest 455 passed, 4 skipped for torch / cupy / pandas); the griffe API check; pre-commit (actionlint and zizmor included); the parity preset (`ctest -L parity` 4/4). Two steps failed for one reason, this record not yet committed: `docs` (Sphinx -W: plan.md's links to `retrospectives/M6b`) and `scaffold` (whose last part builds the docs of the copy; its probes 80/80, 3/3, and the tutorial's `my_bfs` 40/40 had passed). Both are re-run below |
| `ci/check.sh` steps `scaffold` and `docs` after this record | `DYNG_CHECK_ONLY="scaffold docs" ci/check.sh` at `3f6354c`: all checks passed (the scaffold probes 80/80 and 3/3, the tutorial's `my_bfs` 40/40, the copy's docs; `ci/docs.sh`: Doxygen coverage of 138 compounds, 7 snippets, the API baseline of 719 declarations unchanged by this step, Sphinx -W, the link check) |
| `ci/gpu_local.sh` (dev-cuda, GPU 1) | all steps passed in 62 min (RTX A5000, driver 590.48.01, CUDA 13.1; started at `f0c084a`, the summary names `3f6354c`, which only added this record): the build; `ctest -L gpu` 455/455 (the tutorials' CUDA kits included); `ctest -L cpu` in the CUDA build 855/855; the sssp golden corpus with `--configs cuda,cuda-operators` ALL EQUAL (495 cases x 2); the cycle_count CUDA corpus ALL EQUAL (24 cases x 2); compute-sanitizer memcheck on every gpu executable, synccheck (sssp, mosp, cycle_count), racecheck (cycle_count): passed; clang-tidy on the CUDA branches |
| `ci/sanitizers.sh` | at `3f6354c`, `ctest -L cpu -j 8`, no sanitizer report: `asan` (GCC 12) 826/826 (115 s), `tsan` (GCC 12) 733/733 (191 s), `tsan-openmp` (the conda Clang 18 through a local wrapper, Archer) 826/826 (209 s); 10.5 min in all with the builds |
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
- **A kit tests only the batches its generator can make.** The conformance kit was green on
  `dynamic_bfs` and on the tutorial's `my_bfs` while both gave wrong levels for an edge inserted and
  deleted in one batch, because the generator never made such a batch and no preset applied the
  insertions first; a documented name ("the effective changes") hid that the delta lists requested
  operations. The review added the batches and the preset; a new batch-semantics switch needs a
  preset of its own in the kit.
- **A check of a tutorial must follow its prose.** Copying the finished reference files proved
  that the reference works, not that the steps do; the scaffold check now makes the edits where
  the prose puts them.
- **The local toolchain differs from the hosted one.** The conda Clang needs `--gcc-install-dir`,
  `--sysroot=/` and an rpath to use the system libstdc++ and its own libomp (a wrapper script in
  the scratch directory); the hosted jobs install Ubuntu's `clang-18`, `libomp-18-dev` and
  `libclang-rt-18-dev` and need none of it.

## State of 0.2.0 from M6b's side

Ready, with these steps before the release:

1. **Merge** `m6b-hardening` and `m6a-cuda-wheels` into `main` (the orchestrator). The ADR
   numbers no longer collide (M6a: 0030-0032; M6b: 0033, 0034). Expected conflicts, each to be
   resolved by keeping both sides:
   - `GOVERNANCE.md`: both add approvals-log rows after the 2026-10-02 rows, and in "Open
     decisions" M6b removed the ADR 0029 row (accepted) where M6a added the plugin-licence row
     after it: keep M6a's new row, drop the ADR 0029 row.
   - `docs/adr/README.md`: M6b changed the ADR 0029 row and added 0033 and 0034 where M6a added
     0030-0032: keep the 0029 row as accepted, then 0030-0034 in order.
   - `docs/developer/release.md`: M6b changed the role-table row "Zenodo, conda-forge ..." (no Read
     the Docs) next to M6a's changed approval row, and added a paragraph under "Checklist": keep
     both changes.
   - `CHANGELOG.md` (`[Unreleased]`) and `README.md`: both added entries and status texts in
     different places; keep both. `python/CMakeLists.txt` and `python/dyng/__init__.py`: the two
     tutorial modules join M6a's plugin module list.
   - After the merge, the texts that still promise the CUDA plugins in "0.1.x" should say 0.2.0:
     `python/dyng/cli/_common.py` (the `--backend` help, M6a's file; `docs/api/cli.md` is fixed
     here) and whatever M6a's rewrite of `docs/getting_started/install.md`,
     `docs/getting_started/index.md`, `docs/api/python/index.md`, `python/dyng/config.py`,
     `python/dyng/resources.py` and `docs/developer/wheels.md` leaves behind
     (`git grep -n '0\.1\.x'`).
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

1. **ADR 0029** is done: accepted in the review with the author's decision of 2026-10-06
   (`6fab3d9`).
2. **Hosted runs not yet seen** (never pushed): `sanitizers.yml` (in particular the Archer path of
   Ubuntu's `libomp5-18`, found by the workflow under `/usr/lib/llvm-18`, and the `mmap_rnd_bits`
   step), `fuzz.yml`, `property.yml` and the Pages deployment. Each passed actionlint, zizmor and
   the meta check, and its script ran locally.
3. **`pyproject.toml`** (M6a's): add `Documentation = "https://dyng-dev.github.io/dyng/"` under
   `[project.urls]`; optionally exclude `cpp/fuzz/` from the sdist (about 20 KB).
4. **The ADR number collision** is resolved (M6b's ADRs are 0033 and 0034); the other merge items
   are in "State of 0.2.0" item 1.
5. **Remaining "0.1.x" mentions of the CUDA plugins**: see "State of 0.2.0" item 1 (the CLI
   help and the pages M6a rewrites); `docs/api/cli.md` is fixed here.
6. **Tutorial limitations** (ADR 0033): `dynamic_bfs` copies all levels once per update to compute
   `stats.affected` (O(n)) and has no performance gate.
