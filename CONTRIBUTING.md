# Contributing to dynG

Thank you for your interest in dynG! This guide takes you from a fresh clone to a merged pull
request. Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md). Questions go
to [GitHub Discussions](https://github.com/dyng-dev/dyng/discussions) ([SUPPORT.md](SUPPORT.md));
security problems are reported privately ([SECURITY.md](SECURITY.md)).

dynG is **alpha** (0.1.0, the first release): the stable algorithms follow SemVer from 0.1.0, but
the framework, file formats and build options still change. For anything larger
than a small fix, please open an issue (or a discussion) first, so that we can agree on the
approach before you invest time in it.

## Contents

1. [Where to start](#where-to-start)
2. [Set up a development environment](#set-up-a-development-environment)
3. [Build](#build)
4. [Test](#test)
5. [Parity with the original research codes](#parity-with-the-original-research-codes)
6. [Performance and benchmarks](#performance-and-benchmarks)
7. [Style](#style)
8. [Documentation and Doxygen](#documentation-and-doxygen)
9. [Commits](#commits)
10. [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
11. [Pull requests and review](#pull-requests-and-review)
12. [Adding an algorithm or porting research code](#adding-an-algorithm-or-porting-research-code)

## Where to start

- **Good first issues.** Issues labelled
  [`good first issue`](https://github.com/dyng-dev/dyng/labels/good%20first%20issue) are chosen
  to be doable without knowing the whole code base. A second label says what kind of work it is:
  `good-first-issue:docs` (documentation), `good-first-issue:reader` (a file format reader),
  `good-first-issue:backend` (an OpenMP backend for a sequential-only algorithm) and, from
  0.3, `good-first-issue:op` (a new hypergraph pattern). Issues labelled
  [`help wanted`](https://github.com/dyng-dev/dyng/labels/help%20wanted) are larger tasks
  where help is welcome. The full label list is in
  [docs/developer/labels.md](docs/developer/labels.md).
- **Bug reports** with a minimal reproducer are very valuable, especially wrong results
  (please use the issue forms).
- **Documentation**: typos, unclear explanations and missing examples; small documentation
  fixes need no issue first.
- **A new algorithm**: start with the *New algorithm* issue form (see
  [below](#adding-an-algorithm-or-porting-research-code)); the roadmap
  ([docs/roadmap.md](docs/roadmap.md)) lists algorithms that are open to contributors.

Comment on an issue to say that you are working on it, so that nobody duplicates your work.

## Set up a development environment

You need Linux (x86-64), git, and [conda](https://conda-forge.org/download/) (Miniforge or
Anaconda). **No GPU is needed** for the CPU backends (sequential and OpenMP); the CUDA backend
additionally needs the CUDA toolkit (>= 12.4) and an NVIDIA GPU (sm_75 or newer) to run. The
host C++ compiler is your system's (GCC >= 11 or Clang >= 15). From clone to a green build
should take less than 15 minutes.

```bash
git clone https://github.com/<your-user>/dyng.git   # your fork (see "Pull requests")
cd dyng
git remote add upstream https://github.com/dyng-dev/dyng.git   # the main repository
conda env create -f environment.yml   # once: the dyng-dev env (CMake, Ninja, clang-format,
                                      # clang-tidy, Doxygen, pre-commit, reuse, pytest, ...)
source scripts/dev_env.sh             # every shell: activates dyng-dev (and nvcc if installed)
pre-commit install                    # once per clone: format, license and spelling checks
```

After `environment.yml` changes, run `conda env update -f environment.yml`. Tool versions are
pinned in `environment.yml` and `.pre-commit-config.yaml` and are the same in CI.

`scripts/dev_env.sh` also exports a few locations. Their defaults follow the lead maintainer's
machine; set the variable before sourcing the script to put things elsewhere:

| Variable | Default | What lives there |
|---|---|---|
| `DYNG_SCRATCH` | `~/Projects/dyng-work` | the persistent work area of the parity harness: reference builds, goldens, benchmark inputs (a few GB once you run parity) |
| `CPM_SOURCE_CACHE` | `$DYNG_SCRATCH/cpm-cache` | downloads of the build dependencies (GoogleTest, CPM.cmake), shared by all build trees; created by the first configure |
| `DYNG_CUDA_HOME` | `/usr/local/cuda-13.1`, else `/usr/local/cuda` | the CUDA toolkit whose `nvcc` is put on `PATH` (only if it exists) |
| `DYNG_CONDA_ENV` | `dyng-dev` | the conda environment to activate |

For example `export DYNG_SCRATCH=~/.cache/dyng` in your shell profile. The parity harness also
reads `DYNG_ORIGINALS_DIR` (default `~/Projects`), the directory with your clones of the original
research repositories ([below](#parity-with-the-original-research-codes)).

## Build

The build uses CMake presets (`CMakePresets.json`); every preset builds into `build/<preset>`.

```bash
cmake --preset dev              # configure: Debug, tests on, warnings as errors
cmake --build --preset dev      # build
ctest --preset dev              # run the tests
```

| Preset | Use it for |
|---|---|
| `dev` | day-to-day development: Debug, tests, `-Wall -Wextra -Wpedantic -Werror` |
| `cpu-only` | Release build of the CPU backends (what most CI jobs build) |
| `release`, `relwithdebinfo` | optimised builds (with debug information) |
| `asan`, `tsan`, `tsan-openmp` | AddressSanitizer + UndefinedBehaviorSanitizer, ThreadSanitizer (OpenMP off), ThreadSanitizer with the OpenMP backends (Clang; `ci/sanitizers.sh` runs all three) |
| `parity` | the compiler flags of the original research codes, for parity and performance runs |

Build options (`DYNG_ENABLE_OPENMP`, `DYNG_ENABLE_CUDA`, `DYNG_BUILD_TESTS`, ...) are listed at
the top of `CMakeLists.txt`; pass them with `-D` after the preset, for example
`cmake --preset dev -DDYNG_ENABLE_OPENMP=OFF`.

**CUDA.** The presets above never build CUDA. With a CUDA toolkit (>= 12.4; `scripts/dev_env.sh`
puts it on `PATH`) the CUDA backend has its own presets:

```bash
cmake --preset dev-cuda && cmake --build --preset dev-cuda   # Debug, this machine's GPUs (native)
ctest --preset dev-cuda -L gpu                               # the CUDA tests (need a visible GPU)
ci/build_cuda.sh                                             # compile-only, as cuda-build.yml does
```

| Preset | Use it for |
|---|---|
| `dev-cuda` | CUDA development: Debug, `native` architectures, tests, host code `-Werror` |
| `release-cuda` | the release architecture list (sm_75 to sm_120 SASS plus PTX) |
| `parity-cuda` | the flags of MOSP-CUDA (`-O3 -lineinfo`, sm_86): parity and performance runs |
| `sanitize-cuda` | `compute-sanitizer` runs |
| `ci-cuda13`, `ci-cuda12` | the compile-only builds of the hosted `cuda-build` workflow (no GPU) |

`DYNG_CUDA_ARCHITECTURES` chooses `native` (the default), `release` or an explicit list such as
`86`. The README's "Build with CUDA" section has the details.

## Test

**Every change comes with tests.** Tests are GoogleTest programs under `cpp/tests/`, one
executable per module, registered with `dyng_add_test(NAME ... SOURCES ... LABELS ...)`. Labels
select what runs where:

| Label | Meaning |
|---|---|
| `cpu` | runs without a GPU; the default CI set (`ctest --preset dev -L cpu`) |
| `gpu` | needs a CUDA device |
| `parity` | byte-for-byte replay against the original research codes (needs the goldens) |
| `slow` | long randomized or large-input tests |
| a module name (`core`, `graph`, `io`, `sssp`, ...) | select one module: `ctest --preset dev -L sssp` |

Before you open a pull request, run the full local gate. The hosted workflows `lint`, `cpu`
and `docs` run the same steps on every pull request (only the parity replay runs locally):

```bash
ci/check.sh            # clang-format, cpu-only + dev builds and `ctest -L cpu`, clang-tidy
                       # naming rules, REUSE, provenance headers, the Python harness tests,
                       # the documentation build, pre-commit
ci/check.sh --help     # the steps; DYNG_CHECK_SKIP="precommit" skips a step by name,
                       # DYNG_CHECK_ONLY="tidy" runs only the named steps
```

A change that touches CUDA code (or anything the CUDA backend uses) also needs the local GPU
gate, on a machine with an NVIDIA GPU. The hosted `cuda-build` workflow only compiles (hosted
runners have no GPU); if you have no GPU, say so in the pull request and a maintainer runs it:

```bash
ci/gpu_local.sh        # dev-cuda build, `ctest -L gpu` and `-L cpu`, the sssp golden corpus on
                       # the cuda backend (when the goldens exist), compute-sanitizer memcheck
                       # and synccheck, clang-tidy naming on the CUDA branches; fails when the
                       # test GPU is not visible. DYNG_TEST_GPU picks the GPU (default 1),
                       # DYNG_GPU_SKIP="memcheck tidy" skips steps by name
```

Algorithms are tested against an **oracle** (a from-scratch recomputation, or a reference
implementation), on hand-written cases (paper examples, regression seeds, edge cases) and on
randomized batches. A bug fix adds a test that fails without the fix. For memory or threading
changes, also run `ci/sanitizers.sh` (the `asan`, `tsan` and `tsan-openmp` presets, as the
`sanitizers` workflow does on every pull request).

## Parity with the original research codes

Ported algorithms must compute exactly what the pinned original computes (byte-identical outputs
where the algorithm page says so). The parity harness in [parity/](parity/README.md) builds the
original from a `git archive` copy, exports its outputs as goldens and replays them against
dynG:

```bash
git clone https://github.com/SMShovan/MOSP-OpenMP.git ~/Projects/MOSP-OpenMP   # once
parity/build_reference.sh MOSP-OpenMP    # scratch copy of the pinned original, built
parity/export_goldens.py                 # the golden corpus, outside the repository
ci/check.sh --parity                     # the gate plus `ctest --preset parity -L parity`
```

The harness only reads the original (with `git archive`), from `$DYNG_ORIGINALS_DIR/<name>`
(default `~/Projects/<name>`; the clone URL of each original is the `upstream` field of
`parity/references.toml`). The goldens and reference builds live in `$DYNG_SCRATCH` (see
[the table above](#set-up-a-development-environment)), never in the repository. A pull request that changes a ported algorithm needs a green parity run; if
you cannot run it (it needs the original's datasets and, for CUDA ports, a GPU), a maintainer
runs it for you. Never weaken a parity check to make it pass: a difference is either a bug or a
documented, accepted deviation (an ADR and the algorithm page's "Differences from the paper").

## Performance and benchmarks

Changes to kernels or hot paths must not make dynG slower than the gates of the plan (ported
algorithms stay within a stated tolerance of the original). Performance is measured on the lab
GPU machine with the A/B/A/B runner `parity/perf_ab.py` (see [parity/README.md](parity/README.md));
a maintainer adds the `ci:bench` label to request a comparison table on a pull request.
Micro-benchmarks (nvbench / Google Benchmark) arrive with the 0.1 series. Please describe in the
pull request how you measured any performance claim.

## Style

- **Formatting** is automatic: `clang-format` (C++/CUDA, `.clang-format`), `ruff` (Python),
  and whitespace fixers, all through `pre-commit` (`pre-commit run --all-files`).
- **Naming** ([ADR 0004](docs/adr/0004-naming.md), checked by clang-tidy): `snake_case` for
  everything (types, functions, variables, files), no `_t` suffix on types, `_t` only on
  template parameters (`vertex_t`), `DYNG_` prefix for macros, `namespace dyng` for the public
  API and `dyng::detail` for internals. File extensions: `.hpp` / `.cpp` for host code,
  `.cuh` / `.cu` for CUDA code. Public headers (`cpp/include/dyng/`) are host-only.
- **Language:** C++17 (no C++20 features in public headers), CUDA 12.4+.
- **Library rules:** library code never prints, never calls `exit()` or `abort()`, and has no
  global state other than the log level and sink. Errors are exceptions derived from
  `dyng::error` (`DYNG_EXPECTS` for preconditions). Use the resources handle for memory,
  streams and profiling.
- **License headers:** every file starts with the SPDX header (`reuse lint` checks it):

  ```cpp
  // SPDX-FileCopyrightText: 2026 The dynG Authors
  // SPDX-License-Identifier: Apache-2.0
  ```

  Files that cannot hold a comment are covered by `REUSE.toml`. Code **ported** from one of the
  original research repositories also carries a provenance line under the SPDX header
  (`ci/provenance_check.py` checks it):
  `// Derived from <repo>@<commit>:<path>`.
- **Dependencies:** a new dependency needs a justification in the pull request, a compatible
  license, and should be optional where possible. Dependencies are fetched at a pinned version,
  not copied into the repository.
- **Code written by others:** avoid copying it. If it cannot be avoided, the code goes to
  `third_party/<name>/`, **keeps its original copyright and license notices** (its own
  `SPDX-FileCopyrightText` and `SPDX-License-Identifier`, or a `REUSE.toml` annotation, and the
  license text in `LICENSES/`), gets a `NOTICE` line if its license requires one, and is
  recorded in [docs/developer/provenance.md](docs/developer/provenance.md). The dynG header
  above is only for code whose copyright belongs to The dynG Authors.
- **GitHub Actions workflows** (`.github/workflows/`): logic goes into `ci/*.sh` so that CI and
  local runs are the same. Pin every action to a full commit SHA with its tag in a comment
  (`uses: actions/checkout@<40-hex SHA>  # v7.0.1`) and check the pair with
  `python3 ci/github_meta_check.py --verify-pins` (network); a new action owner also goes into
  the Actions allow list (`docs/developer/repository_settings.md`, step 5). Least privilege:
  the top-level `permissions:` grants `read` at most, a job that needs to write asks for it in
  its own `permissions:` with a comment saying why, `actions/checkout` sets
  `persist-credentials: false`, `pull_request_target` is used only by a workflow that neither
  checks out nor runs code from the pull request (today only `welcome.yml`, which greets
  first-time contributors), and values from outside (event data, step outputs) reach a `run:`
  script through `env:`, never as `${{ }}` inside the script. The pre-commit hooks
  `actionlint` (with shellcheck), `zizmor` and `github-meta` check all of this.

## Documentation and Doxygen

- **Every public entity** (everything in `cpp/include/dyng/` outside `detail`) has a Doxygen
  comment: `@brief` (one line), `@tparam` and `@param[in|out|in,out]` for every parameter,
  `@return`, `@throws` for each exception type, and `@ingroup`. Functions that take a
  `resources` handle state `@sync` or `@async`; `compute()` and `update()` also state
  `@backends`, `@determinism` and, for published algorithms, `@paper` with a `@cite` key from
  `docs/references.bib`. `detail::` entities get a one-line `@brief`.
- **Build the documentation** with the documented command (it needs a configured build tree
  and the Sphinx packages of `environment.yml`):

  ```bash
  cmake --preset cpu-only       # once, for the generated version.hpp / config.hpp
  ci/docs.sh                    # Doxygen + checks, the Sphinx site with -W, the link check
  # open build/docs/html/index.html
  ```

  Step 1 runs Doxygen with warnings as errors and then `ci/doxygen_coverage.py`, which fail on
  undocumented public API; step 2 builds the site (Sphinx, MyST, pydata-sphinx-theme, Breathe)
  with warnings as errors, so every page must be in a toctree and every reference and link
  between pages must resolve; step 3 (`ci/docs.sh`'s link check) checks the links to repository
  files (`https://github.com/dyng-dev/dyng/blob/main/<path>`) and the anchors of the built
  pages. `ci/docs.sh --doxygen-only` runs step 1 only.
  `ci/check.sh` and `.github/workflows/docs.yml` run the same command.
- User documentation lives in [docs/](docs/) as MyST Markdown (no reStructuredText needed),
  organised as tutorials, how-to guides, concepts (explanation), reference and developer pages;
  `docs/developer/documentation.md` explains where a page goes and how the C++ reference is
  generated.
  Every algorithm has a page under `docs/algorithms/` with the sections the plan requires,
  including "Differences from the paper" for ported algorithms. Update the documentation in the
  same pull request as the code.
- Architecture decisions are recorded as ADRs in [docs/adr/](docs/adr/README.md).

## Commits

Two kinds of message matter, and only one of them reaches `main`:

- **The commits on your branch** are what reviewers read while the pull request is open. Their
  subject line is `<area>: <what changed>` in the imperative, at most about 72 characters, for
  example `io: reject batches with out-of-range ids`. Areas are the top-level parts of the
  repository (`core`, `graph`, `io`, `sssp`, `parity`, `ci`, `build`, `docs`, `tests`,
  `github`, ...).
- **The pull request title** becomes the one commit on `main`: contributions are
  squash-merged, and the title is the subject of the squashed commit. It follows
  [Conventional Commits](https://www.conventionalcommits.org/), for example
  `fix(io): reject batches with out-of-range ids`
  ([Pull requests and review](#pull-requests-and-review)).

For every commit on your branch:

- One logical change per commit, and a history that builds at every commit.
- The body (wrapped at 72) says **why** the change is made and anything a reviewer should know.
- Use `Co-authored-by:` trailers to credit co-authors, including AI assistants that wrote a
  significant part of the change.
- External contributors sign off every commit (next section).

## Developer Certificate of Origin (DCO)

dynG accepts contributions under the
[Developer Certificate of Origin 1.1](https://developercertificate.org/) (DCO). There is no CLA:
Apache-2.0 section 5 makes contributions "inbound = outbound". By adding a sign-off line to a
commit you certify **all four statements of the DCO 1.1** (read the full text at the link):

- (a) you created the contribution and have the right to submit it under the project's open
  source license; or
- (b) it is based on earlier work that, to the best of your knowledge, is covered by an
  appropriate open source license that lets you submit it, with your modifications, under the
  same license (unless you are permitted to submit it under a different one); or
- (c) it was given to you by someone who certified (a), (b) or (c), and you have not modified
  it; and
- (d) you understand and agree that the project and the contribution are **public**, and that a
  record of the contribution, **including all personal information you submit with it and your
  sign-off (your name and e-mail address)**, is kept indefinitely and may be redistributed.

```text
Signed-off-by: Your Name <your.email@example.org>
```

`git commit -s` adds this line with your git `user.name` and `user.email` (use your real name
and an address you can be reached at). The **DCO** check on every pull request fails if a commit
of an **external contributor** is not signed off; members of the `dyng-dev` organization are
exempt (`.github/dco.yml`). Commits made in GitHub's web editor are signed off automatically.

**Fixing a missing sign-off.**

- *Before a review has started* (or when a maintainer asks for it), rewrite your branch:

  ```bash
  git fetch upstream
  git rebase --signoff upstream/main   # adds the sign-off to every commit of your branch
  git push --force-with-lease
  ```

- *During a review*, do not rewrite the branch: add a **remediation commit**, an empty commit
  that signs off the earlier ones (`.github/dco.yml` enables this). One line per commit that
  lacks the sign-off, with that commit's full SHA, then your own sign-off:

  ```bash
  git commit --allow-empty -s -m "DCO Remediation Commit for Your Name <your.email@example.org>" \
    -m "I, Your Name <your.email@example.org>, hereby add my Signed-off-by to this commit: <sha1>
  I, Your Name <your.email@example.org>, hereby add my Signed-off-by to this commit: <sha2>"
  git push
  ```

  The name and address must be the ones of the commits you sign off. The DCO check's details
  page on the pull request lists the commits and shows the same instructions.

## Pull requests and review

**Process.**

1. Fork `dyng-dev/dyng`, create a branch from the latest `upstream/main`
   (`git fetch upstream && git switch -c fix-io-bounds upstream/main`), and make your change
   with tests and documentation.
2. Run `ci/check.sh` locally, and add an entry to the `Unreleased` section of
   [CHANGELOG.md](CHANGELOG.md) for any user-visible change.
3. Open a pull request against `main` and fill in the template's checklist. Draft pull requests
   are welcome for early feedback.
4. CI runs the required checks (`lint`, `cpu`, `cuda-build` and `docs`; later also `python`
   and `api-check`). Workflows of first-time contributors start after a maintainer approves them.
5. A maintainer reviews. Address comments with new commits (do not force-push during a review
   unless asked); to catch up with `main`, use the pull request's **Update branch** button or
   merge `upstream/main` into your branch. The pull request is **squash-merged**, so the title
   becomes the commit subject on `main`. (Milestone and integration work of the maintainers is
   merged with a merge commit instead, which keeps the commit SHAs cited by parity certificates
   and retrospectives; rebase merging is disabled. See [ADR 0019](docs/adr/0019-merge-policy.md) and
   [GOVERNANCE.md](GOVERNANCE.md#reviews-and-merges).)

**Rules:**

- **Everything goes through a pull request**, including the maintainers' own work (the `main`
  ruleset enforces it; see [GOVERNANCE.md](GOVERNANCE.md#reviews-and-merges)).
- **Titles** follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat(sssp): ...`, `fix(io): ...`, `perf(cycle_count): ...`, `docs: ...`, `test: ...`,
  `build: ...`, `ci: ...`, `refactor: ...`, `chore: ...`. A breaking change adds `!`
  (`feat(core)!: ...`).
- **Required checks** must be green. While the lead maintainer is the only maintainer, they may
  merge their own pull request once the checks pass; from the day a second maintainer exists,
  **one approving review** by someone other than the pull request's author is required.
- **Size:** aim for at most **500 changed lines**, excluding generated files and goldens. A port
  is split into a straight port followed by one pull request per refactor.
- **Extra requirements by kind of change:**

| Change | Also required |
|---|---|
| kernel or hot path | a benchmark comparison (`ci:bench`); a `--resource-usage` diff for fused kernels |
| ported algorithm | a green `parity` run on the GPU machine (a maintainer can run it for you) |
| public API | the `api-change` label, a CHANGELOG entry, updated docs, an ADR if significant, and the [API review checklist](docs/developer/api_review_checklist.md) |
| framework or operators (internal-stable) | an ADR; every in-tree algorithm and tutorial stays green |
| new algorithm | a discussed `new_algorithm` issue, the scaffold, a declared maturity, a CODEOWNERS entry |
| new dependency | a justification, a checked license, optional where possible |

- **Review checklist** (in the pull request template): unit and conformance tests, parity
  where relevant, documentation (Doxygen for new public API, the algorithm page, the example
  still runs), a CHANGELOG entry, DCO, a benchmark delta for kernel changes, no printing /
  `exit()` / global state in library code, provenance headers on ported files.

## Adding an algorithm or porting research code

1. **Open an issue first**: the *New algorithm* form (problem, update model, family, backends,
   static counterpart, oracle, paper) or the *Port research code* form (origin repository and
   commit, license or the authors' permission, parity plan). A maintainer confirms that it fits
   the library.
2. **Decide the family:** a per-element value that converges (distances, labels) is a
   *fixed-point* algorithm; a global count (cycles, triads) is an *aggregate-delta* algorithm.
3. **Scaffold the files of an algorithm** with
   `python3 scripts/new_algorithm.py <algo> --family fixed_point|aggregate_delta --backends seq,omp`:
   the public header `cpp/include/dyng/<algo>.hpp`, the implementation folder
   `cpp/src/algorithms/<algo>/` (`manifest.toml`, `CMakeLists.txt`, the sequential reference
   backend, OpenMP), the tests under `cpp/tests/algorithms/<algo>/` (the `test_traits` of the
   conformance kit and hand cases), the page `docs/algorithms/<algo>.md` and a CHANGELOG entry;
   it then runs `scripts/regen.py` (the algorithm tables, CODEOWNERS, the registries). The result
   builds and passes the conformance kit at once (its update recomputes from scratch); make it
   incremental while `ctest -L <algo>` stays green, then add a CUDA backend, an example under
   `examples/` and the BibTeX entry in `docs/references.bib`. The guide is
   [docs/developer/conformance.md](docs/developer/conformance.md); `cpp/src/algorithms/sssp/` and
   `cycle_count/` are the ported examples.
4. **Ports** keep the original's behaviour exactly: first a straight port with the provenance
   header and parity goldens, then one pull request per refactor, each still passing parity
   ([parity/README.md](parity/README.md)). Code you did not write yourself needs the copyright
   holders' license or written permission: an open-source license that allows redistribution
   under Apache-2.0 (MIT, BSD and Apache-2.0 code also keeps its original copyright and license
   notices, see [Style](#style)), or the written agreement of every copyright holder, recorded
   in [docs/developer/provenance.md](docs/developer/provenance.md).
5. New algorithms usually start as `experimental` (sequential backend, tests, a docs page with
   the problem, the template mapping and the limitations, one example).

Thank you for contributing!
