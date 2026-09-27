# Contributing to dynG

Thank you for your interest in dynG! This guide takes you from a fresh clone to a merged pull
request. Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md). Questions go
to [GitHub Discussions](https://github.com/dyng-dev/dyng/discussions) ([SUPPORT.md](SUPPORT.md));
security problems are reported privately ([SECURITY.md](SECURITY.md)).

dynG is **pre-alpha**: APIs, file formats and build options still change. For anything larger
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
  0.2, `good-first-issue:op` (a new hypergraph pattern). Issues labelled
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
conda env create -f environment.yml   # once: the dyng-dev env (CMake, Ninja, clang-format,
                                      # clang-tidy, Doxygen, pre-commit, reuse, pytest, ...)
source scripts/dev_env.sh             # every shell: activates dyng-dev (and nvcc if installed)
pre-commit install                    # once per clone: format, license and spelling checks
```

After `environment.yml` changes, run `conda env update -f environment.yml`. Tool versions are
pinned in `environment.yml` and `.pre-commit-config.yaml` and are the same in CI.

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
| `asan`, `tsan` | AddressSanitizer + UndefinedBehaviorSanitizer, ThreadSanitizer |
| `parity` | the compiler flags of the original research codes, for parity and performance runs |

Build options (`DYNG_ENABLE_OPENMP`, `DYNG_ENABLE_CUDA`, `DYNG_BUILD_TESTS`, ...) are listed at
the top of `CMakeLists.txt`; pass them with `-D` after the preset, for example
`cmake --preset dev -DDYNG_ENABLE_OPENMP=OFF`.

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

Before you open a pull request, run the full local gate. It is what CI runs:

```bash
ci/check.sh            # clang-format, cpu-only + dev builds and `ctest -L cpu`, clang-tidy
                       # naming rules, REUSE, provenance headers, the Python harness tests,
                       # the Doxygen check, pre-commit
ci/check.sh --help     # the steps; DYNG_CHECK_SKIP="precommit" skips a step by name
```

Algorithms are tested against an **oracle** (a from-scratch recomputation, or a reference
implementation), on hand-written cases (paper examples, regression seeds, edge cases) and on
randomized batches. A bug fix adds a test that fails without the fix. For memory or threading
changes, also run the `asan` and `tsan` presets.

## Parity with the original research codes

Ported algorithms must compute exactly what the pinned original computes (byte-identical outputs
where the algorithm page says so). The parity harness in [parity/](parity/README.md) builds the
original from a `git archive` copy, exports its outputs as goldens and replays them against
dynG:

```bash
parity/build_reference.sh MOSP-OpenMP    # scratch copy of the pinned original, built
parity/export_goldens.py                 # the golden corpus, outside the repository
ci/check.sh --parity                     # the gate plus `ctest --preset parity -L parity`
```

The goldens and reference builds live in `$DYNG_SCRATCH` (set by `scripts/dev_env.sh`), never
in the repository. A pull request that changes a ported algorithm needs a green parity run; if
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
  license, and should be optional where possible. Third-party code is never copied into the
  repository; it is fetched at a pinned version.
- **GitHub Actions workflows** (`.github/workflows/`): logic goes into `ci/*.sh` so that CI and
  local runs are the same. Pin every action to a full commit SHA with its tag in a comment
  (`uses: actions/checkout@<40-hex SHA>  # v7.0.1`) and check the pair with
  `python3 ci/github_meta_check.py --verify-pins` (network); a new action owner also goes into
  the Actions allow list (`docs/developer/repository_settings.md`, step 5). Least privilege:
  the top-level `permissions:` grants `read` at most, a job that needs to write asks for it in
  its own `permissions:` with a comment saying why, `actions/checkout` sets
  `persist-credentials: false`, `pull_request_target` is not used, and values from outside
  (event data, step outputs) reach a `run:` script through `env:`, never as `${{ }}` inside
  the script. The pre-commit hooks `actionlint` (with shellcheck), `zizmor` and `github-meta`
  check all of this.

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
  ci/docs.sh                    # Doxygen + checks, the Sphinx site with -W, the internal link check
  # open build/docs/html/index.html
  ```

  Step 1 runs Doxygen with warnings as errors and then `ci/doxygen_coverage.py`, which fail on
  undocumented public API; step 2 builds the site (Sphinx, MyST, pydata-sphinx-theme, Breathe)
  with warnings as errors, so every page must be in a toctree and every reference must
  resolve; step 3 checks the internal links. `ci/docs.sh --doxygen-only` runs step 1 only.
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

- One logical change per commit, and a history that builds at every commit.
- The subject line is `<area>: <what changed>` in the imperative, at most about 72 characters,
  for example `io: reject batches with out-of-range ids` or `sssp: time the tree import`.
  Areas are the top-level parts of the repository (`core`, `graph`, `io`, `sssp`, `parity`,
  `ci`, `build`, `docs`, `tests`, `github`, ...).
- The body (wrapped at 72) says **why** the change is made and anything a reviewer should know.
- Use `Co-authored-by:` trailers to credit co-authors, including AI assistants that wrote a
  significant part of the change.
- External contributors sign off every commit (next section).

## Developer Certificate of Origin (DCO)

dynG accepts contributions under the
[Developer Certificate of Origin 1.1](https://developercertificate.org/). There is no CLA:
Apache-2.0 section 5 makes contributions "inbound = outbound". By adding a sign-off line to a
commit you certify that you wrote the change, or otherwise have the right to submit it under
the project's license:

```text
Signed-off-by: Your Name <your.email@example.org>
```

`git commit -s` adds this line with your git `user.name` and `user.email` (use your real name
and an address you can be reached at). The DCO check on every pull request fails if a commit of
an **external contributor** is not signed off; members of the `dyng-dev` organization are
exempt (`.github/dco.yml`). To fix a missing sign-off on your branch:

```bash
git rebase --signoff main       # adds the sign-off to every commit of your branch
git push --force-with-lease
```

## Pull requests and review

**Process.**

1. Fork `dyng-dev/dyng`, create a branch from `main` (`fix-io-bounds`, `feat-kcore`, ...), and
   make your change with tests and documentation.
2. Run `ci/check.sh` locally, and add an entry to the `Unreleased` section of
   [CHANGELOG.md](CHANGELOG.md) for any user-visible change.
3. Open a pull request against `main` and fill in the template's checklist. Draft pull requests
   are welcome for early feedback.
4. CI runs the required checks (`lint`, `cpu`, `docs`, and later `cuda-build`, `python` and
   `api-check`). Workflows of first-time contributors start after a maintainer approves them.
5. A maintainer reviews. Address comments with new commits (do not force-push during a review
   unless asked); the pull request is **squash-merged**, so the title becomes the commit
   subject on `main`.

**Rules** (PLAN Section 8.9):

- **Everything goes through a pull request**, including the maintainers' own work.
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
| public API | the `api-change` label, a CHANGELOG entry, updated docs, an ADR if significant, and the API review checklist |
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
   commit, license, parity plan). A maintainer confirms that it fits the library.
2. **Decide the family:** a per-element value that converges (distances, labels) is a
   *fixed-point* algorithm; a global count (cycles, triads) is an *aggregate-delta* algorithm.
3. **Create the files of an algorithm** (PLAN Section 4.8): the public header
   `cpp/include/dyng/<algo>.hpp`, the implementation folder `cpp/src/algorithms/<algo>/`
   (`manifest.toml`, `CMakeLists.txt`, the sequential reference backend first, then OpenMP and
   CUDA), tests under `cpp/tests/algorithms/<algo>/`, an example under `examples/`, the page
   `docs/algorithms/<algo>.md`, the BibTeX entry in `docs/references.bib` and a CHANGELOG entry.
   `cpp/src/algorithms/sssp/` is the model to follow; the scaffolding script
   `scripts/new_algorithm.py` and the conformance kit arrive with the 0.1 framework (milestone
   M3).
4. **Ports** keep the original's behaviour exactly: first a straight port with the provenance
   header and parity goldens, then one pull request per refactor, each still passing parity
   (PLAN Section 6.3).
5. New algorithms usually start as `experimental` (sequential backend, tests, a docs page with
   the problem, the template mapping and the limitations, one example).

Thank you for contributing!
