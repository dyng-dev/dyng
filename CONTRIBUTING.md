# Contributing to dynG

Thank you for your interest! dynG is pre-alpha: the APIs are still moving, so please open an
issue to discuss a change before investing a lot of time in it. Everyone taking part follows the
[Code of Conduct](CODE_OF_CONDUCT.md); questions go to GitHub Discussions ([SUPPORT.md](SUPPORT.md))
and security problems are reported privately ([SECURITY.md](SECURITY.md)).

## Build and test

No GPU is needed for the CPU backends.

```bash
conda env create -f environment.yml   # once; or: conda env update -f environment.yml
source scripts/dev_env.sh             # activates the dyng-dev environment
pre-commit install                    # once per clone: format and license checks on commit

cmake --preset dev && cmake --build --preset dev && ctest --preset dev
ci/check.sh                           # the full local gate, run it before opening a pull request
ci/check.sh --parity                  # ... plus the golden parity replay (see parity/README.md)
```

`ci/check.sh --help` lists its steps (format, build and tests, clang-tidy naming, REUSE, Doxygen,
pre-commit, parity); `DYNG_CHECK_SKIP="precommit"` skips a step by name.

## Style

- C++17; clang-format (`.clang-format`, pinned in `.pre-commit-config.yaml`) and clang-tidy
  (`.clang-tidy`) define the style. Naming follows `docs/adr/0004-naming.md`: snake_case
  everywhere, no `_t` suffix on types, `_t` only on template parameters, `.hpp` for host
  headers and `.cuh` for CUDA headers, `DYNG_` macros, `namespace dyng` / `dyng::detail`.
- Every file starts with the SPDX header:

  ```cpp
  // SPDX-FileCopyrightText: 2026 The dynG Authors
  // SPDX-License-Identifier: Apache-2.0
  ```

  Files ported from the original research repositories also carry a provenance line:
  `// Derived from <repo>@<commit>:<path>`.
- Every public entity has a Doxygen comment (`@brief`, `@param`, `@tparam`, `@return`,
  `@throws`, `@ingroup`; `@backends`, `@determinism` and `@paper` on `compute` / `update`);
  `ci/docs.sh` (Doxygen with warnings as errors, then `ci/doxygen_coverage.py`) fails on
  undocumented public API.
- Library code never prints, never calls `exit()` or `abort()`, and has no global state other
  than the log level and sink. Errors are exceptions derived from `dyng::error`.
- Every change comes with tests (GoogleTest in `cpp/tests/`, labelled `cpu`, `gpu`, `slow`,
  `parity`, ...). Add an entry to the `Unreleased` section of `CHANGELOG.md`.

## Pull requests

- Keep pull requests small (about 500 changed lines or fewer, excluding generated files).
- Titles in Conventional-Commits style: `feat(sssp): ...`, `fix(io): ...`, `docs: ...`.
- All checks (`lint`, `cpu`, ...) must be green.

## Developer Certificate of Origin

External contributions are accepted under the [Developer Certificate of Origin 1.1](https://developercertificate.org/):
sign off every commit with `git commit -s`, certifying that you wrote the change or have the
right to submit it under the project's license (Apache-2.0). There is no CLA; Apache-2.0
section 5 makes contributions "inbound = outbound".
