# ADR 0025: The command line as a Python console script, and the build of the distributions

- **Status:** Accepted under delegation (2026-09-30; GOVERNANCE.md, "Delegation of technical
  ADRs"). It records how PLAN Sections 5.6, 7.7, 7.9 and 8.8 are implemented in M5 and where the
  implementation differs from their sketches. It changes no rule the author approved: the
  performance gates and their protocol, the parity rules, the licence, the names and the
  publishing steps are untouched. Nothing is published: `release.yml` still publishes only for a
  tag the author pushes, TestPyPI first, PyPI only for final versions and only after the author
  approves the `pypi` deployment.
- **Date:** 2026-09-30
- **Deciders:** the AI assistant, on the author's behalf (M5, step "wheels-cli")

## Context

PLAN 5.6 specifies the command line `dyng <algo> <verb> [--kebab-case options]` with the utility
commands `prep` (the `mospPrep` subcommands), `convert` and `generate`, flags one-to-one with the
option fields and the originals' output formats. PLAN 4.2 and 7.2 place it in `tools/cli/` as a
C++ executable (`dyng_cli`, one `cmd_<algo>.cpp` per algorithm), and 11.3 lists `dyng prep` under
M7. The M5 acceptance criteria ask for the CLI "installed as a console script with the wheel (or
a C++ executable; ADR if the choice deviates from the plan)", including `dyng prep`.

PLAN 7.7 and 8.8 specify a manylinux_2_28 x86_64 abi3 CPU wheel built by cibuildwheel, an sdist,
a 90 MB size check, and a `release.yml` that publishes TestPyPI first and PyPI for final
versions. The development machine has no container runtime (PLAN 8.8), so cibuildwheel cannot
run locally.

## Decision

1. **The CLI is a console script of the Python package** (`[project.scripts] dyng =
   "dyng.cli:main"`, and `python -m dyng`), not a C++ executable. Reasons: `pip install dyng` is
   the 0.1 install path, and a script installs with it on every platform without a second
   compiled artifact in the wheel; the flags are generated from the Python `Options` dataclasses
   (below), so a new option field gets its flag without a change to the CLI; the output writers
   already exist in `dyng.io` and are byte-identical to the originals. The C++-only user (CMake
   `find_package(dyng)`) has no `dyng` executable in 0.1; the C++ drop-in clones of the originals'
   CLIs (`tools/compat`, `dyng-compat-mosp`, `dyng-compat-cycle-enum`) stay what the parity harness
   runs. A C++ `dyng_cli` with the same commands and flags remains possible later (it would be
   tested against the same goldens); PLAN 4.2's `tools/cli/` is not created.
2. **Flags are generated.** Every field of `dyng.sssp.Options` and `dyng.cycle_count.Options` is a
   flag with the field's name in kebab case (enumerations take their names as choices, booleans
   get a `--no-` form); the flags of `dyng generate` are the keyword-only arguments of
   `dyng.generators.legacy.mosp_changes()` and `cycle_enum_batch()` in the same way. A test checks
   that every field and argument has its flag, and another that `docs/api/cli.md` mentions every
   flag. `--objective` absent means one tree per objective (MOSP's behaviour).
3. **`dyng prep` keeps `mospPrep`'s syntax**: the six subcommands with their positional arguments,
   their flags (`--changes`, `--ins`, `-k`, ...), their argument checks (exit status 2) and their
   report lines, so existing MOSP scripts only replace the program name. It is implemented on
   the bindings: `mtx2csr` on `read_matrix_market(random_weights=)`, `widen` on MOSP's weight
   stream (a private binding `_core.legacy_mosp_weights`, `detail::legacy_uniform_int` over
   `std::mt19937`), `cache` writing MOSP's binary cache `MOSPCSR2` (including the source
   identity with libstdc++'s file-clock timestamps), `changes` on `mosp_changes()`, `init` and
   `expected` on `sssp.compute()`. This brings `dyng prep` forward from M7 to M5.
4. **Outputs are the originals' formats**: MOSP's `obj<k>/distances*.txt` and `SSSPTree*.txt`,
   CycleEnumeration-GPU's histogram CSV on standard output (its `update_seconds=` and `match=`
   lines on standard error), MOSP's `insert.txt` / `delete.txt`, MOSP's text CSR. Batches of
   `cycle_count update` are read from MOSP's files or from the text of CycleEnumeration-GPU's
   generator (`- u v` / `+ u v`, the golden corpus's format; `docs/api/file_formats.md`), or
   generated. Graph readers choose by `--format` (`auto`: a CSR prefix, `*.mtx`, else an edge
   list; cycle_count reads `*.mtx` with CycleEnumeration-GPU's parser, as the original does).
   Exit status 0, 1 (a failure; `dyng: error: ...`), 2 (usage). The CLI is not a drop-in clone of
   `cycle-enum`: its flags follow rule 2 (`--max-length`, `--num-deletions`, ...), and where the
   original fails on `--openmp-threads 0`, dynG runs (the OpenMP default). *Corrected in the M5
   review:* this item first said that the original also fails on "`--task update` without
   counts" and that dynG then runs an empty batch; the original's case (`cycle-enum --task
   update` without `--max-cycle-length`, fixture `c11`) fails because the bound is missing, and
   dynG ran an unbounded update that did not finish. `dyng cycle_count update` now requires
   `--max-length` too (a usage error, exit status 2), and the test asserts `c11`.
   `docs/api/cli.md` has the flag table, including the renamed CUDA flags and value spellings.
5. **Tests** (`python/tests/test_cli.py`) compare the CLI byte for byte with the committed
   fixtures of the originals: the 34 sssp cases (`compute` = `mospPrep init`, `update` = `mosp`
   from the original's initial trees and from computed ones, `prep init` / `prep expected`), the
   15 `mospPrep changes` batches, the 22 `generate_batch` batches, the 10 successful `cycle-enum`
   cases and 4 of its error cases, `mtx2csr` and `widen` (new fixtures of MOSP-OpenMP c352151's
   `mospPrep widen`, produced by `parity/fixtures/graph_io/make_graph_io_fixtures.sh`), and the
   binary cache against the reference `mospPrep` when it is built (`$DYNG_SCRATCH/ref`).
6. **Wheels in CI**: `[tool.cibuildwheel]` in `pyproject.toml` (build `cp312-manylinux_x86_64` and
   `cp313-manylinux_x86_64`: the abi3 wheel is built once and tested again under 3.13; image
   `manylinux_2_28`; a pytest subset in the built wheel through `test-sources`), run by
   `.github/workflows/wheels.yml` with `pypa/cibuildwheel` pinned by SHA (v4.2.1; a new
   third-party action, added to the Actions allow-list of `docs/developer/repository_settings.md`).
   `ci/wheel_check.py` checks every distribution: the 90 MB budget, the file name and tags, the
   contents (module, typed layer, stubs, console script, licence files, bundled libgomp) and the
   sdist's file set. `wheels.yml` also installs the wheel alone into fresh venvs for 3.12 and 3.13
   and runs the whole suite. `python.yml` runs `ci/python.sh` (editable install, stubs check,
   pytest) and the suite against an installed sdist on 3.12 and 3.13.
7. **Wheels locally**: `ci/wheels.sh` (also `ci/check.sh --wheels`) builds the sdist, the wheel
   from the sdist with `pip wheel`, repairs it with auditwheel against `manylinux_2_28_x86_64`,
   and runs the same checks and install tests. Because the system toolchain links against glibc
   2.36 (`auditwheel show`: manylinux_2_34), it compiles with conda-forge's GCC 12 against a glibc
   2.28 sysroot (`ci/wheel-toolchain.yml`) and links libstdc++ and libgcc statically **with their
   symbols hidden** (`-Wl,--exclude-libs,ALL`): without hiding, the module's references bound to
   the `libstdc++.so.6` NumPy had loaded, and formatting a number into a `std::ostringstream`
   crashed. The differences from CI are listed in `docs/developer/wheels.md`.
8. **`release.yml`** keeps its file name, its environments `testpypi` and `pypi` and its publish
   jobs. A `select` job decides what the tag builds: `v0.0.x` equal to the name-reservation
   version builds `tools/name_reservation` as before (other `v0.0.x` tags fail); every other tag
   must equal `VERSION` and builds the real distributions by calling `wheels.yml`
   (`workflow_call`). A `collect` job checks them again and uploads one artifact `dist`, which
   the unchanged publish jobs upload: TestPyPI always, PyPI for final versions after approval.
9. **Default resources at exit.** The process-wide default resources live in a module global;
   when the interpreter clears that module only after the extension module is finalized,
   nanobind reported the `Resources` as leaked. An `atexit` hook now drops it first, and the
   install tests fail on any nanobind leak report.

10. **Distribution hygiene** (added in the M5 review). The wheel contains third-party code:
    nanobind (BSD-3-Clause) and its robin-map (MIT) linked statically, the GCC runtime
    (libgomp bundled by auditwheel, parts of libstdc++ and libgcc linked statically;
    GPL-3.0-or-later WITH GCC-exception-3.1). Their licences are reproduced in
    `THIRD_PARTY_LICENSES.txt`, which `license-files` puts into `.dist-info/licenses` and
    `ci/wheel_check.py` requires; whether `License-Expression` should also name them (NumPy's
    style) is a licensing decision for the author, so it stays `Apache-2.0` meanwhile. The sdist
    excludes the repository-only files (the CC-BY-SA-4.0 Code of Conduct, governance pages, tool
    configuration), so every file in it is Apache-2.0. The version comes through the standard
    `[[tool.dynamic-metadata]]` table (the `tool.scikit-build.metadata` form is deprecated in
    scikit-build-core 1.1), and the build requirement is bounded (`scikit-build-core>=1.1,<2`)
    so the published sdist keeps building. `VERSION` must be a canonical PEP 440 version:
    `release.yml` checks it (`ci/wheel_check.py --version-info`) before building and takes the
    pre-release flag from the same parse. `wheels.yml` builds the wheel from its sdist, as
    `ci/wheels.sh` does, so the published sdist is built and tested in the release run. The
    local toolchain pins the GCC 12 runtime (`ci/wheel-toolchain.yml`), whose libgomp the local
    wheel bundles.

## Consequences

- One install gives the library and the command line; the CLI's flags cannot drift from the
  options, and its outputs are pinned to the originals' bytes by the tests.
- There is no C++ `dyng` executable in 0.1 (PLAN 4.2's `tools/cli/`); the compat drivers remain
  the C++ tools of the parity harness.
- The hosted wheel build needs `pypa/cibuildwheel` on the Actions allow-list (an account-level
  setting the orchestrator applies), and `python.yml`'s jobs become required checks once they have
  run on the M5 pull request.
- The local wheel differs in compiler and build front end from CI's; the CI wheel is the one that
  is published. Both pass the same checks and the whole test suite.
