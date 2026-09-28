# ADR 0004: Naming conventions

- **Status:** Accepted
- **Date:** 2026-09-27
- **Deciders:** S M Shovan (lead maintainer)

## Context

The six original repositories use camelCase, PascalCase and snake_case, several spellings of
the same concepts (nine graph types, five batch types) and paper acronyms as function names.
One library needs one style that a contributor can guess, that tools can enforce, and that maps
cleanly to Python (decisions O4 and O5 of the plan).

## Decision

The rules of PLAN Section 4.4.1, enforced by clang-format, clang-tidy
`readability-identifier-naming` (`.clang-tidy`) and review:

- **snake_case everywhere** in C++: namespaces (`dyng`, `dyng::sssp`, `dyng::detail`), classes
  and structs, enumerations and their values, functions, variables and files.
- **No `_t` suffix on types** (STL and RMM style; POSIX reserves `*_t`): `graph`, `resources`,
  `array_view`, `sssp::options`. **`_t` only on template parameters**: `vertex_t`, `edge_t`,
  `weight_t`. Member type aliases use the STL `*_type` form (`graph::vertex_type`).
- Private data members end with `_`. Accessors are nouns without `get_` except where the name
  clashes with a type (`get_backend()`, `get_log_level()`, `get_profiler()`); setters use
  `set_`; predicates `is_` / `has_`; counts `num_`.
- Functions use a fixed verb vocabulary: `compute`, `update`, `apply`, `from_*`, `to_*`,
  `read_*`, `write_*`, `generate_*`, `check_*`, `copy`, `synchronize`, `warm_up`.
- **Algorithm names say what is computed**, never the paper acronym: `sssp`, `mosp`,
  `cycle_count`, `triad_count`, `label_propagation`, `hyper_sssp`. Paper names appear in the
  docs, in `@paper` and in `citation()`.
- Files: `.hpp` (host-compilable), `.cuh` (needs nvcc), `.cpp`, `.cu`; `#pragma once`.
- Macros: `DYNG_` + UPPER_SNAKE, only when unavoidable (`DYNG_EXPECTS`, `DYNG_FAIL`,
  `DYNG_HAS_CUDA`, ...).
- Profiler stage and counter names: `<algo>.<hook>[.<sub>]` in dotted lower case, validated at
  run time by `dyng::profiler`.
- CMake: targets `dyng_<module>` with aliases `dyng::<module>`; options `DYNG_*`; functions
  `dyng_*`. CTest labels are lower case (`cpu`, `gpu`, `slow`, `parity`, `sanitize`, algorithm).
- Tests: `<unit>_test.cpp`; GoogleTest suites and fixtures in PascalCase (a GoogleTest
  requirement).
- Python: PEP 8; only the binding layer renames C++ classes to CamelCase (`dyng.Graph`).

## Consequences

- One small deviation from the plain rule, recorded here: the logging function is
  `dyng::log_message()` rather than `dyng::log()`, so that unqualified calls of the math
  function `log()` inside `namespace dyng` keep compiling.
- `dyng::version_info` uses `major_version` / `minor_version` / `patch_version` because glibc
  may define macros named `major` and `minor`.
- Code ported from the originals is renamed on the way in; the mapping from original names is
  kept on each algorithm's documentation page.

## Notes (M1a close-out, 2026-09-27)

These notes record how the decision is enforced at the end of M1a; they do not change it.

- `ci/check.sh` runs clang-tidy with only `readability-identifier-naming` (the `tidy` step) on
  every library source under `cpp/src` and the public headers it includes, with warnings as
  errors. The other checks listed in `.clang-tidy` (bugprone, performance, modernize) are
  advisory until M4 wires clang-tidy into the hosted CI.
- Test and tool sources are not checked by the `tidy` step: GoogleTest requires PascalCase
  fixtures, and the compat driver and the parity tests keep MOSP's `K` (number of objectives) so
  that they read side by side with the original. Library code uses `num_objectives`.
- Ported code is renamed on the way in, including locals (for example `K` became
  `num_objectives` in `cpp/src/graph/apply_host.cpp`); comments that quote the original keep its
  names.

## Notes (M4, 2026-09-27)

- The hosted CI runs the same `tidy` step: the `tidy` job of `.github/workflows/lint.yml`
  (`DYNG_CHECK_ONLY=tidy ci/check.sh` on a configured `cpu-only` tree, with the clang-tidy of
  `environment.yml`), a required check. The naming rules are therefore enforced on every pull
  request. The bugprone, performance and modernize checks of `.clang-tidy` stay advisory until a
  later milestone cleans their findings and adds them to the job.
