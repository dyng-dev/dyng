# API review checklist

Every pull request that adds, changes or removes public API goes through this checklist, and so
does every public header at the 0.1 API freeze (milestone M3) and before 1.0. "Public" means
`cpp/include/dyng/**` outside `detail` (and, from 0.1, the Python names reachable from `dyng`
without a leading underscore, the CLI flags and the file formats). Copy the list into the pull
request, tick what holds, and explain every item you leave unticked.

The rules come from the naming decision ({doc}`../adr/0004-naming`), the algorithm contract
({doc}`../adr/0006-algorithm-contract`) and the stability policy below. A reviewer may accept a
deviation, but it is then recorded: in the pull request for a small one, in an ADR for a
significant one.

## Process

- [ ] The pull request carries the `api-change` label and, for anything beyond a small
      addition, links a discussed *API change* issue.
- [ ] `CHANGELOG.md` has an entry under `Unreleased` (`Added`, `Changed`, `Deprecated` or
      `Removed`); a breaking change has a migration note.
- [ ] A significant change (a new concept, a changed contract, a new dependency in a public
      header) has an ADR in `docs/adr/`.
- [ ] The user documentation is updated in the same pull request: the algorithm page, the
      C++ reference group page ({doc}`documentation`), examples and tutorials that use the API.

## Names

- [ ] snake_case for types, functions and variables; no `_t` suffix on types; `_t` only on
      template parameters (`vertex_t`); member type aliases in STL style (`vertex_type`);
      `enum class` with snake_case values; `DYNG_` + UPPER_SNAKE for the few unavoidable macros
      (ADR 0004; clang-tidy checks most of it).
- [ ] A new function uses a verb from the fixed vocabulary, or the pull request says why none
      fits: `compute` (static solve), `update` (batch + result), `apply` (structural mutation
      only), `from_*` / `to_*` (construction and conversion), `read_*` / `write_*` (I/O),
      `generate_*` (seeded generators), `check_*` (validation that throws), `copy`,
      `synchronize`, `warm_up`.
- [ ] Accessors are nouns without `get_` (unless the name clashes with a type), setters start
      with `set_`, predicates with `is_` / `has_`, counts with `num_`.
- [ ] An algorithm's name says what it computes (`sssp`, `cycle_count`), never the paper
      acronym; the paper appears in `@paper` and the docs.

## The algorithm contract (for algorithm headers)

- [ ] The header has the uniform skeleton: `options`, `result`, `stats`, `compute()` and
      `update()` in a flat `dyng::<algo>` namespace.
- [ ] `options` is an aggregate with defaults; inputs fixed at `compute` are arguments or
      read-only afterwards; tunables change through `result::set_options`.
- [ ] `result` is an opaque, move-only owning class (pimpl) with accessors, `graph_version()`,
      `space()` and `clone(res)`.
- [ ] `stats` derives from `update_stats`; every counter's comment says *deterministic*
      (assertable in tests) or *schedule-dependent* (log only). Timing is not a return value; it
      goes through the profiler.
- [ ] `compute(res, container, inputs, options) -> result` never mutates the container;
      `update(res, container&, batch, result&) -> stats` applies the batch and updates the
      result, and throws `stale_result_error` when the result does not match the container's
      version.
- [ ] The oracle kind (`compute` or `reference`) and the determinism (`bitwise`,
      `exact_value`, `tolerance`) are declared; required graph properties are checked by
      `compute()` with an `invalid_argument_error` that names the property and the fix.
- [ ] `sequential` is implemented; a missing backend throws `not_supported_error` naming the
      available ones.

## Library rules

- [ ] Public headers are host-compilable: no CUDA or CCCL header, no device-only type (the
      header self-containment test builds them with a plain host compiler).
- [ ] No printing, no `exit()` / `abort()`, no new global state; errors are exceptions derived
      from `dyng::error`, preconditions use `DYNG_EXPECTS`.
- [ ] Memory, streams and profiling go through the `resources` handle; functions that take it
      state whether they are `@sync` or `@async`.

## Documentation (Doxygen)

- [ ] Every new public entity has a one-line `@brief`, `@tparam` and `@param[in|out|in,out]`
      for every parameter, `@return`, `@throws` for each exception type, and `@ingroup`;
      `compute()` and `update()` also state `@backends`, `@determinism` and, for published
      algorithms, `@paper` with a key from `docs/references.bib`. `ci/docs.sh` checks this.

## Compatibility (stability policy)

- [ ] Struct fields and enum values are only appended, with defaults; new overloads do not make
      existing calls ambiguous.
- [ ] A changed default value is treated as a breaking change, unless it is a documented bug
      fix.
- [ ] Before 1.0 a stable-API break is allowed in a minor release, with a CHANGELOG "Changed"
      or "Removed" entry and a migration note. From 1.0, breaking changes wait for a major
      release, and removals are deprecated first (`[[deprecated("use X; removed in 2.0")]]`
      and a Python `DeprecationWarning`) for at least one minor release and six months.
- [ ] `dyng::experimental` and algorithms at maturity `experimental` may change in any minor
      release, but the change is still in the CHANGELOG.

The API check workflow (`api-check.yml`, from milestone M5) will diff a snapshot of the public
headers and require the `api-change` label on any pull request that changes it; until then this
checklist and the review are the check.
