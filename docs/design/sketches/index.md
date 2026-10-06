# API sketches of the later algorithms

The 0.1 API is frozen (ADR 0023): `core/*`, `graph/*`, `update.hpp`, `sssp.hpp` and
`cycle_count.hpp`. The algorithms and the container that follow it are designed now, against that
frozen contract, so that the 0.1 API does not have to change for them (PLAN Section 5, risk R4).
The pages below are **reviewed sketches**: each was checked against the API review checklist
({doc}`../../developer/api_review_checklist`) and the uniform algorithm contract (ADR 0006), and
each lists what it needs from the frozen headers and what is still open. They are not contracts:
CI does not check them, and each is frozen by the milestone that implements it (PLAN Section 11).

| Sketch | Release | Container | Family | Frozen in |
|---|---|---|---|---|
| {doc}`mosp` | 0.2 | `graph` (K weight columns) | fixed point (K x `sssp`, then finalize) | M7 |
| {doc}`hypergraph` (with `hyperedge_batch`) | 0.3 | the new `hypergraph` container | - | M8 |
| {doc}`triad_count` | 0.3 | `hypergraph` | aggregate delta | M9 |
| {doc}`label_propagation` | 0.4 | `graph` (slotted or slack rows, vertex batches) | fixed point | M10 |
| {doc}`hyper_sssp` | 0.4 | `hypergraph` (slack CSR, line graph) | fixed point (on the line graph) | M11 |

## What every sketch keeps from the frozen contract

- One header `<dyng/<name>.hpp>`, a flat namespace `dyng::<name>`, and the five names `options`,
  `stats`, `result`, `compute()`, `update()` with the frozen signatures'
  shape: `compute(res, container, inputs..., opt) -> result` never mutates the container;
  `update(res, container&, batch, result&) -> stats` applies the batch and updates the result.
- `options` is an aggregate with defaults (fields appended only); inputs fixed at `compute()` are
  arguments or read-only afterwards; tunables change through `result::set_options()`.
- `stats` derives from `update_stats` and has a `batch` member with what the commit did
  (`apply_summary` for graphs, `hyper_apply_summary` for hypergraphs); every counter is marked
  *deterministic* or *schedule-dependent*.
- `result` is opaque (pimpl), move-only, with `get_options()` / `set_options()`,
  `graph_version()`, `space()` and `clone(res)`; a failed update after the commit poisons it
  (`@guarantee`: strong before the commit, basic after it).
- Several results on one container go through `dyng::update(res, c, batch, r1, r2, ...)`: each
  algorithm specializes `detail::update_traits` for its result, and the hypergraph adds
  `detail::participant_of<hypergraph<...>>`.
- Enumerations are `enum class` with snake_case values and a `to_string()`; errors derive from
  `dyng::error`; profiler stages are `<name>.<hook>[.<sub>]`.
- The registry entry (`manifest.toml`) declares the family, container, backends, determinism,
  oracle kind and graph requirements, and the conformance kit runs C1-C12 from it
  ({doc}`../../developer/conformance`).

The pages use the planned header paths of PLAN Section 4.2. Where a sketch differs from the
plan's own sketch in Section 5, the page says why (usually: the plan's sketch predates the frozen
`stats : update_stats` and `apply_summary` shapes).

```{toctree}
:maxdepth: 1

mosp
hypergraph
triad_count
label_propagation
hyper_sssp
```
