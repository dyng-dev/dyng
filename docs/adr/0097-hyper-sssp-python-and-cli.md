# ADR 0097: hyper_sssp in Python and on the command line, and the container's incidence edits from Python

- **Status:** Technical parts accepted under delegation (2026-11-11; GOVERNANCE.md, "Delegation of
  technical ADRs"). The names are the author's: HD12 (`dyng.hyper_sssp`, `dyng hyper_sssp
  compute|update`, the C++ names the Python ones mirror), HD14 (no openmp backend; every example,
  tutorial and Python test passes explicit resources) and HD10 (the `.dgb` flag), **decided by the
  author on 2026-10-28** (GOVERNANCE.md, approvals log, the rows of 2026-10-28); the names the
  decided list does not spell out are listed in ADR 0096 for the author's acknowledgement. It
  carries out M11b criteria 6 and 7 and the Python half of the container's 0.4 additions (ADR 0087
  kept them refused until now). It changes no approved rule.
- **Date:** 2026-11-11
- **Deciders:** the AI assistant, on the author's behalf for the technical parts (M11b, step
  public-python-cli-docs-freeze); the names: S M Shovan (2026-10-28, HD10, HD12, HD14)
- **Builds on:** ADR 0011 (the Python layer: holders, locks, arrays), ADR 0023 (the uniform
  contract), ADR 0025 (the command line), ADR 0031 (device arrays and plugins), ADR 0046 (the
  hypergraph's Python layer), ADR 0059 (the precedent: triad_count's Python module and command
  line), ADRs 0085, 0087 and 0090 (the incidence edits, their files, hyper_sssp's API).

## Context

The C++ library has had `slack_csr` storage, incidence edits and `hyper_sssp` since M11a; the
Python layer refused them (`NotSupportedError`: `HypergraphProperties` with `"slack_csr"`, a
batch with an edit read from a file, a `.dgb` file with the flag `incidence_edits`) and the
command line had no `hyper_sssp`. HD14 adds a constraint the earlier modules did not have: the
CPU wheel's default resources are OpenMP's (`python/dyng/resources.py`), where hyper_sssp does not
run in 0.4.

## Decision

### 1. `dyng.hyper_sssp` (`python/dyng/hyper_sssp.py`, `python/bindings/hyper_sssp.cpp`)

- `compute(hypergraph, source, *, options=None, resources=None, **kwargs)` and `update(hypergraph,
  batch, result, *, resources=None)`, as `dyng.sssp`'s (the source an argument of compute; option
  fields as keywords; the resources default to the hypergraph's).
- `Options` (a dataclass of the C++ fields and defaults: `work_budget` 1.0, `max_iterations` 4096,
  `on_budget` `"recompute"`, `cuda_engine` `"automatic"`, `validate_inputs` True), `BudgetAction`
  and `FallbackReason` (`StrEnum`s: a member is also its name, as `dyng.triad_count.Pattern`),
  `INFINITE_DISTANCE` (2^61 - 1, sssp's).
- `Stats` (a frozen dataclass: the inherited fields, `batch` a `dyng.HyperApplySummary`,
  `line_graph_insertions`, `line_graph_deletions`, `invalidated`, `relaxations`,
  `fallback_reason` a `FallbackReason`).
- `Result`: `hyperedge_distances` and `hyperedge_parents` are read-only zero-copy `dyng.Array`s of
  the result's state (device memory on cuda; stale after the next update), as sssp's `distances`
  and `parents`; `vertex_distances()` is a method (computed on request, a host `dyng.Array`),
  `distance_to(v)` an int; `source`, `options`, `set_options()`, `graph_version`, `space`,
  `clone(resources)`; `copy` / `deepcopy` are `clone()`, and pickling raises with a hint.
- **The hypergraph behind `distance_to()` and `vertex_distances()`.** The C++ result reads the
  rows of the hypergraph it matches through a weak reference, and the two accessors fill caches of
  the result (ADR 0096 finding 5). The Python result keeps a `weakref` to its `dyng.Hypergraph`
  (re-pointed after every successful update, as the C++ link is) and the binding holds the result's
  lock exclusively and the hypergraph's shared during the two calls, so a concurrent
  `Hypergraph.apply()` or a second accessor call in another thread cannot race with them. A
  hypergraph that is gone gives the C++ `stale_result_error` ("no longer exists"), as in C++.
- Errors are mapped as for sssp (ADR 0011's table); an unweighted hypergraph raises
  `NotSupportedError` in Python (the C++ static_assert's case); openmp resources raise the C++
  `not_supported_error`, whose message names the sequential and cuda backends and the fix
  `dyng.Resources("sequential")` (HD14). The host store serves both host backends, so a hypergraph
  built with OpenMP resources runs hyper_sssp with `resources=dyng.Resources("sequential")`.
- `dyng.update(hypergraph, batch, *results)` takes hyper_sssp results, several at once (several
  sources) and next to triad_count results (refused by triad_count's requirement under
  `slack_csr`, HD9: `NotSupportedError`, nothing changed). A hyper_sssp result passed with a
  `dyng.Graph` raises the `TypeError` of `DYNG_HYPER_SSSP_CONTAINER_MESSAGE`.
- `dyng.testing.hyper_sssp_reference(hypergraph, source)` and `check_hyper_sssp_tree(hypergraph,
  result)` bind the C++ oracle in its public mode (a source vertex), on host copies of the rows by
  id and the weights.

### 2. The container from Python

- `HypergraphProperties(storage="slack_csr", line_graph="slack_csr")` (and `"automatic"`) work;
  the C++ property checks give the matrix's errors (`NotSupportedError` for escher + `slack_csr`
  and `slack_csr` + `sorted_merge`, `InvalidArgumentError` without `store_v2h`).
- `HyperedgeBatch(delete_incidences=(h, v), insert_incidences=(h, v))` (0.3's keywords) apply on
  `slack_csr` storage; the batch gains the properties `delete_incidences`, `insert_incidences`
  (two arrays each; empty without edits) and `num_incidence_edits`; `empty` and `repr` count the
  edits. `HyperApplySummary`'s `inserted_incidences`, `deleted_incidences` and `skipped_ops`
  (0.3's fields) now count them.
- `dyng.io`: the `.dgt` reader returns the edits, the `.dgb` reader reads files with the flag
  `incidence_edits` (`BatchFileHeader.incidence_edits`), `write_binary_batches` sets the flag
  when a batch has an edit (so a file without edits is 0.3's, byte for byte) and the `.dgt`
  writer writes `-i` / `+i`. The native batch tuple of the readers and the generator gains the four
  edit arrays.
- `dyng.generators.random_hyperedge_batch` takes the seven incidence mixes (HD12's enumerators),
  typed by their own `Literal`, `dyng.generators.IncidenceBatchMixName`: appending them to 0.3's
  `HyperedgeBatchMixName` changes the alias's value, which griffe reports as a breaking change of
  the Python API, while a new alias next to it is an addition only.
- `dyng.testing`: `HypergraphModel.apply()` takes the edits with `apply()`'s skip rules, and
  `strategies.hyperedge_batches(model, max_incidence_edits=k)` draws them (Hypothesis: the update
  chain with edits equals compute and the oracle, both id orders).

### 3. `dyng hyper_sssp compute|update` (`python/dyng/cli/_algorithms.py`)

- The option flags are the fields of `dyng.hyper_sssp.Options` one to one (`--work-budget X`,
  `--max-iterations N`, `--on-budget recompute|error`, `--cuda-engine`, `--validate-inputs` /
  `--no-validate-inputs`), generated as for every algorithm (ADR 0025).
- Input: `--input` / `--hypergraph` (a `.hg` file) with `dyng triad_count`'s read flags; the
  properties default to `slack_csr` storage and line graph; the read is weighted (`; w`) unless
  `--random-weights MIN,MAX,SEED` or `--read-options hsosp` (MOSP_ESCHER's loader with its seeded
  weights) draws the weights; a file without weights fails with a message naming those flags.
  `--source S` (default 0). `update`: `--batches FILE ...` (`.dgt` or `.dgb`, incidence edits
  included; a `.dgb` file's expected ids checked, as for triad_count).
- Output: `--out DIR` with `hyperedge_distances.txt` and `hyperedge_parents.txt` (sssp's file
  formats, one line per hyperedge id, `INF` and -1 for none) and, with `--vertex-distances`,
  `vertex_distances.txt`; report lines on standard output (`--quiet`); `update_seconds=` on standard
  error, and with `--compare-recompute` `recompute_seconds=` and `match=yes|no`.
- The default backend is `sequential` (hyper_sssp has no OpenMP backend in 0.4; `--backend openmp`
  fails with exit status 1 and the library's message). Exit statuses 0 / 1 / 2 as for every
  command.
- `add_hyper_sssp` lives in `cli/_algorithms.py` and joins the existing `from ._algorithms
  import` line of `cli/__init__.py`, its registration call after `add_triad_count` (the fixed
  positions of the reconciliation, so M10's `label_propagation` CLI never touches the same line).

### 4. Explicit resources (HD14)

The example `examples/python/hyper_sssp.py`, the tutorial, the doc snippets and every Python test
of hyper_sssp pass `dyng.Resources("sequential")` (or cuda) explicitly; the CLI chooses sequential
itself; the plugin smoke test (`ci/plugin_smoke.py`) runs hyper_sssp on the CPU path's sequential
backend and on cuda with both engines.

## Consequences

- The Python layer has no 0.4 refusal left for hypergraphs; the 0.3 refusal tests are updated
  where `slack_csr` now accepts (escher storage still refuses edits).
- `dyng.hyper_sssp` and `dyng hyper_sssp` are frozen with the C++ header (ADR 0096): a renamed
  parameter or changed default fails griffe's check.
- The weak reference makes the vertex accessors safe across threads at the cost of the two
  exclusive locks; the hyperedge arrays need no lock beyond the result's shared one.
