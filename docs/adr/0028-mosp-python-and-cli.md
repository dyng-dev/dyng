# ADR 0028: mosp in Python and on the command line (0.2)

- **Status:** Accepted under delegation (2026-10-01; GOVERNANCE.md, "Delegation of technical
  ADRs"). It adds the Python and command-line surface of `mosp` (M7 acceptance criterion 2) and
  reviews the additions to the public Python API; it does not change a rule the author approved:
  the PLAN 8.6 gates and their protocol (ADRs 0018, 0021), the parity rules and the C++ API
  baseline are untouched (the C++ baseline was extended by ADR 0027 and is unchanged here, 670
  declarations).
- **Date:** 2026-10-01
- **Deciders:** the AI assistant, on the author's behalf (M7, step python-docs-finish)

## Context

ADR 0027 froze the C++ API of `dyng::mosp`. PLAN 5.4 asks for a Python counterpart of every
stable C++ symbol with the same names (`dyng.<algo>.compute / update / Options / Result /
Stats`, options as keywords with the C++ field names) and PLAN 5.5 sketches
`paths.path_costs.to_numpy()  # shape (n, 3)`. PLAN 5.6 asks for `dyng <algo> <verb>` with one
flag per option field and the originals' output formats. ADR 0011 (the Python bindings) and ADR
0025 (the command line) set the rules; this ADR records the choices that are new with mosp.

## Decision

1. **`dyng.mosp`** (`python/dyng/mosp.py`, `python/bindings/mosp.cpp`): `compute(graph, source,
   *, options=None, resources=None, **kwargs)`, `update(graph, batch, result, *,
   resources=None)`, the dataclass `Options` with the C++ fields and defaults (`preferences:
   list[int]`, `delta`, `cuda_engine`, `compute_path_costs`, `validate_inputs`,
   `num_objectives`; checked equal to `native.MospOptions()` by the defaults test), the frozen
   dataclass `Stats` (the `update_stats` fields, `batch`, `objectives: tuple[dyng.sssp.Stats,
   ...]`, `combined_edges`, `preference_scale`) and `Result`. The per-objective accessors keep the
   C++ shape, methods with the objective index: `Result.distances(k)`, `Result.parents(k)`; the
   other arrays are properties as in `dyng.sssp`: `combined_distances`, `combined_parents`,
   `path_costs`. `Result.from_arrays(graph, source, distances, parents, *, canonicalize=True,
   options=None, resources=None, **kwargs)` takes lists of K arrays (ids converted with a range
   check, as `dyng.sssp.Result.from_arrays`). Constants `MAX_OBJECTIVES`, `MAX_PREFERENCE_SCALE`
   and `INFINITE_DISTANCE` (tested equal to the native values; the module does not load the
   native module at import, so the backend can still be chosen on first use).
2. **Path costs are a 2-D `dyng.Array`** of shape (n, K), int64, host memory: the C++ array is n
   * K values, vertex-major, so the (n, K) view is zero-copy. `dyng.Array` was documented as 1-D;
   it now reports the view's `ndim` and `shape`, and `len()` is `shape[0]` (unchanged for 1-D
   arrays). Every other result array stays 1-D. The staleness and copy-on-write rules of ADR 0011
   apply unchanged.
3. **Composition from Python.** `dyng.update(graph, batch, *results)` accepts mosp results next to
   sssp and cycle_count results (the binding builds `detail::make_mosp_participant`, as ADR 0023
   note 1 prescribes for run-time lists), so one application of the batch updates them all.
4. **Python helpers.** `dyng.io.write_path_costs(path, costs)` (MOSP's `mospCosts.txt`, the C++
   `io::write_path_costs`), `dyng.testing.combined_graph(parents, source, preferences)` and
   `dyng.testing.mosp_path_costs(graph, parents, source, *, num_objectives=0)` (the C++ references
   of `<dyng/testing/mosp_oracle.hpp>`).
5. **`dyng mosp compute|update`** (`python/dyng/cli/_algorithms.py`): the flags are generated from
   `dyng.mosp.Options` as for every algorithm; a `list[int]` field is a comma-separated flag
   (`--preferences 4,1,4`, MOSP's `--pref` syntax), so `mosp --pref` is `--preferences` and
   `mosp -k` is `--num-objectives` (PLAN 5.6: flags map one to one to option fields; the drop-in
   clone with the original's flags stays `dyng-compat-mosp --mosp`). `update` writes
   `obj<k>/distancesUpdated.txt`, `SSSPTreeUpdated.txt` and `combinedGraph/{distancesCsr,
   SSSPTreeCsr,mospCosts}.txt` as the `mosp` driver; `compute` writes the trees as `mospPrep
   init` and the combined files of the static solve. With `--num-objectives` below the graph's
   column count, `mospCosts.txt` covers every column, as the original's does (the other columns
   summed along the same MOSP tree by `dyng.testing.mosp_path_costs`, as `dyng-compat-mosp` does
   with the C++ reference); with `--no-compute-path-costs` no cost file is written.
6. **The engine option** is exposed as before (`cuda_engine=` in Python, `--cuda-engine` on the
   command line, for sssp since M5 and now for mosp); its behaviour is ADR 0026's (`automatic`
   falls back to the operators engine, `operators` runs everywhere). The host backends ignore it.
7. **Build.** `DYNG_BUILD_PYTHON=ON` needs mosp in the build (`DYNG_ALGORITHMS=all`), as it needs
   sssp and cycle_count. The examples gain `examples/python/mosp_update.py` and
   `examples/cpp/mosp_update.cpp` (PLAN 9.6: one example per algorithm), which compare the thesis
   example's combined files with the originals'.

## Consequences

- The Python API gains names only (griffe reports no breaking change against `main`); the
  committed stubs `python/dyng/_core.pyi` are regenerated (the private `MospOptions`, `MospStats`,
  `MospResultI32/I64`, `mosp_compute`, `mosp_update`, `mosp_from_arrays`, `write_path_costs`,
  `testing_combined_graph_*`, `testing_mosp_path_costs`).
- Tests: `python/tests/test_mosp.py` (the thesis example with Pref {4, 1, 4} and {4, 4, 1}, the
  options and their checks, imported trees, clone and pickling, composition, the references, a
  Hypothesis property against a pure-Python oracle: Dijkstra per objective, the combined graph,
  Dijkstra on it and the path costs, before and after a random batch); `test_cli.py` compares
  `dyng mosp update` with the committed files of both originals (8 preference cases), with
  `dyng-compat-mosp --mosp` on every committed case (the 34 SOSP cases and the 8 preference cases)
  and, where it is built, with MOSP-OpenMP's own `mosp`.

## Alternatives considered

- **`Result.distances` as a list of K arrays** (or a 2-D (K, n) array): a list of arrays would
  read each objective's array on every access; a (K, n) array would need a copy (the K trees are
  separate allocations, ADR 0027). Methods with the objective index match the C++ API.
- **`--pref` and `-k` as aliases** on `dyng mosp`: PLAN 5.6 keeps the original spellings in
  `tools/compat`; the command line uses the option names.
- **Path costs over the K objectives only** on the command line when `--num-objectives` is
  smaller: the file would differ from the original's, and the command line's purpose is the
  originals' formats.
