# The conformance kit and the algorithm tooling

Every algorithm in dynG passes the same checks, on every backend of the build and every
instantiated graph type (PLAN Section 8.2). The checks, the registry that drives them, the
scaffold of a new algorithm and the files generated from the manifests are described here.

## The registry

An algorithm is registered in one place, its manifest `cpp/src/algorithms/<name>/manifest.toml`
(PLAN Section 4.8): name, title, what it computes, family, container, tier, maturity, backends,
determinism level, oracle kind, graph requirements, citation keys, maintainers, origin and
performance tolerances. `scripts/regen.py` turns the manifests (and
`cpp/src/algorithms/planned.toml`, the algorithms whose port has not started) into:

| Generated | Content |
|---|---|
| `README.md`, `docs/index.md`, `docs/algorithms/index.md` | the algorithm tables, between `<!-- regen:algorithms begin ... -->` and `<!-- regen:algorithms end -->` |
| `.github/CODEOWNERS` | the per-algorithm block (its folder, header, tests and page), from `maintainers` |
| `cpp/src/core/registry_table.inc` | the table behind `dyng::algorithms()` and `dyng::find_algorithm()` (`<dyng/core/registry.hpp>`); an entry is compiled in when CMake builds the algorithm (`DYNG_ALGORITHM_<NAME>`) |
| `cpp/tests/conformance/registry.hpp` | the conformance kit's list: one tag per manifest and the include of each algorithm's `test_traits` |

```bash
python3 scripts/regen.py            # after editing a manifest
python3 scripts/regen.py --check    # CI: pre-commit, ci/check.sh (step regen), lint.yml
```

`regen.py` also enforces the registration rules (invariant I8): every manifest lists the
`sequential` backend first and declares an oracle kind; its cite keys exist in
`docs/references.bib`; the files every algorithm has exist (public header, `CMakeLists.txt`,
`<name>.cpp`, `problem.hpp`, `sequential.cpp`, `openmp.cpp` / `cuda.cu` for the listed backends,
the test traits, the conformance suite, the docs page); a `stable` algorithm has a CUDA backend or
a written reason; `cpp/src/algorithms/CMakeLists.txt` adds exactly the algorithms that have a
manifest. `dyng_conformance_registry_tests` checks that `dyng::algorithms()` and the kit list the
same algorithms, and CMake refuses to configure an algorithm without a conformance suite.

## The checks

`cpp/tests/conformance/conformance.hpp` defines a typed GoogleTest suite, instantiated for one
algorithm by the line `DYNG_CONFORMANCE_SUITE(<name>);` in
`cpp/tests/algorithms/<name>/<name>_conformance_test.cpp`. Each algorithm has two executables:
`dyng_<name>_conformance_tests` (labels `cpu`, `conformance`, `<name>`: the sequential and OpenMP
backends) and, in CUDA builds, `dyng_<name>_conformance_cuda_tests` (labels `gpu`, `conformance`,
`<name>`: the CUDA backend, compared with the host backends). One test per check and graph type:
`sssp/conformance/v32_e64_w32.C2_UpdateChainsEqualTheOracle`.

| Check | Property |
|---|---|
| C0 | the traits agree with the registry: oracle kind, determinism level; the sequential backend first (I8); a backend of the test binary that the manifest does not list rejects `compute()` with `not_supported_error` |
| C1 | an empty batch changes nothing (`affected == 0`) |
| C2 | the oracle over random graphs (three sizes) x seven batch mixes (insert-only, delete-only, mixed, local, heavy, reweight, vertex growth) x three consecutive batches x the property presets (the defaults, `batch_semantics::set()`, the traits' extras): after every batch the result equals `compute()` on the new graph at the declared level, and `compute()` equals the independent oracle of the traits (sssp: Dijkstra; cycle_count: the brute-force count); for `oracle_kind::reference` both are within the declared tolerance of the converged reference |
| C3 | the backends agree with sequential, results and deterministic counters |
| C4 | the fused engine equals the operators engine on a backend that has both, after `compute()` and after three consecutive batches, for every preset, the small and medium sizes and every batch mix (sssp on CUDA since M7; skipped where a backend has one engine) |
| C5 | a batch followed by its inverse (built from the host model, with the recorded weights) returns the original result |
| C6 | two runs give identical results and deterministic counters |
| C7 | invalid input is rejected with `invalid_argument_error` and changes nothing (negative ids, a batch with the wrong number of weight columns, a missing deletion under `missing_delete::error`, the traits' invalid options and graphs), or is counted as skipped (missing deletions, and under set semantics existing insertions and self-loops); duplicates still give `compute()` |
| C8 | budgets, in `DYNG_DEBUG_BUDGETS` builds (the `dev`, `dev-cuda` and `sanitize-cuda` presets and every Debug build), under strict budgets (an excess throws; a plain Debug build only logs it): after a warm-up with the same shapes the algorithm phase reserves nothing, allocates nothing (the library's memory resources and, through the counting `operator new` of the conformance executables, the host heap) and synchronizes at most the traits' budget, which must equal the problem's `algorithm_budget` (invariant I9); the commit is reported separately. It runs with `engine::automatic` and, where a backend has a second engine, `engine::operators` (M7: a device without cooperative launch runs the operators engine under `engine::automatic`), with the budget of the engine that ran |
| C9 | stats sanity: `0 <= affected <= n`; inserted + updated + ignored insertions + deleted + ignored deletions + dropped self-loops equals the requested operations; the vertex count; `engine_used` is set; `converged` |
| C10 | `dyng::update(res, g, batch, r1, r2)` (both result orders) equals each algorithm's update alone on a graph of its own, for every other registered algorithm of the build on the same graph type and backend; a result left out is stale afterwards |
| C11 | stale results are detected: after a separate `g.apply()`, on another graph with the same edges and version, and on its own graph after the result moved on with a clone (ADR 0006, "Graph identity") |
| C12 | non-default streams (CUDA): two non-blocking streams, alternated between updates, give the host backends' results |

Randomized checks print their seed; `DYNG_TEST_SEED=<seed> ctest -R <test>` replays one and
`DYNG_TEST_SEEDS=<n>` widens a campaign. `DYNG_TEST_ALLOCATION_TRACE=1` prints a backtrace of every
allocation that C8 counts, to find the one a failing C8 reports.

## The traits

An algorithm tells the kit what it needs in `cpp/tests/algorithms/<name>/<name>_traits.hpp`, a
specialisation of `dyng::conformance::test_traits<tags::<name>>` (the vocabulary is in
`cpp/tests/conformance/test_traits.hpp`; sssp's and cycle_count's traits are the examples):
the oracle kind and determinism level, whether it is history-independent (C5), its graph types,
result, stats and a host `snapshot` of a result, its graph requirements (`require`), the sizes of
the generated graphs (`shape`), `compute`, `update`, `take` (the snapshot), the deterministic
counters, the invalid options (C7) and the host synchronizations of its algorithm phase per
backend (C8). Optional: `compare` (required for `determinism::tolerance`), an independent
`oracle_of`, `extra_properties`, for `oracle_kind::reference`, `near_reference`, `num_weights`
(the weight columns of the kit's graphs, default 1; C7 checks `num_weights + 1` columns, C10 uses
the larger count of a pair; mosp's traits set K = 3), and `host_sync_budget(backend, stats)`, the
C8 budget of the engine that ran (`stats::engine_used` and its counters; `run_dependent_budget`
when it depends on counters the stats do not carry, and C8 then checks the problem's own bound).

## Adding an algorithm

```bash
python3 scripts/new_algorithm.py dynamic_kcore --family fixed_point --backends seq,omp \
    --title "Dynamic k-core decomposition" --computes "k-core numbers"
cmake --preset dev -DDYNG_ALGORITHMS=dynamic_kcore
cmake --build --preset dev
ctest --preset dev -L dynamic_kcore
cmake --preset cpu-only && ci/docs.sh --update-api   # the new header joins the API baseline
ci/docs.sh --no-linkcheck                            # the API page builds (Sphinx -W)
```

`scripts/new_algorithm.py` copies `cpp/src/algorithms/_template` (its `README.md` lists the
placeholders and where each file goes), keeps the parts of the chosen family and backends, adds
the `add_subdirectory()` line, a CHANGELOG entry and the API reference page
`docs/api/cpp/<name>.md` (listed in the C++ API toctree), removes a `planned.toml` entry of the
same name, formats the C++ files and runs `regen.py`. The algorithm is green on the first build:
its `update()` applies the batch and recomputes from scratch (`stats.fallback_used` is true), and
the kit checks it from then on. The new public header is not yet in the API baseline, so the docs
check fails until `ci/docs.sh --update-api` adds it (as a `tracked` header; commit the baseline
with the algorithm). `--remove` undoes a scaffold: it deletes only an algorithm whose manifest
still carries the template's "scaffolded by scripts/new_algorithm.py" line (`--force` overrides)
and puts back the `planned.toml` entry the scaffold replaced. Names that cannot compile are
refused: C++ keywords, names already declared in `namespace dyng` (`engine`, `resources`, ...) and
the generated headers `version` and `config`. The public `compute()` and `update()` are inline
wrappers that `static_assert` the instantiated graph types with a plain-English message (the rule
of sssp and cycle_count since the M3 review), so an unsupported graph type fails to compile at the
call instead of failing to link; keep the trait `detail::<name>_supported_v` in step with the
explicit instantiations. In 0.1 the scaffold writes graphs and the host
backends; a CUDA backend is added by hand, and the hypergraph container arrives in 0.2.

Then, in the order of PLAN Section 9.4: write `compute()` (the sequential static solve) and the
traits; make the update incremental in `problem.hpp` (the template card at its top names the hooks
of the family; {doc}`framework`), until the placeholder's `fallback_used` is gone and the kit is
still green; add backends; write the bindings, the CLI command and the example; fill the docs page
(the required sections of PLAN Section 9.5); register a benchmark; add the citation; open the pull
request with the checklist.

`ci/scaffold_check.sh` (CTest `scaffold.new_algorithm`, label `scaffold`; step `scaffold` of
`ci/check.sh`; job `scaffold` of `cpu.yml`) keeps the template honest: in a throwaway copy of the
tree it scaffolds one algorithm of each family, checks `regen.py --check`, the formatting and
`ci/github_meta_check.py` (the generated CODEOWNERS lines), builds every target of the subset
`-DDYNG_ALGORITHMS=<the probes>` (to which the build adds sssp; Debug, warnings as errors, budgets
on), runs their conformance kits, hand
cases and the registry test, runs the docs steps on the result (Doxygen with the coverage check,
`ci/docs.sh --update-api`, which must add only the probes' headers to the baseline, and the Sphinx
build with warnings as errors; skipped with a message when Doxygen or Sphinx is not installed),
removes them again and checks that nothing is left behind.
