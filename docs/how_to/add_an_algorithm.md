# Add an algorithm

This is the outline of the guide for adding a dynamic algorithm to dynG (PLAN Section 9.4). Each
step names the tool or page that does the work today; the steps marked *(from M5)* or
*(from 0.2)* describe what the guide will cover once that part of the library exists. The full
guide, with a worked example (`dynamic_bfs`, one of the tutorial algorithms of M6), replaces this
outline in 0.1.x.

Before you start, read {doc}`../concepts/update_model` (the three steps of every update) and the
framework guide's template card ({doc}`../developer/framework`).

## 1. Decide the family

| Your result is ... | Family | Example |
|---|---|---|
| a value per element (vertex, edge, hyperedge) that is iterated until nothing changes | `fixed_point` | `sssp` |
| a global count that a batch changes by what it destroys and creates | `aggregate_delta` | `cycle_count` |

The family fixes the hooks you fill in (the template card) and the invariants the framework
checks for you (I1-I9, {doc}`../developer/framework`).

## 2. Open a `new_algorithm` issue

The issue form asks for the problem, the update model, the family, the backends, the static
counterpart (`compute()`), the oracle (`compute` or a converged `reference`) and the paper. A
maintainer confirms the name: it says what is computed, never the paper's acronym (ADR 0004).

## 3. Scaffold it

```bash
python3 scripts/new_algorithm.py dynamic_kcore --family fixed_point --backends seq,omp \
    --title "Dynamic k-core decomposition" --computes "k-core numbers"
cmake --preset dev -DDYNG_ALGORITHMS=dynamic_kcore
cmake --build --preset dev
ctest --preset dev -L dynamic_kcore
```

The build adds `sssp` to the list by itself (the library's generators and the composition check
C10 need it). Everything is green on the first build: the placeholder `update()` applies the batch and
recomputes (`stats.fallback_used` is true), and the conformance kit C1-C12 runs on it
({doc}`../developer/conformance`, "Adding an algorithm"). `--remove` undoes the scaffold.

## 4. Write `compute()` and the test traits

`compute()` is the sequential static solve and the oracle of every later check. The traits
(`cpp/tests/algorithms/<name>/<name>_traits.hpp`) give the kit its graph types, generators, the
comparator, the oracle kind and the determinism level.

## 5. Make it incremental

Fill the hooks of `problem.hpp` (Tier A: framework operators) until the kit is green without the
placeholder's recompute. The kit tells you which invariant a wrong hook breaks (C1 to C12, C8 for
the steady-state budgets).

## 6. Add backends

OpenMP first, then CUDA: hooks with operators first, a fused engine only when a benchmark shows
the need (Tier B, the verbatim kernels of a port). The scaffold writes the host backends; a CUDA
backend is added by hand in 0.1.

## 7. Bindings, CLI and example *(from M5)*

The Python binding (`python/src/`), the `dyng <name> compute|update` command and an example in
`examples/`.

## 8. The documentation page

`docs/algorithms/<name>.md` with every required section (PLAN Section 9.5): what it computes, the
update model and its hooks, the stages and their paper regions, graph requirements, backends,
determinism, the oracle, parity with the original (for a port), performance, and the citation.
The public header follows the Doxygen conventions (`@backends`, `@determinism`, `@paper`,
`@guarantee`; `ci/docs.sh` checks them) and, once reviewed, its declarations enter the API
baseline ({doc}`../developer/api_review_checklist`, "Updating the API baseline").

## 9. Benchmark it

Register a micro-benchmark; a published algorithm also gets a paper suite and its timed regions
(`parity/timed_regions/<name>.toml`) *(the benchmark suites from 0.2)*.

## 10. Credit

The BibTeX entry in `docs/references.bib` (and the manifest's `cite`, which feeds
`dyng::citation()`), the `CITATION.cff` reference, the CHANGELOG entry.

## 11. Open the pull request

With the pull-request checklist and, for a new public header, the API review checklist. Choose the
maturity in the manifest: `experimental` is fine for a first merge.

## Porting research code instead

A port follows the same steps from the other side: first the original is pinned and
cross-checked, then ported verbatim behind the hooks, then proved equal to it by the parity harness
(PLAN Section 6.3; {doc}`run_parity`). The "port research code" guide is planned for 0.1.x.
