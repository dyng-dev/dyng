# ADR 0022: The conformance kit, the algorithm registry and the scaffold

- **Status:** Accepted under delegation (2026-09-29; GOVERNANCE.md, "Delegation of technical
  ADRs"). It sets up the kit and the tooling of PLAN Sections 4.8, 8.2 and 9.4 and does not change
  a rule the author approved: the performance gates, the measurement protocol (ADRs 0018, 0021)
  and the parity rules are untouched.
- **Date:** 2026-09-29
- **Deciders:** the AI assistant, on the author's behalf (M3, step kit-scaffold)

## Context

M3 turns the two ports into a library other people can add algorithms to (PLAN Section 11.2,
acceptance criteria 4 and 5): the conformance kit C1-C12 for every registered algorithm,
backend and graph type; a registry-driven setup where the manifest is the one place an algorithm
is registered; `scripts/new_algorithm.py` producing an algorithm that is green on the first
build; `scripts/regen.py` for everything generated from the manifests. The plan sketches these;
building them required some choices the plan leaves open, and a few places where its sketch did
not fit what exists.

## Decision

1. **Where the kit lives.** `cpp/tests/conformance/` (`conformance.hpp`, `test_traits.hpp`,
   `generators.hpp`, `type_list.hpp`, the counting `operator new`), not
   `cpp/tests/support/conformance.hpp` as PLAN Section 4.2 sketches: the kit has several files and
   is its own layer. The per-algorithm files are where PLAN Section 4.8 puts them
   (`cpp/tests/algorithms/<name>/<name>_traits.hpp` and the one-line
   `<name>_conformance_test.cpp`).
2. **One typed suite, one executable per algorithm and variant.** `DYNG_CONFORMANCE_SUITE(name)`
   instantiates a typed GoogleTest suite over the algorithm's graph types; every test runs on each
   backend of its executable: `dyng_<name>_conformance_tests` (labels `cpu`, `conformance`,
   `<name>`: sequential and OpenMP) and `dyng_<name>_conformance_cuda_tests` (labels `gpu`, ...:
   cuda, compared with the host backends in C3). Per-algorithm executables keep `ctest -L <name>`
   meaningful (PLAN Section 9.4); the backends are the manifest's, intersected with the build's.
3. **Registry-driven.** `scripts/regen.py` generates, from the manifests, the kit's list
   (`cpp/tests/conformance/registry.hpp`: a tag per algorithm and the include of each traits
   header for the algorithms CMake builds) and the library's registry table behind a new public
   header `<dyng/core/registry.hpp>` (`dyng::algorithms()`, `find_algorithm()`; listed in PLAN
   Section 4.2 as `core/registry.hpp`; its review is part of the 0.1 API review). The Python
   registry (`python/dyng/_algorithms.py`, PLAN Section 4.8) is generated from the same data once
   the Python package exists (M5). The kit reads the backends from the registry, and C0 checks
   that the traits' oracle kind and determinism agree with the manifest; the composition check
   C10 pairs an algorithm with every other registered algorithm of the build on the same graph
   type. The manifests gain `computes`, `paper` and `since` (the columns of the tables), and the
   algorithms whose port has not started are listed in `cpp/src/algorithms/planned.toml`, so the
   tables are generated completely.
4. **Registration rules (invariant I8), in three places.** `regen.py` rejects a manifest without
   the sequential backend first, an oracle kind, known cite keys and the files every algorithm has;
   CMake refuses an algorithm without a conformance suite; `DYNG_CONFORMANCE_SUITE` static_asserts
   `compute()`, `update()`, the oracle kind, the determinism level and the stats' batch summary.
5. **Generated batches from a host model.** The kit keeps a host model of the graph (a map of
   edges and weights) and generates each batch mix from it, so every batch is valid under both
   semantics presets (deletions of existing edges, insertions of absent ones, no duplicates or
   self-loops, reweights only under upsert semantics) and has an exact inverse for C5. Degenerate
   batches are C7's business.
6. **C8, "once reserved", made precise.** The budget counters gain two notions: a *reservation*
   (`note_reservation()`: a workspace created or enlarged, a scratch buffer or per-thread list
   grown, a result grown for new vertices), after which a phase's allocations are reported but not
   held against the budget; and *container work* (`container_scope`: the graph's device copy
   uploaded on first use after a host commit, its host copy downloaded), which is counted but never
   charged to a problem (PLAN 4.5.5: container growth is reported, not failed). C8 warms the
   handle up with the same shapes (a twin result takes the same batch first; up to five warm-ups
   for the OpenMP engines, whose per-thread lists follow the dynamic schedule) and then requires a
   run that reserves nothing, allocates nothing (the library's memory resources and, through a
   counting `operator new` linked into the conformance executables, the host heap; off under the
   sanitizers) and synchronizes at most the traits' budget, which must equal the problem's
   `algorithm_budget`. C8 found two things, both fixed: the OpenMP sssp engine allocated a
   `std::vector` per gather (`list_gather` now keeps its offsets inline), and cycle_count's CUDA
   phase was charged for the graph's upload after a host commit (now container work).
7. **C4 is skipped in 0.1.** No backend of the two algorithms has both a fused and an operators
   engine (the CUDA operators engines arrive in 0.2); the check runs as soon as one does.
8. **The scaffold's scope in 0.1.** `new_algorithm.py` writes graph algorithms on the host
   backends (`--backends seq[,omp]`); a CUDA backend is added by hand after the host backends pass
   the kit, and `--container hypergraph` waits for the hypergraph container (0.2). The template is
   one set of files with `//@@ <family|backend>` blocks, so both families share the lifecycle,
   the result and the tests; its placeholder computation (hop levels, or reciprocal pairs) runs in
   the family's hooks (Tier A), recomputing on G_{t+1} with `stats.fallback_used = true`. It is
   verified by `ci/scaffold_check.sh` (a build of one algorithm of each family in a copy of the
   tree, with sssp, through `-DDYNG_ALGORITHMS`), which is why subset builds now configure.

## Consequences

- A new algorithm is registered by its manifest alone; `regen.py --check` (pre-commit,
  `ci/check.sh`, `lint.yml`) keeps the tables, CODEOWNERS and both registries in step with it.
- The kit runs in every `ctest -L cpu` (and `-L gpu`) run; its C8 part only in
  `DYNG_DEBUG_BUDGETS` builds (Debug: the `dev`, `dev-cuda` and `sanitize-cuda` presets).
- `<dyng/core/registry.hpp>` is new public API in the 0.1 set; the API review decides its final
  shape before the freeze.
- The budget refinement is internal (`core/budget_counters.hpp`, `framework/budgets.hpp`); the
  library behaves the same outside `DYNG_DEBUG_BUDGETS` builds apart from two counter snapshots
  per update (around the commit) and a note in each growth path.
