# ADR 0033: The tutorial algorithms: registered with maturity `tutorial`, one source for every backend, undirected graphs in the kit

- **Status:** Accepted under delegation (2026-10-06; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out the M6b acceptance criteria for the tutorial algorithms (created with
  `scripts/new_algorithm.py`, maturity `tutorial`, on every backend, the full conformance kit) and
  records where they differ from PLAN 9.3. It changes no rule the author approved: no gate, no
  parity rule, no licence, no name of the project or of a published distribution; the public API
  only grows (one enumerator, two new headers).
- **Date:** 2026-10-06
- **Deciders:** the AI assistant, on the author's behalf (M6b, step tutorials)
- **Number:** written as 0030 and renumbered 0033 in the M6b review: the parallel milestone M6a
  (branch `m6a-cuda-wheels`) uses 0030, 0031 and 0032.

## Context

PLAN 9.3 plans two teaching algorithms, `dynamic_bfs` (fixed point) and `triangle_delta`
(aggregate delta), written on the extracted framework, "built and tested in CI on every backend",
living in `examples/tutorial_algorithms/` and "not installed or registered as library
algorithms". The M6b acceptance criteria ask for them to be created with
`scripts/new_algorithm.py`, completed on the sequential, OpenMP and CUDA backends with an oracle,
test traits and the full conformance kit C1-C12, with maturity `tutorial` (or `experimental`) in
the registry and the generated tables, and not presented as research algorithms. The scaffold
writes into `cpp/src/algorithms/`, and the conformance kit runs the registered algorithms only, so
both cannot hold together. Three further questions came up:

1. how to write a CUDA backend without a second copy of every hook (the scaffold writes the host
   backends only, PLAN 4.5.4 asks for "hook bodies written with operators");
2. `triangle_delta` counts triangles of an undirected graph, and the kit generated directed graphs
   only;
3. whether the tutorials get Python bindings and CLI commands.

## Decision

1. **Registered, maturity `tutorial`.** Both algorithms live with the others
   (`cpp/src/algorithms/{dynamic_bfs,triangle_delta}`, public headers, manifests), so the scaffold
   creates them, the kit (C0-C12) runs them on every backend and composes them with the other
   algorithms (C10), and `dyng::algorithms()` lists them. A new maturity level marks them as
   teaching material: `maturity_level::tutorial` (appended to the enumeration; `"tutorial"` in the
   manifests, the registry, Python's `AlgorithmInfo.maturity` and the generated tables, which
   also say "teaching material" and "– (tutorial)" for the paper). A tutorial algorithm is covered
   like `experimental` (no SemVer guarantee), cites no paper (the Python registry test accepts an
   empty `cite` for it) and has no benchmark or performance gate. PLAN 9.3's
   `examples/tutorial_algorithms/` holds the tutorial's reference solution instead: the files the
   reader of "Your first dynamic algorithm" writes into a scaffolded `my_bfs` (decision 5). The
   algorithms are installed with the library (their headers are part of the public API baseline,
   as `tracked` headers until they are reviewed).
2. **One source for every backend: executors (Tier A).** Each pass of an algorithm is a small
   functor with a `DYNG_HD` call operator that does the work of one element; an *executor* runs it
   for every element: `operators::sequential_exec` (a loop), `openmp_exec` (an OpenMP loop),
   `cuda_exec` (a grid-stride kernel on the handle's stream). Executors also read a scalar back
   (`read`, the only host synchronization: on CUDA a copy, a stream synchronization and
   `note_host_sync()`), write, fill, copy and upload. Shared words change through the atomics of
   `operators/execution.hpp` (CUDA atomics on the device, `__atomic` builtins on the host). Each
   algorithm's `engine.hpp` holds its passes as one template on the executor
   (`<algo>_engine_impl<exec_t, V, E>`, behind a small virtual interface the problem calls), and
   `sequential.cpp`, `openmp.cpp` and `cuda.cu` are one instantiation each. The executors and the
   atomics go to `cpp/src/operators/` because both tutorials use them (the rule of two, PLAN
   4.5.3); the operators with one user stay in the algorithm folders: `dynamic_bfs`'s frontier
   push (round stamps), `advance` and `invalidate_subtree` (a level-by-level walk), and
   `triangle_delta`'s sorted-row intersection. sssp's and cycle_count's engines are unchanged (no
   operator moved out of them).
3. **Undirected graphs in the kit.** An algorithm whose `require()` sets
   `graph_properties::directed = false` gets undirected models: each edge once, as (u, v) with
   u < v; the edge count and the `apply_summary` counters of C2, C7 and C9 count each stored
   direction (as `apply_summary` documents); C10 builds the pair's model with the merged
   direction, so every other algorithm is also checked on undirected graphs when it is paired
   with `triangle_delta` (all pass). Directed models, and the random streams that generate them,
   are unchanged.
4. **Python yes, CLI no.** Both algorithms have Python modules (`dyng.dynamic_bfs`,
   `dyng.triangle_delta`: `Options`, `Stats`, `Result`, `compute`, `update`), because the
   registry test requires a module for every registered algorithm and the bindings are cheap (the
   models of `cycle_count`). `dyng.update()` (several results in one call) does not accept their
   results in Python (in C++ `dyng::update` does, through their `update_traits`), and there is no
   `dyng <algo>` command: neither adds anything to the lesson. Both are recorded in
   `docs/developer/python_gaps.md`. Nor is there an `examples/cpp/<algo>_update.cpp` program (PLAN
   9.6): the tutorial, the algorithm pages and the hand cases are the examples of teaching
   algorithms.
5. **The tutorial is checked by CI.** `docs/tutorials/your_first_dynamic_algorithm.md` takes the
   reader from a fresh clone to a green kit: it scaffolds `my_bfs` and changes three files. The
   finished files are in `examples/tutorial_algorithms/my_bfs/` with `// [tutorial: ...]`
   markers; the page quotes them by marker (`literalinclude`; a missing marker fails the Sphinx
   build with warnings as errors), and `ci/scaffold_check.sh` scaffolds `my_bfs`, makes the
   tutorial's edits in the scaffold's own files (`ci/tutorial_edits.py`, from the M6b review:
   copying the finished files in could not catch a prose step that does not compile), builds it
   and runs its kit and hand cases next to the scaffold probes. `my_bfs` is a
   simpler algorithm than `dynamic_bfs` (sequential, no BFS tree: it invalidates every vertex
   below a deleted shortest-path edge), so that the lesson fits one sitting; its last section
   shows what `dynamic_bfs` adds.

## Consequences

- The registry, `README.md`, the landing page and the algorithms index list five algorithms;
  two of them say `tutorial`. `dyng.algorithms()` returns them as well.
- The public API baseline grows by `<dyng/dynamic_bfs.hpp>`, `<dyng/triangle_delta.hpp>` and the
  enumerator `maturity_level::tutorial`.
- A change of the scaffold template or of the framework that breaks the tutorial's `my_bfs` fails
  `ci/scaffold_check.sh`; the files in `examples/tutorial_algorithms/my_bfs/` are updated with it.
- New algorithms can follow the executor pattern for their CUDA backend; `scripts/new_algorithm.py`
  still writes the host backends only (a CUDA option of the scaffold would follow this pattern).
