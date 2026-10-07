# Your first dynamic algorithm

In this tutorial you add a dynamic algorithm to dynG, from a fresh clone to a green
conformance kit. The algorithm is `my_bfs`: the BFS level (hop distance) of every vertex from a
source, kept up to date while edges are inserted and deleted. You generate it with
`scripts/new_algorithm.py`, which gives you a correct but slow algorithm (it recomputes after
every batch), and then make its update incremental by filling in the hooks of the framework's
fixed-point template. The conformance kit checks your work at every step.

The tutorial is new in 0.2.0 and follows the main branch: clone it, not the 0.1.0 release. It
takes about an hour and needs no GPU. You should be comfortable with C++17 and with
breadth-first search; you do not need to know dynG's internals.

:::{note}
The library has the finished version of this algorithm: {doc}`../algorithms/dynamic_bfs`, a
**teaching algorithm** (maturity `tutorial`) that also runs on OpenMP and CUDA. Its companion for
the other family of the template, aggregate deltas, is {doc}`../algorithms/triangle_delta`. The
last section of this tutorial shows how `dynamic_bfs` goes further than `my_bfs`.
:::

## 1. Get the code and the tools

```bash
git clone https://github.com/dyng-dev/dyng.git
cd dyng
conda env create -f environment.yml    # once: CMake, Ninja, clang-format, Python (no compiler)
source scripts/dev_env.sh              # activates the environment (dyng-dev)
git switch -c my-first-algorithm
```

The environment does not include a compiler: with or without conda, you need a C++17 compiler on
the system. Without conda, also install CMake, Ninja, Python and clang-format yourself. The
requirements table of {doc}`../getting_started/install` lists the minimum versions.

## 2. The idea

A dynamic algorithm in dynG answers a query on a graph (here: the BFS levels from a source) and
keeps the answer current while batches of changes arrive. Every update follows the same template
({doc}`../concepts/update_model`): the framework applies the batch to the graph (the *commit*),
then the algorithm repairs its result in two steps:

- **Step 1, find what the batch affected.** For BFS levels: an inserted edge `u -> v` can only
  make `v` closer to the source; a deleted edge can only make vertices farther, and only the
  vertices below the deleted edge on a shortest path.
- **Step 2, propagate until nothing changes** (a *fixed point*). A frontier of vertices whose level
  dropped offers `level + 1` to its out-neighbours, round after round, until the frontier is empty.

In the framework this is the **fixed-point template card**. Each box is a *hook*, a member function
of your *problem* class that the framework's *enactor* calls in this order:

```text
commit -> identify_affected -> seed -> { loop until the frontier is empty } -> finalize
```

The enactor also opens a profiler stage per hook, enforces the convergence policy, checks the
allocation and synchronization budget, and keeps the result consistent when something throws
({doc}`../developer/framework`).

## 3. Scaffold the algorithm

```bash
python3 scripts/new_algorithm.py my_bfs --family fixed_point --backends seq,omp \
    --title "My first dynamic BFS" --computes "BFS levels from a source"
```

The script writes the algorithm's files and registers it (the registry, the algorithm tables, the
conformance kit's list):

| File | What it is |
|---|---|
| `cpp/include/dyng/my_bfs.hpp` | the public API: `options`, `stats`, `result`, `compute()`, `update()` |
| `cpp/src/algorithms/my_bfs/manifest.toml` | the registry entry: family, backends, oracle, maturity |
| `cpp/src/algorithms/my_bfs/problem.hpp` | **the problem: its state, workspace and hooks** |
| `cpp/src/algorithms/my_bfs/my_bfs.cpp` | **the hook bodies**, the result, `compute()` / `update()` |
| `cpp/src/algorithms/my_bfs/sequential.cpp` | the static solve (a plain BFS): `compute()`, the oracle of every update |
| `cpp/src/algorithms/my_bfs/openmp.cpp` | the OpenMP backend (calls the sequential solve for now) |
| `cpp/tests/algorithms/my_bfs/my_bfs_traits.hpp` | what the conformance kit needs to know (graph types, comparator, budgets) |
| `cpp/tests/algorithms/my_bfs/my_bfs_conformance_test.cpp` | one line: `DYNG_CONFORMANCE_SUITE(my_bfs);` |
| `cpp/tests/algorithms/my_bfs/my_bfs_test.cpp` | hand-written test cases |
| `docs/algorithms/my_bfs.md` | the algorithm's page |

Build only this algorithm (the build adds `sssp`, which the library's generators need) and run its
tests:

```bash
cmake --preset dev -DDYNG_ALGORITHMS=my_bfs
cmake --build --preset dev
ctest --preset dev -L my_bfs
```

Every test passes: `100% tests passed`. A few are reported as `Skipped`, once per graph type: C4
compares the two engines of an algorithm that has both (fused and operators; `my_bfs` has one),
and C12 checks CUDA streams (`my_bfs` has no CUDA backend). The generated algorithm is correct
from the start because
its `update()` cheats: the `loop` hook runs the static solve again, and `finalize` reports
`stats.fallback_used = true`. The rest of this tutorial replaces the cheat with a real update. The
conformance kit (C0-C12, {doc}`../developer/conformance`) keeps checking the result after every
batch against `compute()` on the new graph, on random graphs, batch mixes and property presets.

## 4. Make the update incremental

You change three files: `problem.hpp`, `my_bfs.cpp` and the hand test. The complete files of the
finished tutorial are in `examples/tutorial_algorithms/my_bfs/`; the snippets below are quoted
from them (the `// [tutorial: ...]` comments mark the quoted parts, you need not type them).

### 4.1 Scratch space: the workspace

The update needs two frontier lists (the input and the output of a loop round) and a flag per
vertex, so that a vertex enters a frontier only once per round. Scratch arrays live in a
*workspace* that the resources handle pools across calls, so that a steady stream of updates
allocates nothing (invariant I9, which the kit's check C8 enforces). In `problem.hpp`, add to
`my_bfs_workspace`, after `before`:

```{literalinclude} ../../examples/tutorial_algorithms/my_bfs/problem.hpp
:language: cpp
:start-after: "// [tutorial: workspace]"
:end-before: "// [tutorial: workspace end]"
```

and size them in `my_bfs_workspace::reserve()` (in `my_bfs.cpp`), counting their bytes too:

```{literalinclude} ../../examples/tutorial_algorithms/my_bfs/my_bfs.cpp
:language: cpp
:start-after: "// [tutorial: reserve]"
:end-before: "// [tutorial: reserve end]"
```

### 4.2 A frontier

The scaffold uses `framework::internal_frontier`, with which `loop` is called once and runs to the
fixed point by itself. A real frontier lets the enactor run the rounds: it calls `loop(in, out)`,
swaps the two frontiers and calls it again until `empty()` is true. In `problem.hpp`, before the
problem class, add:

```{literalinclude} ../../examples/tutorial_algorithms/my_bfs/problem.hpp
:language: cpp
:start-after: "// [tutorial: frontier]"
:end-before: "// [tutorial: frontier end]"
```

and in the problem class, use it:

```cpp
  using frontier_type = my_bfs_frontier;  ///< the loop's frontier (the workspace's lists)
```

The seed will read the in-edges of the graph, so ask the commit to build them. The scaffold's
class already has a `reads_prepared_graph()` that returns `false` (its engines read only the
out-edges); replace that definition with:

```cpp
  /// The seed reads the in-edges: the commit builds them once for every result.
  [[nodiscard]] bool reads_prepared_graph() const noexcept {
    return true;
  }
```

### 4.3 Declare the hooks

Replace the declaration of `loop` (under `// ---- Step 2 ----`) with the three hooks of the
update:

```{literalinclude} ../../examples/tutorial_algorithms/my_bfs/problem.hpp
:language: cpp
:start-after: "// [tutorial: hooks]"
:end-before: "// [tutorial: hooks end]"
```

In the private part, declare a helper and remember what the commit did:

```cpp
  /// Offer the level `from + 1` to `v` (`from`: the level of an in-neighbour, -1 if unreached).
  void offer(std::int64_t from, std::int64_t v, frontier_type& f);

  const applied* applied_ = nullptr;  ///< what the commit did (identify_affected, seed)
```

A hook the problem does not declare is not called (the base class's default returns
`not_provided`), so `my_bfs` has no `normalize`, `prepare` or `before_apply` stage.

### 4.4 Step 1: what did the batch invalidate?

`resume()` runs right after the commit. The scaffold already grows the levels for new vertices
and copies the old levels into `ws.before`; add one line at its start:

```cpp
  applied_ = &applied;  // what the commit did: the insertions and deletions the batch requested
```

(and give the parameter its name: `const applied& applied`). Now the first hook; put its
definition in `my_bfs.cpp` next to the other hooks (before `loop`, say), like every definition
below. Deleting
`u -> v` can make `v` farther only if the edge was on a shortest path
(`before[v] == before[u] + 1`), and then every vertex reached from `v` along shortest-path edges
may be farther too. Those are *invalidated*: their level becomes -1 and they join the queue.
Every vertex outside that set keeps a shortest path that the batch did not touch, so its level is
still right.

```{literalinclude} ../../examples/tutorial_algorithms/my_bfs/my_bfs.cpp
:language: cpp
:start-after: "// [tutorial: identify_affected]"
:end-before: "// [tutorial: identify_affected end]"
```

`applied.delta` lists the insertions and deletions the batch *requested* (without self-loops; for
an undirected graph, both directions). It is not the net change of the batch: a deletion may name
an edge that does not exist, and with `batch_semantics::deletions_first = false` a batch can
insert an edge and then delete it again, so the edge is listed as inserted but is not in the
graph. That is harmless for the deletions here (invalidating too much is safe), but the seed below
has to check its insertions. Algorithms that need the net change compute it from the two graphs
(`compute_structural_change`, as `triangle_delta` does). `graph_access::out_view` is the host CSR
of G_{t+1}, the graph after the batch.
Everything after the commit reads G_{t+1}: the framework gives this hook a `new_view`, and a hook
that subtracts on G_t would get an `old_view` (invariant I1).

### 4.5 Step 1: the first frontier

A vertex enters a frontier when its level drops. One helper does the offering:

```{literalinclude} ../../examples/tutorial_algorithms/my_bfs/my_bfs.cpp
:language: cpp
:start-after: "// [tutorial: offer]"
:end-before: "// [tutorial: offer end]"
```

The `seed` hook builds the first frontier: every invalidated vertex takes the best level that
its in-neighbours offer, and every inserted edge `u -> v` offers `level[u] + 1` to `v`. Because
`applied.delta` lists requested insertions, the seed first checks that `u -> v` is in G_{t+1}
(that `v` is in `u`'s out-row); without that check, a batch that inserts and deletes the same edge
would give `v` a level along an edge that does not exist:

```{literalinclude} ../../examples/tutorial_algorithms/my_bfs/my_bfs.cpp
:language: cpp
:start-after: "// [tutorial: seed]"
:end-before: "// [tutorial: seed end]"
```

### 4.6 Step 2: the loop

Replace the scaffold's `loop` (the recompute) with one round of offers:

```{literalinclude} ../../examples/tutorial_algorithms/my_bfs/my_bfs.cpp
:language: cpp
:start-after: "// [tutorial: loop]"
:end-before: "// [tutorial: loop end]"
```

Levels only drop in Step 2 and never below the true BFS level, so the rounds end, and the fixed
point is the BFS level of every vertex whatever order the offers come in.

### 4.7 compute(), finalize and a requirement

`compute()` runs the same problem through the *static* enactor: `reset`, `seed_static`, then
`loop` until the frontier is empty. The scaffold's static solve used to run in `loop`; move it
into `seed_static`, so that the frontier stays empty and `loop` does not run:

```{literalinclude} ../../examples/tutorial_algorithms/my_bfs/my_bfs.cpp
:language: cpp
:start-after: "// [tutorial: seed_static]"
:end-before: "// [tutorial: seed_static end]"
```

In `finalize`, delete the line `stats.fallback_used = true;`: the update no longer recomputes.
(`finalize` keeps the scaffold's count of the vertices whose level changed, `stats.affected`.)

Some of the scaffold's doc comments in `problem.hpp` now describe the placeholder: the comment at
the top of the file ("The placeholder ... so update() recomputes"), the one of `finalize` ("the
placeholder sets fallback_used") and the one of `seed_static` ("nothing to seed, the loop solves
from the source"). Update them to describe what the code does now; the reference solution shows one
way to word them.

Finally, the seed needs the in-edges, so reject graphs that do not store them before anything
changes. At the end of `begin_update` (which runs before the commit, so a rejected batch leaves the
graph and the result unchanged):

```{literalinclude} ../../examples/tutorial_algorithms/my_bfs/my_bfs.cpp
:language: cpp
:start-after: "// [tutorial: requirement]"
:end-before: "// [tutorial: requirement end]"
```

### 4.8 The hand test

The generated hand test (`cpp/tests/algorithms/my_bfs/my_bfs_test.cpp`) checks that the
placeholder recomputes. It no longer does:

```cpp
  EXPECT_FALSE(s.fallback_used);  // incremental: no recomputation
```

Replace the test's `TODO(my_bfs)` comment with one that says what it checks, and add a case for
the trap of 4.5, an edge inserted and deleted in one batch:

```{literalinclude} ../../examples/tutorial_algorithms/my_bfs/my_bfs_test.cpp
:language: cpp
:start-after: "// [tutorial: hand test]"
:end-before: "// [tutorial: hand test end]"
```

### 4.9 Build and run the kit

```bash
cmake --build --preset dev
ctest --preset dev -L my_bfs
```

`100% tests passed` again, and this time the update is incremental. Among the kit's checks:

- **C2** compares a chain of three updates with `compute()` on the new graph, on random graphs of
  three sizes under eight batch mixes (insertions only, deletions only, mixed, local, heavy,
  reweight, vertex growth, and `cancel`: edges inserted and deleted in the same batch) and three
  batch semantics (the default, the default with the insertions applied first, and sets);
- **C3** compares the OpenMP backend with the sequential one. For `my_bfs` it passes trivially:
  the scaffold's `openmp.cpp` calls the sequential solve and the hooks are serial, so both backends
  run the same code. It starts to matter once `openmp.cpp` has a parallel path of its own, as in
  `dynamic_bfs`;
- **C5** applies a batch and its inverse and expects the original levels;
- **C8** checks that, once the workspace is sized, an update allocates nothing;
- **C10** runs `my_bfs` and `sssp` on one graph in one `dyng::update(res, g, batch, r1, r2)` call
  and compares each with an update of its own.

If a test fails, compare your files with `examples/tutorial_algorithms/my_bfs/`.

## 5. Break it on purpose

See what the kit does with a wrong update. In `identify_affected`, stop the walk below the
deleted edges: change the condition in the second loop to `if (false && ...)`. Rebuild and run
`ctest --preset dev -L my_bfs`: C2, C5, C8 and the hand test fail. C2's message shows the levels
`compute()` expects and the levels the update chain produced, and its trace names the case: the
backend, the batch semantics, the seed (`seed 2000 (replay: DYNG_TEST_SEED=2000 ctest -R
<test>)`), the graph size, the batch mix and the batch. `DYNG_TEST_SEED=2000 ctest --preset dev
-R 'my_bfs.*C2'` replays that case alone while you debug. Undo the change before you go on.

## 6. Clean up, or keep going

To delete the experiment, remove the algorithm and configure the build again with every
algorithm:

```bash
python3 scripts/new_algorithm.py my_bfs --remove
cmake --preset dev -DDYNG_ALGORITHMS=all
```

The first command removes every file the scaffold wrote and its registrations (it refuses to
delete an algorithm whose manifest no longer has the scaffold line, which every finished algorithm
drops). The second resets the `DYNG_ALGORITHMS=my_bfs` of step 3, which the build directory keeps
until you change it (a configure that still names `my_bfs` fails, since it no longer exists).

To keep going, {doc}`../how_to/add_an_algorithm` lists the remaining steps of a real
contribution (the docs page, the Python bindings, the CLI, the benchmark, the citation, the pull
request). Two improvements that `dynamic_bfs` makes are worth reading first:

- **A BFS tree.** `my_bfs` invalidates every vertex below a deleted shortest-path edge, even when
  another shortest path survives. `dynamic_bfs` keeps a parent per vertex and invalidates only the
  subtree under a deleted *tree* edge (the `invalidate_subtree` operator), then
  repairs the parents of the vertices it touched with a deterministic rule (the lowest-id
  in-neighbour one level up), so every backend keeps the same tree.
- **Every backend from one source.** `dynamic_bfs` writes each pass once, as a small functor that
  does the work of one element, and runs it with an *executor* of the framework operators
  (`cpp/src/operators/execution.hpp`): a loop on the sequential backend, an OpenMP parallel loop,
  a CUDA kernel. Words that several elements may write in one pass change through atomics, so the
  same functor is correct on all three. The offer of `my_bfs` becomes:

```{literalinclude} ../../cpp/src/algorithms/dynamic_bfs/engine.hpp
:language: cpp
:start-at: "struct offer {"
:end-at: "};"
```

and each backend file is one instantiation:

```{literalinclude} ../../cpp/src/algorithms/dynamic_bfs/cuda.cu
:language: cpp
:start-at: "template <typename vertex_t, typename edge_t>"
:end-at: "}"
```

The aggregate-delta family works the same way with other hooks: `count` on the old graph before
the commit (what the deletions destroy) and on the new graph after it (what the insertions
create), with an ownership rule so that a pattern through several changed edges counts once.
{doc}`../algorithms/triangle_delta` walks through it.
