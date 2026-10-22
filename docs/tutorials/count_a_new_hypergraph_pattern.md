# Count a new hypergraph pattern (an Op)

In this tutorial you count something on a hypergraph that dynG does not count yet, and keep the
count exact while hyperedges are deleted and inserted. You write it as an **Op**: a small struct
whose callbacks say what to count for a triangle of hyperedges, a pair of hyperedges or a single
hyperedge. `dyng::experimental::count_local_patterns<op_t>` runs your Op with the engine
`triad_count` uses (ESCHER's wedge engine), on the GPU and on the CPU, and gives you the full count
and the exact update under batches for free.

:::{warning}
`count_local_patterns` is **experimental** ({doc}`../developer/plan`, Section 5.9): its API may
change in any minor release, and the changes are listed in the changelog. It is C++ only and
needs nvcc: its header `<dyng/experimental/local_patterns.cuh>` holds the kernels that are
compiled for your Op. There is no Python counterpart ({doc}`../developer/python_gaps`).
:::

You should be comfortable with C++17 and know what a hypergraph is ({doc}`../concepts/hypergraphs`).
The code of this tutorial is in `examples/tutorial_algorithms/local_patterns/`: the Ops
(`patterns.hpp`), a brute-force oracle (`oracle.hpp`) and the programs that run them
(`tutorial_test.cu` on every backend, `tutorial_host_test.cpp` without nvcc). The snippets below
are quoted from those files. The local GPU gate (`ci/gpu_local.sh`, run on a machine with nvcc and
a GPU) builds and runs `tutorial_test.cu` on every backend and both hypergraph types; hosted CI has
no nvcc, so it runs the host half (`tutorial_host_test.cpp`, which drives the Ops' callbacks by
hand against the brute force) and the drift check, which fails when a quoted line changes without
this page (`ci/tutorial_quotes.py`).

## 1. The idea: a pattern is an Op over wedges, pairs or keys

`triad_count` counts the **triads** of a hypergraph: three hyperedges that pairwise share a
vertex, which are the triangles of its line graph (the graph whose vertices are the hyperedges,
with an edge between two hyperedges that share a vertex). Its engine enumerates them by
**wedges**: a wedge `(a, b)` is a hyperedge `a` and one of its neighbours `b` in the line graph,
and every common neighbour `c` of `a` and `b` closes a triangle `{a, b, c}`. The same
enumeration gives every adjacent pair `{a, b}`, and the hyperedges themselves.

An Op picks one of these **shapes** and says what to count:

| `shape` | the engine calls | once per |
|---|---|---|
| `shape::triangles` | `begin_wedge(rows, a, b)`, then `triangle(rows, c, bins)` for each `c` | triangle `{a, b, c}` of the line graph |
| `shape::pairs` | `pair(rows, a, b, bins)` | pair of hyperedges that share a vertex |
| `shape::keys` | `key(rows, a, bins)` | live hyperedge |

and every worker (a GPU thread, a host source) ends with `flush(bins)`. `a`, `b` and `c` are
0-based hyperedge ids; `rows` gives the members of a hyperedge; `bins` is where the counts go: an
Op has `num_bins` unsigned 64-bit bins, and `bins.add(bin, value)` adds to one.

`triad_count`'s four patterns are four built-in Ops of this engine (the 30 h-motifs and StatHyper's
three types); a new pattern is a new Op, not a new `triad_count::pattern`.

## 2. A one-bin Op: the triangles of the line graph

The smallest triangles Op counts every triangle once:

% quote: examples/tutorial_algorithms/local_patterns/patterns.hpp#line_graph_triangles dc8c14593254
```{literalinclude} ../../examples/tutorial_algorithms/local_patterns/patterns.hpp
:language: cpp
:start-after: "// [tutorial: line_graph_triangles]"
:end-before: "// [tutorial: line_graph_triangles end]"
```

- `num_bins` and `shape` are `static constexpr`; `count_local_patterns` checks every part of the
  Op with its own `static_assert` message, so a missing or misspelled member is reported by name.
- The callbacks are `DYNG_HD` (`__host__ __device__` under nvcc): the same Op runs on the GPU and
  on the host. The ones that receive `bins` are templates, because the GPU and the host hand the
  Op different bins handles.
- Write every callback `DYNG_HD`, never `__device__` alone. nvcc reports a host-only callback, but
  **not a `__device__`-only one**: the library calls the host callbacks from a template, where nvcc
  compiles such a call to a stub, so the program compiles, counts on the GPU, and on `sequential`
  and `openmp` ends with exit status 1 and no message. Put
  `DYNG_EXPERIMENTAL_CHECK_HOST_CALLBACKS(your_op);` after each Op, at namespace scope of the `.cu`
  file: it makes nvcc report a `__device__`-only callback ("calling a `__device__` function ...
  from a `__host__` function is not allowed") and adds no code that runs. The tutorial's test does
  it for its four Ops:

% quote: examples/tutorial_algorithms/local_patterns/tutorial_test.cu#host check 5209cf030820
```{literalinclude} ../../examples/tutorial_algorithms/local_patterns/tutorial_test.cu
:language: cpp
:start-after: "// [tutorial: host check]"
:end-before: "// [tutorial: host check end]"
```

- `count` is a partial sum in the Op's own member; `flush()` adds it to the bin once per worker.
  The Op must be trivially copyable and default-constructible: the engine copies the Op value you
  give it (here the default, `count = 0`) into every worker.

Count it, and a pattern of the next section, on a hypergraph:

% quote: examples/tutorial_algorithms/local_patterns/tutorial_test.cu#compute aa19cb0ceab3
```{literalinclude} ../../examples/tutorial_algorithms/local_patterns/tutorial_test.cu
:language: cpp
:start-after: "// [tutorial: compute]"
:end-before: "// [tutorial: compute end]"
```

The result holds `num_bins` counts in host memory (`counts()`, `count(bin)`, `total()`), the Op
value it was computed with (`op()`), and the hypergraph version it matches. The triangles of the
line graph are exactly the triads `triad_count` classifies into its 30 h-motifs, so the one bin
equals `triad_count`'s `total()`, and a brute force over every triple of hyperedges agrees:

% quote: examples/tutorial_algorithms/local_patterns/tutorial_test.cu#check 92abc482e0fe
```{literalinclude} ../../examples/tutorial_algorithms/local_patterns/tutorial_test.cu
:language: cpp
:start-after: "// [tutorial: check]"
:end-before: "// [tutorial: check end]"
```

## 3. A pattern `triad_count` does not have

A pairs Op that counts the pairs of hyperedges sharing at least `k` vertices: the threshold is a
member of the Op, set by the value you pass to `compute()` (`strong_pairs{2}` above) and kept by
the result for every update.

% quote: examples/tutorial_algorithms/local_patterns/patterns.hpp#strong_pairs 13c03619acae
```{literalinclude} ../../examples/tutorial_algorithms/local_patterns/patterns.hpp
:language: cpp
:start-after: "// [tutorial: strong_pairs]"
:end-before: "// [tutorial: strong_pairs end]"
```

`rows.row(id)` is a `member_span`: the members of a hyperedge as **opaque, ascending codes**. The
set helpers `intersect_size()`, `venn3()` and `intersects3()` merge the codes directly, without
decoding them, so the hot loop reads the store's own rows. Every Op is checked against a brute
force on the host (`oracle.hpp`):

% quote: examples/tutorial_algorithms/local_patterns/oracle.hpp#oracle 1310e6c44915
```{literalinclude} ../../examples/tutorial_algorithms/local_patterns/oracle.hpp
:language: cpp
:start-after: "// [tutorial: oracle]"
:end-before: "// [tutorial: oracle end]"
```

Several bins: a triangles Op that sorts the triangles by the size of their core (the vertices
all three hyperedges share), with `venn3()`. It remembers the wedge in `begin_wedge()` and adds to
a bin directly in `triangle()`, so its `flush()` is empty:

% quote: examples/tutorial_algorithms/local_patterns/patterns.hpp#core_triads 753d0e633b11
```{literalinclude} ../../examples/tutorial_algorithms/local_patterns/patterns.hpp
:language: cpp
:start-after: "// [tutorial: core_triads]"
:end-before: "// [tutorial: core_triads end]"
```

When an Op needs a vertex id, `rows.decode(code)` gives it. The pairs of hyperedges that both
contain a given vertex:

% quote: examples/tutorial_algorithms/local_patterns/patterns.hpp#hub_pairs 2f8f73ee3435
```{literalinclude} ../../examples/tutorial_algorithms/local_patterns/patterns.hpp
:language: cpp
:start-after: "// [tutorial: hub_pairs]"
:end-before: "// [tutorial: hub_pairs end]"
```

## 4. Updates under hyperedge batches, and the stats

`count_local_patterns<op_t>::update(res, h, batch, result)` applies a batch of hyperedge
deletions and insertions to the hypergraph and changes the counts by what the batch destroyed and
created, without counting the rest again. A result composes with every other hypergraph result in
one `dyng::update()`, so one commit updates several patterns (and `triad_count`) together:

% quote: examples/tutorial_algorithms/local_patterns/tutorial_test.cu#update a0e3120d0ae6
```{literalinclude} ../../examples/tutorial_algorithms/local_patterns/tutorial_test.cu
:language: cpp
:start-after: "// [tutorial: update]"
:end-before: "// [tutorial: update end]"
```

The stats (`experimental::local_patterns_stats`) hold the batch summary (`batch.inserted_ids`:
the ids the store gave the inserted hyperedges), the frontier sizes (`deleted_frontier`,
`inserted_frontier`), `removed` and `added` (the two phases' counts, summed over the bins) and
`delta`, the change of every bin. The test updates all four Ops (the other two, with
`triad_count`, in a second `dyng::update()` over a copy of the hypergraph), under both id-reuse
orders (`hypergraph_properties::reuse`: `erase_first` and `insert_first`), through an empty, a
deletion-only and an insertion-only batch and three mixed ones. After every batch it checks every
updated result against the brute force and against a fresh `compute()` ("baseline + deltas ==
recount"). **Test every new Op this way, under updates, on every backend you claim**: a check of
`compute()` alone cannot catch an Op that depends on the order of its ids (section 6). A result
that was not part of an update is stale: using it on the new hypergraph throws
`stale_result_error`, and it must be computed again.

## 5. The same Op on the host and on the GPU

The Op is compiled for both: run it on the GPU with `dyng::resources::cuda()` and on the CPU with
`sequential()` or `openmp()`, with identical counts:

% quote: examples/tutorial_algorithms/local_patterns/tutorial_test.cu#backends 282cee499579
```{literalinclude} ../../examples/tutorial_algorithms/local_patterns/tutorial_test.cu
:language: cpp
:start-after: "// [tutorial: backends]"
:end-before: "// [tutorial: backends end]"
```

In your own project, put the Op and the calls in a `.cu` file and build it with nvcc against the
installed library:

% quote: ci/consumer/op.CMakeLists.txt#op consumer cb2e261b55c8
```{literalinclude} ../../ci/consumer/op.CMakeLists.txt
:language: cmake
:start-after: "# [tutorial: op consumer]"
:end-before: "# [tutorial: op consumer end]"
```

(`ci/cmake_consumer.sh` builds this file, unchanged, against an installed build with a user Op as
`my_patterns.cu`, and runs it on every backend of the build; in CI without nvcc it is skipped.) A
program that only reads the rows (`experimental::access_rows()`) or names the types includes the
host header `<dyng/experimental/local_patterns.hpp>` and needs no nvcc.

The same Op value runs on every backend because it holds only scalars. An Op that needs data of
your own (vertex labels, a vertex set) gets it through a pointer member you set: on `cuda` it must
point to device memory, on `sequential` and `openmp` to host memory, and the data must stay valid
and unchanged for the result's whole lifetime, because the result keeps the Op value and every
`update()` counts with it. Such an Op value belongs to one backend.

## 6. Why the Op never sees the frontier or the sign

An update counts in two phases: on the hypergraph **before** the batch, everything through the
deleted hyperedges is counted and subtracted; the batch is applied once; on the hypergraph
**after** it, everything through the inserted hyperedges is counted and added. A triangle through
two changed hyperedges would be counted twice, so the engine counts it once, at its smallest
changed hyperedge (the ownership rule `ownership::min_member` of the framework): from a changed
hyperedge `f` it skips every wedge and every closing hyperedge that is a smaller changed one.

All of this is the engine's: the frontier (the changed hyperedges), the skips, the sign of each
phase and the bins' arithmetic. The Op only says what one triangle (pair, hyperedge) is worth.
That is why an Op written for `compute()` is already correct for `update()`, and why it cannot get
the update wrong: it holds no view of the hypergraph, so the subtraction cannot read the
hypergraph after the batch. What the Op must guarantee is that a structure's count does not depend
on who counts it or in which order: the engine may hand a triangle to any worker.

**The ids of a call come in no particular order.** A full count happens to pass `a < b` to
`pair()` and `begin_wedge()`, and `a < b < c` for a triangle, because it walks each hyperedge's
larger neighbours. An update's phases start from the changed hyperedges instead: `a` is a changed
hyperedge and `b` any neighbour, smaller or larger (and `c` any common neighbour above `b`). So an
Op's count must be **symmetric in `a` and `b`** (pairs) and **in `a`, `b` and `c`** (triangles).
This Op is not:

% quote: examples/tutorial_algorithms/local_patterns/tutorial_test.cu#asymmetric 81dded2cbe11
```{literalinclude} ../../examples/tutorial_algorithms/local_patterns/tutorial_test.cu
:language: cpp
:start-after: "// [tutorial: asymmetric]"
:end-before: "// [tutorial: asymmetric end]"
```

On the hyperedges `{0, 1, 2}`, `{2, 3}` and `{3, 4, 5, 6}` its `compute()` gives 1, as a brute
force over every pair `i < j` does. Deleting hyperedge 2 subtracts the pair counted from the
changed hyperedge as `pair(2, 1)`, which this Op counts (|2| > |1|) although the full count did
not (`pair(1, 2)`): `update()` gives 0, a recount gives 1, and nothing is reported. The test
`AnOpThatDependsOnTheOrderOfItsIdsIsWrongAfterAnUpdate` shows it on every backend. A symmetric
version counts the pairs of different sizes, `rows.size(a) != rows.size(b)`. (A bin that the
subtraction would take below zero is reported as an `internal_error`, but an asymmetric Op need
not take one there.)

## 7. Limits

- **`num_bins` is at most `experimental::max_pattern_bins` (1024)**: the bins of a GPU block live
  in its shared memory (8 KiB at 1024 bins), and those of a host thread on its stack. A bin
  outside `[0, num_bins)` is not written: `compute()` or `update()` throws
  `invalid_argument_error` (after the commit, the result is left unusable).
- **The Op is at most `experimental::max_op_size` (4,096) bytes**: its value is a parameter of
  every counting kernel and every GPU thread holds a copy of it, so it is small per-worker state,
  not a table (a larger Op stops at its own `static_assert`). Keep tables in memory the Op points
  to (section 5).
- **Data of your own** comes through a pointer member, into the memory of the result's backend,
  valid for the result's lifetime (section 5); a pointer-free Op is the only kind that runs on
  every backend unchanged.
- **`__device__`-only callbacks are not reported by nvcc** unless the Op goes through
  `DYNG_EXPERIMENTAL_CHECK_HOST_CALLBACKS` (section 2); without it the host backends end the
  program with exit status 1.
- **C++ and nvcc only**, by design: the Op is compiled into kernels. No Python binding
  ({doc}`../developer/python_gaps`).
- **Experimental**: the API may change in any minor release; `ci/api_snapshot.py` tracks it and
  reports changes without failing on them.
- **Closed patterns only**: the engine enumerates the triangles, edges and vertices of the line
  graph. A triad with a pair that shares nothing (the open h-motifs, bins 20-25 of
  `triad_count`, always 0) is not a triangle of the line graph, so no Op of this engine sees it.
- **The members' codes end at INT_MAX**: the set helpers merge rows up to `INT_MAX`, so every Op
  is refused on a hypergraph with 2,147,483,647 vertices (the vertex ids up to 2^31 - 3 are
  supported).
- **Types**: `hypergraph<int32_t, unweighted>` and `hypergraph<int32_t, int32_t>` (the weights are
  not passed to the Op). The shapes `triangles` and `pairs` need the line graph and the
  vertex-to-hyperedge rows (the default properties); `keys` needs neither.
- **The counts are unsigned 64-bit sums**: a count past 2^64 - 1 throws `capacity_error` (an Op
  that computes a product can report its own overflow with `bins.overflow()`).

## 8. Run the tutorial's code

In a CUDA build of the repository:

```bash
cmake --preset dev-cuda && cmake --build --preset dev-cuda
ctest --preset dev-cuda -R 'Tutorial'      # the Ops on every backend, against the brute force
```

and without nvcc, the host half (the Ops' callbacks driven by hand on the host rows):

```bash
cmake --preset cpu-only && cmake --build --preset cpu-only
ctest --preset cpu-only -R 'TutorialHost'
```

The ADR that introduced the extension point is {doc}`../adr/0058-experimental-local-patterns`;
the API reference is {doc}`../api/cpp/experimental`. A new Op makes a good first contribution:
the label `good-first-issue:op` marks such issues ({doc}`../developer/labels`).
