# Shortest hyperpaths

In this tutorial you compute the **shortest hyperpaths** of a weighted hypergraph from a source
vertex, keep them exact while hyperedges are deleted and inserted and vertices leave and join
hyperedges, and read the distances of the hyperedges and the vertices, the paths and what each
update did. You do it in C++ and in Python, with the same input and the same output. It takes
about 20 minutes.

You need dynG 0.4 or newer installed ({doc}`../getting_started/install`): for C++ a compiler and
CMake, for Python `pip install "dyng>=0.4"` (with a CUDA plugin, upgrade both:
`pip install -U "dyng[cu13]"`). You should know what a hypergraph is
({doc}`../concepts/hypergraphs`). The algorithm is {doc}`../algorithms/hyper_sssp`; this page shows
how to use it.

The code is two example programs, `examples/cpp/hyper_sssp_update.cpp` and
`examples/python/hyper_sssp.py`, their input `examples/data/hyper_sssp.hg` and
`examples/data/hyper_sssp.dgt`, and two tests that hold the rest of the page's code,
`cpp/tests/algorithms/hyper_sssp/hyper_sssp_tutorial_test.cpp` and
`python/tests/test_shortest_hyperpaths_tutorial.py`. CI builds the C++ program against the
installed package and runs both programs on the sequential backend (the local GPU gate,
`ci/gpu_local.sh`, also on cuda), and compares what they print with
`examples/data/hyper_sssp_update.expected.txt`; a test rebuilds that file from the oracle,
MOSP_ESCHER's test oracle `dyng.testing.hyper_sssp_reference`, after every batch
(`python/tests/test_examples.py`). The snippets below are quoted from those files, and the
documentation check fails when a quoted line changes without this page
(`ci/tutorial_quotes.py`). Clone the repository and run the commands from its root:

```bash
git clone --branch v0.4.0 https://github.com/dyng-dev/dyng.git
cd dyng
```

## 1. The problem in one paragraph

The **line graph** of a hypergraph has one node per live hyperedge, and two hyperedges are
adjacent when they share a vertex. Entering hyperedge h costs its weight w(h). From a source
vertex s, the distance D(h) of a hyperedge is the cost of the cheapest walk of overlapping
hyperedges h_1, ..., h_k = h with s in h_1: D(h) = w(h_1) + ... + w(h_k), so a hyperedge that
holds s has D(h) = w(h). Its parent is the hyperedge before it on such a walk (the lowest id among
equals), -1 when it holds s. The distance of a **vertex** is the smallest D(h) over the hyperedges
that hold it, and 0 for s.

## 2. The input: a weighted `.hg` file and a batch file

A `.hg` file has one hyperedge per line; `; w` ends a line with its weight. The example's 9
hyperedges over the vertices 1-12 (ids 1..n in the file become the vertices 0..n-1):

% quote: examples/data/hyper_sssp.hg#L6-L14 0acc5eac2b80
```{literalinclude} ../../examples/data/hyper_sssp.hg
:language: text
:lines: 6-14
```

The batches are a `.dgt` file ({doc}`../api/file_formats`), with dynG's 0-based ids: `-h id`
deletes a hyperedge, `+h v1 v2 ... ; w` inserts one, and the **incidence edits** `-i h v` and
`+i h v` remove vertex v from, or add it to, the live hyperedge h:

% quote: examples/data/hyper_sssp.dgt#L8-L18 1c6e92ca0f2b
```{literalinclude} ../../examples/data/hyper_sssp.dgt
:language: text
:lines: 8-18
```

Inside a batch the order is fixed, whatever the order of the lines: the hyperedge deletions,
then the incidence removals, the incidence additions, and the insertions last
({doc}`../concepts/hypergraphs`). An edit that cannot apply (a dead hyperedge, a vertex that is
not a member, the last member of a hyperedge) is skipped and counted.

## 3. Build the hypergraph (C++)

% quote: examples/cpp/hyper_sssp_update.cpp#read 58539031b18d
```{literalinclude} ../../examples/cpp/hyper_sssp_update.cpp
:language: cpp
:start-after: "// [tutorial: read]"
:end-before: "// [tutorial: read end]"
```

- `read_hypergraph<std::int32_t, std::int32_t>` reads the weights (`; w`): hyper_sssp needs
  integer weights in [1, 2^28].
- `hypergraph_storage::slack_csr` is the storage of incidence edits and of hyper_sssp (the ESCHER
  store, `escher`, keeps its rows sorted for the triad counts and refuses incidence edits), and
  `line_graph_policy::slack_csr` keeps its line graph up to date under every batch.

## 4. Compute from two sources (C++)

% quote: examples/cpp/hyper_sssp_update.cpp#compute c8116afa5b98
```{literalinclude} ../../examples/cpp/hyper_sssp_update.cpp
:language: cpp
:start-after: "// [tutorial: compute]"
:end-before: "// [tutorial: compute end]"
```

A result belongs to one source vertex. `hyperedge_distances()` and `hyperedge_parents()` are its
arrays by hyperedge id (device memory on cuda: the program copies them with `to_vector()`), and
`distance_to(v)` is the distance of a vertex. The program prints every reached hyperedge as
`id:D/parent` and the distance of vertex 11:

% quote: examples/data/hyper_sssp_update.expected.txt#L1-L3 137df216d0bd
```{literalinclude} ../../examples/data/hyper_sssp_update.expected.txt
:language: text
:lines: 1-3
```

From vertex 0, hyperedge 0 = {0, 1, 2} holds the source, so D = w = 4 and its parent is -1;
hyperedge 1 = {2, 3} overlaps it, D = 4 + 2 = 6, parent 0; and so on. Vertex 11 is in hyperedges
7 (D = 21) and 8 (D = 21), so its distance is 21.

`options::work_budget` (here 0.5 for the second result) bounds the work of an update, as a
multiple of the line graph's live entries: past it the update recomputes from scratch (exact
either way; section 7).

## 5. Apply the batches to both results at once (C++)

% quote: examples/cpp/hyper_sssp_update.cpp#update 3d35eb600668
```{literalinclude} ../../examples/cpp/hyper_sssp_update.cpp
:language: cpp
:start-after: "// [tutorial: update]"
:end-before: "// [tutorial: update end]"
```

- `dyng::update(res, hg, batch, a, c)` applies the batch to the hypergraph **once** and brings
  both results up to date; it returns one `hyper_sssp::stats` per result. (One result alone is
  `hyper_sssp::update(res, hg, batch, a)`; the other would then be stale.)
- `stats::batch` is what the batch did to the hypergraph (the same for both results):
  `deleted_hyperedges`, `inserted_hyperedges` and `inserted_ids`, `deleted_incidences` and
  `inserted_incidences`, `skipped_ops`. `line_graph_deletions` / `line_graph_insertions` are the
  pairs of overlapping hyperedges the batch removed and added, and `invalidated` the hyperedges
  whose path the update had to rebuild (each result its own).

The whole output:

% quote: examples/data/hyper_sssp_update.expected.txt#L1-L9 3b52b3e56991
```{literalinclude} ../../examples/data/hyper_sssp_update.expected.txt
:language: text
:lines: 1-9
```

Batch 0 deletes hyperedge 3, and its insertion {0, 11} gets id 3 back (the store reuses the slot;
read the ids from `inserted_ids`, never predict them). The new hyperedge 3 holds vertex 0, so from
vertex 0 the vertex 11 is now at distance 6. After every update the results equal a fresh
`compute()` on the new hypergraph, bit for bit, on every backend.

Build and run it against your installed dynG (the CMake block of {doc}`../getting_started/install`,
with `add_executable(hyper_sssp_update hyper_sssp_update.cpp)`):

```console
$ ./hyper_sssp_update examples/data/hyper_sssp.hg examples/data/hyper_sssp.dgt sequential
```

The last argument is the backend: `sequential` (the default) or `cuda` (the GPU of
`CUDA_VISIBLE_DEVICES`; the program exits with 77 if there is none); the output is the same.
hyper_sssp has **no OpenMP backend in 0.4**: OpenMP resources raise `not_supported_error`, whose
message names the fix.

## 6. A batch in code, vertex distances and a path (C++)

The rest of the C++ code is in `hyper_sssp_tutorial_test.cpp`, on the same hypergraph. A batch
built in code, the first batch of the file:

% quote: cpp/tests/algorithms/hyper_sssp/hyper_sssp_tutorial_test.cpp#batch 8ee1d3282ece
```{literalinclude} ../../cpp/tests/algorithms/hyper_sssp/hyper_sssp_tutorial_test.cpp
:language: cpp
:start-after: "// [tutorial: batch]"
:end-before: "// [tutorial: batch end]"
```

The distance of every vertex is computed on request, once per hypergraph version (a host array):

% quote: cpp/tests/algorithms/hyper_sssp/hyper_sssp_tutorial_test.cpp#vertex distances b62e29a6fb07
```{literalinclude} ../../cpp/tests/algorithms/hyper_sssp/hyper_sssp_tutorial_test.cpp
:language: cpp
:start-after: "// [tutorial: vertex distances]"
:end-before: "// [tutorial: vertex distances end]"
```

A shortest hyperpath is the chain of parents back to a hyperedge that holds the source:

% quote: cpp/tests/algorithms/hyper_sssp/hyper_sssp_tutorial_test.cpp#path 27def46a267b
```{literalinclude} ../../cpp/tests/algorithms/hyper_sssp/hyper_sssp_tutorial_test.cpp
:language: cpp
:start-after: "// [tutorial: path]"
:end-before: "// [tutorial: path end]"
```

## 7. Tune the work budget

An update rebuilds the paths through what the batch changed; when a batch changes much of the
hypergraph, recomputing is cheaper. `stats::relaxations` counts the work an update did (the
neighbours its pulls and pushes read), and `options::work_budget` caps it: before each push round
(cuda) or expansion (sequential), an update whose relaxations exceed `work_budget` times the line
graph's live entries recomputes from scratch (`stats::fallback_used`,
`fallback_reason == work_budget`), or raises `convergence_error` under
`on_budget = budget_action::error`. The work before the first check (the invalidation and the
seed's pulls) is never refused, so even a budget of 0 lets an update whose work all lies in its
seed complete; 0 acts at the first push round or expansion. The result is exact either way:

% quote: cpp/tests/algorithms/hyper_sssp/hyper_sssp_tutorial_test.cpp#budget 72860df1978e
```{literalinclude} ../../cpp/tests/algorithms/hyper_sssp/hyper_sssp_tutorial_test.cpp
:language: cpp
:start-after: "// [tutorial: budget]"
:end-before: "// [tutorial: budget end]"
```

To tune it, run your batches with `work_budget = std::numeric_limits<double>::infinity()` (never
fall back), time the updates and a `compute()`, and find the batch size where the update stops
being faster than the recompute. The budget is a multiple of the line graph's live entries, not a
count of relaxations: set it near that batch's `stats::relaxations` divided by the live entries,
which is the size of the line graph's column array (`hg.h2h().num_entries()` in C++,
`len(hg.h2h().col_ind)` in Python). The default 1.0, MOSP_ESCHER's, means about one pass over the
line graph. On cuda, a batch close to its budget may fall back in one run and not in another
(the relaxations of a parallel round depend on the schedule); the results are the same.

## 8. Check against the oracle

`dyng::testing::hyper_sssp_reference` and `check_hyper_sssp_tree` are MOSP_ESCHER's test oracle (a
Dijkstra over the line graph rebuilt from the hyperedges; it shares no code with hyper_sssp and
the container), for tests of your own:

% quote: cpp/tests/algorithms/hyper_sssp/hyper_sssp_tutorial_test.cpp#oracle 499adab546f6
```{literalinclude} ../../cpp/tests/algorithms/hyper_sssp/hyper_sssp_tutorial_test.cpp
:language: cpp
:start-after: "// [tutorial: oracle]"
:end-before: "// [tutorial: oracle end]"
```

Link `dyng::testing` for it (`target_link_libraries(... dyng::dyng dyng::testing)`).

## 9. The same in Python

The example program, with explicit resources (the CPU wheel's default resources are OpenMP's,
which hyper_sssp does not run on in 0.4):

% quote: examples/python/hyper_sssp.py#read 6c090251cee1
```{literalinclude} ../../examples/python/hyper_sssp.py
:language: python
:start-after: "# [tutorial: read]"
:end-before: "# [tutorial: read end]"
```

% quote: examples/python/hyper_sssp.py#compute ff87404820bf
```{literalinclude} ../../examples/python/hyper_sssp.py
:language: python
:start-after: "# [tutorial: compute]"
:end-before: "# [tutorial: compute end]"
```

% quote: examples/python/hyper_sssp.py#update 1d3256224aba
```{literalinclude} ../../examples/python/hyper_sssp.py
:language: python
:start-after: "# [tutorial: update]"
:end-before: "# [tutorial: update end]"
```

```console
$ python examples/python/hyper_sssp.py examples/data/hyper_sssp.hg examples/data/hyper_sssp.dgt
```

prints the nine lines above, byte for byte. The rest, from
`python/tests/test_shortest_hyperpaths_tutorial.py`:

% quote: python/tests/test_shortest_hyperpaths_tutorial.py#batch 56ee74166629
```{literalinclude} ../../python/tests/test_shortest_hyperpaths_tutorial.py
:language: python
:start-after: "# [tutorial: batch]"
:end-before: "# [tutorial: batch end]"
```

% quote: python/tests/test_shortest_hyperpaths_tutorial.py#vertex distances c108ec400ecc
```{literalinclude} ../../python/tests/test_shortest_hyperpaths_tutorial.py
:language: python
:start-after: "# [tutorial: vertex distances]"
:end-before: "# [tutorial: vertex distances end]"
```

% quote: python/tests/test_shortest_hyperpaths_tutorial.py#path a814d7de7b37
```{literalinclude} ../../python/tests/test_shortest_hyperpaths_tutorial.py
:language: python
:start-after: "# [tutorial: path]"
:end-before: "# [tutorial: path end]"
```

% quote: python/tests/test_shortest_hyperpaths_tutorial.py#budget 0d80f5ae2848
```{literalinclude} ../../python/tests/test_shortest_hyperpaths_tutorial.py
:language: python
:start-after: "# [tutorial: budget]"
:end-before: "# [tutorial: budget end]"
```

% quote: python/tests/test_shortest_hyperpaths_tutorial.py#oracle 848f024c6c38
```{literalinclude} ../../python/tests/test_shortest_hyperpaths_tutorial.py
:language: python
:start-after: "# [tutorial: oracle]"
:end-before: "# [tutorial: oracle end]"
```

The Python names are the C++ ones: `Options`, `Stats`, `Result`; the arrays are read-only
`dyng.Array` views (`.to_numpy()` copies them), and `dyng.update()` takes several results and
returns their stats as a tuple.

## 10. From the command line

`dyng hyper_sssp` does the same without a program and writes the distances and parents by
hyperedge id in sssp's file formats ({doc}`../api/cli`):

```console
$ dyng hyper_sssp update --input examples/data/hyper_sssp.hg --batches examples/data/hyper_sssp.dgt \
      --source 0 --out sp --vertex-distances --compare-recompute
```

writes `sp/hyperedge_distances.txt`, `sp/hyperedge_parents.txt` and `sp/vertex_distances.txt`, and
prints one line per batch.

## Next

- {doc}`../algorithms/hyper_sssp`: the update step by step, the two CUDA engines, the gates
  against MOSP_ESCHER and what differs from the paper.
- {doc}`../concepts/hypergraphs`: the storages, the incidence edits and the ids.
