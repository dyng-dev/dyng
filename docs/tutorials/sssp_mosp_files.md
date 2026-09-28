# Dynamic shortest paths on MOSP files (C++)

In this tutorial you build and run `examples/cpp/sssp_update.cpp`. It reads a graph and a change
batch in the text formats of the original research code (MOSP-OpenMP), computes a shortest-path
tree, updates it with the batch, and writes the result in MOSP's own output format, so you can
compare it byte for byte with the original's output. It takes about ten minutes and needs no GPU.

## 1. Build

From a clone with the development environment active ({doc}`../getting_started/install`):

```bash
cmake --preset dev
cmake --build --preset dev --target dyng_example_sssp_update
```

## 2. The input

The test data has tiny cases in MOSP's formats ({doc}`../api/file_formats`). `testCase0` is a
graph with five vertices and one weight per edge, stored as three files with the prefix
`graphCsr` (`RowPtr`, `ColInd`, `Values`), plus a batch: `insert.txt` holds `0 2 2` (insert the
edge 0 -> 2 with weight 2, or change its weight if it exists) and `delete.txt` holds `3 4`
(delete the edge 3 -> 4).

## 3. Read the graph and compute the tree

The example picks the backend from its last argument and reads the CSR files into a graph with
MOSP's batch semantics (parallel edges and self-loops kept, append row order; ADR 0010):

```{literalinclude} ../../examples/cpp/sssp_update.cpp
:language: cpp
:start-at: const std::string backend
:end-at: auto tree = dyng::sssp::compute
```

`compute()` returns the **canonical** tree: every vertex's parent is its lowest-id in-neighbour
on a shortest path, on every backend.

## 4. Apply the batch

```{literalinclude} ../../examples/cpp/sssp_update.cpp
:language: cpp
:start-at: dyng::io::legacy_batch_options options
:end-at: res.attach_profiler(nullptr)
```

`update()` does the whole update: it normalizes the batch, applies it to `g` exactly once, finds
the vertices whose shortest paths may have changed, and repairs only those
({doc}`../concepts/update_model`). The attached `profiler` records one stage per step of that
template (`sssp.prepare`, `sssp.commit`, `sssp.identify_affected`, `sssp.seed`, `sssp.loop`,
`sssp.finalize`).

## 5. Write the result and compare

```{literalinclude} ../../examples/cpp/sssp_update.cpp
:language: cpp
:start-at: const std::string out = argv[4]
:end-at: prof.write_csv
```

Run it on both CPU backends and compare with the output of the original tool, which is stored
next to the case:

```bash
case=cpp/tests/data/mosp_graph_io/testCase0
expected=cpp/tests/data/mosp_sssp/testCase0/updated/obj0
for backend in sequential openmp; do
  mkdir -p out/$backend
  build/dev/examples/cpp/sssp_update $case/graphCsr $case/insert.txt $case/delete.txt out/$backend $backend
  cmp out/$backend/SSSPTreeUpdated.txt $expected/SSSPTreeUpdated.txt
  cmp out/$backend/distancesUpdated.txt $expected/distancesUpdated.txt
done
```

`cmp` prints nothing: the files are identical. CTest runs the same comparisons, both files on
every CPU backend (`ctest --preset dev -R example.sssp_update`).

With a GPU and a CUDA build (the `dev-cuda` preset, {doc}`../getting_started/install`), the
same program runs on the `cuda` backend: the graph and the tree then live in device memory, the
fused kernel of the original MOSP-CUDA code runs the update, and the output files are the same
bytes:

```bash
cmake --build --preset dev-cuda --target dyng_example_sssp_update
mkdir -p out/cuda
build/dev-cuda/examples/cpp/sssp_update $case/graphCsr $case/insert.txt $case/delete.txt out/cuda cuda
cmp out/cuda/SSSPTreeUpdated.txt $expected/SSSPTreeUpdated.txt
```

## What you learned

- One `resources` object selects the backend for every call.
- `compute()` gives the static result; `update()` applies a batch and repairs the result, and the
  result must match the graph's version.
- The profiler's stage names follow the update template, the same for every algorithm.

Next: {doc}`../how_to/choose_a_backend`, {doc}`../how_to/profile_an_update`, and the
{doc}`../algorithms/sssp` page for the details of the algorithm.
