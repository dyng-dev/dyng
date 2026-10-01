# Read and write graph and batch files

The readers and writers are in `dyng::io` (`<dyng/dyng.hpp>` includes them). Every reader
validates its input and throws `dyng::io_error` with the path, line and column of the problem;
the formats are specified in {doc}`../api/file_formats`.

## A graph in MOSP's text CSR format

```cpp
using vertex = std::int32_t; using edge = std::int32_t; using weight = std::int32_t;
auto csr = dyng::io::read_csr_triplet<vertex, edge, weight>("data/graphCsr");  // RowPtr/ColInd/Values
auto g = dyng::graph<vertex, edge, weight>::from_csr(res, csr.view(),
                                                     dyng::graph_properties::mosp_compatible());
```

`graph_properties::mosp_compatible()` reproduces how the original applies batches (append row
order, parallel edges and self-loops kept); without it the graph uses dynG's defaults (ADR 0010).

## A Matrix Market file

```cpp
dyng::io::matrix_market_options options;   // weights: from the file, or seeded random weights
auto edges = dyng::io::read_matrix_market<vertex, weight>("roadNet-CA.mtx", options);
auto g = dyng::graph<vertex, edge, weight>::from_edges(res, edges.view());
```

With `options.weights = dyng::io::matrix_market_weights::random` and the default
`random_weights`, the edges, their order and their weights are bit-identical to
`mospPrep mtx2csr <in.mtx> <prefix> K 1 100 12345` of the original tools.

## A batch

```cpp
dyng::io::legacy_batch_options bopts;
bopts.num_weights = g.num_weights();
bopts.num_vertices = g.num_vertices();      // ids are checked against it
auto batch = dyng::io::read_legacy_batch<vertex, weight>("insert.txt", "delete.txt", bopts);
dyng::sssp::update(res, g, batch.view(), tree);
```

A sequence of batches in the library's text format (`.dgt`, {doc}`../api/file_formats`):

```cpp
dyng::io::batch_file_options fopts;
fopts.num_vertices = g.num_vertices();
for (const auto& b : dyng::io::read_batches<vertex, weight>("stream.dgt", fopts)) {
  dyng::sssp::update(res, g, b.view(), tree);
}
```

## Results

```cpp
dyng::io::write_distances("out/distances.txt", tree.distances());
dyng::io::write_parents("out/SSSPTree.txt", tree.parents());
```

The writers produce the same bytes as the original tools, which is what the parity harness
compares ({doc}`run_parity`).
