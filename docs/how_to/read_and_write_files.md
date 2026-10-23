# Read and write graph, hypergraph and batch files

The readers and writers are in `dyng::io` (`<dyng/dyng.hpp>` includes them). Every reader
validates its input and throws `dyng::io_error` with the path, line and column of the problem;
the formats are specified in {doc}`../api/file_formats`. Graphs and their batches come first;
hypergraphs, their batches and triad counts are in the second half
({ref}`how-to-hypergraph-files`). From Python the same readers are in `dyng.io`
({doc}`use_hypergraphs_from_python` for the hypergraph ones).

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

(how-to-hypergraph-files)=
## A hypergraph (`.hg`)

```cpp
using hypergraph_t = dyng::hypergraph<std::int32_t, dyng::unweighted>;
auto ro = dyng::io::hypergraph_read_options::escher_gpu();  // ESCHER-GPU's loader: ids 1..n kept
ro.dedupe_hyperedges = true;                                 // and repeated member sets dropped
dyng::io::hypergraph_file_info info;                         // optional: what the reader dropped
auto hes = dyng::io::read_hypergraph<std::int32_t, dyng::unweighted>("coauth-DBLP.hg", ro, &info);
dyng::hypergraph_properties hp;
hp.line_graph = dyng::line_graph_policy::sorted_merge;       // for h_motif and types 2-3
hypergraph_t h = hypergraph_t::from_hyperedges(res, hes.view(), hp);
dyng::io::write_hypergraph("out.hg", h.to_hyperedge_list(res).view());  // live, in id order
```

`hypergraph_read_options::hsosp()` reads a file as H-SOSP does (at most 25 members, vertices by
first appearance, seeded weights U[1, 100]; a weighted `weight_t`); the fields (`index_base`,
`dedupe_hyperedges`, `max_cardinality`, `relabel`, `random_weights`, `num_vertices`) override
an option set. A weighted file (`v1 v2 .. ; w`) needs a weighted `weight_t`. The reader refuses a
`.dgb` or `.dgt` batch file and a Matrix Market file passed by mistake, naming the format.

## Hyperedge batches (`.dgt` and `.dgb`)

```cpp
dyng::io::hyperedge_batch_file_options bo;
bo.num_vertices = h.num_vertices();       // members are checked against the fixed vertex set
for (const auto& b : dyng::io::read_hyperedge_batches<std::int32_t, dyng::unweighted>(
         "batches.dgt", bo)) {             // text: "+h v1 v2 ..", "-h id", "%batch n"
  h.apply(res, b.view());
}
dyng::io::batch_reader reader("batches.dgb");                  // binary, streamed, CRC-checked
while (reader.next()) {
  h.apply(res, reader.hyperedge_batch<std::int32_t, dyng::unweighted>());
}
```

`write_hyperedge_batches()` and `write_binary_hyperedge_batches()` write them. A `.dgb` record can
also carry the members of the deleted hyperedges and the ids its insertions are expected to get
(`reader.annotations<std::int32_t>()`), which the parity harness checks.

## Triad counts (the triad-bins JSON)

```cpp
dyng::triad_count::result motifs = dyng::triad_count::compute(res, h);
dyng::io::triad_bins bins;
bins.kind = std::string(dyng::triad_count::to_string(motifs.kind()));   // "h_motif"
bins.graph_version = motifs.graph_version();
const auto counts = motifs.counts();                                  // a host array
bins.counts.assign(counts.begin(), counts.end());
dyng::io::write_triad_bins_json("motifs.json", bins);   // format "dyng.triad_bins", version 1
const dyng::io::triad_bins back = dyng::io::read_triad_bins_json("motifs.json");
```

`bins.deltas` may hold one row of `num_bins` values per batch (a `stats::delta` each). The
{doc}`../tutorials/hypergraph_triads` tutorial runs these steps on a small file, in C++ and
Python.
