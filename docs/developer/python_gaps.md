# Python gaps

PLAN Section 5.4, rule 8: every stable C++ symbol has a Python counterpart, or it is listed here.
The Python API is described in {doc}`../adr/0011-python-bindings` (ADR 0011). This page lists what
the Python package does not bind, and why; each entry names the release that closes it, or
"by design". The hypergraph rows (0.3, M8b) are in their own table below.

## C++ symbols without a Python counterpart

| C++ | Python in 0.1 | Why, and when |
|---|---|---|
| `memory_resource_ref`, `resources::set_memory_resource()`, `default_*_memory_resource()` | none | custom allocators from Python need a callback design (RMM-style); after 0.2.0 (not in 0.2.0) |
| `buffer<T>`, `copy.hpp` (`copy()`, `copy_async()`) | NumPy / `dyng.Array` | owning device buffers matter with the CUDA plugins; after 0.2.0 (0.2.0 has device result arrays, ADR 0031) |
| `array_view<T>` | `dyng.Array` (outputs), any array (inputs) | by design: arrays are NumPy / DLPack objects |
| `stream_ref` | an integer handle, a CuPy / PyTorch stream, `__cuda_stream__` (`Resources.cuda(stream=)`) | by design |
| `set_log_sink()`, `log_message()`, `log_enabled()`, `parse_log_level()` | `dyng.set_log_level()`, `dyng.get_log_level()` | a sink calling Python from native threads needs care; the level covers users; after 0.2.0 (not in 0.2.0) |
| `profiler::begin_stage()`, `end_stage()`, `add_counter()`, `scoped_stage` | none (reading: `dyng.profile()`) | recording is for algorithm authors, who write C++ |
| `graph::view()` (the zero-copy CSR view) | `Graph.to_csr()` (a copy), `Graph.edges()` | a zero-copy view of a graph that the next batch may reallocate needs the Array staleness rule for graphs; after 0.2.0 (not in 0.2.0) |
| `graph(props)` (an empty graph) | `Graph.from_edges([], [], ...)` | by design |
| `csr<V,E,W>`, `edge_list<V,W>` and their views | `dyng.CSR`, `dyng.io.EdgeList` (NumPy arrays) | by design |
| `edge_batch::insert_edge()`, `delete_edge()`, `reserve()`, `clear()` (and the `initializer_list` overloads, ADR 0023 note 3) | `dyng.EdgeBatch(insert=..., delete=...)` from arrays | by design (a Python loop over single edges would be slow) |
| `update_each()` | `dyng.update(graph, batch, *results)` | by design (one call for any list) |
| `find_algorithm()` | `[a for a in dyng.algorithms() if a.name == ...]` | by design |
| `header_version()`, `library_version()` | `dyng.__version__` | by design |
| `to_string(...)` of the enumerations | enumerations are strings in Python | by design |
| `testing::edge_set_after_batch()` | `dyng.testing.oracles.GraphModel.apply()` (pure Python) | by design |
| `testing::check_sssp_tree()` on arrays (without a result) | `dyng.testing.check_sssp_tree(graph, result)` | after 0.2.0 (not in 0.2.0) |
| `dyng::update(res, g, batch, r...)` with `dynamic_bfs::result` or `triangle_delta::result` | `dyng.update()` accepts sssp, cycle_count and mosp results only; call `dyng.dynamic_bfs.update()` / `dyng.triangle_delta.update()` | by design: teaching algorithms (ADR 0033) |
| `experimental::count_local_patterns<op_t>`, `local_patterns_result<op_t>`, `local_patterns_stats`, the Op concept, `member_span`, `hyperedge_rows`, the set helpers, `experimental::access_rows()` (`<dyng/experimental/local_patterns.hpp>`, `.cuh`) | none | by design: C++ and nvcc only. An Op is compiled into the engine's kernels, so it cannot come from Python; the experimental tier is not bound (ADR 0058). The four built-in Ops are `dyng.triad_count`'s patterns |

## PLAN 5.4 features that are not in the 0.1 Python package

| Feature | State | When |
|---|---|---|
| CUDA: `Resources.cuda()`, `__cuda_array_interface__`, DLPack stream ordering (rule 4) | done in 0.2 (M6a; ADR 0030, 0031): `Resources.cuda()` with a CUDA plugin wheel (`pip install "dyng[cu13]"`), device result arrays with `__cuda_array_interface__`, DLPack >= 1.0 exports ordered on the consumer's stream, host copies; inputs in device memory are accepted | – |
| `copy=False` on inputs (rule 1: raise instead of copying) | inputs of another dtype or layout are converted with one copy; inputs in CUDA device memory are copied to the host once (graphs and batches are built from host arrays), and that copy is not yet subject to `Resources.copy_policy`; `Resources.copy_policy` governs the library's own host/device copies | 0.3 (with device-resident batches) |
| Inputs that only have `__cuda_array_interface__` (Numba device arrays) | inputs in device memory are read through DLPack (CuPy, PyTorch, JAX); `cupy.asarray(x)` wraps a Numba array without a copy | 0.3 |
| `dyng.interop.from_networkx()`, `from_scipy()`, `from_cudf()` (rule 6) | `Graph.from_csr(csr.indptr, csr.indices, csr.data)` covers SciPy | after 0.2.0 (not in 0.2.0) |
| `.dgb` binary batch files and the streaming `batch_reader` (PLAN 5.7) | done in 0.3 (M8b): `dyng.io.write_binary_batches()`, `read_binary_batches()` and the streaming `dyng.io.BatchReader` (graph and hypergraph batches, the annotations); the batches are host objects (see the hypergraph table) | – |
| `Graph.with_capacity()`, vertex insertions and deletions | `EdgeBatch(insert_vertices=...)` exists; applying it raises `NotSupportedError` | 0.4 |
| `dyng.Hypergraph`, `HyperedgeBatch` | done in 0.3 (M8b): `dyng.Hypergraph.from_hyperedges()` / `from_lists()`, `dyng.HyperedgeBatch`, `dyng.HypergraphProperties`, `dyng.CapacityPolicy` with its presets, `dyng.HyperApplySummary` (`inserted_ids` an int64 array), `dyng.io.read_hypergraph()` / `write_hypergraph()`, the `.dgt` hyperedge batches, the generators (legacy included) and the Hypothesis strategies | – |

## Hypergraph symbols without a Python counterpart (0.3, M8b)

Every stable C++ symbol of `<dyng/hypergraph/*.hpp>` (frozen at M8b, ADR 0047) and the tracked
hypergraph parts of `io`, `generators` and `testing` has a counterpart in `dyng.Hypergraph`,
`dyng.HyperedgeBatch`, `dyng.HypergraphProperties`, `dyng.CapacityPolicy`,
`dyng.HyperApplySummary`, `dyng.StorageStats`, `dyng.io`, `dyng.generators` and `dyng.testing`,
except:

| C++ | Python in 0.3 | Why, and when |
|---|---|---|
| `dyng::update(res, h, batch, results...)` over a hypergraph (`participant_of<hypergraph<V,W>>`, `hyper_update_participant`, `run_update`) | none: `dyng.update()` takes a `dyng.Graph`; `Hypergraph.apply()` applies a batch to the structure | there is no hypergraph algorithm before M9: `dyng.update()` over hypergraphs arrives with `triad_count` (0.3, M9), the first result type it can update |
| `hypergraph::h2v()`, `v2h()`, `h2h()` (zero-copy views in the container's memory space) | `Hypergraph.h2v()`, `v2h()`, `h2h()` return host copies (`dyng.Incidence`, `dyng.LineGraph`) | a view is invalidated by the next `apply()` (the rule of `graph::view()` above); zero-copy device views need the Array staleness rule for containers; after 0.3 |
| `line_graph_view::overlaps` | none | always empty under `hypergraph_storage::escher` (ADR 0041, section 7); with `slack_csr` (0.4, M11) |
| `line_graph_view::weights` | `Hypergraph.weights` (the same array: the weights by id) | by design |
| `hyperedge_batch::delete_hyperedge()`, `insert_hyperedge()`, `delete_incidence()`, `insert_incidence()`, `reserve()`, `clear()` (and the `initializer_list` overloads) | `dyng.HyperedgeBatch(delete_hyperedges=..., insert_hyperedges=..., insert_offsets=..., insert_members=..., delete_incidences=..., insert_incidences=...)` from arrays | by design (as `edge_batch`'s builders) |
| `hyperedge_list<V,W>`, `hyperedge_list::add_hyperedge()`, `hyperedge_list_view<V,W>`, `hyperedge_batch_view<V,W>` | `dyng.HyperedgeList` (NumPy arrays), `Hypergraph.from_lists()` | by design |
| `hypergraph<int32, W>` with another vertex type | int32 vertex ids only; int64 member arrays raise `NotSupportedError` unless `vertex_dtype="int32"` (checked) | the library instantiates (int32, unweighted) and (int32, int32) only (PLAN 4.4.3) |
| `io::batch_reader(res, path)` (device views staged through pinned buffers) | `dyng.io.BatchReader(path)` gives host batches | Python batches are host objects (`dyng.Hypergraph.apply()` stages them; the same rule as graph batches above); with device-resident batches (after 0.3) |
| `io::batch_writer` (record by record) | `dyng.io.write_binary_batches(path, batches, ...)` writes a whole list | a streaming writer from Python is not needed by the parity or benchmark tools; after 0.3 if asked |
| `io::edge_batch_file_header()`, `hyperedge_batch_file_header()`, `batch_file_flags` | `dyng.io.BatchFileHeader` (read side; the flags as `deleted_members`, `expected_ids`, `insert_first`); `write_binary_batches()` sets them from its arguments | by design |
| `generators::random_hyperedge_batch(num_vertices, h2v, weights, opt)` (the rows overload) | `dyng.generators.random_hyperedge_batch(hypergraph, ...)` (the hypergraph overload) | by design: the rows of a Python hypergraph come from the hypergraph |
| `generators::legacy::escher_batch_options::batch_size()` and the `escher_hypergraph_options` / `escher_batch_options` aggregates | keywords of `escher_hypergraph()` and `EscherBatchGenerator(..., batch_size=...)` | by design |
| `testing::cbst_host_model` and its policy, layout and event types | none | a model of the store's trees for the store's own tests (C++ contributors who change the store); not a user-facing oracle; by design |
| `to_string(hypergraph_storage)`, `to_string(line_graph_policy)`, `to_string(id_reuse)`, `to_string(vertex_relabel)`, `to_string(batch_dtype)`, `to_string(hyperedge_batch_mix)`, `to_string(escher_insert_mode)` | enumerations are strings in Python | by design |
