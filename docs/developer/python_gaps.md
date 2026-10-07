# Python gaps

PLAN Section 5.4, rule 8: every stable C++ symbol has a Python counterpart, or it is listed here.
The Python API is described in {doc}`../adr/0011-python-bindings` (ADR 0011). This page lists what
the 0.1 Python package does not bind, and why; each entry names the release that closes it, or
"by design".

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

## PLAN 5.4 features that are not in the 0.1 Python package

| Feature | State | When |
|---|---|---|
| CUDA: `Resources.cuda()`, `__cuda_array_interface__`, DLPack stream ordering (rule 4) | done in 0.2 (M6a; ADR 0030, 0031): `Resources.cuda()` with a CUDA plugin wheel (`pip install "dyng[cu13]"`), device result arrays with `__cuda_array_interface__`, DLPack >= 1.0 exports ordered on the consumer's stream, host copies; inputs in device memory are accepted | – |
| `copy=False` on inputs (rule 1: raise instead of copying) | inputs of another dtype or layout are converted with one copy; inputs in CUDA device memory are copied to the host once (graphs and batches are built from host arrays), and that copy is not yet subject to `Resources.copy_policy`; `Resources.copy_policy` governs the library's own host/device copies | 0.3 (with device-resident batches) |
| Inputs that only have `__cuda_array_interface__` (Numba device arrays) | inputs in device memory are read through DLPack (CuPy, PyTorch, JAX); `cupy.asarray(x)` wraps a Numba array without a copy | 0.3 |
| `dyng.interop.from_networkx()`, `from_scipy()`, `from_cudf()` (rule 6) | `Graph.from_csr(csr.indptr, csr.indices, csr.data)` covers SciPy | after 0.2.0 (not in 0.2.0) |
| `.dgb` binary batch files and the streaming `batch_reader` (PLAN 5.7) | `.dgt` text batches are read and written (`dyng.io.read_batches()` / `write_batches()`), MOSP's `insert.txt` / `delete.txt` by `read_legacy_batch()` | 0.3 (M8; moved from M7, which PLAN 11.3 listed) |
| `Graph.with_capacity()`, vertex insertions and deletions | `EdgeBatch(insert_vertices=...)` exists; applying it raises `NotSupportedError` | 0.4 |
| `dyng.Hypergraph`, `HyperedgeBatch` | none | 0.3 |
