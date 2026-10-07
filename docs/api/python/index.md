# Python API

`import dyng` gives the whole public Python API: the names of the `dyng` package and the
modules `dyng.sssp`, `dyng.cycle_count`, `dyng.mosp` (0.2), the tutorial algorithms
`dyng.dynamic_bfs` and `dyng.triangle_delta` (0.2), `dyng.io`, `dyng.generators` and
`dyng.testing`. The
names mirror the C++ API ({doc}`../cpp/index`): `dyng::sssp::compute` is `dyng.sssp.compute`,
`dyng::sssp::options` is `dyng.sssp.Options`, and option keywords are the C++ field names
(`max_length=`, `delta=`). Names with a leading underscore (`dyng._core`, the native module) are
private and may change in any release. What the C++ library has and the Python
package does not yet bind is listed in {doc}`../../developer/python_gaps`.

```{toctree}
:maxdepth: 1

reference/dyng/index
```

The pages below the reference are generated from the docstrings and signatures of
`python/dyng` (and the committed stubs of the native module). The rest of this page explains the
rules that hold across the whole package (ADR 0011).

## Resources and backends

A {py:class}`dyng.Resources` chooses the backend: `dyng.Resources.sequential()`,
`dyng.Resources.openmp(num_threads=0)` (0 is the OpenMP default, which honours
`OMP_NUM_THREADS`) or `dyng.Resources("openmp")`. The CPU wheel has no CUDA backend:
`dyng.Resources.cuda()` raises {py:class}`dyng.NotSupportedError` with a message that names the
CUDA plugin wheels of 0.1.x. Every function takes `resources=None` as a keyword; `None` means
the resources the graph was built with for every call that takes a graph (`compute`, `update`,
`Graph.apply`), and the process-wide default ({py:func}`dyng.get_default_resources`, set with
{py:func}`dyng.set_default_resources`) for the rest (building graphs, the readers). All backends
return the same results bit for bit.

## Dtype dispatch

The compiled library holds five graph types, `(vertex id, edge offset, weight)` =
`(int32, int32, int32)`, `(int32, int64, int32)`, `(int64, int64, int32)`,
`(int32, int32, unweighted)` and `(int32, int64, unweighted)`. The Python layer picks one from the
inputs:

- The vertex id type comes from the declared dtype of the id arrays: int32 selects int32 ids and
  int64 selects int64 ids; narrower integer types widen to int32 and uint32 to int64. **An int64
  array never selects int32 ids on its own**; pass `vertex_dtype="int32"` to ask for them, and
  every value is then checked. Inputs without a dtype (Python lists, ranges) take int32 when every
  value fits, int64 otherwise.
- The edge offset type is int32 whenever the stored edge count fits (ADR 0009), int64 with int64
  ids or when `row_ptr` is declared int64.
- Where the type is already fixed (a batch, a tree or parents for an existing graph), ids are
  converted to the graph's type with a range check: a value that does not fit raises
  {py:class}`dyng.InvalidArgumentError` instead of wrapping. So an `EdgeBatch` of int64 NumPy
  arrays updates an int32 graph.
- Weights are int32 integers (range-checked); floating-point and boolean weights are rejected.
- An unsupported combination raises {py:class}`dyng.NotSupportedError` listing the supported
  ones.

## Result arrays

Result arrays (`tree.distances`, `tree.parents`, `hist.counts`, `paths.path_costs`) are
{py:class}`dyng.Array` objects (1-D, except the (n, K) path costs of `dyng.mosp`): read-only, zero-copy views with `__dlpack__`, `__dlpack_device__` and
`__array_interface__`, so `np.asarray(a)`, `np.from_dlpack(a)` and `torch.from_dlpack(a)` see the
library's memory without a copy, and every view keeps that memory alive. An Array belongs to one
state of its result: once the result is updated, using it raises
{py:class}`dyng.StaleResultError`; read the property again. `a.to_numpy()` returns a copy (the
default), `a.to_numpy(copy=False)` the read-only view. Views that NumPy or another library made
keep showing the state they were made from, unchanged: while such a view is alive, an update
copies the result's state first and changes the copy (copy-on-write), so exported memory is
never changed or freed under a view. That costs one copy of the result's arrays per update while
a view is alive; drop views you no longer need to update in place. DLPack consumers that ask for
DLPack >= 1.0 get a read-only zero-copy capsule; the unversioned capsule of older consumers,
which cannot be marked read-only, is a copy. PyTorch and CuPy have no read-only arrays: a tensor
from `a.to_torch()` or `torch.from_dlpack(a)` aliases the result and must not be written.
`copy.copy` / `copy.deepcopy` of a result or a graph are `clone()`; batches, resources and arrays
pickle, graphs and results explain how to send them (`to_csr()`, `Result.from_arrays()`).

## Results, versions and stale results

A graph has a version that grows by one with every applied batch, and every result remembers the
version it matches ({doc}`../../concepts/results_and_versions`). `dyng.sssp.update(g, batch,
tree)` applies the batch to `g` and brings `tree` up to date; another result on the same graph is
then stale, and using it raises {py:class}`dyng.StaleResultError`. To keep several results
current, update them together: `dyng.update(g, batch, tree, hist)` applies the batch once and
returns one `Stats` per result (any mix of `dyng.sssp`, `dyng.cycle_count` and `dyng.mosp`
results).

## Exceptions

Every error of the library is a {py:class}`dyng.Error`, and each class is also the built-in
exception a Python user expects:

| C++ | Python | Also a |
|---|---|---|
| `dyng::error` | `dyng.Error` | `Exception` |
| `invalid_argument_error` | `dyng.InvalidArgumentError` | `ValueError` |
| `stale_result_error` | `dyng.StaleResultError` | `dyng.InvalidArgumentError` |
| `io_error` | `dyng.FileFormatError` (`.path`, `.line`, `.column`) | `OSError` |
| `capacity_error` | `dyng.CapacityError` | `MemoryError` |
| `not_supported_error` | `dyng.NotSupportedError` | `NotImplementedError` |
| `convergence_error` | `dyng.ConvergenceError` | `RuntimeError` |
| `cuda_error` | `dyng.CudaError` (`.code`) | `RuntimeError` |
| `out_of_memory_error` | `dyng.OutOfMemoryError` | `MemoryError` |
| `internal_error` | `dyng.InternalError` | `RuntimeError` |

A call that raises leaves the graph and its results as they were, except where the C++
documentation says otherwise (a result poisoned by a failed update raises `StaleResultError`
until it is recomputed).

## Threads and the GIL

The GIL is released around every native call that runs an algorithm or I/O, so several Python
threads can run dynG at once. Graphs and results are not thread-safe in C++; the binding keeps
them memory-safe with a reader/writer lock per object: calls that change a graph or a result
(`apply`, the updates) are serialized against every other call on it, and reads run in parallel.
Setting `Resources.copy_policy` and entering or leaving `dyng.profile()` change the shared
resources handle, so they wait until no native call runs in any thread. A batch reads its arrays
at every use: do not change them while a call that uses the batch runs.

## Profiling

`with dyng.profile() as p:` attaches a profiler to the default resources (or to the resources
given) for the calls inside the block; `p.stages` lists the stages (`sssp.update`,
`sssp.identify_affected`, ...), `p.to_csv()` and `p.to_json()` export them, and `p.to_dataframe()`
returns a pandas DataFrame when pandas is installed ({doc}`../../how_to/profile_an_update`).

## Citing

`dyng.citation("sssp")` returns the BibTeX entries of an algorithm's papers (and
`dyng.citation()` the entry of dynG itself); `dyng.algorithms()` lists the algorithms of the
installed package with their maturity, backends and cite keys.
