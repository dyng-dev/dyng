# ADR 0011: The Python package (nanobind bindings and the typed layer)

- **Status:** Accepted under delegation (2026-09-30; GOVERNANCE.md, "Delegation of technical
  ADRs"). It records how PLAN Sections 5.4, 5.5 and 7.7 are implemented in M5 and where the
  implementation differs from their sketches. It changes no rule the author approved: the
  performance gates, their measurement protocol, the parity rules, the licence, the names and
  the publishing steps are untouched, and nothing is published by it.
- **Date:** 2026-09-30
- **Deciders:** the AI assistant, on the author's behalf (M5, step "bindings")

## Context

PLAN 5.4 fixes the technology (nanobind, scikit-build-core, one root `pyproject.toml`, abi3
wheels, a private extension module under a typed pure-Python layer, committed stubs) and the
Python names of the 0.1 API. ADR 0023 (notes 1-6) lists what needs a deliberate binding: the
variadic `dyng::update()`, the lifetime of the result views, the `initializer_list` builders,
per-thread default streams, the profiler's records and the enumerations' string names. This ADR
records the decisions taken while binding them.

## Decision

1. **Build.** The root `pyproject.toml` uses scikit-build-core (>= 1.1) and nanobind **pinned
   exactly** (3.1.0: the committed stubs are its output). The version comes from `VERSION`
   through the regex provider. CMake gets `DYNG_BUILD_PYTHON=ON` (a new `python/` subdirectory),
   `BUILD_SHARED_LIBS=OFF`, `DYNG_ENABLE_CUDA=OFF`, and the tests, examples, tools and install
   rules off; only the target `_core` is built and only its install component is installed, so the
   wheel holds `dyng/_core.abi3.so` and the typed layer, nothing of the C++ SDK. The module is
   `nanobind_add_module(_core STABLE_ABI NB_STATIC NB_DOMAIN dyng_cpu ...)` linking `libdyng` and
   `dyng::testing` statically. The bindings compile with the project's warnings (`-Werror` in the
   dev and CI presets); nanobind's own sources are exempt from `-Werror`.
2. **Two layers.** `dyng._core` is private and flat (one class per C++ instantiation:
   `GraphI32I64U`, `SsspResultI64`, `EdgeBatchI32I32`, ...; overloads per graph type). The typed
   layer (`python/dyng/*.py`) owns every public name, docstring and keyword. Its modules are those
   of PLAN 4.2 with four changes: `_update.py` (the module of `dyng.update`, so the function and a
   module do not share the name), `_registry.py` (`algorithms()`, `citation()`), `config.py`
   (`show_config()`, logging, `use_cpu_only()`), `generators/` a package with `legacy.py`
   (mirroring `dyng::generators::legacy`); `hypergraph.py` and the later algorithms come with
   their milestones. `_algorithms.py` is generated from the manifests by `scripts/regen.py`.
3. **Dtype dispatch** (PLAN 5.4 rule 2) picks the instantiation from the declared dtypes of the
   inputs: int32 selects int32 ids, int64 selects int64 ids; narrower integer types widen to
   int32 and uint32 to int64; inputs without a dtype (lists) take the smallest width that holds
   their values. **An int64 array never selects int32 ids on its own**; `vertex_dtype="int32"`
   asks for it, and every value is then checked. Where the instantiation is already fixed (a
   batch, a tree, parents for an existing graph), ids are converted to the graph's type with a
   range check that raises `InvalidArgumentError` instead of wrapping; this is what lets PLAN
   5.5's `EdgeBatch(insert=(np.array([0, 5]), ...))` (int64 arrays) update an int32 graph. The
   edge offset type is int32 whenever the stored edge count fits (ADR 0009), int64 with int64 ids
   or when `row_ptr` is declared int64. Weights are int32 (range-checked; floats are rejected).
   An unsupported combination raises `NotSupportedError` listing the five supported ones.
4. **Arrays.** Result arrays are `dyng.Array`, a read-only view with `__dlpack__`,
   `__dlpack_device__` and `__array_interface__` (nanobind's array-API object underneath). ADR
   0023 note 2 asked to "check the owner's version on access or copy": an Array belongs to one
   state of its result and raises `StaleResultError` when used after the result was updated. The
   generation counter lives in the native result holder and is advanced, under the result's
   exclusive lock, by every update path before the update runs (so a failed update also retires
   its arrays, and every Python wrapper of one native result sees it). *Amended in the M5 review
   (2026-09-30):* the first version kept the counter in the Python wrapper and made the native
   holder the owner of every export, so `copy.copy(result)` bypassed the check and views exported
   through NumPy or DLPack read freed memory after an update that grew the vertex set. Now the
   result's state lives in reference-counted storage; every export (the Array's DLPack capsule,
   and so every NumPy, PyTorch or buffer-protocol view made from it) holds a reference to the
   state it views, and an update **copies the state first while such a reference exists**
   (copy-on-write), so exported memory is never changed or freed under a view: an old view keeps
   showing the state it was made from. Without live exports an update works in place, as before.
   `Array.to_numpy()` still **copies by default** (as PLAN 5.5's comment on `.to_numpy()` says);
   `to_numpy(copy=False)` returns the read-only view. DLPack exports are read-only: a consumer
   that asks for the unversioned (legacy) capsule, which cannot carry DLPack 1.0's read-only
   flag, gets a copy (as NumPy's own `__dlpack__` refuses a read-only legacy export). PyTorch and
   CuPy have no read-only arrays, so `to_torch()` stays a zero-copy view that is documented as
   not writable. `Result.__copy__` / `__deepcopy__` are `clone()`.
5. **The GIL** is released around every native call that runs algorithms or I/O (rule 3). Two
   locks keep concurrent Python threads memory-safe, because the C++ containers are not
   thread-safe: every graph and result holder has a reader/writer lock (exclusive for `apply` and
   the updates, shared otherwise; taken in address order after the GIL is released), and every
   native call holds a process-wide lock in shared mode that reading a profiler's records takes
   exclusively (ADR 0023 note 5: the records are copied while no call can be recording). Two
   Python threads updating one graph are therefore serialized: each call sees the state the
   previous one left (a result that the other thread's call did not update is then stale and
   raises `StaleResultError`, the library's normal contract).
6. **Exceptions.** `dyng.errors` defines the hierarchy of PLAN 5.4 and registers the classes with
   the native module, whose translator maps each C++ type (most derived first). `io_error` becomes
   `FileFormatError` with `.path`, `.line` and `.column`; `cuda_error` `CudaError` with `.code`.
   `internal_error`, which PLAN 5.4's table leaves out, becomes `InternalError(Error,
   RuntimeError)`.
7. **`dyng.update(graph, batch, *results)`** takes a run-time list of results of mixed types. The
   binding builds each result's participant with the factories the algorithm headers declare for
   `dyng::update()` (`detail::make_sssp_participant`, `detail::make_cycle_count_participant`) and
   runs them through `detail::participant_of<graph>::run()`, the path of the C++ `update()` (ADR
   0023 note 1). Duplicates and stale results are rejected before anything changes.
8. **Resources.** `resources=None` means the resources the graph was built with for every call
   that takes a graph (`compute`, `update`, `Graph.apply`, ...) and the process-wide default
   (`get_default_resources()`) for everything else (building graphs, readers). A result
   remembers its resources for `clone()`. `Resources.cuda()` raises `NotSupportedError` in the CPU
   wheel with a message naming the CUDA plugin wheels of 0.1.x. `dyng.profile()` is the only way
   to attach a profiler from Python (it keeps the profiler alive, restores the previous one on
   exit, and nests).
9. **Options, stats, enumerations.** `Options` are mutable dataclasses with the C++ field names
   and defaults; the keywords of `compute()` override their fields and unknown keywords raise
   `TypeError`. `Stats` and `ApplySummary` are frozen dataclasses. Enumerations are accepted and
   returned as their C++ enumerators' names (`"fused"`, `"append"`), the `to_string` names of ADR
   0023 finding 5. `GraphProperties` keeps the C++ field names (`order`, `parallel_edges`), and
   `Graph.from_edges` takes PLAN 5.4's keyword names for them (`row_order=`, `multi_edges=`).
10. **Stubs.** `python/dyng/_core.pyi` is generated by nanobind's `stubgen` and committed.
    `python scripts/regen.py --stubs` regenerates it and `--stubs --check` fails on drift. PLAN
    5.4 names CMake's `nanobind_add_stub`; the stub is generated by the script instead, because
    the check must regenerate and diff the committed file against an importable `dyng._core`
    (the editable install), and the lint job, which runs `regen.py --check` without a compiled
    module, must not need one.
11. **Readers** return a `dyng.Graph` (PLAN 5.5's `dyng.io.read_matrix_market(...)` usage); the
    `*_arrays` variants return the arrays. `generators.legacy.mosp_changes()` returns
    `(batch, report)`. The builders with `std::initializer_list` are not bound (ADR 0023 note 3):
    batches are built from arrays.
12. **What 0.1 leaves out** is listed in `docs/developer/python_gaps.md` (PLAN 5.4 rule 8), for
    example `copy=False` on inputs, `__cuda_array_interface__` and DLPack stream ordering (CUDA,
    0.1.x), the `.dgt` batch files, `dyng.interop` and memory resources.
13. **Copying and pickling** (added in the M5 review). `EdgeBatch` and the option, stats and
    record dataclasses are plain values: they copy and pickle. `Graph` and the results copy
    (`copy.copy`, `copy.deepcopy`) as `clone()`, never as aliases of one native object; pickling
    them raises `TypeError` naming the supported path (`to_csr()` + `Graph.from_csr()`,
    `Result.from_arrays()`, `counts.to_numpy()`), because they carry a graph version and may live
    in device memory. `Resources` copies share the handle (as in C++); pickling rebuilds equal
    resources in the receiving process. A `dyng.Array` pickles and deep-copies as a NumPy copy.

## Consequences

- One abi3 wheel per platform for CPython >= 3.12, with the sequential and OpenMP backends; the
  CUDA plugins of 0.1.x reuse the same bindings under another `NB_DOMAIN`.
- The typed layer is what `griffe` checks for breaking changes; the native names may change.
- The array rule is stricter than NumPy's usual views: an Array that outlives an update raises
  instead of showing changed or freed memory, views made from it keep showing their state, and
  the default `to_numpy()` is a copy. The price of the copy-on-write is one copy of the result's
  arrays per update while an export of the current state is alive.
- Concurrent Python threads cannot corrupt a graph or a result; they pay a lock per call, which
  is negligible next to the native work of a call.
- A nanobind upgrade regenerates the stubs in its own commit (`regen.py --stubs`).
