# ADR 0031: Choosing a CUDA plugin, the CPU fallback, and arrays in device memory from Python

- **Status:** Accepted under delegation (2026-10-06; GOVERNANCE.md, "Delegation of technical
  ADRs"). It records how PLAN Section 5.4 ("CPU core + CUDA plugins", rules 1 and 4 of the Python
  API) and ADR 0011 item 14 are implemented in M6a, step "discovery-tests", and where the
  implementation refines them. It changes no rule the author approved: the gates and their
  measurement protocol, the parity rules, the licence, the names and the publishing steps are
  untouched, and nothing is published by it.
- **Date:** 2026-10-06
- **Deciders:** the AI assistant, on the author's behalf (M6a, step "discovery-tests")

## Context

ADR 0030 builds the plugin wheels `dyng-cu12` / `dyng-cu13`; their import packages register the
entry point group `dyng.backends` and offer `available()`, `native`, `CUDA_MAJOR`,
`__version__` and `status()`. Before this step `dyng/_backend.py` took the first available
plugin in name order (cu12 before cu13), fell back to the CPU module silently and never compared
versions; and a result array of the CUDA backend reported `device == "cuda:0"` but could not be
read from Python (`shape`, `to_numpy()` failed with "Unsupported device in DLTensor"; no
`__cuda_array_interface__`; `__dlpack__` ignored the consumer's stream). PLAN 5.4 asks: "if both
plugins are installed, the one matching the driver's CUDA major wins, and `dyng.show_config()`
says so"; rule 1, inputs from any DLPack / CUDA array interface producer; rule 4, "DLPack
consumers pass their stream through `__dlpack__(stream=...)`, and `dyng.Array` orders its stream
before the consumer's with an event".

## Decision

1. **Selection rules** (`dyng/_backend.py`, on the first use, as before ADR 0011 item 14):
   1. `DYNG_CPU_ONLY=1` or `dyng.use_cpu_only()` (before the first use) chooses `dyng._core`
      without loading any plugin package.
   2. **Version pin.** A plugin whose `__version__` differs from the installed `dyng`
      distribution's version (`importlib.metadata`) is never imported; after import the module's
      own `__version__` is checked as well (a plugin built from a source tree has no metadata).
      When `dyng` itself runs from a source tree without metadata the check is skipped.
   3. **Candidates** are the plugins whose `available()` is true (ADR 0030 item 3: a driver of the
      plugin's CUDA major or newer and a visible device). The plugin of the driver's CUDA major
      (`status().driver_version // 1000`) wins; otherwise the newest older major (CUDA's
      backward compatibility: a cu12 plugin runs on a CUDA 13 driver); then by name.
   4. A candidate whose module fails to import, reports another version or was built without
      CUDA is reported as `failed` and the next candidate is tried.
   5. **Fallback.** No plugin installed: `dyng._core`, silently (the CPU-only install behaves as
      in 0.1.0). Plugins installed but none usable: `dyng._core` and **one
      `dyng.BackendWarning`** (a new `UserWarning` subclass, exported as `dyng.BackendWarning`),
      naming each plugin and why it cannot run (no driver, too old a driver, no visible device,
      another version, an import error), with the remedy. The warning is attributed to the
      caller's line (`skip_file_prefixes`), and the usual warning filters silence it; rule 1's
      explicit choices never warn.
   The minimal contract of ADR 0011 item 14 (`available()` and `native`) still suffices; the
   extra attributes only refine the choice and the messages.
2. **Reporting.** `dyng._backend.selection()` returns the choice (`module_name`, `plugin`,
   `reason`, one `PluginReport` per installed plugin: name, module, version, CUDA major, the
   driver's CUDA version, `state` in `chosen` / `available` / `unusable` / `version mismatch` /
   `failed` / `not considered`, and the reason). `dyng.config()` gains `selection` and
   `plugins`; `dyng.show_config()` prints "chosen because" and one line per plugin.
   `Resources.cuda()` in the CPU module appends what the selection found to its message.
3. **Device arrays.** `dyng.Array` handles memory whose DLPack device is CUDA (2) or CUDA managed
   (13):
   - `shape`, `dtype`, `ndim`, `size` come from a new native `array_info()` (no element read);
   - `__cuda_array_interface__` (version 3, read-only, `strides` None) carries the stream of the
     call that last wrote the result (1 for the legacy default stream, 2 for the per-thread
     default stream, else the handle); host arrays do not have it, and device arrays do not have
     `__array_interface__`;
   - `__dlpack__(stream=s)` orders the consumer's stream after the writer's with a CUDA event
     (native `order_stream()`: `detail::cuda_stream_fence`, record then wait; the host does not
     block), following the DLPack Python specification's values (None and 1: the legacy default
     stream, 2: the per-thread default stream, -1: no ordering, 0: refused as ambiguous); device
     memory is exported through DLPack >= 1.0 only (the capsule is read-only, ADR 0011 item 4);
     `dl_device=(1, 0)` gives a host copy, `copy=True` of device memory is refused;
   - `to_numpy()` copies to the host on the writer's stream (native `array_to_host()`:
     `dyng::copy` then a stream synchronization); `to_numpy(copy=None)` (new) is the read-only
     host view of host memory, or a read-only host copy of device memory kept by the Array;
     `to_numpy(copy=False)` raises for device memory (NumPy's meaning of "never copy");
     indexing, iteration, comparison, `repr` and `np.asarray(a)` read through that kept copy;
   - `to_torch()` / `to_cupy()` pass the Array itself to `from_dlpack`, so the framework's
     current stream is ordered.
   The writer's resources are kept by the native result holder (`result_holder::writer()`, set by
   compute, `from_arrays`, `clone` and every update path under the result's exclusive lock;
   Python reads `handle.writer`). A Resources made with a stream object
   (`Resources.cuda(stream=cupy_stream)`) keeps that object alive, and so do the graphs, results
   and Arrays made with it (the C++ rule "the stream must outlive every copy of the handle" made
   automatic for Python objects; exports such as a PyTorch tensor keep the memory, not the
   stream).
4. **Device inputs** (rule 1). Arrays whose DLPack device is CUDA (CuPy, PyTorch, JAX, a
   `dyng.Array` of the CUDA backend) are accepted wherever an array is: they are copied to the
   host once, with the producer's work ordered through `__dlpack__(stream=1)` and the copy on the
   legacy default stream of a cached per-device handle (PyTorch refuses the per-thread default
   stream as a consumer stream; the legacy stream is ordered with every blocking stream), since
   graphs and batches are built from host arrays in this release. Their dtype is read without a
   copy (`torch.int32` is read as `int32`). With the CPU module a device input raises
   `NotSupportedError` naming the plugins. Such a copy is not yet subject to
   `Resources.copy_policy` (python_gaps.md).
5. **Tests.** `python/tests/test_backend_selection.py` (no GPU): every rule on `_select()` with
   fake plugins, and fresh processes for the first use, the warning (category, attribution,
   filtering), `DYNG_CPU_ONLY`, `use_cpu_only()`, `show_config()` and `Resources.cuda()`.
   `python/tests/test_cuda.py` (marker `gpu`; skipped without a CUDA module, an error with
   `DYNG_REQUIRE_CUDA=1`): sssp (both engines), cycle_count and mosp on CUDA equal the sequential
   backend element by element through compute, update and `dyng.update`; the goldens subset of
   `test_parity.py` on CUDA (MOSP's files byte for byte for six sssp cases on both engines, the
   23 CycleEnumeration-GPU fixture graphs' `.cuda` histograms, ten random cases); device arrays
   (metadata, copies, the interface, stream ordering with the calls recorded, staleness); device
   inputs; PyTorch and CuPy round trips including a user stream (skipped when not installed);
   the fallback with no visible device and `use_cpu_only()` next to an installed plugin.
   `ci/plugin_wheels.sh` runs them in its fresh venvs (and, with `DYNG_PLUGIN_INTEROP_PYTHON`, in
   a venv with PyTorch / CuPy); `ci/gpu_local.sh` gains the step `plugin`.

## Amendments (M6a review, 2026-10-06; accepted under delegation)

1. **The probe initializes no CUDA in the calling process.** `status()` of ADR 0030 item 3 called
   `cuInit` through ctypes, so any first use of dynG with a plugin installed (even
   `dyng.__version__` or CPU work) left the process unable to fork workers that use CUDA (a child
   forked after `cuInit` gets `CUDA_ERROR_NOT_INITIALIZED`). The probe now reads the driver's
   version with `cuDriverGetVersion` (no `cuInit` needed) and the devices through NVML
   (`libnvidia-ml.so.1`: count, compute capability, UUID, MIG mode), applying
   `CUDA_VISIBLE_DEVICES` as CUDA does (ordinals and `GPU-` UUID prefixes up to the first entry
   that names no device; a repeated entry hides every device). When NVML cannot tell which
   devices CUDA will see (no NVML, e.g. a container without the driver's utility libraries; MIG;
   ordinals on GPUs of different kinds without `CUDA_DEVICE_ORDER=PCI_BUS_ID`, since CUDA numbers
   them fastest first), a short child process (`python -I -S dyng_cu<N>/_devices.py`) asks the
   driver; only if no child can be started is `cuInit` called in the process itself.
   `Status.probe` says which way was taken. In C++, a `cudaErrorInitializationError` (and a
   device count of 0 caused by it) now names fork and the remedy (the `spawn` / `forkserver`
   start methods), since a parent that really used CUDA still forks children that cannot.
2. **Compute capability floor.** A plugin is usable only when at least one visible device has
   compute capability 7.5 or newer (`ARCHITECTURE_FLOOR`, the first entry of every release list
   of `cmake/cuda_architectures.cmake`; a test ties the two). Before, a Volta or Pascal GPU (a V100
   on a 525-580 driver) made the plugin the default backend, and every default call failed with
   `cudaErrorNoKernelImageForDevice`; now rule 5 falls back with the reason "the visible GPU
   (sm_70) is older than the oldest architecture of dyng-cu12, sm_75". On a machine with mixed
   GPUs the plugin is used when one of them is supported; the error of a kernel without code
   for the current device names its compute capability.
3. **`DYNG_CPU_ONLY`** is read as a boolean: `1`, `true`, `yes`, `on` (any case) are on; `0`,
   `false`, `no`, `off` and empty are off; any other value is ignored with a
   `dyng.BackendWarning` (before, every value but `0` and empty forced the CPU module).

## Consequences

- Upgrading `dyng` without its plugin, or a plugin without a driver, gives a working CPU
  installation and one clear warning instead of a crash or a silent CPU run.
- Device results are usable from Python without a host round trip (PyTorch, CuPy and any
  `__cuda_array_interface__` consumer), and correctly ordered when the consumer runs on another
  stream; host code still gets copies with `to_numpy()`.
- A result keeps one more reference-counted handle (its writer's resources).
- The test-suite modules that are about host memory (`test_arrays.py`, `test_threads.py`) use the
  sequential backend as their default resources, so the whole suite also passes with a plugin
  active (whose default backend is CUDA).
- The ADR number may collide with M6b, which runs at the same time; renumber at merge if needed.
