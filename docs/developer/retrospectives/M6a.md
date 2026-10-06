# Retrospective M6a: the CUDA plugin wheels (dyng-cu12, dyng-cu13)

M6a ships the CUDA backends to Python as the plugin wheels of PLAN 5.4 (`pip install
"dyng[cu13]"`), their discovery from `dyng`, their CI builds and the release workflow (release
0.2.0, PLAN Appendix F). This page is written step by step; the final summary and the open items
for the lead maintainer close it at the end of the milestone.

## Step plugin-build (2026-10-06)

**Done.** ADR 0030 (accepted under delegation). The plugins are built from the core sdist with a
`pyproject.toml` rendered from the root one (`ci/plugin_pyproject.py`), the CMake option
`DYNG_PYTHON_PLUGIN` (`dyng_cu<N>._core`, `NB_DOMAIN dyng_cu<N>`, every backend, refused with a
toolkit of another major or a shared CUDA runtime), the static CUDA runtime for every target
(`DYNG_CUDART_TARGET`), the release architectures, one plugin package source
(`python/plugin/dyng_plugin`: identity from its name, a ctypes probe of the driver, the module
imported on first access), the extras `cu12` / `cu13` pinned through dynamic metadata (and
`cupy-cu12` / `cupy-cu13`), the plugin checks of `ci/wheel_check.py` (including the module's
`DT_NEEDED` entries), the licence files `THIRD_PARTY_LICENSES_CUDA.txt` and the toolkit's EULA,
and the local build `ci/plugin_wheels.sh`.

Local result: `dyng_cu13` (CUDA 13.1, 5.5 MB) and `dyng_cu12` (CUDA 12.9, 5.6 MB), both
`manylinux_2_28_x86_64`, SASS for sm_75/80/86/89/90/100/120 plus sm_120 PTX, no `libcudart` or
`libcuda` needed or bundled; `twine check --strict` and `ci/wheel_check.py` clean. Installed
with the core wheel into fresh venvs (3.12, 3.13): the plugin is selected (`dyng.show_config()`:
`native module : dyng_cu13 (plugin cu13)`), sssp, cycle_count and mosp run on the CUDA backend
of GPU 1, a process without a visible device falls back to `dyng._core`, and the pytest suite
passes with `DYNG_CPU_ONLY=1` (447 passed, 4 skipped) next to the installed plugin (both
plugins built from this step's last commit; log `$DYNG_SCRATCH/runs/m6a-plugin-wheels.log`,
wheels in `$DYNG_SCRATCH/wheels/0.2.0.dev0-plugins/dist`). `ci/check.sh` passes, and so do the
portability pre-checks (Clang 18 syntax over the dev build with the bindings: 187 files, no
failure; the `cpu-only` build with `-D_FORTIFY_SOURCE=3` and the bindings). The `dev-cuda`
preset with `-DCMAKE_CUDA_RUNTIME_LIBRARY=Static -DBUILD_SHARED_LIBS=OFF` passes `ctest -L gpu`
on GPU 1.

**Deviations from the plan's sketch** (each recorded in ADR 0030):

1. PLAN 7.7's "release script rewrites name / wheel.packages / defines" is a tested renderer
   (`ci/plugin_pyproject.py`, `ci/tests/test_plugin_pyproject.py`) that writes a whole
   `pyproject.toml` from the parsed root one, applied to an unpacked core sdist, not an in-place
   edit of the checkout. Same intent (one source of metadata), and the plugins and the core come
   from one source archive.
2. `wheel.packages` cannot map `dyng_cu13` to `python/plugin/dyng_plugin`: scikit-build-core
   requires the directory to carry the package's name ("must match in the last component of
   the paths"). The renderer copies the package to `python/plugin/dyng_cu<N>` in the disposable
   tree instead of committing two identical packages.
3. PLAN 7.7's extras `cu12` / `cu13` ("filled in by the release script") are filled by
   scikit-build-core's template provider from `VERSION` (`optional-dependencies` is dynamic), so
   they can never drift from the version, also in the sdist and in development builds.
4. The CUDA Toolkit's EULA, which the plugins must carry with the static CUDA runtime, is copied
   from the toolkit at build time rather than committed (NVIDIA's text, version-specific, and the
   repository's files are Apache-2.0); the build fails without it.
5. The cu12 wheel was built locally with conda-forge's CUDA 12.9 packages
   (`$DYNG_SCRATCH/tools/cuda-12.9`, created for this step; 1.3 GB): the machine has no
   system-wide CUDA 12 toolkit, and the `cuda_12.8` conda environment is not the latest 12.x
   (PLAN 7.8). CI uses NVIDIA's RHEL 8 packages (the CI step).

**Found for the next steps.**

- Result arrays in device memory: `Array.device` reports `cuda:0`, but `to_numpy()` (and every
  NumPy-based accessor: `shape`, `dtype`, indexing) fails with "Unsupported device in DLTensor"
  and there is no `__cuda_array_interface__`; the install test of `ci/plugin_wheels.sh`
  therefore compares cycle_count's total and mosp's host path costs only. The discovery / GPU
  test step must give device arrays a host copy and the CUDA array interfaces (acceptance
  criterion 2), then compare the arrays element by element.
- `dyng/_backend.py` still takes the first available plugin in name order (cu12 before cu13),
  falls back silently, and does not check the plugin's version against `dyng`. The plugin
  package offers what the selection needs: `CUDA_MAJOR`, `__version__`, `status()` (with a
  human-readable `reason`) and `driver_version()`.
- `ci/plugin_wheels.sh` does not run in `ci/check.sh --wheels` (it needs a CUDA toolkit and a
  GPU); the CI build is the CI step's (`wheels.yml`: the toolkits in `CIBW_BEFORE_ALL`, the core
  wheel offered to the plugin's test through `PIP_FIND_LINKS`, since `dyng==<dev version>` is not
  on PyPI; the EULA from the toolkit's packages, e.g. NVIDIA's `cuda-documentation` RPM, or
  `--cuda-eula`).

**Open item for the lead maintainer** (GOVERNANCE.md, open decisions): the licence expression of
the plugin wheels. They keep the CPU wheel's expression until the author decides whether to name
the CUDA runtime's EULA (`LicenseRef-NVIDIA-End-User-License-Agreement`, conda-forge's
identifier) and CCCL's `Apache-2.0 WITH LLVM-exception` in it; the licence files are complete
either way.

## Step discovery-tests (2026-10-06)

**Done.** ADR 0031 (accepted under delegation). `dyng/_backend.py` now chooses among the
installed plugins by the rules of PLAN 5.4 and ADR 0011 item 14: the driver's CUDA major wins
(else the newest older major), a plugin of another version than `dyng` is never imported (the
metadata versions, then the module's own `__version__`), a module that fails to import passes to
the next candidate, and installed-but-unusable plugins give the CPU module with one
`dyng.BackendWarning` (new public name) that names each plugin's reason; without plugins the
CPU-only install is unchanged and silent. `dyng.show_config()` / `dyng.config()` report the
choice and every plugin's state, and `Resources.cuda()` in the CPU module says why no plugin is
used. `dyng.Array` handles device memory: `shape` / `dtype` from the new native `array_info()`,
`__cuda_array_interface__` (v3, with the writer's stream), DLPack >= 1.0 exports ordered on the
consumer's stream by a CUDA event (`order_stream()`, `detail::cuda_stream_fence`), host copies on
the writer's stream (`array_to_host()`), `to_numpy(copy=None)`. The native result holder keeps
the resources of its last writer. Inputs in device memory are copied to the host once (rule 1).

Tests: `python/tests/test_backend_selection.py` (every selection rule with fake plugins, and fresh
processes for the first use, the warning, `DYNG_CPU_ONLY`, `use_cpu_only()`, `show_config()`);
`python/tests/test_cuda.py` (marker `gpu`, 46 tests: CUDA == sequential for sssp on both engines,
cycle_count and mosp through compute / update / `dyng.update`; the goldens subset on CUDA; device
arrays and stream ordering; device inputs; PyTorch / CuPy round trips; the fallback);
host-memory tests of the new Array rules in `test_arrays.py`. Wired into `ci/plugin_wheels.sh`
(the install test now compares the device arrays element by element, checks the warning of the
no-device fallback, and runs `pytest -m gpu` with `DYNG_REQUIRE_CUDA=1`; `DYNG_PLUGIN_INTEROP_PYTHON`
adds a venv with PyTorch / CuPy) and into `ci/gpu_local.sh` as the step `plugin` (cu13, one
Python, about five minutes including the build; `DYNG_GPU_PLUGIN_WHEELS` reuses built wheels).

Results: see "Verification of the step discovery-tests" below.

**Deviations and refinements** (each recorded in ADR 0031):

1. PLAN 5.4 names the per-thread default stream for `Resources.cuda()`'s default; for inputs in
   device memory dynG asks the producer for the **legacy** default stream (DLPack stream 1) and
   copies on it, because PyTorch refuses the per-thread default stream as a DLPack consumer stream
   ("per-thread default stream is not supported"); the legacy stream is ordered with every
   blocking stream, the per-thread default stream included.
2. `to_numpy(copy=False)` raises for device memory (NumPy's "never copy"), and a new
   `copy=None` gives the read-only view of host memory or a read-only host copy of device memory;
   dynG's own writers (`dyng.io.write_*`) use it. Indexing, iteration, comparison, `repr` and
   `np.asarray()` of a device Array read through one host copy that the Array keeps.
3. A plugin whose module was built without CUDA, or whose module version differs, is refused like
   a version mismatch; ADR 0011 item 14's minimal contract (`available()`, `native`) still works.
4. Device inputs are copied to the host without consulting `Resources.copy_policy` (the input
   conversion does not know the call's resources yet), and inputs with only
   `__cuda_array_interface__` (Numba) are not read; both are in `python_gaps.md`.
5. `Resources.cuda(stream=<object>)` keeps the stream object alive (and so do the graphs, results
   and Arrays made with those resources): without it a CuPy stream destroyed before the result
   crashed the process when the result freed its stream-ordered memory (found by
   `test_a_user_stream_orders_the_results`). The stream is held in a slot of a private base class
   of `Resources`, so that the native handle (the subclass's slot, cleared first) is freed before
   the stream; a weak-key dictionary released the stream first and crashed the same way.
   `griffe` (ci/api_check.sh) reports any change of a public class's `__slots__` tuple as a
   breaking change, so `Resources.__slots__` and `Array.__slots__` are unchanged (the Array keeps
   its writer and its result's resources in a private `_Owner` behind `_is_current`).
6. The test modules about host memory (`test_arrays.py`, `test_threads.py`) now use the
   sequential backend as their default resources (fixture `host_default_resources`), and
   `test_plugin_package.py` restores an installed plugin's modules in `sys.modules`: the whole
   suite passes with a plugin active (519 passed, 4 skipped in the interop venv), not only with
   `DYNG_CPU_ONLY=1`.
7. The ADR is numbered 0031; M6b, running at the same time, may take the same number: renumber at
   merge if needed.

**Found for the next steps.** The CI build (`wheels.yml`, `release.yml`) has to run the GPU-free
part: an import / selection smoke test on the hosted runners falls back to `dyng._core` with a
`dyng.BackendWarning` ("no CUDA driver"), which the smoke test should expect, and
`test_backend_selection.py` runs there unchanged. The wheel-vs-parity timing row (PLAN 7.7) is
still open.
