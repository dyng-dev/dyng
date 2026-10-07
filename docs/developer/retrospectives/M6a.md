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
   `test_a_user_stream_orders_the_results`). The stream is held in a weak-key dictionary per
   `Resources`, and `Resources.__del__` drops the native handle before the entry, so the stream
   goes last (without `__del__` the weak reference's callback released the stream before the
   slots were cleared and crashed the same way; a private base class with a slot worked but the
   Sphinx reference cannot resolve a private base, warnings being errors).
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

**Verification of the step discovery-tests** (from `cd6f8b7`, GPU 1, RTX A5000, driver 590.48.01):
`ci/plugin_wheels.sh` with `DYNG_PLUGINS="cu12 cu13"` (CUDA 12.9 and 13.1) and the interop venv
(log `$DYNG_SCRATCH/runs/m6a-discovery-plugin-wheels.log`, wheels in
`$DYNG_SCRATCH/wheels/m6a-discovery`): both wheels clean (5.65 MB, 5.51 MB); in fresh venvs on
Python 3.12 and 3.13 for each plugin, the smoke test's device arrays equal the sequential
backend's, the no-device fallback gives `dyng._core` with one `dyng.BackendWarning`,
`pytest -m gpu` gives 43 passed and 3 skipped (PyTorch / CuPy absent), and the suite with
`DYNG_CPU_ONLY=1` 473 passed, 50 skipped; in the interop venv (PyTorch 2.14.1+cu130, CuPy
14.2.0) `pytest -m gpu` gives 46 passed for each plugin. Both plugins installed together:
`dyng_cu13` is chosen ("it matches the driver's CUDA major (CUDA 13.1)") and `dyng_cu12` is
reported as available. The whole suite with the cu13 plugin active: 519 passed, 4 skipped.
`ci/gpu_local.sh` with only its step `plugin` passes (log
`$DYNG_SCRATCH/runs/m6a-discovery-gpu-local-plugin.log`). `ci/check.sh`: all checks passed
(`ci/python.sh`, `ci/docs.sh`, the griffe API check and pre-commit included; log
`$DYNG_SCRATCH/runs/m6a-discovery-check.log`). Portability pre-checks (C++ unchanged since):
the Clang 18 syntax pass over the dev build with the bindings, 188 files, no failure; the
`cpu-only` build with `-D_FORTIFY_SOURCE=3` and the bindings is clean.

## Step ci-release (2026-10-06)

**Done.** ADR 0032 (accepted under delegation). `wheels.yml` builds `dyng-cu12` and `dyng-cu13`
from the run's sdist with cibuildwheel in the manylinux_2_28 image (`ci/cibuildwheel-plugin.toml`,
`before-all` = `ci/cibw_plugin.sh`), with the CUDA toolkits pinned file by file in
`ci/cuda_toolkits.toml` (CUDA 12.9 and 13.4, every RPM of NVIDIA's RHEL 8 repository with its
SHA-256; `ci/cuda_toolkit.py` locks, downloads, verifies and unpacks them; the downloads cached
except for tags), checks them (`twine check --strict`, `ci/wheel_check.py` with the 90 MB
budget) and install-tests each with the CPU wheel in fresh 3.12 / 3.13 venvs on the GPU-less
runners (`ci/plugin_smoke.py --expect fallback --reason "no CUDA driver"`, the pytest suite).
`release.yml` builds core and plugins for tags from v0.2.0 (release candidates included),
checks the release set and publishes each distribution through its own environments
(`testpypi` / `pypi`, `testpypi-cu12` / `pypi-cu12`, `testpypi-cu13` / `pypi-cu13`), TestPyPI
first and PyPI only for final versions, one approval per environment; v0.0.x / v0.1.x tags
publish `dyng` alone as before (`ci/wheel_check.py --release-distributions`, `--release-set`,
`--split`, with tests). `ci/plugin_wheels.sh` now uses the same smoke script (`--expect cuda`,
`--expect fallback` with no device and with the driver hidden by `ci/without_cuda_driver.sh`).
The informational wheel-vs-parity row is `parity/wheel_vs_parity.py`. Docs: `wheels.md`,
`release.md`, `repository_settings.md` (the four new environments), the install guide,
CHANGELOG, ADR index, GOVERNANCE approvals log.

**What could not be run here.** There is no container runtime on the machine, so the container
part of the CI build (`dnf` in the image, its gcc-toolset 14, cibuildwheel's isolated build) has
not run; the first pull request run of `wheels.yml` is its first real test. Checked instead:
actionlint, zizmor, `ci/github_meta_check.py --verify-pins` (no new action: `actions/cache` is
already used and pinned); cibuildwheel 4.2.1's own option parsing of the plugin configuration
(one identifier, `cp312-manylinux_x86_64`; the image pinned at 2026.09.05-1, whose `PATH` puts
gcc-toolset 14 first); the `select` job's script for `v0.0.1` and `v0.2.0.dev0` tags; the pinned
toolkits unpacked (`ci/cuda_toolkit.py extract`) and used for local builds of both plugins with
the usual GCC 12.4 toolchain **and** with conda-forge's GCC 14 against the glibc 2.28 sysroot
(the image's compiler major; logs `$DYNG_SCRATCH/runs/m6a-ci-toolkits-plugin-wheels.log`,
`m6a-ci-gcc14-plugin-wheels.log`): every build passed `wheel_check` and the GPU tests on GPU 1;
the CI smoke test and the whole suite with a plugin installed and the driver hidden (473 passed,
50 skipped: the same suite the `plugin-install-test` job runs).

**Deviations and refinements** (each recorded in ADR 0032):

1. PLAN 7.7: "CUDA toolkit installed in `before-all` from NVIDIA's RHEL8 repository". The packages
   are NVIDIA's RHEL 8 RPMs, but pinned by SHA-256 in a committed lock file, downloaded on the
   runner (where `actions/cache` can keep them) and installed from those files, not from the
   live repository; dependabot does not see the lock, so re-pinning to a new 12.x / 13.x is a
   maintainer's step (`ci/cuda_toolkit.py lock`, `docs/developer/wheels.md`).
2. The cu13 wheel of CI is built with CUDA 13.4 (the latest 13.x on 2026-10-06), not with the
   development machine's 13.1; it runs on the 13.1 driver here (minor-version compatibility).
3. No test runs inside cibuildwheel for the plugins (`test-skip = "*"`): the plugin needs the
   core wheel of the same version, which is not on PyPI before the release; a separate job
   installs both wheels (`--no-index`) and tests them, as the CPU wheel's install-test does.
4. The uploads are ordered on each index: the plugins first, `dyng` only after both
   succeeded (revised after the review; the first version was one unordered matrix, which a
   failed plugin leg would have left with `dyng[cuN]` pointing at a missing distribution).
5. PLAN Appendix F names the environments `pypi` / `testpypi` for the plugins' pending
   publishers; the author's setup (relayed by the orchestrator) has one pair per plugin
   (`*-cu12`, `*-cu13`), because PyPI refuses two identical pending publishers. The workflow
   follows the setup; the plan's line is out of date (an account-level fact, not decided here).
6. The wheel-vs-parity row's cycle_count update batch is drawn by the script (numpy, seed 1),
   not by the original's generator, which is not bound to Python; both sides use the same batch.
7. The ADR is numbered 0032; M6b, running at the same time, may take the same number: renumber
   at merge if needed.

**The wheel-vs-parity row** (PLAN 7.7, informational; `parity/results/M6a-wheel-vs-parity.json`,
log `$DYNG_SCRATCH/runs/m6a-wheel-vs-parity.log`): commit `476ab72`, GPU 0 (RTX A5000, driver
590.48.01), clocks locked at boost (SM 1695 MHz, memory 7601 MHz), the exclusive perf lock, 21
alternating rounds after one warm-up round, medians. Sides: **parity-cuda** (the `parity-cuda`
preset's library: shared `libdyng.so`, `-O3 -lineinfo`, sm_86, shared CUDA runtime, CUDA 13.1,
system GCC 12.2; with the bindings), **wheel-cu13** (the cu13 wheel built locally with CUDA 13.1:
release architectures, static runtime, conda GCC 12.4) and **wheel-cu13-ci** (the same with the
pinned CI toolkit, CUDA 13.4). Each side computed the same results (checked).

| Region (ms, median) | parity-cuda | wheel-cu13 | ratio | wheel-cu13-ci | ratio |
|---|---|---|---|---|---|
| sssp roadNet-CA unsafe 50K: `sssp.enact_fused` obj 0 | 8.676 | 8.579 | 0.989 | 8.785 | 1.013 |
| obj 1 | 8.919 | 8.799 | 0.986 | 9.062 | 1.016 |
| obj 2 | 8.799 | 8.667 | 0.985 | 8.882 | 1.010 |
| sum of the three | 26.400 | 26.054 | 0.987 | 26.734 | 1.013 |
| the whole `dyng.update` (commit included) | 174.5 | 180.5 | 1.034 | 181.1 | 1.037 |
| cycle_count DD k = 4: `cycle_count.count` | 0.979 | 0.975 | 0.996 | 0.974 | 0.995 |
| `cycle_count.update` (25K + 25K) | 3.809 | 3.765 | 0.988 | 4.146 | 1.088 |

The wheel built with the parity build's toolkit is within 1.5 % of the parity build on the
CUDA regions (slightly faster), and 3.4 % slower on the whole update, whose host-side commit is
compiled by another GCC with libdyng linked into the module. The CI-toolkit wheel (CUDA 13.4) is
1-2 % slower on sssp and 8.8 % slower on the cycle_count update: a compiler-version difference
(the same build otherwise), worth a look before 0.2.0 if it persists in the CI-built wheel. No
gate applies (PLAN 7.7).

**Verification** (HEAD `476ab72`, before this section was committed):

| Check | Result |
|---|---|
| `ci/plugin_wheels.sh`, cu12 + cu13 with the pinned CI toolkits (12.9, 13.4), Python 3.12 and 3.13, the interop venv | rc 0 (log `$DYNG_SCRATCH/runs/m6a-ci-release-plugin-wheels.log`, wheels in `$DYNG_SCRATCH/wheels/m6a-final-ci`): `dyng_cu12` 5.65 MB, `dyng_cu13` 5.92 MB, twine and `wheel_check` clean; per plugin and Python: the GPU smoke test (arrays equal to the sequential backend's), the fallback without a visible device and with the driver hidden, `pytest -m gpu` 43 passed / 3 skipped (no torch / cupy), the suite with `DYNG_CPU_ONLY=1` 473 passed / 50 skipped; interop venv (PyTorch, CuPy): 46 passed per plugin |
| the same for cu13 with `/usr/local/cuda-13.1`, Python 3.12 | rc 0 (`$DYNG_SCRATCH/wheels/m6a-final-cu131`; 5.51 MB) |
| `ci/check.sh` in a fresh clone (`$DYNG_SCRATCH/m6a-verify/clone`, `git checkout m6a-cuda-wheels`) | all checks passed (log `$DYNG_SCRATCH/runs/m6a-ci-release-check.log`): `cpu-only` 678/678 and `dev` 707/707 (`ctest -L cpu`), clang-tidy, reuse, provenance, regen, the harness and CI-script tests (155 passed), `ci/python.sh` (stubs, mypy, pytest 473 passed / 50 skipped), griffe, the scaffold check, `ci/docs.sh` (Doxygen, Sphinx -W, links), pre-commit (actionlint and zizmor included) |
| `ci/github_meta_check.py --verify-pins` | OK (every action SHA equals its tag) |
| Portability pre-checks (no C++ change in this step) | the Clang 18 syntax pass over a `dev` + `DYNG_BUILD_PYTHON=ON` database: 188 files, 0 failures; the `cpu-only` build with `-D_FORTIFY_SOURCE=3` and the bindings: clean |

## Step review-fixes (2026-10-06)

Independent reviewers confirmed seventeen findings (some reported twice, by different lenses). All
are fixed in this step; none is deferred.

| Finding | Fix |
|---|---|
| Choosing the plugin called `cuInit` (ctypes), so after any use of dynG (even `dyng.__version__` or CPU work) a fork-started worker could not use CUDA, with a misleading "no device" message (reported twice) | `6dc5f30`: the driver version from `cuDriverGetVersion`, the devices from NVML (with `CUDA_VISIBLE_DEVICES` applied) or a short child process, `cuInit` only as the last resort; `a53a8c3`: the C++ errors name fork and the remedy; `3d94e37`: `show_config()` no longer initializes CUDA either (the fork test's `show_config` case failed before it). Fresh-process fork tests: nothing, `__version__`, `show_config()`, CPU work, then CUDA in the child |
| A GPU below sm_75 got the plugin and failed every default call with error 209 (reported twice) | `6dc5f30`: a plugin is usable only when a visible GPU has compute capability 7.5 or newer (the release lists' floor, tied by a test); the missing-kernel error names the device's compute capability; troubleshooting rows |
| `DYNG_CPU_ONLY=false/no/off` forced the CPU module (reported twice) | `34cfbaa`: a boolean (1/true/yes/on, 0/false/no/off/empty, any case), other values ignored with a warning |
| Stream handle 0 was the per-thread stream in Python, the legacy stream in C++ | `508dfc5`: `None` is the per-thread stream, 0 (and the frameworks' default streams) the legacy stream |
| A result updated on another stream kept a dangling stream (CAI named a dead handle, `to_numpy()` crashed) | `d14aa03` kept the last writer's resources; **that was not enough**, see below: `17b4142` |
| The stream-ordering tests passed with `order_stream` disabled | `466c563`: a kernel spinning on the writer's stream after the call; the consumer must finish after it, a control without the event first |
| PyPI uploads unordered: `dyng[cu13]` could resolve to an older `dyng` without CUDA (reported twice) | `5c79770`: plugins first on each index, `dyng` only after both succeeded; recovery in `release.md` |
| No test of `release.yml`'s `select` step | `5c79770`: `ci/tests/test_release_select.py` runs the step's script from the workflow for v0.0.1, v0.0.2, v0.1.1, v0.1.2rc1, v0.2.0rc1, v0.2.0, v1.0.0 and refused tags |
| `release.md` step 7 named only `testpypi` | `5c79770` |
| The pinned CUDA RPMs were never checked against NVIDIA's signature | `c0e2787`: gpg on the runner (key accepted only by its pinned fingerprint, header signature and signed payload digest), `rpm -K` and `localpkg_gpgcheck` in the container |
| No CI check of the published wheels' SASS and PTX | `9d288f1`: `ci/cibw_plugin_repair.sh` runs `ci/wheel_check.py --code-objects` with the pinned toolkit's cuobjdump in the container; the smoke test checks the reported architectures in both modes; the lists are tied to the CMake file |
| The CI wheels were linked differently (no `--exclude-libs`), had no export check, and were never GPU-tested before release | `9d288f1`: `-Wl,--exclude-libs,ALL` in `ci/cibuildwheel-plugin.toml`, an export-list rule in `wheel_check`; `d39b925`: release steps 8 and 10 run the GPU tests of both CI-built plugins (section "The GPU tests of the published plugin wheels") |
| `wheels.yml` never built the plugins for a `cpp/` change | `67f99c8`: `cpp/**` and the licence files in the pull-request paths |

What the step found beyond the reports:

1. **The writer fix of `d14aa03` crashed with CuPy.** Its GPU test skips without CuPy, and the
   earlier verification venv had none; in the interop venv it segfaulted. A call releases the
   memory it adds to a graph or result on its own stream, possibly long after the call, so
   keeping only the last writer's resources is not enough: the graph was not covered at all
   (freeing it after the update's stream was dropped called `cudaFreeAsync` on a destroyed
   stream), and an Array that outlived its result released the stream before the result's memory
   (its owner's slots are cleared in sorted order, and the tuple it keeps releases its items last
   to first). `17b4142`: every graph and result keeps one `Resources` per distinct stream object
   used on it (so the resources' cached workspaces go before the stream too), registered by every
   call that takes the object and resources; the native object is freed first (`__del__`), the
   Array's memory before its owner, the owner's callables (which hold the native result) before
   what it keeps. ADR 0031 amendment 5 is revised.
2. **`d14aa03` had changed the results' `__slots__`**, which `ci/api_check.sh` (griffe) reports
   as a breaking change (the fresh-clone `ci/check.sh` failed on it; this milestone's earlier
   steps had avoided it on purpose). The bookkeeping is now a side table with weak keys, read
   through private properties; the slots are those of 0.1.
3. **The CI linking was checked locally**: a cu13 module built with only `-Wl,--exclude-libs,ALL`
   (the CI flags, dynamic libstdc++) exports the same 125 symbols as the local wheel, all
   accepted by the new rule, while a dev build's module (1882 exports, 1459 of them libdyng's) is
   rejected; the pinned cuobjdump 12.9 / 13.4 read the CI-toolkit wheels (SASS sm_75-sm_120, PTX
   sm_120).

**What could not be run here:** the container half of the new CI steps (gpg and `rpm -K` in the
manylinux_2_28 image, `dnf --setopt=localpkg_gpgcheck=1`, cuobjdump inside cibuildwheel's repair
step): no container runtime on this machine. The same code runs on the host (gpg, cuobjdump), and
the first pull-request run of `wheels.yml` is the real test (open item 3).

**Verification** (HEAD `2c1d710`):

| Check | Result |
|---|---|
| `ci/plugin_wheels.sh`, cu12 + cu13 with the pinned CI toolkits (12.9, 13.4), Python 3.12 and 3.13, the interop venv (PyTorch 2.14, CuPy 14.2) | rc 0 (log `$DYNG_SCRATCH/runs/m6a-fix-plugin-wheels.log`, wheels in `$DYNG_SCRATCH/wheels/m6a-fix-ci`): `dyng_cu12` 5.66 MB, `dyng_cu13` 5.93 MB; code objects, export list, twine and `wheel_check` clean; per plugin and Python: the GPU smoke test, both fallbacks, `pytest -m gpu` 53 passed / 7 skipped (no torch / cupy), the suite with `DYNG_CPU_ONLY=1` 526 passed / 64 skipped; interop venv: 60 passed per plugin |
| The release-step GPU check as documented (`DYNG_PLUGIN_TEST_ONLY=1`, the wheels sorted by `wheel_check --release-set --split` as `collect` does) | rc 0, both plugins, both Pythons (log `$DYNG_SCRATCH/runs/m6a-fix-release-gpu-check.log`) |
| `ci/check.sh` in a fresh clone (`git checkout m6a-cuda-wheels`) | all checks passed (log `$DYNG_SCRATCH/runs/m6a-fix-check.log`): `cpu-only` 678/678, `dev` 707/707, clang-tidy, reuse, provenance, regen, the harness and CI-script tests, `ci/python.sh` (mypy, pytest 526 passed / 64 skipped), griffe (no breaking change), the scaffold, `ci/docs.sh`, pre-commit (actionlint, zizmor) |
| `ci/gpu_local.sh` (dev-cuda: build, `ctest -L gpu`, `ctest -L cpu`) in the same clone, GPU 1 | all passed (log `$DYNG_SCRATCH/runs/m6a-fix-gpu-local.log`); the sanitizers, parity and the timing row were not re-run: the C++ change of this step (`a53a8c3`) touches error messages only |
| Portability pre-checks | the Clang 18 syntax pass over a `dev` + `DYNG_BUILD_PYTHON=ON` database: 188 files, 0 failures; the `cpu-only` build with `-D_FORTIFY_SOURCE=3` and the bindings: clean |

## Milestone acceptance

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | Plugin build: documented and reproducible, `dyng_cu12` / `dyng_cu13` with every backend, own `NB_DOMAIN`, static cudart, no libcuda, SASS + PTX per the toolkit's release list, `dyng==` same version, entry point `dyng.backends`, core extras `cu12` / `cu13` pinned; wheels < 90 MB; `twine check --strict`; licence files | step plugin-build (ADR 0030); `docs/developer/wheels.md`; the wheels of this step 5.5-5.9 MB, every check clean | met, except the licence **expression** of the plugins, which is the author's open decision (the licence files are complete) |
| 2 | Discovery and selection: `Resources.cuda()` from Python, the driver's major wins, version mismatch / missing driver fall back with a warning, `use_cpu_only()`, `show_config()`, the CPU-only install unchanged; tests with fake plugins and real GPU tests on GPU 1 against a local cu13 wheel (sssp / cycle_count / mosp equal to CPU and to the goldens subset; device arrays, CuPy / PyTorch round trips) | step discovery-tests (ADR 0031); re-run in this step's verification (both plugins, both Pythons, interop venv) | met; the version-mismatch and too-old-driver fallbacks are tested with fake plugins only (no such driver here) |
| 3 | CI: `wheels.yml` builds both plugins with cibuildwheel in manylinux_2_28 with the CUDA toolkits in `CIBW_BEFORE_ALL`, a GPU-free selection smoke test, artifacts, the size check; `release.yml` for tags >= v0.2.0 with per-distribution environments, TestPyPI first, PyPI only for finals, v0.0.x / v0.1.x unchanged; actionlint / zizmor clean, actions pinned and allowed | this step (ADR 0032) | met as far as it can be checked without a push: the workflows have not run on GitHub yet (see "What could not be run here") |
| 4 | Local verification: both plugin wheels in fresh 3.12 / 3.13 venvs with the core wheel, GPU tests on GPU 1; the wheel-vs-parity row under the exclusive lock with locked clocks | this step's verification and the row above | met |
| 5 | Docs (install guide, `wheels.md`, `release.md`), CHANGELOG, ADRs, this retrospective; a fresh clone passes `ci/check.sh`, `ci/python.sh`, `ci/docs.sh` and the portability pre-checks | this step | met |

## Open items for the lead maintainer

1. **The licence expression of the plugin wheels** (GOVERNANCE.md, open decisions; ADR 0030
   item 9): keep the CPU wheel's expression, or add `Apache-2.0 WITH LLVM-exception AND
   LicenseRef-NVIDIA-End-User-License-Agreement`. Due before the first `v0.2.0*` tag.
2. **The four new environments** `testpypi-cu12`, `pypi-cu12`, `testpypi-cu13`, `pypi-cu13`
   need the protection rules of `pypi` / `testpypi` (required reviewer on `pypi-*`, tags `v*`,
   no administrator bypass): `docs/developer/repository_settings.md` section 11, item 11b.
3. **The first CI run of the plugins** happens on the pull request: watch the `plugin` jobs
   (the container part: the signature checks with gpg and `rpm -K`, `dnf` installing the pinned
   RPMs, nvcc 12.9 / 13.4 with gcc-toolset 14, the code-object check in the repair step, the
   build time on four cores) and `plugin-install-test`. A manual `wheels.yml` run before the
   release candidate is the rehearsal (`docs/developer/release.md` step 4).
4. **The cycle_count update in the CUDA 13.4 wheel** was 8.8 % slower than the parity build
   (13.1) in the informational row, while the 13.1 wheel was 1.2 % faster: worth a look (an
   Nsight comparison of the update kernels under 13.1 and 13.4) before 0.2.0; no gate applies.
5. **Re-pinning the CUDA toolkits** is manual (`ci/cuda_toolkit.py lock`): check for a newer
   12.x / 13.x before each release.
6. **Disk in the work area** (not in the repository): kept are the pinned toolkits unpacked
   under `$DYNG_SCRATCH/tools/ci-cuda` (1.1 GB, used by `wheels.md`'s "CI toolkit locally") and
   the verification wheels and venvs `$DYNG_SCRATCH/wheels/m6a-final-ci` (570 MB),
   `m6a-final-cu131` (146 MB) and the review fixes' `m6a-fix-ci` (571 MB), besides the earlier
   steps' `m6a-discovery` and the interop venv.
   Deleted after use: the GCC 14 toolchain of the compiler check, the parity-cuda build and
   overlay of the timing row (`parity/wheel_vs_parity.py build-parity` rebuilds them in a few
   minutes), the fresh clone, and the intermediate wheel trees.
