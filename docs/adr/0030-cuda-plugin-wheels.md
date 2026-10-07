# ADR 0030: The CUDA plugin wheels (dyng-cu12, dyng-cu13): how they are built from the same sources

- **Status:** Accepted under delegation (2026-10-06; GOVERNANCE.md, "Delegation of technical
  ADRs"). It records how PLAN Sections 5.4 ("CPU core + CUDA plugins"), 7.7 and 7.8 and ADR 0011
  items 14-15 are implemented in M6a, step "plugin-build", and where the implementation refines
  the plan's sketch. It changes no rule the author approved: the gates and their measurement
  protocol, the parity rules, the licence and the licence expression, the names and the
  publishing steps are untouched, and nothing is published by it. One licensing question it
  meets (item 9) was left to the author, who extended the plugins' expression on 2026-10-06.
- **Date:** 2026-10-06
- **Deciders:** the AI assistant, on the author's behalf (M6a, step "plugin-build")

## Context

PLAN 5.4 ships the CUDA backends to Python as plugin wheels in the JAX pattern: `dyng` (the CPU
module `dyng._core`) is always installed; `dyng-cu12` and `dyng-cu13` each contain an import
package (`dyng_cu12`, `dyng_cu13`) with an extension module built from the same sources with
CUDA on, registered under the entry point group `dyng.backends`; `pip install "dyng[cu13]"`
pulls the plugin of the same version. PLAN 7.7 sketches the build as "the release script
rewrites `name` to `dyng-cu13`, `wheel.packages` to `dyng_cu13`, the defines to
`DYNG_ENABLE_CUDA=ON`, `DYNG_PYTHON_PLUGIN=cu13`, and adds the `dyng.backends` entry point", with
cudart linked statically, SASS per toolkit plus PTX for the highest architecture, `libcuda.so`
never bundled, the plugin depending on `dyng` of the same version and a 90 MB size budget. ADR
0011 item 14 fixed the selection contract in `dyng/_backend.py` (an entry point names a module
with `available() -> bool` and `native`), item 15 left the extras `cu12` / `cu13` out until the
plugins exist.

## Decision

1. **One source, a rendered `pyproject.toml`.** `ci/plugin_pyproject.py --plugin cu<N>
   --project-dir <tree>` turns a source tree of `dyng` (an unpacked core sdist, or a disposable
   checkout) into the plugin's: it renders the plugin's `pyproject.toml` from the root one (so
   the licence, authors, URLs, build requirements and scikit-build-core settings have one
   source), copies the plugin package into place (item 3) and the CUDA toolkit's EULA into the
   tree (item 9). The rendered file: `name = "dyng-cu<N>"`, its own description and readme
   (`python/plugin/README.md`), the CPU package's keywords and classifiers plus
   `Environment :: GPU :: NVIDIA CUDA` / `:: <N>` (without `Typing :: Typed`: the plugin package
   is private), `dependencies = ["dyng==<VERSION>"]` (the version read from `VERSION` when the
   file is rendered; the plugin's own version still comes from `VERSION` through the regex
   provider), no extras and no console script, `[project.entry-points."dyng.backends"] cu<N> =
   "dyng_cu<N>"`, `wheel.packages = ["python/plugin/dyng_cu<N>"]`, and the CPU package's CMake
   defines plus `DYNG_ENABLE_CUDA=ON`, `DYNG_ENABLE_OPENMP=ON`, `DYNG_PYTHON_PLUGIN=cu<N>`,
   `CMAKE_CUDA_RUNTIME_LIBRARY=Static` and `DYNG_CUDA_ARCHITECTURES=release`. The plugins are
   built **from the core sdist** (which therefore contains `python/plugin/` and
   `THIRD_PARTY_LICENSES_CUDA.txt`; `ci/wheel_check.py` requires them), so a release builds every
   distribution from one source archive. No plugin sdist is published (PLAN 7.7's table: the
   plugins are wheels; building one from source is documented in `docs/developer/wheels.md`).
   This is PLAN 7.7's "release script", as a tested script rather than an in-place `sed` of the
   checkout; RAPIDS automates the same with `rapids-build-backend`.
2. **CMake: `DYNG_PYTHON_PLUGIN`** (cache, empty | `cu12` | `cu13`, `python/CMakeLists.txt`).
   Empty builds `dyng._core` as before (`NB_DOMAIN dyng_cpu`, installed to `dyng/`). `cu<N>`
   builds the same target `_core` from the same bindings under `NB_DOMAIN dyng_cu<N>`, installed
   to `dyng_cu<N>/`, so the plugin's module is `dyng_cu<N>._core` (the module keeps the name
   `_core`: `NB_MODULE(_core, ...)` is unchanged). Configuring fails unless the build has CUDA
   and OpenMP (a plugin module replaces `dyng._core`, so it has every backend), the CUDA
   compiler's major equals `<N>`, and the CUDA runtime is static. The module's `build_config`
   gains `plugin`, `cuda_toolkit`, `cuda_architectures` and `cuda_runtime` (empty strings in the
   CPU module), which `dyng.show_config()` prints for a CUDA module.
3. **The plugin package** is one directory, `python/plugin/dyng_plugin/`, used by both plugins:
   `ci/plugin_pyproject.py` copies it to `python/plugin/dyng_cu<N>/` in the (disposable) plugin
   tree, because scikit-build-core requires a package's directory to carry the package's name
   (`wheel.packages` as a table `{dyng_cu13 = "python/plugin/dyng_plugin"}` is refused: "must
   match in the last component of the paths"). The package learns which plugin it is from its
   own name (`PLUGIN`, `CUDA_MAJOR`, `DISTRIBUTION`, `__version__` from the distribution's
   metadata). It implements ADR 0011 item 14's contract and adds what the selection needs to
   explain itself: `status()` probes the driver once per process through ctypes
   (`libcuda.so.1`: `cuDriverGetVersion`, `cuInit`, `cuDeviceGetCount`) and returns
   `Status(usable, driver_version, device_count, reason)`; `available()` is `status().usable`:
   a driver of CUDA `<N>`.0 or newer (CUDA's minor-version compatibility lets a plugin built with
   the latest `<N>`.x run on any `<N>`.x driver, its SASS covering the GPUs; a cu12 plugin also
   runs on a CUDA 13 driver) and at least one visible device (amended by ADR 0031, "Amendments":
   no `cuInit` in the calling process, the devices through NVML or a child process, and a
   compute capability of 7.5 or newer). `native` imports `dyng_cu<N>._core`
   on first access (a module `__getattr__`), so asking a plugin loads neither the extension
   module nor the CUDA runtime. Which plugin wins when both are installed, the warnings of a
   fallback and the version check against `dyng` are the selection's (`dyng/_backend.py`, the
   next step of M6a); the package gives it `CUDA_MAJOR`, `driver_version()` and the reason.
4. **The static CUDA runtime.** `CMAKE_CUDA_RUNTIME_LIBRARY=Static` alone only governs the links
   nvcc drives; libdyng, its CUDA modules and the test libraries linked `CUDA::cudart`
   (the shared runtime) explicitly. They now link `${DYNG_CUDART_TARGET}`: `CUDA::cudart_static`
   when `CMAKE_CUDA_RUNTIME_LIBRARY` is `Static`, `CUDA::cudart` when it is `Shared` (the default
   of every preset, unchanged); any other value is refused at configure time, and so is `Static`
   with `BUILD_SHARED_LIBS=ON`: a shared libdyng would carry its own copy of the runtime and every
   program using it another, two CUDA runtimes with separate error states in one process (the
   suite `CudaApiErrors` fails exactly so in such a build). The plugin's module
   then needs no `libcudart.so` and no `libcuda.so` (the runtime loads the driver at run time):
   its `DT_NEEDED` entries are the C library's and the bundled libgomp only.
5. **Architectures** come from `cmake/cuda_architectures.cmake`'s release list for the toolkit in
   use (`DYNG_CUDA_ARCHITECTURES=release`): for CUDA 12.8+ and 13.x
   `75-real;80-real;86-real;89-real;90-real;100-real;120` (SASS for seven architectures, PTX for
   sm_120). The cu12 wheel is built with the latest 12.x, the cu13 wheel with the latest 13.x
   (PLAN 7.8); a 12.4-12.7 toolkit would give the shorter list, which a release does not use.
6. **Repair.** `auditwheel repair --plat manylinux_2_28_x86_64` bundles libgomp as
   `dyng_cu<N>.libs/libgomp-*.so*` and is told `--exclude libcuda.so.1 --exclude libcuda.so`
   (never needed after item 4, excluded so that a future change cannot bundle the user's
   driver). The module links libstdc++ and libgcc statically with their symbols hidden, as the
   CPU wheel does (ADR 0025).
7. **Checks** (`ci/wheel_check.py`, recognising `dyng_cu<N>-*.whl` by name): the 90 MB budget;
   the file name and tags (`cp312-abi3`, the platform); only `dyng_cu<N>/` and
   `dyng_cu<N>.libs/` besides the `.dist-info` (never `dyng/`); the module and `__init__.py`; the
   entry point `[dyng.backends] cu<N> = dyng_cu<N>` and no console script; `Name: dyng-cu<N>`;
   `Requires-Dist` exactly `dyng==<VERSION>`; the licence files and expression of item 9; no
   bundled `libcuda*`, `libcudart*` or `libnvidia-*`; and, read from the module's ELF dynamic
   section, no `DT_NEEDED` entry for them. The CPU wheel's `METADATA` must offer the extras
   `cu12` and `cu13` requiring `dyng-cu<N>==<VERSION>`.
8. **The extras of `dyng`.** `optional-dependencies` becomes dynamic in the root
   `pyproject.toml`: scikit-build-core's template provider fills it, with `cu12 =
   ["dyng-cu12=={project[version]}"]` and `cu13 = ["dyng-cu13=={project[version]}"]`, so the pin
   always equals `VERSION` without a release-time rewrite (PLAN 7.7's "filled in by the release
   script"); the other extras are unchanged (`test`, `torch`, `pandas`), and PLAN 7.7's
   `cupy-cu12 = ["cupy-cuda12x"]` / `cupy-cu13 = ["cupy-cuda13x"]` are added now that the plugins
   exist (ADR 0011 item 15 deferred them to this point). The sdist's `PKG-INFO` carries the same.
9. **Licences.** A plugin wheel contains, besides what `THIRD_PARTY_LICENSES.txt` lists, the CUDA
   runtime (`libcudart_static.a`, which the CUDA Toolkit EULA's Attachment A lists as
   distributable), device code from the toolkit's headers, the CUDA C++ Core Libraries compiled
   in (CUB: BSD-3-Clause; Thrust: Apache-2.0; libcu++: Apache-2.0 WITH LLVM-exception) and NVTX 3
   (Apache-2.0 WITH LLVM-exception). Its licence files are the CPU wheel's plus
   `THIRD_PARTY_LICENSES_CUDA.txt` (committed: what is in the module, CUB's notice, the LLVM
   exception) and `NVIDIA_CUDA_EULA.txt`, the EULA **of the toolkit the wheel is built with**,
   copied in by `ci/plugin_pyproject.py` (from `<toolkit>/EULA.txt`, a conda prefix's `LICENSE`,
   or the `cuda-cudart-static` package that a conda-forge environment's `conda-meta` names;
   `--cuda-eula` overrides; the build fails without it). The EULA is not committed: it is
   NVIDIA's text, it changes with the toolkit, and the repository's files are Apache-2.0 (and
   one CC-BY-SA-4.0). **The licence expression** of the plugins is the CPU wheel's plus what they
   add: `Apache-2.0 AND BSD-3-Clause AND MIT AND GPL-3.0-or-later WITH GCC-exception-3.1 AND
   Apache-2.0 WITH LLVM-exception AND LicenseRef-NVIDIA-End-User-License-Agreement` (libcu++ and
   NVTX; the CUDA runtime under NVIDIA's EULA, with conda-forge's identifier; CUB's BSD-3-Clause
   and Thrust's Apache-2.0 are already named). This is the author's decision of 2026-10-06
   (GOVERNANCE.md, approvals log); until then the plugins carried the CPU wheel's expression.
   It is `ci/wheel_check.py`'s `PLUGIN_LICENSE_EXPRESSION`, which `ci/plugin_pyproject.py`
   writes into the rendered `pyproject.toml` and `ci/wheel_check.py` checks in the wheel's
   `METADATA`; the CPU wheel and the sdist keep their expression.
10. **Local build.** `ci/plugin_wheels.sh` is the plugins' counterpart of `ci/wheels.sh`: the core
    sdist and wheel (ci/wheels.sh, build only), then per plugin the tree of item 1, `pip wheel`
    with the glibc 2.28 conda toolchain of `ci/wheel-toolchain.yml` as host compiler and the
    plugin's CUDA toolkit, item 6's repair, `twine check --strict`, `ci/wheel_check.py`, and an
    install test per Python version (3.12, 3.13) in a fresh venv holding only the two wheels:
    the plugin is selected, sssp, cycle_count and mosp run on the CUDA backend (GPU 1), no
    visible device falls back to `dyng._core`, and the pytest suite passes with
    `DYNG_CPU_ONLY=1`. CI builds the same wheels with cibuildwheel in the manylinux_2_28 image
    (the CI step of M6a).

## Consequences

- One code base, one sdist, three wheels per platform. The plugins add no Python code of their
  own beyond `dyng_plugin` (about 150 lines); everything user-facing stays in `dyng`.
- The wheels are small: 5.5 MB (cu13) and 5.6 MB (cu12), well below the 90 MB budget (the module
  is 28 MB uncompressed, 24 MB of it the device code of the eight fatbin entries).
- Upgrading `dyng` without its plugin leaves a plugin of another version installed; the
  selection must refuse it (the next step of M6a) rather than load a module whose bindings
  differ from the typed layer.
- `CMAKE_CUDA_RUNTIME_LIBRARY=Static` now works for every C++ build with a static libdyng
  (embedding without `libcudart.so`), not only for the wheels; the `dev-cuda` preset with
  `-DCMAKE_CUDA_RUNTIME_LIBRARY=Static -DBUILD_SHARED_LIBS=OFF` passes `ctest -L gpu`.
- The licence expression of the plugins awaits the author's decision (item 9); the licence
  files are complete in either case.
