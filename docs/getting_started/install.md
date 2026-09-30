# Install

Python users need only `pip install dyng` (section [Python](#python) below). The rest of the
page builds the C++ library from source, with or without CUDA, and uses it from CMake.

## Requirements

| Tool | Version |
|---|---|
| Operating system | Linux (x86-64); other platforms are untested |
| C++ compiler | GCC 11 or newer, or Clang 15 or newer (CI builds with GCC 12/13 and Clang 17/18) |
| CMake | 3.30 or newer, and Ninja |
| OpenMP | optional; enables the `openmp` backend |
| Python | 3.12 or newer, for the Python package (NumPy >= 1.26) |
| CUDA | optional; the toolkit 12.4 or newer builds the `cuda` backend, which runs on an NVIDIA GPU of compute capability 7.5 (Turing) or newer |

The conda environment in `environment.yml` provides CMake, Ninja, the formatters, Doxygen and
the documentation tools at the versions CI uses; the compiler and the CUDA toolkit come from the
system.

## Build from source

```bash
git clone https://github.com/dyng-dev/dyng.git
cd dyng
conda env create -f environment.yml    # once; `conda env update -f environment.yml` after it changes
source scripts/dev_env.sh              # activates dyng-dev, adds nvcc if present, sets DYNG_SCRATCH

cmake --preset dev                     # Debug, tests on, warnings as errors
cmake --build --preset dev
ctest --preset dev                     # unit, randomized and fixture-parity tests (no GPU)
```

Presets (`cmake --list-presets`):

| Preset | Use |
|---|---|
| `dev` | Debug development build: tests on, warnings as errors, allocation budgets on |
| `cpu-only` | Release build of the CPU backends (hosted CI, the CPU wheel) |
| `release`, `relwithdebinfo` | optimized builds |
| `asan`, `tsan` | host sanitizers (AddressSanitizer + UBSan; ThreadSanitizer with OpenMP off) |
| `parity` | the flags of the original research codes, for parity and performance runs |

`ci/check.sh` runs the whole local gate (formatting, the `cpu-only` and `dev` presets, the
clang-tidy naming checks, REUSE, the documentation build and pre-commit); the hosted workflows
`lint`, `cpu` and `docs` run the same steps on every pull request.

## Build with CUDA

The CPU presets never build CUDA. With the CUDA toolkit on `PATH` (`scripts/dev_env.sh` adds
`/usr/local/cuda-13.1/bin` when it exists; set `DYNG_CUDA_HOME` for another toolkit), the CUDA
presets build the library with the `cuda` backend next to the CPU ones:

```bash
cmake --preset dev-cuda                # Debug, this machine's GPUs (native), host code -Werror
cmake --build --preset dev-cuda
ctest --preset dev-cuda -L gpu         # the CUDA tests (they need a visible GPU)
```

| Preset | Use |
|---|---|
| `dev-cuda` | CUDA development: Debug, `native` architectures, tests on |
| `release-cuda` | the release architecture list (sm_75 to sm_120 SASS plus PTX) |
| `parity-cuda` | the flags of MOSP-CUDA (`-O3 -lineinfo`, sm_86), for parity and performance runs |
| `sanitize-cuda` | for `compute-sanitizer` runs |
| `ci-cuda13`, `ci-cuda12` | the compile-only builds of the hosted `cuda-build` workflow |

`DYNG_CUDA_ARCHITECTURES` chooses `native` (the default; without a visible GPU the configure
step warns and uses the release list), `release` or an explicit list such as `86`.
`ci/gpu_local.sh` is the local GPU gate (the CUDA tests, the golden corpus on the `cuda` backend,
`compute-sanitizer`); the hosted `cuda-build` workflow compiles the CUDA presets with CUDA 13
and 12 but runs no test, because hosted runners have no GPU.

## Use dynG from CMake

Install a build (`cmake --install build/cpu-only --prefix <prefix>`) and find it from your
project:

```cmake
find_package(dyng 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE dyng::dyng)
```

Or add the source tree with `add_subdirectory()`; the target name is the same. Everything is in
namespace `dyng`, and `#include <dyng/dyng.hpp>` includes the whole public API
({doc}`../api/cpp/index`).

## Python

The Python package `dyng` is one wheel for CPython 3.12 and newer on Linux x86-64 (abi3,
manylinux_2_28; OpenMP's runtime is bundled). It contains the sequential and OpenMP backends and
the `dyng` command line ({doc}`../api/cli`); NumPy is its only dependency.

```bash
pip install dyng                   # from PyPI once 0.1.0 is published
python -c "import dyng; dyng.show_config()"
```

Until 0.1.0 is on PyPI, build the same package from a clone (it compiles the C++ core, so it
needs CMake >= 3.30, Ninja and a C++17 compiler, for example from the `dyng-dev` environment):

```bash
pip install .                      # the wheel of this checkout
python -m build                    # or: the sdist and the wheel into dist/
```

For development, an editable install rebuilds the extension when `dyng` is imported after a C++
change:

```bash
source scripts/dev_env.sh
pip install -e . --no-build-isolation -Ceditable.rebuild=true -Cbuild-dir=build
ci/python.sh                       # the editable install, the stubs check, pytest
```

`dyng.show_config()` prints the backends of the active native module; `dyng.Resources("cuda")`
raises `NotSupportedError` in the CPU wheel. The CUDA backends come as plugin wheels in 0.1.x
(`pip install "dyng[cu13]"`, PLAN Section 7.7); until then, use CUDA from C++. The PyPI version
0.0.1 is only the name reservation and contains no library. {doc}`first_update_python` runs a
first update; the local build of the release wheels is described in
{doc}`../developer/wheels`.
