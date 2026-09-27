# Install

## Requirements

| Tool | Version |
|---|---|
| Operating system | Linux (x86-64); other platforms are untested |
| C++ compiler | GCC 11 or newer, or Clang 15 or newer (CI builds with GCC 12/13 and Clang 17/18) |
| CMake | 3.30 or newer, and Ninja |
| OpenMP | optional; enables the `openmp` backend |
| CUDA | 12.4 or newer, for the CUDA backend (arrives in milestone M1b) |

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

`ci/check.sh` runs the whole local gate (formatting, the `cpu-only` and `dev` presets, naming
checks, REUSE, the documentation build and pre-commit); CI runs the same steps.

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

Planned for 0.1: `pip install dyng` (CPU backends), and CUDA plugin wheels
(`pip install "dyng[cu13]"`) in 0.1.x. The `dyng` 0.0.1 package on PyPI only reserves the name.
