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
| `asan`, `tsan`, `tsan-openmp` | host sanitizers (AddressSanitizer + UBSan; ThreadSanitizer with OpenMP off; ThreadSanitizer with OpenMP, Clang and Archer) |
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
find_package(dyng 0.2 REQUIRED)
target_link_libraries(my_app PRIVATE dyng::dyng)
```

Before 1.0 a minor release may break the API, so the package accepts only a request of its own
minor version: `find_package(dyng 0.2)` finds 0.2.x (the release candidates included) and refuses
0.1 and 0.3.

Or add the source tree with `add_subdirectory()`; the target name is the same. Everything is in
namespace `dyng`, and `#include <dyng/dyng.hpp>` includes the whole public API
({doc}`../api/cpp/index`).

## Python

The Python package `dyng` is one wheel for CPython 3.12 and newer on Linux x86-64 (abi3,
manylinux_2_28; OpenMP's runtime is bundled). It contains the sequential and OpenMP backends and
the `dyng` command line ({doc}`../api/cli`); NumPy is its only dependency.

```bash
pip install dyng                   # from PyPI (0.1.0)
python -c "import dyng; dyng.show_config()"
```

The release candidate of 0.2.0, **0.2.0rc1**, is on TestPyPI only (its dependency NumPy comes
from PyPI):

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ dyng==0.2.0rc1
```

The same package can be built from a clone (it compiles the C++ core, so it needs CMake >=
3.30 and Ninja, for example from the `dyng-dev` environment, and a C++17 compiler from the
system):

```bash
pip install .                      # the wheel of this checkout
python -m build                    # or: the sdist and the wheel into dist/
```

For development, an editable install rebuilds the extension when `dyng` is imported after a C++
change:

```bash
source scripts/dev_env.sh
pip install -e . --no-build-isolation -Ceditable.rebuild=true -Cbuild-dir=build
ci/python.sh                       # the editable install, the stubs check, mypy, pytest
```

`dyng.show_config()` prints the backends of the active native module; `dyng.Resources("cuda")`
raises `NotSupportedError` in the CPU wheel. The PyPI release 0.0.1 was only the name
reservation and contains no library. {doc}`first_update_python` runs a first update; the local
build of the release wheels is described in {doc}`../developer/wheels`.

### CUDA from Python: the plugin wheels

From 0.2 the CUDA backend comes as a plugin wheel next to `dyng`: `dyng-cu13` for an NVIDIA
driver of CUDA 13, `dyng-cu12` for CUDA 12 (ADR 0030). The extras install the plugin of the same
version:

```bash
pip install "dyng[cu13]"           # NVIDIA driver 580 or newer (CUDA 13.x)
pip install "dyng[cu12]"           # NVIDIA driver 525 or newer (CUDA 12.x)
python -c "import dyng; dyng.show_config()"
```

Until 0.2.0 is on PyPI, the plugins are in the release candidate 0.2.0rc1 on TestPyPI only:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "dyng[cu13]==0.2.0rc1"
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "dyng[cu12]==0.2.0rc1"
```

A plugin contains the CUDA runtime (no CUDA toolkit is needed), code for the GPU architectures
sm_75 to sm_120 (Turing to Blackwell) and PTX for newer ones. Which one to install: `nvidia-smi`
prints the driver's "CUDA Version"; take the plugin of that major (a cu12 plugin also runs on a
CUDA 13 driver; a cu13 plugin needs a CUDA 13 driver). With both installed, the one of the
driver's CUDA major is used.

| `nvidia-smi` "CUDA Version" (Linux driver) | Install | Built with |
|---|---|---|
| 13.x (driver 580.65.06 or newer) | `pip install "dyng[cu13]"` | CUDA 13.4 |
| 12.x (driver 525.60.13 or newer) | `pip install "dyng[cu12]"` | CUDA 12.9 |
| older, or no NVIDIA GPU | `pip install dyng` (CPU backends) | |

Requirements: Linux x86-64 with glibc 2.28 or newer (manylinux_2_28), CPython 3.12 or newer, an
NVIDIA GPU of compute capability 7.5 or newer (Turing, 2018, and later; the CUDA 13 toolkits no
longer support older GPUs), and the NVIDIA driver. CUDA's minor-version compatibility lets a
plugin built with the latest toolkit of its major run on any driver of that major. In a
container the GPU and the driver's libraries must be passed in (`docker run --gpus all`, the
NVIDIA Container Toolkit). The optional extras `cupy-cu13` / `cupy-cu12` and `torch` install
CuPy and PyTorch for the device-array round trips.

```python
import dyng

res = dyng.Resources.cuda(device=0)                 # or Resources.cuda(stream=torch_or_cupy_stream)
g = dyng.Graph.from_edges([0, 0, 1], [1, 2, 2], [4, 1, 1], resources=res)
tree = dyng.sssp.compute(g, 0)
print(tree.distances.device, tree.distances.to_numpy())   # cuda:0 [0 4 1]
```

Results of the CUDA backend stay in device memory: `torch.from_dlpack(tree.distances)` and
`cupy.asarray(tree.distances)` view them without a copy, and `to_numpy()` copies them to the host
({doc}`../api/python/index`).

**Troubleshooting.** `dyng.show_config()` names the active module (`dyng_cu13 (plugin cu13)` or
`dyng._core`), why it was chosen, and the state of every installed plugin. If a plugin is
installed but cannot be used, dynG runs on the CPU backends and issues one `dyng.BackendWarning`
with the reason:

| Reason in the warning | Remedy |
|---|---|
| `no CUDA driver: libcuda.so.1 cannot be loaded` | install the NVIDIA driver (in a container: run it with the GPU, e.g. `--gpus all`) |
| `the CUDA driver supports CUDA 12.x, but dyng-cu13 needs CUDA 13.0 or newer` | update the driver, or `pip install "dyng[cu12]"` |
| `no CUDA device is visible` | check `CUDA_VISIBLE_DEVICES` and `nvidia-smi` |
| `the visible GPU (sm_70) is older than the oldest architecture of dyng-cu12, sm_75` | the plugins need compute capability 7.5 or newer (Turing and later): use the CPU backends (`pip install dyng`), or make a newer GPU visible |
| `version 0.2.0 does not match dyng 0.2.1` | `pip install "dyng[cu13]==<the dyng version>"` (a plugin of another version is never loaded) |
| `its module cannot be loaded (...)` | reinstall the plugin; report the message if it persists |
| `Resources.cuda: the active native module is dyng._core` (an error) | no plugin is installed, or the warning above says why it is not used |

Errors of the CUDA backend itself (a `dyng.CudaError` or `NotSupportedError` after the plugin was
chosen):

| Message | Cause and remedy |
|---|---|
| `... cannot be initialized in this process (cudaErrorInitializationError); CUDA cannot be initialized in a process forked after its parent initialized CUDA ...` | a worker process started with `fork` (the default of `multiprocessing` on Linux before Python 3.14) whose parent had already used CUDA; start the workers with `multiprocessing.get_context("spawn")` (or `"forkserver"`), or use CUDA in the parent only after the workers are started. Choosing the plugin does not initialize CUDA, so a parent that only imported dynG, read `dyng.__version__` or ran CPU work can fork workers that use CUDA |
| `cudaErrorNoKernelImageForDevice ... this build of dynG has no code for device 0 (compute capability X.Y)` | the device is not covered by the plugin's architectures (sm_75 to sm_120, PTX for newer ones); with several GPUs, choose a supported one with `Resources.cuda(device=...)` or `CUDA_VISIBLE_DEVICES` |

On a machine without a GPU (a CI runner, a login node) an installed plugin is harmless: dynG
falls back to the CPU backends with the warning; set `DYNG_CPU_ONLY=1` there to silence it.

`DYNG_CPU_ONLY=1` in the environment (also `true`, `yes` or `on`, in any case; `0`, `false`,
`no`, `off` or empty leave the choice to dynG, and any other value is ignored with a
`dyng.BackendWarning`), or `dyng.use_cpu_only()` before the first use of dynG,
chooses the CPU backends without the warning; `warnings.filterwarnings("ignore",
category=dyng.BackendWarning)` silences it. Building a plugin from source is described in
{doc}`../developer/wheels`.
