# Wheels, the sdist and the release workflow

How the Python distributions of dynG are built, checked and published (PLAN Sections 7.7, 7.9,
8.8 and 10.3; ADR 0011, ADR 0025, and for the CUDA plugins ADRs 0030 to 0032). Nothing in this
page publishes anything by itself: uploads happen only in `release.yml`, for a release tag
(pushed by the author, or by the AI assistant on the author's behalf, GOVERNANCE.md,
2026-09-30), and PyPI needs the author's approval of each `pypi*` deployment.

## What is built

| Distribution | Contents | Built by |
|---|---|---|
| `dyng-<version>.tar.gz` | the source of the root `pyproject.toml`: `VERSION`, `CMakeLists.txt`, `cmake/`, `cpp/`, `python/` and the licence files, and `docs/references.bib` (compiled into the library for `dyng::citation()`; no other part of `docs/`, no `parity/`, `tools/`, `ci/`, `.github/`, and none of the repository-only files: the CC-BY-SA-4.0 Code of Conduct, governance pages, tool configuration: every file of the sdist is Apache-2.0) | `python -m build --sdist` |
| `dyng-<version>-cp312-abi3-manylinux_2_28_x86_64.whl` | `dyng/_core.abi3.so` (nanobind stable ABI, sequential + OpenMP backends, libdyng and libstdc++ linked in), the typed layer `dyng/*.py` with `py.typed` and `_core.pyi`, the `dyng` console script, `dyng.libs/libgomp-*.so*` bundled by auditwheel, and in `.dist-info/licenses` `LICENSE`, `NOTICE`, `LICENSES/Apache-2.0.txt` and `THIRD_PARTY_LICENSES.txt` (the licences of nanobind, robin-map and the GCC runtime, which the wheel contains) | cibuildwheel from the sdist (CI), `ci/wheels.sh` from the sdist (locally) |

| `dyng_cu12-<version>-cp312-abi3-manylinux_2_28_x86_64.whl`, `dyng_cu13-...` (from 0.2.0) | the CUDA plugins (PLAN 5.4; ADR 0030): `dyng_cu<N>/_core.abi3.so` (every backend: sequential, OpenMP, CUDA; nanobind domain `dyng_cu<N>`; the CUDA runtime, libdyng and libstdc++ linked in; SASS for sm_75, 80, 86, 89, 90, 100, 120 and PTX for sm_120), the package `dyng_cu<N>/__init__.py` (from `python/plugin/dyng_plugin`), `dyng_cu<N>.libs/libgomp-*.so*`, the entry point `[dyng.backends] cu<N> = dyng_cu<N>`, `Requires-Dist: dyng==<version>`, and the licence files of the CPU wheel plus `THIRD_PARTY_LICENSES_CUDA.txt` and `NVIDIA_CUDA_EULA.txt`. Never `libcuda` (the user's driver) or `libcudart` (linked statically) | the core sdist turned into the plugin's tree by `ci/plugin_pyproject.py`; cibuildwheel with the pinned CUDA toolkit (CI, ADR 0032), `ci/plugin_wheels.sh` (locally), see below |

One abi3 wheel serves every CPython from 3.12. The CPU wheel's extras `cu12` and `cu13` require
the plugin of exactly its version (`pip install "dyng[cu13]"`; dynamic metadata, ADR 0030 item
8).

Every distribution passes `twine check --strict` and `ci/wheel_check.py`: the size budget of
PLAN 7.7 (a wheel above 90 MB fails; the CPU wheel is about 1.8 MB), the file name and tags
(`cp312-abi3`, `manylinux_2_28_x86_64`), the contents listed above and, for the sdist, the files
a source build needs and none of the excluded trees or repository-only files. `ci/wheel_check.py
--version-info` checks that `VERSION` is a canonical PEP 440 version (`release.yml` refuses a tag
otherwise).

**Licence metadata.** Both distributions carry core metadata 2.4 (PEP 639) with

```text
License-Expression: Apache-2.0 AND BSD-3-Clause AND MIT AND GPL-3.0-or-later WITH GCC-exception-3.1
License-File: LICENSE
License-File: NOTICE
License-File: LICENSES/Apache-2.0.txt
License-File: THIRD_PARTY_LICENSES.txt
```

dynG itself is Apache-2.0; the other terms are the licences of what the wheel contains besides
dynG's code (`THIRD_PARTY_LICENSES.txt`): nanobind (BSD-3-Clause) and robin-map (MIT), linked
into `_core.abi3.so`, and the GCC runtime (libstdc++ and libgcc linked statically, libgomp bundled
by auditwheel; GPL-3.0-or-later WITH GCC-exception-3.1). The expression is the author's decision of
2026-09-30 (GOVERNANCE.md, approvals log; until then it was `Apache-2.0`). It comes from the one
`[project]` table of `pyproject.toml` (`license`), so the sdist, whose own files are all
Apache-2.0, carries the same expression as the wheel: for the sdist it is broader than its
files need (it names the licences of the wheels built from it), never narrower; a repackager
of the sources alone (for example the conda-forge recipe from 0.3) can declare `Apache-2.0` for
them (`THIRD_PARTY_LICENSES.txt` says the same). `ci/wheel_check.py` fails when
`pyproject.toml`'s `license` differs from its `LICENSE_EXPRESSION`, when a wheel's `METADATA` or
an sdist's `PKG-INFO` has another `License-Expression`, a legacy `License:` field, or no
`License-File:` line for `LICENSE`, `NOTICE` or `THIRD_PARTY_LICENSES.txt`. Changing the
expression (for example when a bundled component changes) is a licensing decision of the author:
change `pyproject.toml`, `LICENSE_EXPRESSION` and this page together.

## In CI

| Workflow | Trigger | What it does |
|---|---|---|
| `python.yml` | pull requests, `main` | `ci/python.sh` (the editable development install, the stubs check `scripts/regen.py --stubs --check`, `mypy --strict`, the pytest suite with the Hypothesis profile `ci`); the sdist built, installed into a fresh venv (an isolated build from PyPI) and tested on Python 3.12 and 3.13 |
| `wheels.yml` | pull requests that touch the packaging, the library (`cpp/`) or the licence files, manual, and called by `release.yml` | the sdist; the wheel built **from that sdist** (its artifact, cibuildwheel's `package-dir`) with cibuildwheel in the manylinux_2_28 image (`[tool.cibuildwheel]` in `pyproject.toml`: built for cp312, a pytest subset run in the built wheel under cp312 and again under cp313); `twine check` and `ci/wheel_check.py`; the wheel alone installed into fresh venvs on 3.12 and 3.13 with the whole pytest suite; artifacts `sdist` and `wheel-cpu-manylinux_2_28_x86_64`. With the input `plugins` (default; `release.yml` sets it from v0.2.0) also the CUDA plugins from the same sdist (jobs `plugin` and `plugin-install-test`, {ref}`wheels-plugins-ci`): artifacts `wheel-cu12-manylinux_2_28_x86_64` and `wheel-cu13-manylinux_2_28_x86_64` |
| `release.yml` | a tag `v*` | see below |

`ci/python.sh` locally installs into the active environment, except when that environment's
`dyng` is an editable install of another checkout (the shared `dyng-dev` environment seen from a
fresh clone or a review worktree): re-pointing it would break the environment once the clone is
deleted, and concurrent runs would race on it, so the script then uses a throwaway venv
`build/py-venv` of the checkout (`--system-site-packages`, reusing the environment's packages).
`DYNG_PYTHON_INSTALL=force` re-points the environment anyway, `=venv` always uses the venv, `=0`
installs nothing.

## Locally (no container runtime)

cibuildwheel's Linux builds need Docker or Podman, which the development machine does not have
(PLAN 8.8). `ci/wheels.sh` builds the same distributions without a container:

```bash
source scripts/dev_env.sh
# once: the glibc 2.28 toolchain, the wheel tools and a Python 3.13
conda env create -p "$DYNG_SCRATCH/tools/manylinux228-tc" -f ci/wheel-toolchain.yml
python -m venv "$DYNG_SCRATCH/tools/wheeltools"
"$DYNG_SCRATCH/tools/wheeltools/bin/pip" install auditwheel patchelf twine
conda create -p "$DYNG_SCRATCH/tools/py313" -c conda-forge python=3.13
# every time
flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/wheels.sh     # or: ci/check.sh --wheels
```

It builds the sdist, then the wheel **from the sdist** with `pip wheel`, repairs it with
`auditwheel repair --plat manylinux_2_28_x86_64` (bundling libgomp), runs `twine check` and
`ci/wheel_check.py`, and installs the wheel alone into a fresh venv for each Python version
(3.12 and 3.13): an import from an empty directory, sssp and cycle_count on both host backends,
`dyng --version` and `dyng config`, then the whole pytest suite against the installed wheel (a
report of leaked nanobind objects fails the run). The distributions land in
`$DYNG_SCRATCH/wheels/<version>/dist`.

The differences from the CI build:

| | CI (`wheels.yml`) | locally (`ci/wheels.sh`) |
|---|---|---|
| environment | the manylinux_2_28 image (AlmaLinux 8, glibc 2.28) | Debian 12 (glibc 2.36) with conda-forge's GCC 12 against a glibc 2.28 sysroot (`ci/wheel-toolchain.yml`) |
| compiler | the image's gcc-toolset (libstdc++ newer than the system's through `libstdc++_nonshared`) | GCC 12 with libstdc++ and libgcc linked statically and their symbols hidden (`-static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL`) |
| build front end | `build` (isolated, dependencies from PyPI) | `pip wheel --no-build-isolation` from the sdist, with the pinned scikit-build-core and nanobind of `environment.yml` |
| repair | auditwheel inside the image | auditwheel from `$DYNG_SCRATCH/tools/wheeltools`, `--plat manylinux_2_28_x86_64 --only-plat`, libgomp from the conda toolchain |
| tests | a pytest subset in the built wheel (cibuildwheel), then the whole suite (install-test job) | the whole suite in fresh venvs |
| bundled OpenMP runtime | the image's libgomp (GCC 12 toolset) | the conda toolchain's libgomp, pinned to GCC 12 (`libgomp=12.*` in `ci/wheel-toolchain.yml`) |

The system GCC alone cannot produce a manylinux_2_28 wheel: linked against glibc 2.36 and the
system libstdc++, `auditwheel show` reports `manylinux_2_34`. Two pitfalls found while setting this
up (ADR 0025): a statically linked libstdc++ must have its symbols hidden, or the module's
references bind to a `libstdc++.so.6` another extension (NumPy) loaded first and formatting a
number into a `std::ostringstream` crashes; and the process-wide default resources are dropped at
interpreter exit, or nanobind reports them as leaked when the module is finalized.

## The CUDA plugin wheels

The plugins `dyng-cu12` and `dyng-cu13` are built from the same sources as `dyng` (ADR 0030):

1. **The source tree.** Unpack the core sdist (or use a disposable checkout) and run
   `python3 ci/plugin_pyproject.py --plugin cu13 --project-dir <tree> --cuda-root <toolkit>`.
   It replaces the tree's `pyproject.toml` with the plugin's, rendered from the root one
   (`--print` shows it): `name = "dyng-cu13"`, `dependencies = ["dyng==<VERSION>"]`, the entry
   point, `wheel.packages = ["python/plugin/dyng_cu13"]` (a copy of `python/plugin/dyng_plugin`
   it makes in the tree), and the defines `DYNG_ENABLE_CUDA=ON`, `DYNG_ENABLE_OPENMP=ON`,
   `DYNG_PYTHON_PLUGIN=cu13`, `CMAKE_CUDA_RUNTIME_LIBRARY=Static`,
   `DYNG_CUDA_ARCHITECTURES=release`. It also copies the toolkit's EULA into the tree as
   `NVIDIA_CUDA_EULA.txt` (found as `<toolkit>/EULA.txt`, a conda prefix's `LICENSE`, or through
   a conda-forge environment's `conda-meta`; `--cuda-eula <file>` otherwise).
2. **The wheel.** Build that tree with the toolkit of the plugin's major: `CUDACXX=<toolkit>/bin/nvcc
   pip wheel --no-deps <tree>` (CMake refuses a toolkit of another major, a shared CUDA runtime
   or a build without OpenMP). The cu12 wheel is built with the latest CUDA 12.x, the cu13 wheel
   with the latest 13.x (PLAN 7.8).
3. **Repair and checks.** `auditwheel repair --plat manylinux_2_28_x86_64 --exclude libcuda.so.1`,
   `twine check --strict`, `ci/wheel_check.py` (it recognises a plugin wheel by its name).

`ci/plugin_wheels.sh` does all of it locally, with the toolchain of `ci/wheel-toolchain.yml` as
host compiler, and then installs each plugin with the core wheel (and nothing else from dynG)
into fresh venvs for Python 3.12 and 3.13: the plugin must be selected and run sssp,
cycle_count and mosp on the CUDA backend (GPU 1) with arrays equal to the sequential backend's,
a process without a visible device must fall back to `dyng._core` with a `dyng.BackendWarning`,
the GPU tests of `python/tests` must pass (`pytest -m gpu`, `test_cuda.py`, with
`DYNG_REQUIRE_CUDA=1` so that they cannot pass by skipping; ADR 0031), and the whole suite must
pass with `DYNG_CPU_ONLY=1`. The fresh venvs hold no PyTorch or CuPy, so the round trips with
them skip there; `DYNG_PLUGIN_INTEROP_PYTHON=<python of a venv with torch / cupy>` reinstalls the
core and plugin wheels into that venv, runs the GPU tests there too, and uninstalls the plugin
again (on the development machine: `$DYNG_SCRATCH/venvs/m6a-interop-3.12`, PyTorch 2.14 cu130
and CuPy 14 for CUDA 13; 5.6 GB, so it is made once and not per run):

```bash
source scripts/dev_env.sh
# once, besides the tools of the CPU wheel: a CUDA 12.x toolkit for cu12 (none is installed
# system-wide on the development machine; conda-forge's packages work; cuda-cuobjdump is needed
# for the check of the module's SASS and PTX, and the script refuses a toolkit without it)
conda create -p "$DYNG_SCRATCH/tools/cuda-12.9" -c conda-forge --override-channels \
  cuda-version=12.9 cuda-nvcc cuda-cudart-dev cuda-cudart-static cuda-cccl cuda-nvtx-dev \
  cuda-cuobjdump
# (an environment made before 2026-10-06 without it:
#  conda install -p "$DYNG_SCRATCH/tools/cuda-12.9" -c conda-forge --override-channels \
#    cuda-version=12.9 cuda-cuobjdump)
# every time: cu13 with /usr/local/cuda-13.1 (the default), cu12 with that toolkit
DYNG_PLUGINS="cu12 cu13" DYNG_CUDA12_ROOT="$DYNG_SCRATCH/tools/cuda-12.9" \
  flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/plugin_wheels.sh
# once: the interop venv for the PyTorch / CuPy round trips (CUDA 13 builds of both)
python3.12 -m venv "$DYNG_SCRATCH/venvs/m6a-interop-3.12"
"$DYNG_SCRATCH/venvs/m6a-interop-3.12/bin/pip" install numpy "pytest>=8" "hypothesis==6.167.1" \
  cupy-cuda13x
"$DYNG_SCRATCH/venvs/m6a-interop-3.12/bin/pip" install torch \
  --index-url https://download.pytorch.org/whl/cu130
DYNG_PLUGIN_INTEROP_PYTHON="$DYNG_SCRATCH/venvs/m6a-interop-3.12/bin/python" \
  flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/plugin_wheels.sh
```

The install test also runs the smoke test of the hosted runners (below): with the driver hidden
by `ci/without_cuda_driver.sh` (a user and mount namespace in which `libcuda.so.1` is an empty
file; no root needed), `ci/plugin_smoke.py --expect fallback --reason "no CUDA driver"` must
pass. The GPU smoke test itself is `ci/plugin_smoke.py --expect cuda`.

The local GPU gate `ci/gpu_local.sh` has the same as its step `plugin` (cu13 and one Python by
default, about four minutes; `DYNG_GPU_PLUGIN_WHEELS=<dir>` tests wheels built before).

The wheels land in `$DYNG_SCRATCH/wheels/<version>-plugins/dist` (the core sdist and wheel they
were built with in `.../core/dist`). `DYNG_WHEEL_SKIP_TESTS=1` builds and checks only;
`DYNG_PLUGIN_TEST_ONLY=1` tests the wheels built before. The script's header lists every
setting.

The local builds of 2026-10-06 (M6a): cu13 with CUDA 13.1 (`/usr/local/cuda-13.1`), cu12 with
CUDA 12.9 (conda-forge's packages, the recipe above in a fresh prefix, `cuda-cuobjdump`
included; re-run after the acceptance found the recipe without it); 5.5 MB and 5.7 MB; both
`manylinux_2_28_x86_64` (auditwheel reports the module itself consistent with
`manylinux_2_17`), `twine check --strict` and `ci/wheel_check.py` clean; the install tests pass
on Python 3.12 and 3.13 with the RTX A5000 (driver 590.48.01, CUDA 13.1). How these builds
differ from CI's is the table of the CPU wheel above, plus: the toolkits (CI: NVIDIA's RHEL 8
packages of the latest 12.x and 13.x installed in the manylinux_2_28 image; locally: the system
CUDA 13.1 and conda-forge's CUDA 12.9).

(wheels-plugins-ci)=
### The plugins in CI

`wheels.yml` builds both plugins from the run's sdist (ADR 0032):

- **job `plugin`** (matrix `cu12`, `cu13`): the sdist unpacked into `plugin-src/`; the CUDA
  toolkit's RPM files downloaded on the runner by `python ci/cuda_toolkit.py download --plugin
  cu<N> --dest .cuda-rpms` (cached between runs with `actions/cache`, except for tags; every
  file checked against its SHA-256 and against NVIDIA's OpenPGP signature, see below); then
  cibuildwheel with `package-dir: plugin-src` and
  `config-file: ci/cibuildwheel-plugin.toml`. In the manylinux_2_28 container, `before-all`
  (`ci/cibw_plugin.sh`) checks the files again (digests; signatures with gpg and `rpm -K`),
  installs them with `dnf` (`localpkg_gpgcheck=1`), links
  `/usr/local/cuda`, checks that nvcc is the pinned release and compiles and links a kernel with
  the image's gcc-toolset and the static runtime, and renders the plugin's tree with
  `ci/plugin_pyproject.py --cuda-root /usr/local/cuda`; cibuildwheel builds the abi3 wheel once
  (cp312, four compile jobs; the module linked with `-Wl,--exclude-libs,ALL`) and repairs it
  with `ci/cibw_plugin_repair.sh` (`auditwheel repair --plat manylinux_2_28_x86_64 --only-plat
  --exclude libcuda.so.1 --exclude libcuda.so`, then `ci/wheel_check.py --code-objects` with
  the toolkit's cuobjdump: the module must carry exactly the SASS of the toolkit's release list
  and the PTX of its last entry, so a lost architecture fails the build). `twine check
  --strict` and `ci/wheel_check.py` (90 MB budget and the module's export list included) check
  the wheel; artifact
  `wheel-cu<N>-manylinux_2_28_x86_64`. About 10 to 20 minutes per plugin.
- **job `plugin-install-test`** (plugin x Python 3.12, 3.13): the run's CPU wheel and plugin
  wheel in a fresh venv (`--no-index`: `dyng==<version>` is the run's core wheel; NumPy and the
  test tools from PyPI). The runners have no GPU and no driver (checked), so the test is the
  fallback: `ci/plugin_smoke.py --expect fallback --reason "no CUDA driver"` (one
  `dyng.BackendWarning`, `dyng._core` active, the plugin `unusable` for that reason, the plugin's
  module importing without a driver and reporting plugin, toolkit, static runtime and
  architectures, which must be the release list of its toolkit, sssp / cycle_count / mosp on
  the CPU backends), `dyng config`, and the whole
  pytest suite (its GPU tests skip).

**The pinned toolkits** are `ci/cuda_toolkits.toml`: per plugin, the toolkit release and every
RPM file of NVIDIA's RHEL 8 repository the build installs (nvcc, the CUDA runtime with its static
library, CCCL, NVTX, the documentation package with `EULA.txt`, cuobjdump for the check of the
code objects, and their dependencies inside
NVIDIA's repository), with SHA-256 and size. The pins make the build reproducible; that the
files are NVIDIA's is checked separately, every time they are used: each RPM's header must carry
a valid OpenPGP signature by NVIDIA's repository key (`D42D0685.pub`, fetched from the
repository next to the packages and accepted only with the fingerprint
`610C7B14E068A878070DA4E99CD0A493D42D0685` pinned in `ci/cuda_toolkit.py`), and the payload
must match the digest that the signed header records (`download` and `verify --signatures`, with
gpg; in the container also `rpm -K` and dnf's `localpkg_gpgcheck`). So a lock taken from a
tampered response would fail at its first use. Now: CUDA 12.9 (nvcc 12.9.86) for `cu12`, CUDA
13.4 (nvcc 13.4.92) for `cu13`. When NVIDIA ships a newer 12.x or 13.x (PLAN 7.8: the wheels
are built with the latest of each major), re-pin in a pull request; dependabot does not see the
file:

```bash
python3 ci/cuda_toolkit.py lock --plugin cu13 --release 13.5     # reads NVIDIA's repository
python3 ci/cuda_toolkit.py show
python3 -m pytest ci/tests/test_cuda_toolkit.py
```

**The CI toolkit locally.** `ci/cuda_toolkit.py extract` unpacks the pinned files without root
(bsdtar needed, e.g. conda's `libarchive`), which builds the plugins locally with exactly the
toolkits of CI:

```bash
for p in cu12 cu13; do
  python3 ci/cuda_toolkit.py download --plugin $p --dest "$DYNG_SCRATCH/tools/ci-cuda/rpms-$p"
  python3 ci/cuda_toolkit.py extract --plugin $p --rpms "$DYNG_SCRATCH/tools/ci-cuda/rpms-$p" \
    --dest "$DYNG_SCRATCH/tools/ci-cuda/root-$p"
done
DYNG_PLUGINS="cu12 cu13" \
  DYNG_CUDA12_ROOT="$DYNG_SCRATCH/tools/ci-cuda/root-cu12/usr/local/cuda-12.9" \
  DYNG_CUDA13_ROOT="$DYNG_SCRATCH/tools/ci-cuda/root-cu13/usr/local/cuda-13.4" \
  flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/plugin_wheels.sh
```

How the plugins' CI build differs from `ci/plugin_wheels.sh`:

| | CI (`wheels.yml`) | locally (`ci/plugin_wheels.sh`) |
|---|---|---|
| CUDA toolkit | the pinned RPMs of `ci/cuda_toolkits.toml` (12.9, 13.4) installed in the image | `DYNG_CUDA12_ROOT` / `DYNG_CUDA13_ROOT`: by default the system's 13.1 and conda-forge's 12.9; the pinned RPMs unpacked as above give CI's toolkits |
| host compiler | the image's gcc-toolset 14 (cibuildwheel 4.2.1's manylinux_2_28 image, 2026.09.05-1); the system's libstdc++ / libgcc, other static libraries hidden (`-Wl,--exclude-libs,ALL`) | conda-forge's GCC 12.4 against a glibc 2.28 sysroot, libstdc++ / libgcc static and hidden (`-static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL`) |
| checks of the module | `ci/wheel_check.py`: no libcuda / libcudart, the export list (only `PyInit__core`, std, nanobind, type_info); in the container, `ci/cibw_plugin_repair.sh`: SASS + PTX equal to the toolkit's release list (cuobjdump) | the same checks (`--code-objects` with the toolkit's cuobjdump) |
| build front end | `build` (isolated, from PyPI) | `pip wheel --no-build-isolation` with `environment.yml`'s scikit-build-core and nanobind |
| tests | the fallback smoke test and the whole suite, no GPU (the release then runs the GPU tests on the CI-built wheels: {doc}`release`, steps 8 and 10) | the GPU smoke test, the fallback with no visible device and with the driver hidden, `pytest -m gpu` (GPU 1), the suite with `DYNG_CPU_ONLY=1`; optionally the PyTorch / CuPy round trips |

Building a plugin from source (for another toolkit or architecture list) is the same three
steps on a checkout; `-Ccmake.define.DYNG_CUDA_ARCHITECTURES=86` (for example) replaces the
release list. A plugin always has the version of the `dyng` it is installed with.

## Releasing

The release checklist is {doc}`release` (PLAN Section 10.3), which says who does each step. For
the Python distributions (ADR 0032 for the plugins):

1. Bump `VERSION` (for example `0.1.0rc1`, then `0.1.0`), `CITATION.cff` and the CHANGELOG, in a
   pull request.
2. Push the tag `v<VERSION>` (the author, or the AI assistant on the author's behalf).
   `release.yml` selects the package: a `v0.0.x` tag equal to
   `tools/name_reservation`'s version builds the name-reservation package (ADR 0014); every other
   tag must equal `VERSION` and builds the real distributions by calling `wheels.yml`, from
   v0.2.0 (release candidates included) with the CUDA plugins (`ci/wheel_check.py
   --release-distributions` names them). The `collect` job checks them again, checks that they
   are exactly the distributions of the release (`--release-set`), and uploads one artifact
   `dist` with a directory per distribution (`--split`).
3. Each distribution is uploaded to TestPyPI through its own environment (Trusted
   Publishing), the CUDA plugins first: `publish-testpypi-plugins` uploads `dyng-cu12` through
   `testpypi-cu12` and `dyng-cu13` through `testpypi-cu13`, then `publish-testpypi` uploads
   `dyng` through `testpypi` only if both succeeded (`dyng`'s extras pin the plugins: a `dyng`
   on an index without its plugin would make `pip install "dyng[cu13]"` fall back silently to
   an older `dyng`; ADR 0032 item 6). Pre-releases (`a`, `b`, `rc`, `dev`) stop here:
   smoke-test the RC from TestPyPI ({doc}`release`, step 8).
4. For a final version, once `dyng` reached TestPyPI, `publish-pypi-plugins` uploads the plugins
   to PyPI through `pypi-cu12` and `pypi-cu13`, then `publish-pypi` uploads `dyng` through
   `pypi`; each waits for the author's approval (three approvals for a release with the
   plugins: the two plugins in one "Review deployments" dialog, then `dyng`).

| Distribution | TestPyPI environment | PyPI environment | From |
|---|---|---|---|
| `dyng` (sdist, CPU wheel) | `testpypi` | `pypi` | 0.0.1 (the reservation), 0.1.0 |
| `dyng-cu12` (wheel) | `testpypi-cu12` | `pypi-cu12` | 0.2.0 |
| `dyng-cu13` (wheel) | `testpypi-cu13` | `pypi-cu13` | 0.2.0 |

The workflow file name `release.yml` and the six environment names are bound to the trusted
publishers on TestPyPI and PyPI (PyPI refuses two identical pending publishers, so each plugin
has its own pair): never rename them.
