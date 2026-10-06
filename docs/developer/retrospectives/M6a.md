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
