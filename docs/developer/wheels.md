# Wheels, the sdist and the release workflow

How the Python distributions of dynG are built, checked and published (PLAN Sections 7.7, 7.9,
8.8 and 10.3; ADR 0011, ADR 0025). Nothing in this page publishes anything by itself: uploads
happen only in `release.yml`, for a release tag (pushed by the author, or by the AI assistant on
the author's behalf, GOVERNANCE.md, 2026-09-30), and PyPI needs the author's approval of the
`pypi` deployment.

## What is built

| Distribution | Contents | Built by |
|---|---|---|
| `dyng-<version>.tar.gz` | the source of the root `pyproject.toml`: `VERSION`, `CMakeLists.txt`, `cmake/`, `cpp/`, `python/` and the licence files (no `parity/`, `docs/`, `tools/`, `ci/`, `.github/`, and none of the repository-only files: the CC-BY-SA-4.0 Code of Conduct, governance pages, tool configuration: every file of the sdist is Apache-2.0) | `python -m build --sdist` |
| `dyng-<version>-cp312-abi3-manylinux_2_28_x86_64.whl` | `dyng/_core.abi3.so` (nanobind stable ABI, sequential + OpenMP backends, libdyng and libstdc++ linked in), the typed layer `dyng/*.py` with `py.typed` and `_core.pyi`, the `dyng` console script, `dyng.libs/libgomp-*.so*` bundled by auditwheel, and in `.dist-info/licenses` `LICENSE`, `NOTICE`, `LICENSES/Apache-2.0.txt` and `THIRD_PARTY_LICENSES.txt` (the licences of nanobind, robin-map and the GCC runtime, which the wheel contains) | cibuildwheel from the sdist (CI), `ci/wheels.sh` from the sdist (locally) |

One abi3 wheel serves every CPython from 3.12. The CUDA plugin wheels (`dyng-cu12`,
`dyng-cu13`) follow in 0.1.x.

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
Apache-2.0, carries the same expression as the wheel. `ci/wheel_check.py` fails when
`pyproject.toml`'s `license` differs from its `LICENSE_EXPRESSION`, when a wheel's `METADATA` or
an sdist's `PKG-INFO` has another `License-Expression`, a legacy `License:` field, or no
`License-File:` line for `LICENSE`, `NOTICE` or `THIRD_PARTY_LICENSES.txt`. Changing the
expression (for example when a bundled component changes) is a licensing decision of the author:
change `pyproject.toml`, `LICENSE_EXPRESSION` and this page together.

## In CI

| Workflow | Trigger | What it does |
|---|---|---|
| `python.yml` | pull requests, `main` | `ci/python.sh` (the editable development install, the stubs check `scripts/regen.py --stubs --check`, `mypy --strict`, the pytest suite with the Hypothesis profile `ci`); the sdist built, installed into a fresh venv (an isolated build from PyPI) and tested on Python 3.12 and 3.13 |
| `wheels.yml` | pull requests that touch the packaging, manual, and called by `release.yml` | the sdist; the wheel built **from that sdist** (its artifact, cibuildwheel's `package-dir`) with cibuildwheel in the manylinux_2_28 image (`[tool.cibuildwheel]` in `pyproject.toml`: built for cp312, a pytest subset run in the built wheel under cp312 and again under cp313); `twine check` and `ci/wheel_check.py`; the wheel alone installed into fresh venvs on 3.12 and 3.13 with the whole pytest suite; artifacts `sdist` and `wheel-cpu-manylinux_2_28_x86_64` |
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

## Releasing

The release checklist is {doc}`release` (PLAN Section 10.3), which says who does each step. For
the Python distributions:

1. Bump `VERSION` (for example `0.1.0rc1`, then `0.1.0`), `CITATION.cff` and the CHANGELOG, in a
   pull request.
2. Push the tag `v<VERSION>` (the author, or the AI assistant on the author's behalf).
   `release.yml` selects the package: a `v0.0.x` tag equal to
   `tools/name_reservation`'s version builds the name-reservation package (ADR 0014); every other
   tag must equal `VERSION` and builds the real distributions by calling `wheels.yml`. The
   `collect` job checks them again and uploads one artifact `dist`.
3. `publish-testpypi` uploads to TestPyPI (environment `testpypi`, Trusted Publishing).
   Pre-releases (`a`, `b`, `rc`, `dev`) stop here: smoke-install the RC from TestPyPI on a clean
   machine (CPU and GPU machine, CPU backends).
4. For a final version, `publish-pypi` waits for the author's approval of the `pypi` environment
   and uploads the same files to PyPI.

The workflow file name `release.yml` and the environment names `testpypi` and `pypi` are bound
to the trusted publishers on TestPyPI and PyPI: never rename them.
