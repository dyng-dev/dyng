# ADR 0113: The cu13 plugin re-pinned to CUDA 13.5; the EULA without a documentation package

- **Status:** Accepted under delegation (2026-11-20; R040, step release-changes; GOVERNANCE.md,
  "Delegation of technical ADRs"), technical scope only. It applies PLAN 7.8's existing rule (the
  plugin wheels are built with the latest 12.x and the latest 13.x; ADR 0032) and changes no rule
  the author approved: no gate, parity rule, licence term, package name or publishing step.
- **Date:** 2026-11-20
- **Deciders:** the AI assistant, on the author's behalf
- **Builds on:** ADR 0030 (the plugin wheels; item 9, the toolkit's EULA shipped as a licence
  file), ADR 0032 (the toolkits pinned file by file in `ci/cuda_toolkits.toml`).

## Context

Each release checks before its tag that `ci/cuda_toolkits.toml` still pins the newest 12.x and
13.x `nvcc` of NVIDIA's RHEL 8 repository (R030 and R031 found 12.9 and 13.4 both times). On
2026-11-20 the repository's newest are `cuda-nvcc-12-9` 12.9.86 (unchanged) and
`cuda-nvcc-13-5` 13.5.37 (new); the shared, version-independent package
`cuda-toolkit-config-common` is now 13.5.37 as well.

`ci/cuda_toolkit.py lock --plugin cu13 --release 13.5` failed: its root packages included
`cuda-documentation-{s}`, the package that carried the toolkit's `EULA.txt`, and the repository
has no `cuda-documentation-13-5` (it has one for every release from 10.1 to 13.4). Its file lists
show where the EULA went: `/usr/local/cuda-13.5/EULA.txt` is in `cuda-cudart-13-5`, which the
runtime's closure (`cuda-cudart-devel-13-5` requires it) already contains.

## Decision

1. `ci/cuda_toolkit.py` takes the documentation package as an **optional** root package
   (`OPTIONAL_ROOT_PACKAGES`): it is pinned when the release has one (every release up to 13.4,
   so a 12.9 lock is unchanged) and not required otherwise. The required roots stay `nvcc`, the
   runtime's development package, NVTX and cuobjdump.
2. The EULA stays checked where it is used: `extract` fails unless the pinned files give
   `<root>/EULA.txt` (besides `bin/nvcc`), `ci/cibw_plugin.sh` already fails the CI build
   without it, and `ci/plugin_pyproject.py` accepts only a file that is the CUDA Toolkit EULA.
   So a release whose EULA is in neither package fails the build instead of shipping a wheel
   without it.
3. Re-pinned: `cu13` to CUDA 13.5 (nvcc 13.5.37; 13 RPMs, 127 MB, no documentation package);
   `cu12` stays CUDA 12.9 (nvcc 12.9.86), its lock changing only in `cuda-toolkit-config-common`
   (13.4.92 to 13.5.37), the newest of that shared package. Every pinned RPM was downloaded and
   passed `verify --signatures` (NVIDIA's key, fingerprint
   `610C7B14E068A878070DA4E99CD0A493D42D0685`).
4. The release list of architectures is the same for 13.5 as for 12.8 and later
   (`ci/wheel_check.py`'s `RELEASE_ARCHITECTURES`); no change.
5. `cuda-build.yml`'s compile matrix (`CUDA 13.4.1 (ci-cuda13)` beside 13.1.1 and 12.9.2) is not
   changed here: its job names are required status checks of the branch protection (repository
   settings, the orchestrator's and the author's), and that job compiles from conda-forge's
   packages, not from the pinned RPMs. Moving it to 13.5 is left to a pull request that changes
   the required checks with it (R040's open items).

## Consequences

- The cu13 plugin of 0.4.0 is built with CUDA 13.5 and statically links its runtime; CUDA's
  minor-version compatibility keeps the driver requirement of 13.x (580.65.06 or newer). The local
  rebuild with the unpacked CI toolkits (`ci/plugin_wheels.sh`, both plugins) and its install
  tests on GPU 1, under a CUDA 13.1 driver, are recorded in R040's retrospective.
- Tests: `ci/tests/test_cuda_toolkit.py` locks a 13.5-style repository without a documentation
  package, keeps the 13.4-style lock with it, still fails on a missing required root, and checks
  that `extract` refuses a toolkit without `EULA.txt`.
- The docs that name the toolkit (`docs/getting_started/install.md`'s table,
  `docs/developer/wheels.md`) say 13.5.
