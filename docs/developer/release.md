# Release process

How a dynG release is made (PLAN Sections 10.3 and 10.4). Releases are cut from `main`, which is
always releasable; `release/X.Y` branches are created only when an older minor release needs a
patch. Versions follow SemVer 2.0.0's meaning from 0.1.0 (PLAN Section 5.9 says what the
stability promise covers), **spelled in canonical PEP 440 form**, which the Python distributions
require: `0.1.0`, `0.1.0rc1`, `0.1.0.dev0`, never SemVer's `0.1.0-rc.1`. The single source of the
version is the file `VERSION`; tags are `v` + `VERSION` (`v0.1.0rc1`, `v0.1.0`), and
`release.yml` refuses a `VERSION` that is not canonical.

**Who does what.** Tags, the approval of the PyPI upload, GitHub Releases and the registrations
(Zenodo, Read the Docs, conda-forge) are the **author's** (the lead maintainer's): they are
publishing steps, which are never delegated (GOVERNANCE.md). Everything before the tag (the
checks, the version bump, the CHANGELOG, the certificates) can be prepared by a maintainer or the
AI assistant in a pull request.

## Checklist

For each release, copy this list into the release pull request and tick it.

| # | Step | Command or place | Who |
|---|---|---|---|
| 1 | Full parity on the GPU machine: the golden corpora of every ported algorithm on every backend | `ci/check.sh --parity` (CPU backends), `ci/gpu_local.sh` (CUDA: the corpora, the sanitizers) | maintainer |
| 2 | The performance gates against the originals (manual, under the exclusive lock), the result tables and the parity certificate committed | `parity/perf_ab.py run ...` (`parity/README.md`); the certificate under `parity/results/` and, from the first release on, the machine-readable `benchmarks/results/<version>/parity.json` | maintainer |
| 3 | Sanitizers and mutation checks green | the `asan` / `tsan` presets, `compute-sanitizer` in `ci/gpu_local.sh`, the CTests `*.mutation.*` | maintainer |
| 4 | The distributions build and install | `ci/check.sh --wheels` locally; `wheels.yml` (manual run) on hosted runners | maintainer |
| 5 | The release-candidate pull request: set `VERSION` to `X.Y.ZrcN` (for example `0.1.0rc1`; `release.yml` requires the tag to equal `VERSION`); move the CHANGELOG's `Unreleased` entries under `## [X.Y.ZrcN] - <date>` (the draft summary of the release is kept at the top while it is prepared) | a pull request | maintainer |
| 6 | `version: X.Y.ZrcN` and `date-released` in `CITATION.cff` | the same pull request | maintainer |
| 7 | Release candidate: tag `vX.Y.ZrcN` (equal to `VERSION`) on `main` after the merge; `release.yml` builds the sdist and the wheels through `wheels.yml` and uploads them to TestPyPI (environment `testpypi`); pre-releases stop there | `git tag -s vX.Y.Zrc1 && git push origin vX.Y.Zrc1` | **author** |
| 8 | Smoke-install the RC from TestPyPI in a clean venv on a CPU machine and on the GPU machine (CPU backends): import, `dyng.show_config()`, `sssp` and `cycle_count` on both backends, `dyng --version` | `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ dyng==X.Y.ZrcN` | author or maintainer |
| 9 | The final-release pull request: set `VERSION` to `X.Y.Z`, rename the CHANGELOG section to `## [X.Y.Z] - <date>`, and set `version: X.Y.Z` and `date-released` in `CITATION.cff` (without a release candidate, steps 5-6 do this directly and 7-8 are skipped) | a pull request | maintainer |
| 10 | Final release: tag `vX.Y.Z` (equal to `VERSION`) on `main` after that merge; `release.yml` uploads to TestPyPI, then waits for the **approval of the `pypi` environment** and uploads the same files to PyPI | `git tag -s vX.Y.Z && git push origin vX.Y.Z`; approve the deployment in the Actions run | **author** |
| 11 | The GitHub Release from the CHANGELOG section, with the distributions attached | `gh release create vX.Y.Z --title "dynG X.Y.Z" --notes-file <the CHANGELOG section> dist/*` | **author** |
| 12 | The Zenodo DOI of the release (once Zenodo is connected, checkpoint A4), then the DOI in `CITATION.cff` and the README | Zenodo's GitHub integration | **author** |
| 13 | conda-forge: merge the bot's feedstock pull request (from 0.3) | the `dyng-feedstock` repository | author |
| 14 | Announce (Discussions); curate the next "good first issue" backlog; write the milestone retrospective; bump `VERSION` to the next `.dev0` | a pull request | maintainer |

## The release workflow

`release.yml` runs on a tag `v*` (only the author can create one: the `release tags` ruleset):

1. `select` decides what the tag builds. A `v0.0.x` tag equal to the version of
   `tools/name_reservation` builds the name-reservation package (ADR 0014); every other tag must
   equal `VERSION`, which must be a canonical PEP 440 version (`ci/wheel_check.py
   --version-info`; a SemVer spelling such as `0.1.0-rc.1` fails here, before anything is built,
   instead of producing distributions named `0.1.0rc1` that no later check expects), and builds
   the real distributions by calling `wheels.yml` (the sdist, and the manylinux_2_28 x86_64 abi3
   CPU wheel built from that sdist, checked by `ci/wheel_check.py`, install-tested on Python 3.12
   and 3.13). Whether the version is a pre-release comes from the same parse. A tag that matches
   neither fails.
2. `collect` checks the files again (`twine check`, `ci/wheel_check.py`) and uploads one artifact
   `dist`.
3. `publish-testpypi` uploads to TestPyPI through Trusted Publishing (environment `testpypi`).
4. `publish-pypi` runs only for final versions (no `a`, `b`, `rc` or `dev` part) and waits for
   the required reviewer of the environment `pypi` (the author) before uploading.

The file name `release.yml` and the environment names `testpypi` and `pypi` are bound to the
trusted publishers on TestPyPI and PyPI: never rename them. {doc}`wheels` describes the wheel
builds and the local build of this machine; {doc}`pypi_name_reservation` the 0.0.1 reservation.

## Versioning rules

- `VERSION` holds the next version with a `.dev0` suffix between releases (`0.1.0.dev0`); the
  CMake package version, the Python package version and the documentation's version all come from
  it.
- Before 1.0, a minor release may break the stable API, with a CHANGELOG "Changed" or "Removed"
  entry and a migration note; patch releases only fix bugs. `api-check.yml` reports the Python
  API's changes against the base branch (griffe) and `ci/docs.sh` the C++ API's against the
  committed baseline.
- Output stability: generator streams are bit-exact within a major version; results change only
  for documented bug fixes.

The readiness list of 0.1.0, with what is done and what waits for the author, is in the M5
retrospective ({doc}`retrospectives/M5`).
