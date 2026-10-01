# Release process

How a dynG release is made (PLAN Sections 10.3 and 10.4). Releases are cut from `main`, which is
always releasable; `release/X.Y` branches are created only when an older minor release needs a
patch. Versions follow SemVer 2.0.0's meaning from 0.1.0 (PLAN Section 5.9 says what the
stability promise covers), **spelled in canonical PEP 440 form**, which the Python distributions
require: `0.1.0`, `0.1.0rc1`, `0.1.0.dev0`, never SemVer's `0.1.0-rc.1`. The single source of the
version is the file `VERSION`; tags are `v` + `VERSION` (`v0.1.0rc1`, `v0.1.0`), and
`release.yml` refuses a `VERSION` that is not canonical.

**Who does what.** Publishing steps belong to the **author** (the lead maintainer) unless the
author delegates one in the approvals log of GOVERNANCE.md. For 0.1.0 the author did so on
2026-09-30:

| Step | 0.1.0 (`v0.1.0rc1`, `v0.1.0`) | later releases |
|---|---|---|
| the checks, the certificate, the version bump, the CHANGELOG, `CITATION.cff` (steps 1-6, 9) | the AI assistant, in the release branch and its pull request | a maintainer or the AI assistant |
| merging the release pull request into `main` (merge commit, ADR 0019) | the AI assistant, through the pull request (the `main` ruleset) | the author |
| pushing the tag (steps 7, 10), which uploads to TestPyPI | the **AI assistant**, on the author's behalf | the **author** |
| the approval of the `pypi` deployment (step 10), which uploads to PyPI | the **author** (never delegated) | the **author** |
| the GitHub Release (step 11) | the **AI assistant**, on the author's behalf | the **author** |
| Zenodo, Read the Docs, conda-forge (steps 12, 13; checkpoint A4) | the **author** | the **author** |

So the AI assistant can take a release candidate to TestPyPI on its own, but nothing reaches PyPI
before the author approves the deployment in the `release.yml` run.

## Checklist

For each release, copy this list into the release pull request and tick it.

| # | Step | Command or place | Who |
|---|---|---|---|
| 1 | Full parity on the GPU machine: the golden corpora of every ported algorithm on every backend, each replay written to `benchmarks/results/<version>/parity-<set>-<backend>.json` | `ci/check.sh --parity` (CPU backends), `ci/gpu_local.sh` (CUDA: the corpora, the sanitizers); the largest corpora with `parity/compare.py ... --json` ({doc}`benchmarks`) | maintainer |
| 2 | The performance gates against the originals (manual, under the exclusive lock; CUDA at locked clocks, the default-clock readings recorded, ADRs 0018 and 0021): every benchmark suite of the release's algorithms, run from a clean checkout of the release commit | `parity/bench_suite.py run benchmarks/paper/<suite>.yaml --version <version>` ({doc}`benchmarks`): the suite summaries and the compacted records in `benchmarks/results/<version>/` | maintainer |
| 3 | Sanitizers and mutation checks green, each recorded; then the parity certificate written and committed | the `asan` / `tsan` presets, `compute-sanitizer` in `ci/gpu_local.sh`, the CTests `*.mutation.*` and `parity/mutate.py run` (the mutations that need the golden corpus), each recorded with `parity/certify.py check`; then `parity/certify.py write --version <version>` writes `benchmarks/results/<version>/parity.json` and the tables of its `README.md` (it fails if any part failed, is missing, or was measured on other library code than the release commit's) | maintainer |
| 4 | The distributions build and install | `ci/check.sh --wheels` locally; `wheels.yml` (manual run) on hosted runners | maintainer |
| 5 | The release-candidate pull request: set `VERSION` to `X.Y.ZrcN` (for example `0.1.0rc1`; `release.yml` requires the tag to equal `VERSION`); move the CHANGELOG's `Unreleased` entries under `## [X.Y.ZrcN] - <date>` (the draft summary of the release is kept at the top while it is prepared) | a pull request | maintainer |
| 6 | `version: X.Y.ZrcN` and `date-released` in `CITATION.cff` | the same pull request | maintainer |
| 7 | Release candidate: tag `vX.Y.ZrcN` (equal to `VERSION`) on `main` after the merge; `release.yml` builds the sdist and the wheels through `wheels.yml` and uploads them to TestPyPI (environment `testpypi`); pre-releases stop there. Before pushing, build the distributions locally as `release.yml` would ({ref}`release-local-build`) | `git tag -s -m "dynG X.Y.ZrcN" vX.Y.ZrcN && git push origin vX.Y.ZrcN` | **author** (0.1.0rc1: the AI assistant on the author's behalf) |
| 8 | Smoke-install the RC from TestPyPI in a clean venv on a CPU machine and on the GPU machine (CPU backends): import, `dyng.show_config()`, `sssp` and `cycle_count` on both backends, `dyng --version` | `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ dyng==X.Y.ZrcN` | author or maintainer |
| 9 | The final-release pull request: set `VERSION` to `X.Y.Z`, rename the CHANGELOG section to `## [X.Y.Z] - <date>`, and set `version: X.Y.Z` and `date-released` in `CITATION.cff` (without a release candidate, steps 5-6 do this directly and 7-8 are skipped) | a pull request | maintainer |
| 10 | Final release: tag `vX.Y.Z` (equal to `VERSION`) on `main` after that merge; `release.yml` uploads to TestPyPI, then waits for the **approval of the `pypi` environment** and uploads the same files to PyPI | `git tag -s -m "dynG X.Y.Z" vX.Y.Z && git push origin vX.Y.Z`; approve the deployment in the Actions run (**Review deployments** → **Approve and deploy**) | the tag: **author** (0.1.0: the AI assistant on the author's behalf); the approval: **author** |
| 11 | The GitHub Release from the CHANGELOG section, with the distributions of the `release.yml` run attached (a release candidate may get a GitHub pre-release, `--prerelease`) | `gh release create vX.Y.Z --title "dynG X.Y.Z" --notes-file <the CHANGELOG section> dist/*` | **author** (0.1.0: the AI assistant on the author's behalf) |
| 12 | The Zenodo DOI of the release (once Zenodo is connected, checkpoint A4), then the DOI in `CITATION.cff` and the README | Zenodo's GitHub integration | **author** |
| 13 | conda-forge: merge the bot's feedstock pull request (from 0.3) | the `dyng-feedstock` repository | author |
| 14 | Announce (Discussions); curate the next "good first issue" backlog; write the milestone retrospective; bump `VERSION` to the next `.dev0` | a pull request | maintainer |

## The release workflow

`release.yml` runs on a tag `v*` (the `release tags` ruleset lets only repository administrators
create one: the author, or the AI assistant on the author's behalf where the approvals log says
so):

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

(release-local-build)=
## Building a release locally, as `release.yml` would

Before a tag is pushed, the distributions of that tag are built and tested on the development
machine with the steps of `release.yml` (no tag is created; for 0.1.0rc1 this was done in R010,
{doc}`retrospectives/R010`):

```bash
source scripts/dev_env.sh
# the select job, for the tag about to be pushed: its run block, extracted from release.yml
python -c 'import yaml; wf = yaml.safe_load(open(".github/workflows/release.yml")); print(next(
    s["run"] for s in wf["jobs"]["select"]["steps"] if s.get("id") == "select"))' > /tmp/select.sh
GITHUB_REF_NAME=v0.1.0rc1 GITHUB_OUTPUT=/dev/stdout bash /tmp/select.sh
# the package job (wheels.yml): the sdist, the wheel from the sdist, wheel_check, fresh venvs
flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/wheels.sh
# the collect job
twine check --strict "$DYNG_SCRATCH"/wheels/0.1.0rc1/dist/*
python ci/wheel_check.py "$DYNG_SCRATCH"/wheels/0.1.0rc1/dist/* \
    --platform manylinux_2_28_x86_64 --require-libgomp
```

`select` must print `kind=package`, `version=0.1.0rc1` and `prerelease=true`, and fail for a tag
that differs from `VERSION`. Then install the wheel alone into fresh Python 3.12 and 3.13 venvs
and run the wheel's test subset and the README quickstart from an empty directory.

The readiness list of 0.1.0, with what is done and what waits for the author, is in the M5
retrospective ({doc}`retrospectives/M5`) and the R010 retrospective ({doc}`retrospectives/R010`).
