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
| Zenodo, conda-forge (steps 12, 13; checkpoint A4; the documentation is on GitHub Pages, not Read the Docs) | the **author** (Zenodo is not connected for 0.1.0, so 0.1.0 has no DOI: GOVERNANCE.md, 2026-10-01) | the **author** |

So the AI assistant can take a release candidate to TestPyPI on its own, but nothing reaches PyPI
before the author approves the deployment in the `release.yml` run.

## Checklist

For each release, copy this list into the release pull request and tick it.

The documentation site is not a step of its own: the `deploy` job of `docs.yml` publishes
<https://dyng-dev.github.io/dyng/> from every push to `main`. The final-release pull request
(step 9) also updates the status blocks of `README.md` and `docs/index.md`, which name the latest
release; after its merge, check that the `deploy` job succeeded and that the site shows the
release without the development-version banner (`docs/conf.py` shows the banner while `VERSION`
is a development or pre-release version, with the latest release read from `CHANGELOG.md`).

| # | Step | Command or place | Who |
|---|---|---|---|
| 1 | Full parity on the GPU machine: the golden corpora of every ported algorithm on every backend, each replay written to `benchmarks/results/<version>/parity-<set>-<backend>.json` | `ci/check.sh --parity` (CPU backends), `ci/gpu_local.sh` (CUDA: the corpora, the sanitizers); the largest corpora with `parity/compare.py ... --json` ({doc}`benchmarks`) | maintainer |
| 2 | The performance gates against the originals (manual, under the exclusive lock; CUDA at locked clocks, the default-clock readings recorded, ADRs 0018 and 0021): every benchmark suite of the release's algorithms, run from a clean checkout of the release commit | `parity/bench_suite.py run benchmarks/paper/<suite>.yaml --version <version>` ({doc}`benchmarks`): the suite summaries and the compacted records in `benchmarks/results/<version>/` | maintainer |
| 3 | Sanitizers and mutation checks green, each recorded; then the parity certificate written and committed | the `asan` / `tsan` / `tsan-openmp` presets (`ci/sanitizers.sh`; from 0.2.0 also the hosted `sanitizers.yml` of the release commit), `compute-sanitizer` in `ci/gpu_local.sh`, the CTests `*.mutation.*` and `parity/mutate.py run` (the mutations that need the golden corpus), each recorded with `parity/certify.py check`; then `parity/certify.py write --version <version>` writes `benchmarks/results/<version>/parity.json` and the tables of its `README.md`. It fails if any part failed or is missing (a suite summary must cover its whole suite, with verified inputs; every committed fixture set of `cpp/tests/data` needs its tests passed in a check), or if a part was measured on other code than the release commit's in the paths of its scope; the one exception is a library difference confined to the generated algorithm metadata, accepted for the gates, the replays and the golden mutations (never for a test-suite check) with a passed `parity/certify.py equivalence` record ({doc}`benchmarks`, "The release certificate") | maintainer |
| 4 | The distributions build and install | `ci/check.sh --wheels` locally; `wheels.yml` (manual run) on hosted runners | maintainer |
| 5 | The release-candidate pull request: set `VERSION` to `X.Y.ZrcN` (for example `0.1.0rc1`; `release.yml` requires the tag to equal `VERSION`); move the CHANGELOG's `Unreleased` entries under `## [X.Y.ZrcN] - <date>` (the draft summary of the release is kept at the top while it is prepared) | a pull request | maintainer |
| 6 | `version: X.Y.ZrcN` and `date-released` in `CITATION.cff` | the same pull request | maintainer |
| 7 | Release candidate: tag `vX.Y.ZrcN` (equal to `VERSION`) on `main` after the merge; `release.yml` builds the sdist and the wheels through `wheels.yml` and uploads them to TestPyPI (environment `testpypi`); pre-releases stop there. Before pushing: build the distributions locally as `release.yml` would ({ref}`release-local-build`), and check that the CHANGELOG heading's date and `CITATION.cff`'s `date-released` are **the day of the tag** (UTC); if the merge fell on a later day, fix both in a small pull request first (the tagged files cannot be changed afterwards) | `python ci/wheel_check.py --release-metadata --release-date today`; `git tag -s -m "dynG X.Y.ZrcN" vX.Y.ZrcN && git push origin vX.Y.ZrcN` | **author** (0.1.0rc1: the AI assistant on the author's behalf) |
| 8 | Smoke-install the RC from TestPyPI in a clean venv on a CPU machine and on the GPU machine (CPU backends): import, `dyng.show_config()`, `sssp` and `cycle_count` on both backends, `dyng --version` | `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ dyng==X.Y.ZrcN` | author or maintainer |
| 9 | The final-release pull request ({ref}`release-final-pr`): `VERSION` `X.Y.Z`; the CHANGELOG section renamed `## [X.Y.Z] - <date>` with its link references; `version: X.Y.Z` and `date-released` in `CITATION.cff`; the texts the release candidate wrote for its own state (the status lines, the install lines, `SECURITY.md`); and the certificate under `benchmarks/results/X.Y.Z/` (the release candidate's carried over, or measured again). Without a release candidate, steps 5-6 do this directly and 7-8 are skipped | a pull request; `python ci/wheel_check.py --release-metadata`; `parity/certify.py write --version X.Y.Z` | maintainer |
| 10 | Final release: tag `vX.Y.Z` (equal to `VERSION`) on `main` after that merge; `release.yml` uploads to TestPyPI, then waits for the **approval of the `pypi` environment** and uploads the same files to PyPI. Before pushing, as in step 7: the release date is the day of the tag ({ref}`release-late-tag-day` when that day is later than the final-release pull request's), and the author's open decisions due before the release are taken (GOVERNANCE.md, "Open decisions"; for 0.1.0 these were O3, the `NOTICE` wording, confirmed as drafted on 2026-10-01, and the licence expression, to which the author raised no objection by the tag day, the approval of the `pypi` deployment being the final confirmation: both in the approvals log) | `python ci/wheel_check.py --release-metadata --release-date today`; `git tag -s -m "dynG X.Y.Z" vX.Y.Z && git push origin vX.Y.Z`; approve the deployment in the Actions run (**Review deployments** → **Approve and deploy**) | the tag: **author** (0.1.0: the AI assistant on the author's behalf); the approval: **author** |
| 11 | The GitHub Release from the CHANGELOG section, with the distributions of the `release.yml` run attached (a release candidate may get a GitHub pre-release, `--prerelease`) | `gh release create vX.Y.Z --title "dynG X.Y.Z" --notes-file <the CHANGELOG section> dist/*` | **author** (0.1.0: the AI assistant on the author's behalf) |
| 12 | The Zenodo DOI of the release (once Zenodo is connected, checkpoint A4), then the DOI in `CITATION.cff` and the README. Zenodo archives only the GitHub Releases published after it is switched on: for a DOI of this release, the author switches it on **before step 11**; otherwise the release can only be uploaded to Zenodo by hand ({doc}`repository_settings`, section 14). **0.1.0:** Zenodo is not connected, so 0.1.0 has no DOI (the author, 2026-10-01); it may be connected later (A4), and the releases published after that get DOIs | Zenodo's GitHub integration | **author** |
| 13 | conda-forge: merge the bot's feedstock pull request (from 0.4) | the `dyng-feedstock` repository | author |
| 14 | Announce (Discussions); curate the next "good first issue" backlog; write the milestone retrospective; bump `VERSION` to the next `.dev0` | a pull request | maintainer |

## The release workflow

`release.yml` runs on a tag `v*` (the `release tags` ruleset lets only repository administrators
create one: the author, or the AI assistant on the author's behalf where the approvals log says
so):

1. `select` decides what the tag builds. A `v0.0.x` tag equal to the version of
   `tools/name_reservation` builds the name-reservation package (ADR 0014); every other tag must
   equal `VERSION`, which must be a canonical PEP 440 version (`ci/wheel_check.py
   --version-info`; a SemVer spelling such as `0.1.0-rc.1` fails here, before anything is built,
   instead of producing distributions named `0.1.0rc1` that no later check expects), whose
   CHANGELOG section and `CITATION.cff` must agree with it (`ci/wheel_check.py
   --release-metadata`), and builds the real distributions by calling `wheels.yml` (the sdist,
   and the manylinux_2_28 x86_64 abi3 CPU wheel built from that sdist, checked by
   `ci/wheel_check.py`, install-tested on Python 3.12 and 3.13). Whether the version is a pre-release comes from the same parse. A tag that matches
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

(release-final-pr)=
## The final-release pull request (step 9)

The release candidate's pull request wrote some texts for the state "the release candidate is on
TestPyPI, the release is not on PyPI yet"; the final-release pull request brings each to the
release's state:

- `VERSION` `X.Y.Z`; `CITATION.cff` `version: X.Y.Z` and `date-released` (the planned tag day);
- `CHANGELOG.md`: `## [X.Y.ZrcN] - <date>` renamed `## [X.Y.Z] - <date>` (or a new `## [X.Y.Z]`
  section above the release candidate's, with what changed since), the empty `## [Unreleased]`
  kept above it, and the link references: `[Unreleased]: .../compare/vX.Y.Z...main` and
  `[X.Y.Z]: .../compare/<previous tag>...vX.Y.Z`; the summary's install line (`pip install
  dyng`, no longer TestPyPI);
- the status lines: `README.md` (the "Alpha: 0.1 release candidate" block, the install lines
  "once 0.1.0 is published"), `docs/index.md` (the same block and install line),
  `docs/getting_started/install.md` (`pip install dyng ... once 0.1.0 is published`),
  `SECURITY.md` (the status paragraph and the supported-versions row of the release
  candidates), `CONTRIBUTING.md` and `SUPPORT.md` (the "0.1 release candidate" wording),
  `docs/roadmap.md` ("Where we are") and `docs/developer/plan.md` (the status line);
- the certificate (`benchmarks/results/X.Y.Z/`), one of:
  - **carried over**, when the library did not change since the release candidate's measured
    commits (`git diff <measured> HEAD` over the library paths is empty, which `certify.py
    write` verifies for every part in the paths of its scope): copy the release candidate's
    directory, `cp -r benchmarks/results/X.Y.ZrcN benchmarks/results/X.Y.Z`, and run
    `parity/certify.py write --version X.Y.Z` on the committed final tree. It checks every
    measured commit against the final commit in the paths of its scope and writes a
    `parity.json` naming `X.Y.Z` and the final commit. `VERSION`, `CHANGELOG.md` and
    `CITATION.cff` are outside the library scope, so the gates, the replays and the golden
    mutations carry over; the checks whose scope the final pull request touches are **run
    again** on the final tree and recorded with `certify.py check` first: those of the
    `packaging` and `repo` scopes (the distributions, `ci/check.sh`, `ci/gpu_local.sh`) always
    (they see the new `VERSION`), and the C++ test suites of the `tests` scope (the sanitizer
    presets, `ctest -L mutation`) when `README.md` changed, whose C++ quickstart is a test;
  - **measured again**, when code changed: steps 1-3 for `X.Y.Z`.
  Commit the directory with the release; `release.yml` publishes the version the certificate
  names.

`python ci/wheel_check.py --release-metadata` (also run by the `select` job of `release.yml` and
by `ci/tests`) checks `VERSION`, the CHANGELOG section and its links and `CITATION.cff` against
each other; the status texts are read by the reviewer of the pull request.

(release-late-tag-day)=
## A tag day later than the final-release pull request

The CHANGELOG heading's date and `CITATION.cff`'s `date-released` must be the day the tag is
pushed (UTC): `python ci/wheel_check.py --release-metadata --release-date today` fails on any
other day, and the tagged files cannot be changed afterwards. When the tag falls on a later day
than the one the final-release pull request wrote (0.1.0: written 2026-10-01, tagged
2026-10-02, {doc}`retrospectives/R012`), fix both in a small pull request before the tag, and
bring the certificate to its tree:

1. Set the date in both files; `python ci/wheel_check.py --release-metadata --release-date
   <tag day>` passes (and `--release-date today` on the tag day). Record any decision the
   author took meanwhile (GOVERNANCE.md) in the same pull request.
2. `CHANGELOG.md` and `CITATION.cff` are in the `packaging` scope and in the `repo` scope (the
   whole tree but `benchmarks/results/`), so after the last change outside
   `benchmarks/results/`, from fresh clones of that commit, run again: the distributions
   ({ref}`release-local-build`: `select` for the tag, `ci/wheels.sh`, `twine check --strict`,
   `ci/wheel_check.py --platform manylinux_2_28_x86_64 --require-libgomp`, fresh 3.12 and 3.13
   venvs with the wheel's test subset and the README quickstart), `ci/check.sh --parity` and
   `ci/gpu_local.sh`. Record each with `parity/certify.py check --version X.Y.Z` (the names
   `distributions`, `check-parity` and `gpu_local`, replacing the records of the same names).
   The `tests` records stand while `README.md` and the test paths are unchanged; otherwise the
   sanitizer presets and `ctest -L mutation` are run again too.
3. `parity/certify.py write --version X.Y.Z` on the committed tree, which checks every measured
   commit against it in the paths of its scope; commit `benchmarks/results/X.Y.Z/`.

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
GITHUB_REF_NAME=v0.1.0rc1 GITHUB_OUTPUT=/tmp/select.out bash /tmp/select.sh && cat /tmp/select.out
# the package job (wheels.yml): the sdist, the wheel from the sdist, wheel_check, fresh venvs
flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/wheels.sh
# the collect job
twine check --strict "$DYNG_SCRATCH"/wheels/0.1.0rc1/dist/*
python ci/wheel_check.py "$DYNG_SCRATCH"/wheels/0.1.0rc1/dist/* \
    --platform manylinux_2_28_x86_64 --require-libgomp
```

`select` must print `kind=package`, `version=0.1.0rc1` and `prerelease=true`, and fail for a tag
that differs from `VERSION` or when `VERSION`, `CHANGELOG.md` and `CITATION.cff` disagree
(`ci/wheel_check.py --release-metadata`; a release date other than today is a warning there, and
an error with `--release-date today` before the tag). Then install the wheel alone into fresh Python 3.12 and 3.13 venvs
and run the wheel's test subset and the README quickstart from an empty directory.

The readiness list of 0.1.0, with what is done and what waits for the author, is in the M5
retrospective ({doc}`retrospectives/M5`), the R010 retrospective ({doc}`retrospectives/R010`)
and the R011 retrospective, the final-release record ({doc}`retrospectives/R011`), and the R012
retrospective, the tag day ({doc}`retrospectives/R012`).
