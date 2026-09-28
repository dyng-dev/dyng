# Retrospective: INT1 (integration of M1b and M4)

Status: **complete (2026-09-28)** on `main`. M1b (the CUDA `sssp` backend) was finished and
accepted on `main`; M4 (the project infrastructure) was finished and accepted on the branch
`m4-infra`, which ran in parallel. INT1 merges the two, makes the documentation site cover
M1b, applies the author's credit and citation facts, and records the merge policy in force.
The branch `m2-cycle` (M2a, `cycle_count` on the CPU) is not part of INT1; it is merged in M2b.

## What was done

| Commit | What |
|---|---|
| `92e5f54` | `git merge --no-ff m4-infra` ("Merge branch m4-infra: M4 project infrastructure"). Three textual conflicts, resolved by combining both sides: `CONTRIBUTING.md` (M4's guide, plus M1b's CUDA presets and `ci/gpu_local.sh`; `cuda-build` joins the required checks), `README.md` (M1b's CUDA build and parity sections, plus M4's clone step for the originals, now for MOSP-CUDA too) and `ci/check.sh` (M4's steps `tidy`, `harness`, `DYNG_CHECK_ONLY` and the Sphinx docs under M1b's shared-lock `heavy` wrapper; `--help` prints the whole header) |
| `37f7a5d` | `cuda-build.yml` (from M1b) meets M4's workflow checks: `timeout-minutes: 120`, `persist-credentials: false`, one grouped redirect to `GITHUB_ENV` (actionlint's shellcheck SC2129) |
| `b450a45` | `examples/cpp/first_update.cpp` (from M4) uses `graph<>` (the int32 default of ADR 0009, decided in M1b), reads the result with `to_vector()` and runs on `cuda`; CTest `example.first_update.cuda` (label `gpu`, skipped without a device) |
| `d7801ba` | the site covers M1b: an API page for the new Doxygen group `generators`; the core page names the CUDA pieces; install, choose-a-backend, backends-and-resources, getting-started, roadmap, tutorial and the short plan describe the CUDA backend instead of "arrives in M1b"; the algorithm tables (README, landing page, algorithms index) agree; the parity how-to has the CUDA replay and links to the M1a and M1b certificates |
| `cace0ff` | the author's facts of 2026-09-27: ESCHER (IPDPS 2026) title and authors, TruCy as a submitted manuscript, S M Ferdous's affiliation without an e-mail address, the placeholder-identity commits credited to S M Shovan, no funding line; `GOVERNANCE.md` approvals log |
| `a44bc71` | the merge policy (ADR 0019) in `GOVERNANCE.md`, `CONTRIBUTING.md`, the pull request template and the repository settings guide, which also marks the settings applied on 2026-09-27 and lists the real job names of `cuda-build` and `docs` among the required checks |
| `59fc861` | CHANGELOG |

Things the merge brought together that needed no edit: M1b's ADRs (0003, 0009, 0015-0018) and
retrospective reach the site through the globs of `docs/adr/index.md` and
`docs/developer/retrospectives/index.md`, and their rows were already in `docs/adr/README.md`;
M1b's CUDA headers (`stream.hpp`, `buffer.hpp`, `copy.hpp`, `memory.hpp`) are in the `core`
group, so the Breathe page `docs/api/cpp/core.md` renders them (`stream_ref`,
`cuda_async_memory_resource`, `pinned_host_memory_resource`, `to_vector`); the sssp page of
M1b builds under `-W -n` with one exception (below); `parity/README.md` merged without a
conflict and is included by `docs/developer/parity.md`.

## Deviations (each a pragmatic choice with the plan's intent)

| Plan | What was done | Why |
|---|---|---|
| Sections 8.9 and 10.1: squash merges only | ADR 0019: merge commits for milestone and integration pull requests, squash for external contributions, no rebase merging; the `main` ruleset allows Merge and Squash and does not require a linear history | the parity certificates, retrospectives and ADRs cite milestone-branch SHAs (for example the M1b gate records and M4's trial merges); a squash would leave them outside `main`. This is the policy the lead maintainer set in the repository settings on 2026-09-27 |
| M4 notes: the sssp page links `../../parity/results/M1b.md` (M1b wrote it before the site existed) | the link goes through the repository URL (`github.com/dyng-dev/dyng/blob/main/parity/results/M1b.md`), which `ci/docs_links.py` checks | files outside `docs/` are not pages of the site; M4's convention for them is the repository URL |
| Section 3.2: `CITATION.cff` references per paper | TruCy is `type: unpublished` with `status: submitted` and a note naming the journal (BibTeX `@unpublished`) | the author asked for a manuscript, not an article; CFF 1.2.0 has no `manuscript` type (`cffconvert --validate` rejects it), and `unpublished` is the schema's type for a manuscript |
| (no plan item) the parity harness tests in `ci/check.sh` | `pytest` of the `harness` step runs under the shared lock like the other heavy steps | the machine rules: test suites take the shared lock |

## Validation

Fresh clone of `main` at `59fc861` (the commit before this retrospective), GPU 1 for the tests,
the shared lock for every heavy step:

| Check | Result |
|---|---|
| `ci/check.sh --parity` | all checks passed in 2 minutes: clang-format; `cpu-only` 234/234 and `dev` 246/246 (`ctest -L cpu`); clang-tidy naming; REUSE; provenance (88 files); harness 26 passed; docs (Doxygen, coverage of 105 compounds, Sphinx `-W -n`, `docs_links` over 1156 files); pre-commit (actionlint, zizmor, the `.github` metadata check on `cuda-build.yml` included); parity preset golden replay |
| `ci/gpu_local.sh` (dev-cuda) | all steps passed in 8 minutes: build; `ctest -L gpu` 92/92 (with `example.first_update.cuda`); `ctest -L cpu` 246/246 in the CUDA build; the sssp golden corpus on `cuda`, 495/495 cases byte-equal; compute-sanitizer memcheck (0 errors, 0 leaks) and synccheck (53/53, 0 errors); clang-tidy on the CUDA branches |
| `ci/docs.sh` (inside `ci/check.sh`, and again on the final commit with this page) | builds the site with every M1b page: ADRs 0003, 0009, 0015-0019, the M1b retrospective (and this one), the CUDA headers on the core page (`stream_ref`, `cuda_async_memory_resource`, `pinned_host_memory_resource`, `buffer`, `to_vector`) and `generators::legacy` on its own page |
| `cffconvert --validate -i CITATION.cff` (cffconvert in a scratch venv) | valid against CFF 1.2.0 |
| `git grep -nE '^(<<<<<<<\|=======\|>>>>>>>)( \|$)'` | empty; `git branch --merged main` lists `m4-infra` |
| read-only GitHub API (`gh api repos/dyng-dev/dyng`, `.../actions/permissions*`, `.../rulesets`, `.../environments`, `.../private-vulnerability-reporting`, `.../vulnerability-alerts`, `.../automated-security-fixes`, `gh api orgs/dyng-dev`, `orgs/dyng-dev/installations`) | the settings marked done in the guide's checklist; no ruleset or branch protection on `main`; no setting was changed |

## Open items and notes for the next steps

- **The `main` ruleset** (repository settings guide, step 9) waits for the first pull request:
  every required check (including the three `cuda-build` jobs, `site` and `DCO`) must have run
  once. The next milestone merge (M2b, the branch `m2-cycle`) is the natural first pull request;
  it is merged with a merge commit (ADR 0019).
- **Hosted runs.** `cuda-build.yml` and `docs.yml` have not run on GitHub yet (only `lint` and
  `cpu` have, on the M1a history); the first push of `main` runs them. The labels of
  `.github/labels.yml` are applied by the `labels` workflow on that push. After it, run the
  `docs` workflow once with **Run workflow** for the external-link job (M4 open item).
- **Settings still open** (guide): the optional restriction of repository creation in the
  organization (step 1.3), CodeQL (optional, step 6.5), the Discussions categories and the
  pinned roadmap (step 12), Read the Docs and Zenodo (checkpoint A4).
- **ADR numbers.** The next free number is 0020; `m2-cycle` must not reuse 0019 when it is
  merged in M2b.
- The algorithm table still exists three times (README, landing page, algorithms index) until
  `scripts/regen.py` generates it (M3); INT1 edited all three by hand.
- `cycle_count` is described as "planned; the CPU backends in progress on a branch" until M2b
  merges `m2-cycle`; M2b updates the three tables, the roadmap and the short plan.
- Author decisions still open (`GOVERNANCE.md`): O3 (the NOTICE institution line and years) and
  whether a co-author wrote code of the originals outside git. The funding acknowledgement is
  added by the author later.
