# Governance

## Model

dynG follows a **lead-maintainer** model while it has fewer than three active maintainers.

- **Maintainers** are listed, with their areas, in [MAINTAINERS.md](MAINTAINERS.md).
- **Lead maintainer:** S M Shovan. The lead maintainer owns releases, merges to `main`, and
  breaks ties.
- **Algorithm maintainers** (optional): the co-authors of an algorithm's papers may review
  changes to that algorithm.
- **Becoming a maintainer:** sustained, reviewed contributions and an invitation from the
  existing maintainers.
- **Decisions** are made by lazy consensus in issues and pull requests. Architecture decisions
  are recorded as ADRs in `docs/adr/` (context, decision, consequences, status); an ADR is
  superseded by a newer one, never rewritten.
- **Reviews and merges:** see [below](#reviews-and-merges).
- When there are three active maintainers, this model is revisited in an ADR (for example a
  steering group).

## Reviews and merges

- Every change goes through a pull request with the required checks green (CONTRIBUTING.md,
  "Pull requests and review"). While the lead maintainer is the only maintainer, they may merge
  their own pull requests once the checks pass; from the day a second maintainer exists, one
  approving review by someone other than the author is required.
- **Merge policy** (ADR 0019; the repository settings allow exactly these methods):
  - pull requests of **milestone and integration work** are merged with a **merge commit**
    (`git merge --no-ff`), so that the commit SHAs cited by parity certificates, milestone
    retrospectives and ADRs stay reachable from `main`;
  - **external contributions** (and small maintainer fixes) are **squash-merged**; the pull
    request title (Conventional Commits) becomes the commit subject;
  - **rebase merging** is disabled.
- **Since the `main` ruleset exists** (created after pull request #1, INT1, merged with the
  merge commit `eda8b8b`; docs/developer/repository_settings.md, step 9), every change reaches
  `main` through a pull request with the required checks green, merged as above; the history
  before it, from the first commit through M4, was pushed directly by the lead maintainer.

## Contacts

| For | Contact |
|---|---|
| Code of Conduct reports | e-mail the lead maintainer, sm.shovan@gmail.com (decision O13). A report about the lead maintainer: GitHub's **Report content** / **Report abuse** (GitHub Trust & Safety), until a second moderator exists (CODE_OF_CONDUCT.md) |
| Security vulnerabilities | GitHub private vulnerability reporting (the repository's **Security** tab), or sm.shovan@gmail.com (SECURITY.md) |
| Questions and support | GitHub Discussions (SUPPORT.md); there is no private e-mail support |

## Approval checkpoints

Actions that are irreversible or public wait for the lead maintainer's explicit approval
(`docs/adr/0014-approval-checkpoints.md`):

| Checkpoint | Action |
|---|---|
| A1 | create the GitHub organization and repository |
| A2 | make the repository public |
| A3 | any upload to TestPyPI or PyPI, including the name-reserving 0.0.1 |
| A4 | other public registrations: Read the Docs, Zenodo, conda-forge, a domain |
| A5 | publishing each ported algorithm |

## Approvals log

One line per decision: date, checkpoint or decision, what was approved, by whom.

| Date | Checkpoint / decision | Decision | Approved by |
|---|---|---|---|
| 2026-09-27 | O1 (name) | The project is named **dynG** (identifiers `dyng`). | S M Shovan |
| 2026-09-27 | O2 (license) | Apache-2.0 + NOTICE + CITATION.cff. | S M Shovan |
| 2026-09-27 | O1′ (GitHub home) | `dyng-dev/dyng`, a new GitHub organization. | S M Shovan |
| 2026-09-27 | A1 | Approved: create the `dyng-dev` organization and the `dyng` repository. | S M Shovan |
| 2026-09-27 | A2 | Approved: the repository may be public from the start. | S M Shovan |
| 2026-09-27 | A3 | Approved: reserve `dyng` on PyPI with a real 0.0.1 through Trusted Publishing (workflow `release.yml`, environments `pypi` and `testpypi`). | S M Shovan |
| 2026-09-27 | O26 / IP clearance | The university IP office has asked for the code to be public; IP clearance is done. | S M Shovan |
| 2026-09-27 | A5 / consent | Consent e-mails are not needed; ports may be published as soon as they pass their gates. | S M Shovan |
| 2026-09-27 | O22 (history) | Fresh git history; the originals' histories are not imported. | S M Shovan |
| 2026-09-27 | Plan | The plan (version 2) is approved; execute milestone by milestone, asking only for account-level actions or real blockers. | S M Shovan |
| 2026-09-27 | O13 (contact) | `sm.shovan@gmail.com` is the published contact for the Code of Conduct and security reports; GitHub private vulnerability reporting is to be enabled in the repository settings. | S M Shovan |
| 2026-09-27 | Accounts | Done by the author: the GitHub organization `dyng-dev` and the empty public repository `dyng-dev/dyng`; pending trusted publishers on PyPI and TestPyPI (project `dyng`, repository `dyng-dev/dyng`, workflow `release.yml`, environments `pypi` and `testpypi`). Pushing tag `v0.0.1` publishes the name reservation. | S M Shovan |
| 2026-09-27 | Credit and citations | The 22 commits under the placeholder identity `CUDA <user@example.com>` in the originals were made by the author or by an AI assistant working on the author's behalf: credited to S M Shovan. No funding acknowledgement for now (none found; the author adds it later). ESCHER is cited as the IPDPS 2026 paper by S. M. Shovan, A. Khanda, S. Bhowmick and S. K. Das, "ESCHER: Efficient and Scalable Hypergraph Evolution Representation with Application to Triad Counting"; TruCy (IEEE Transactions on Computers) as a submitted manuscript. S M Ferdous is listed with the affiliation Pacific Northwest National Laboratory only (no third-party e-mail address in the repository). | S M Shovan |
| 2026-09-27 | Merge policy | Milestone and integration pull requests are merged with merge commits, external contributions are squash-merged, rebase merging is disabled (ADR 0019); applied in the repository settings together with About and topics, features, the Actions allow-list with SHA pinning, fork approval, the read-only workflow token, private vulnerability reporting, Dependabot alerts and security updates, secret scanning with push protection, web sign-off, the `release tags` ruleset, the `pypi` and `testpypi` environments, organization 2FA and base permission Read; the DCO app installed. The `main` ruleset follows the first pull request. | S M Shovan |
| 2026-09-28 | `main` ruleset | Created after pull request #1 (INT1, merge commit `eda8b8b`): pull request required (0 approvals, stale approvals dismissed, conversation resolution; merge methods merge and squash), 17 required checks (lint, cpu, cuda-build, docs), strict; deletions and force pushes blocked; the Repository admin role may bypass only through a pull request. `DCO` is not required yet, because the author's organization membership is private; it is added once the membership is public. | S M Shovan |
| 2026-09-28 | ADR 0018 | Accepted, option B: CUDA performance gates are read with the GPU clocks locked for the whole A/B (both programs, every busy sample checked); default-clock readings are recorded and published, not gated. | S M Shovan |
| 2026-09-29 | Organization membership | The author's `dyng-dev` membership is public. The DCO app exempts members only for signed commits, so the maintainer's commits are SSH-signed from 2026-09-29 (a dedicated signing key); `DCO` becomes a required check after the first signed pull request passes it. | S M Shovan |
| 2026-09-29 | ADR 0020 | Accepted: the resident device graph under set semantics (device batch apply, lazily downloaded host copy) and Step 0 once per update. | S M Shovan |
| 2026-09-29 | ADR 0021 | Accepted: a CUDA gate case whose GPU cannot hold the boost clock lock under the power cap is read at the base lock, applied equally to both programs (the COLLAB `cycle_count` update). | S M Shovan |
| 2026-09-29 | Delegation of technical ADRs | The AI assistant may accept a technical ADR on the author's behalf when its decision does not change a rule the author approved, and reports it to the author afterwards (recorded in this log as "accepted under delegation"). Rules the author approved (gates, the measurement protocol), licensing, naming, publishing and account-level actions stay with the author. | S M Shovan |
| 2026-09-29 | ADR 0022 | Accepted under delegation (M3, step kit-scaffold): the conformance kit in `cpp/tests/conformance` (one executable per algorithm and variant), registry-driven from the manifests (`scripts/regen.py`; the public `<dyng/core/registry.hpp>`, to be reviewed with the 0.1 API), C8's reservations and container work, C4 skipped while no backend has two engines, and the scope of `scripts/new_algorithm.py` in 0.1 (graphs, host backends). No approved rule changes. | AI assistant (delegated) |
| 2026-09-29 | ADR 0023 (and ADR 0006) | Accepted under delegation (M3, step api-freeze): the 0.1 API review of `core/*`, `graph/*`, `update.hpp`, `sssp.hpp`, `cycle_count.hpp` and the top-level headers, its fixes (`to_string` of the core enumerations, `@guarantee` exception guarantees checked by the docs job, allocation failures translated in header-inline code, the no-CUDA check of the public headers), and the freeze: the committed public-API listing `cpp/tests/api/api_snapshot/public_api.txt` checked by `ci/docs.sh`, changed only through the procedure of the API review checklist. ADR 0006 (the algorithm contract) is accepted as part of the freeze. No approved rule changes. | AI assistant (delegated) |
| 2026-09-29 | ADR 0023 amendments, ADR 0022 amendments | Accepted under delegation (M3 review fixes): the API corrections A1-A11 of ADR 0023 (the `participant_of` dispatch of `dyng::update()`, sssp's supported-type `static_assert`s and `distance_t`, `to_string` of ten more enumerations, strong batch builders and `@guarantee` on every mutating member, the snapshot of the `detail` contract, macros and the umbrella's includes, the thread-safe profiler recording and the threading docs, `edge_list`'s field order, `update_each()`'s checks, `ci/docs.sh --update-api`), and in ADR 0022 the run-time placement of the "sequential backend" rule, C0's check of unlisted backends and C4's test on a fake two-engine algorithm. The framework changes (engine chosen before the commit, backend-aware `engine::automatic`, per-thread budgets that throw only under strict budgets, the half before the commit in the budget) are recorded in `docs/developer/framework.md`. No approved rule changes. | AI assistant (delegated) |
| 2026-09-30 | ADR 0022 amendment | Accepted under delegation (M3 acceptance fixes): sssp is part of every build of a subset of the algorithms (`-DDYNG_ALGORITHMS` without it gets it added, because `libdyng`'s MOSP batch generator and the shared suites call it), and `ci/scaffold_check.sh` builds every target of the probes-only subset. No approved rule changes. | AI assistant (delegated) |
| 2026-09-30 | ADR 0024 | Accepted under delegation (M3 acceptance fixes): a dynG-against-dynG A/B (a milestone's refactor bar) cycles its rounds through heap layouts (`--layouts`) and reads bimodal regions mode by mode with the equality of the mode fractions (`parity/ab_modes.py`); the fixed-layout readings stay recorded. The PLAN 8.6 gates against the originals and the protocol of ADRs 0018 and 0021 are unchanged. No approved rule changes. | AI assistant (delegated) |
| 2026-09-30 | ADR 0011 | Accepted under delegation (M5, the Python bindings): the root `pyproject.toml` (scikit-build-core, nanobind pinned exactly, abi3), the private `dyng._core` under the typed layer, dtype dispatch that never narrows ids silently (checked conversions where the graph's type is fixed), `dyng.Array` views that raise `StaleResultError` after their result is updated (`to_numpy()` copies by default), GIL release with per-object and profiler locks, the exception translation (plus `InternalError`), `dyng.update` over a run-time list of results, and the stubs regenerated by `scripts/regen.py --stubs`. No approved rule changes; nothing is published. | AI assistant (delegated) |
| 2026-09-30 | ADR 0025 | Accepted under delegation (M5, wheels and CLI): the `dyng` command line as a console script of the Python package (flags generated from the option fields, `dyng prep` with `mospPrep`'s syntax, the originals' output formats) instead of PLAN 4.2's C++ `tools/cli`; the wheels built by cibuildwheel in `wheels.yml` and locally by `ci/wheels.sh` (a glibc 2.28 toolchain, hidden static libstdc++), checked by `ci/wheel_check.py` (90 MB budget); `python.yml`; `release.yml` building the real distributions for tags v0.1.0 and later through `wheels.yml`, with the file name, the environments and the publish jobs unchanged. The new action `pypa/cibuildwheel` needs the Actions allow-list (account-level, for the orchestrator). No approved rule changes; nothing is published. | AI assistant (delegated) |
| 2026-09-30 | Repository state: `DCO` required | `DCO` is a required check of the `main` ruleset (id 24131378) from 2026-09-30, the 18th required check (the DCO app, integration id 1861). The maintainer's commits are SSH-signed with the key "dynG commit signing", registered as a signing key on the author's account on 2026-09-29, so the DCO app exempts them as signed commits of an organization member; external contributors sign off. The CI `harness` job installs `pyyaml==6.0.3` (the fix of pull request #3). Recorded in `docs/developer/repository_settings.md` (steps 7 and 9). | S M Shovan |
| 2026-09-30 | Release tags and the GitHub Release (0.1.0) | The AI assistant (the orchestrator of the milestones) may push the release tags `v0.1.0rc1` and `v0.1.0` and create the GitHub Release of 0.1.0 on the author's behalf; the author keeps the approval of the `pypi` deployment (the required reviewer of the environment `pypi`), so nothing reaches PyPI without it. Pushing a tag also uploads its distributions to TestPyPI (`release.yml`), which this decision covers. This amends the "publishing stays with the author" part of the delegation of 2026-09-29 for these two tags only. `docs/developer/release.md` says who does each step. | S M Shovan |
| 2026-09-30 | Licence metadata of the distributions | The `License-Expression` of the wheel (and of the sdist, which shares the metadata) names the bundled licences: `Apache-2.0 AND BSD-3-Clause AND MIT AND GPL-3.0-or-later WITH GCC-exception-3.1` (dynG; nanobind; robin-map; the GCC runtime, `THIRD_PARTY_LICENSES.txt`). The recommended default, recorded as the author's decision pending objection before the final PyPI approval (readiness item 15 of the M5 retrospective). Checked by `ci/wheel_check.py`; `docs/developer/wheels.md`. | S M Shovan |
| 2026-09-30 | Maturity for 0.1.0 | `sssp` and `cycle_count` become `stable` for 0.1.0 once the benchmark-suite record of PLAN 8.5 exists (it does: `benchmarks/paper/ipdps25_dynamosp_sosp.yaml` and `benchmarks/paper/ieee_tc_dyntrucy.yaml`, R010). SemVer applies to both from 0.1.0. | S M Shovan |
| 2026-10-01 | Repository state: M5 merged, 24 required checks | Pull request #4 (M5) merged with the merge commit `8842acf`. The Actions allow-list includes `pypa/cibuildwheel@*` (added by the maintainer through the API on 2026-09-30). The `main` ruleset (id 24131378) requires **24** checks: the 18 before, plus `scaffold` (`cpu.yml`), `Editable install, stubs, mypy, pytest (gcc)`, `Editable install, stubs, mypy, pytest (clang-18)`, `From the sdist (Python 3.12)`, `From the sdist (Python 3.13)` (`python.yml`) and `Python API (griffe)` (`api-check.yml`); `wheels.yml` stays unrequired (path-filtered). Recorded in `docs/developer/repository_settings.md` (steps 5 and 9). | S M Shovan |
| 2026-09-30 | ADR 0026 | Accepted under delegation (M7, step sssp-operators): the CUDA operators engine of `sssp` (decision O24), MOSP_ESCHER@4b86159's multi-kernel host loop with MOSP-CUDA@e220ee2's semantics, byte-identical to the fused engine (C4); `engine::automatic` runs it on a device without cooperative launch, `engine::operators` everywhere, `engine::fused` still needs cooperative launch (replaces ADR 0017 item 2); its budget 3 + iterations + epochs host synchronizations; the device building blocks shared by both engines in `kernels.cuh` (the fused kernel's SASS unchanged), the operators in the sssp folder (rule of two); `--cuda-engine`, `compare.py --configs cuda-operators`, `perf_ab.py engines`. No public signature changes; the engine is measured, not gated. No approved rule changes. | AI assistant (delegated) |
| 2026-09-30 | ADR 0027 | Accepted under delegation (M7, step mosp-core): the mosp API frozen from the M3 sketch (plus `options::num_objectives`; `delta` of the K updates only; host path costs over the K objectives; `affected` against the previous MOSP tree; `<dyng/mosp.hpp>` frozen), mosp as a participant that owns K sssp participants and runs the combined step through sssp's static enactor on a pooled CSR, the additive `io::write_path_costs()`, `<dyng/testing/mosp_oracle.hpp>` and the umbrella include, `sssp::update()` accepting batches without insertions whatever their weight count, the kit's `test_traits::num_weights`, and `dyng-compat-mosp --mosp` (the default mode, which the sssp gates measure, unchanged). No approved rule changes. | AI assistant (delegated) |
| 2026-10-01 | ADR 0028 | Accepted under delegation (M7, step python-docs-finish): the Python module `dyng.mosp` (`Options` with the C++ fields, `Stats` with the K `dyng.sssp.Stats`, `Result` with `distances(k)` / `parents(k)`, the combined arrays and the path costs as a zero-copy (n, K) `dyng.Array`, `from_arrays` over lists of arrays), `dyng.Array` reporting 2-D shapes, mosp results in `dyng.update`, `dyng.io.write_path_costs` and the references `dyng.testing.combined_graph` / `mosp_path_costs`, the command line `dyng mosp compute|update` (`--preferences` as a comma-separated list flag, the `mosp` driver's output files, every column's costs with `--num-objectives` below the graph's columns), the engine option of sssp and mosp in Python and on the command line, `DYNG_BUILD_PYTHON` needing mosp, and the mosp examples. Additions only; the C++ API baseline unchanged. No approved rule changes. | AI assistant (delegated) |
| 2026-10-01 | O3 (`NOTICE`) | The institution line and the years of `NOTICE` are confirmed as drafted ("software developed at the Missouri University of Science and Technology", "Copyright 2023-2026 their authors"); `NOTICE` is unchanged. This closes O3 before the tag `v0.1.0` is pushed (`docs/developer/release.md` step 10). | S M Shovan |
| 2026-10-01 | Zenodo not connected for 0.1.0 (no DOI) | Zenodo is not connected for 0.1.0, so 0.1.0 has no DOI. It may be connected later (checkpoint A4, `docs/developer/repository_settings.md` section 14); the releases published after that get DOIs. | S M Shovan |
| 2026-10-02 | Licence metadata of the distributions (0.1.0) | The `License-Expression` of 2026-09-30 stands for 0.1.0: the author was told that it is in the 0.1.0rc1 distributions on TestPyPI and raised no objection by the tag day; the author's approval of the `pypi` deployment of 0.1.0 is the final confirmation. | no objection by the tag day (S M Shovan); final confirmation: the `pypi` approval |
| 2026-10-02 | PyPI deployment of 0.1.0 | The author approved the `pypi` deployment of release run 36956746032 (tag `v0.1.0`); dyng 0.1.0 is on PyPI. The approval is the final confirmation of the wheel's licence expression (`Apache-2.0 AND BSD-3-Clause AND MIT AND GPL-3.0-or-later WITH GCC-exception-3.1`). The GitHub Release v0.1.0 was created by the AI assistant per the decision of 2026-09-30. | S M Shovan |
| 2026-10-02 | Release re-grouping after 0.1.0 (PLAN Appendix F) | Smaller releases: **0.2.0** = `mosp` and the `sssp` operators engine (M7) + the CUDA plugin wheels `dyng-cu12` / `dyng-cu13`, the tutorial algorithms, the hosted documentation, fuzzers and mutation checks (M6, carried out as M6a and M6b); **0.3.0** = the ESCHER store and the hypergraph container (M8) + `triad_count` and `count_local_patterns` (M9); **0.4.0** = `label_propagation` (M10) + `hyper_sssp` (M11); the later items of PLAN 11.5-11.6 move one minor version up. The M6 hardening, planned as 0.1.x, ships in 0.2.0 (0.1.x patch releases stay possible for fixes). `docs/roadmap.md` and `docs/developer/plan.md` follow it. | S M Shovan |
| 2026-10-02 | O14 (docs hosting) | GitHub Pages, <https://dyng-dev.github.io/dyng/>, the plan's fallback to Read the Docs: `docs.yml` deploys the site of every push to `main` (`actions/deploy-pages`, environment `github-pages`). Read the Docs is not connected (`.readthedocs.yaml` is kept, so it can be added later through checkpoint A4). | S M Shovan |
| 2026-10-02 | Publishing the CUDA plugins (0.2.0) | PyPI does not accept two identical pending publishers, so each plugin publishes through its own GitHub environments: `pypi-cu12` / `pypi-cu13` (required reviewer SMShovan, no administrator bypass, tags `v*` only) and `testpypi-cu12` / `testpypi-cu13` (tags `v*` only), created by the maintainer through the API. The author registered the pending publishers on PyPI and TestPyPI on 2026-10-02: `dyng-cu12` (environments `pypi-cu12` / `testpypi-cu12`) and `dyng-cu13` (`pypi-cu13` / `testpypi-cu13`), repository `dyng-dev/dyng`, workflow `release.yml`. `release.yml` publishes each plugin through its own environments (M6a); the `dyng` publisher (`pypi` / `testpypi`) is unchanged. Recorded in `docs/developer/repository_settings.md` (step 11). | S M Shovan |
| 2026-10-06 | ADR 0030 (the tutorial algorithms; branch `m6b-hardening`) | Accepted under delegation (M6b, step tutorials): `dynamic_bfs` and `triangle_delta` registered in `cpp/src/algorithms` with the new `maturity_level::tutorial` (appended to the enumeration) and installed headers, one source for the three backends through the framework operators' executors (`cpp/src/operators`), undirected graphs in the conformance kit, no citation, no CLI command and no `dyng.update()` support for them; the tutorial's reference solution in `examples/tutorial_algorithms/my_bfs`. No approved rule changes. (Its number collides with M6a's ADR 0030; one of them is renumbered at the merge.) | AI assistant (delegated) |
| 2026-10-06 | ADR 0032 | Accepted under delegation (M6b, step docs-pages-fuzz): the reader fuzzers in `cpp/fuzz` (preset `fuzz`, `ci/fuzz.sh`, `fuzz.yml`; no binary-reader target before M8), the cycle_count and mosp mutation suites of `parity/mutate.py`, the Hypothesis profile `full` and `property.yml`. No approved rule changes. | AI assistant (delegated) |
| 2026-10-06 | Repository state: GitHub Pages enabled | The author enabled GitHub Pages (Settings → Pages → Source: **GitHub Actions**): <https://dyng-dev.github.io/dyng/>, HTTPS enforced; GitHub created the environment `github-pages`, deployable from `main` only. The first deployment follows the merge of M6b. Recorded in `docs/developer/repository_settings.md` (step 13). | S M Shovan |

## Open decisions

Decisions the lead maintainer still has to take; each moves to the approvals log when taken.

| Decision | Default until then |
|---|---|
| Whether a co-author wrote code of the originals outside git (docs/developer/provenance.md) | co-authors credited as research collaborators only |
| ADR 0029 (M7 review), the packing window: for non-canonical imported trees inside the window (n - 1) * W <= max_distance < n * W the host backends follow MOSP-OpenMP's packing rule and the CUDA engines MOSP-CUDA's, so their parents differ (each byte-identical to its original). Option A: keep each backend's original and document the exception to cross-backend equality (C3); option B: one rule on every backend, giving up byte parity with one original on those inputs. A parity rule, so not delegated | option A (the behaviour since M1b), documented and pinned by a test |
| Layout control for the 0.2 gates (retrospective open item 1): the parity preset's readings move with code placement and the heap layout on the Xeon Gold 6258R (`parity/results/M3.md` sections 5.4 and 6.3). Keep the one-layout protocol of ADRs 0018 / 0021 for the gates against the originals, read them over heap layouts as ADR 0024 does for refactors, or add a function / branch alignment flag to the parity and release presets (a PLAN 8.6 per-algorithm flag)? (M3's criterion 3 no longer waits on it: its 2 % bar is met under ADR 0024, M3.md section 6.5.) | the one-layout protocol; a measured proposal before the 0.2 gates |

## License consent and IP record

- **University IP office:** asked for the code to be made public (recorded 2026-09-27).
- **Co-author consent:** not required by the lead maintainer's decision of 2026-09-27; the
  co-authors are credited in `AUTHORS.md` and `CITATION.cff`.
