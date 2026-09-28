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
- **Until the `main` ruleset exists:** the history of `main` so far, from the first commit
  through the integration of M1b and M4 (INT1, itself a `--no-ff` merge of `m4-infra`), was
  pushed directly by the lead maintainer. The ruleset (docs/developer/repository_settings.md,
  step 9) is created after the first pull request has run every required check; from then on
  milestone and integration work also goes through pull requests, merged as above.

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
| 2026-09-27 | Credit and citations | The 22 commits under the placeholder identity `CUDA <user@example.com>` in the originals were made by the author or by an AI assistant working for him: credited to S M Shovan. No funding acknowledgement for now (none found; the author adds it later). ESCHER is cited as the IPDPS 2026 paper by S. M. Shovan, A. Khanda, S. Bhowmick and S. K. Das, "ESCHER: Efficient and Scalable Hypergraph Evolution Representation with Application to Triad Counting"; TruCy (IEEE Transactions on Computers) as a submitted manuscript. S M Ferdous is listed with the affiliation Pacific Northwest National Laboratory only (no third-party e-mail address in the repository). | S M Shovan |
| 2026-09-27 | Merge policy | Milestone and integration pull requests are merged with merge commits, external contributions are squash-merged, rebase merging is disabled (ADR 0019); applied in the repository settings together with About and topics, features, the Actions allow-list with SHA pinning, fork approval, the read-only workflow token, private vulnerability reporting, Dependabot alerts and security updates, secret scanning with push protection, web sign-off, the `release tags` ruleset, the `pypi` and `testpypi` environments, organization 2FA and base permission Read; the DCO app installed. The `main` ruleset follows the first pull request. | S M Shovan |

## Open decisions

Decisions the lead maintainer still has to take; each moves to the approvals log when taken.

| Decision | Default until then |
|---|---|
| O3: confirm the institution line and the years of `NOTICE` ("software developed at the Missouri University of Science and Technology", "Copyright 2023-2026 their authors") | the draft wording stays in `NOTICE` |
| Whether a co-author wrote code of the originals outside git (docs/developer/provenance.md) | co-authors credited as research collaborators only |

## License consent and IP record

- **University IP office:** asked for the code to be made public (recorded 2026-09-27).
- **Co-author consent:** not required by the lead maintainer's decision of 2026-09-27; the
  co-authors are credited in `AUTHORS.md` and `CITATION.cff`.
