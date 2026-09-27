# Governance

## Model

dynG follows a **lead-maintainer** model while it has fewer than three active maintainers.

- **Lead maintainer:** S M Shovan. The lead maintainer owns releases, merges to `main`, and
  breaks ties.
- **Algorithm maintainers** (optional): the co-authors of an algorithm's papers may review
  changes to that algorithm.
- **Becoming a maintainer:** sustained, reviewed contributions and an invitation from the
  existing maintainers.
- **Decisions** are made by lazy consensus in issues and pull requests. Architecture decisions
  are recorded as ADRs in `docs/adr/` (context, decision, consequences, status); an ADR is
  superseded by a newer one, never rewritten.
- **Reviews:** every change goes through a pull request with the required checks green. While
  the lead maintainer is the only maintainer, they may merge their own pull requests once the
  checks pass; from the day a second maintainer exists, one approving review by someone other
  than the author is required.
- When there are three active maintainers, this model is revisited in an ADR (for example a
  steering group).

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

## License consent and IP record

- **University IP office:** asked for the code to be made public (recorded 2026-09-27).
- **Co-author consent:** not required by the lead maintainer's decision of 2026-09-27; the
  co-authors are credited in `AUTHORS.md` and `CITATION.cff`.
