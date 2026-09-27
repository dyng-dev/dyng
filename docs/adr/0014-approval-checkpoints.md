# ADR 0014: Approval checkpoints

- **Status:** Accepted
- **Date:** 2026-09-27
- **Deciders:** S M Shovan (lead maintainer)

## Context

Much of the work on dynG is prepared by an AI assistant. Creating accounts and repositories,
uploading packages and making code public cannot be undone, and some depend on the university's
IP position. The author's standing instruction is that nothing irreversible or public happens
without an explicit "yes".

## Decision

Preparing is always allowed; doing waits for a checkpoint approved by the lead maintainer:

| Checkpoint | Action that waits for approval | Prepared beforehand |
|---|---|---|
| A1 | create the GitHub organization and repository | README, LICENSE, NOTICE and CITATION.cff drafts |
| A2 | make the repository public | IP office answer and consent records; code of conduct contact |
| A3 | any upload to TestPyPI or PyPI, including a name-reserving 0.0.1 | the pure-Python 0.0.1 package and the Trusted Publishing workflow |
| A4 | other public registrations: Read the Docs, Zenodo, conda-forge, a domain | configuration files |
| A5 | publishing each ported algorithm | the consent record for its source repository |

Each approval is logged as one line in `GOVERNANCE.md` (date, checkpoint, "approved by").

**Approvals given on 2026-09-27** (plan Appendix E): A1 (organization `dyng-dev`, repository
`dyng`), A2 (public from the start), A3 (reserve `dyng` on PyPI through Trusted Publishing),
IP clearance (the university IP office asked for the code to be public) and A5 (no consent
e-mails needed; ports are published once they pass their gates). A4 is not yet approved.

## Consequences

- The assistant asks the author only for account-level actions (GitHub organization and
  repository, the PyPI trusted-publisher entry, Read the Docs, Zenodo) and real blockers.
- The name-reservation package and `release.yml` are ready; the upload itself happens when the
  author creates the PyPI and TestPyPI trusted publishers (project `dyng`, repository
  `dyng-dev/dyng`, workflow `release.yml`, environments `pypi` and `testpypi`) and pushes a
  `v0.0.1` tag.
