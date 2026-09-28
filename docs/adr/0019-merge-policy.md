# ADR 0019: Merge policy (merge commits for milestone work, squash for contributions)

- **Status:** Accepted (INT1)
- **Date:** 2026-09-28 (the repository settings were applied on 2026-09-27)
- **Deciders:** S M Shovan (lead maintainer)

## Context

The plan (Sections 8.9 and 10.1) asked for **squash merges only** on `main`, with the pull
request title as the commit subject. Two things it did not foresee:

- Milestone work is done on long-lived branches (`m4-infra`, `m2-cycle`, ...) in many small,
  reviewed commits, and the records of that work cite those commits by SHA: the parity
  certificates (`parity/results/<milestone>.md`: "measured at `<sha>`"), the milestone
  retrospectives (the review-fix tables, the trial merges) and the ADRs. A squash merge replaces
  the branch by one new commit; the cited SHAs are then not in any clone of `main` (once the
  branch is deleted, only GitHub's hidden pull-request ref still holds them), so the records
  would point at commits a reader cannot find.
- The milestone branches merge each other (M1b and M4 in integration INT1), and a squashed
  history cannot tell afterwards which commit of which branch introduced a line.

External contributions are different: they are small, their intermediate commits ("fix typo",
"address review") are noise on `main`, and nothing cites them.

## Decision

1. **Pull requests of milestone and integration work** (branches of the maintainers that carry
   a milestone, or merge milestones together) are merged with a **merge commit**
   (`git merge --no-ff`; GitHub's **Create a merge commit**). The commit title is the pull
   request title, and the message its description.
2. **External contributions** (and small maintainer fixes) are **squash-merged**; the pull
   request title (Conventional Commits) becomes the commit subject on `main`.
3. **Rebase merging is disabled**: it rewrites the SHAs of the branch as well, with the same
   problem as squashing for milestone work and no benefit for contributions.
4. Repository settings: **Allow merge commits** and **Allow squash merging** on, **Allow rebase
   merging** off; both default messages are the pull request title and description. The `main`
   ruleset allows the merge methods **Merge** and **Squash** and does **not** require a linear
   history (docs/developer/repository_settings.md, steps 4 and 9).

The maintainer who merges picks the method by this rule; the pull request template and
CONTRIBUTING.md tell contributors that their pull request is squash-merged.

## Consequences

- Every SHA cited by a parity certificate, a retrospective or an ADR stays reachable from
  `main`, so the records remain checkable (`git show <sha>`, `git merge-base --is-ancestor`).
- `main` is not linear. `git log --first-parent main` shows one entry per merged pull request
  (the merge commits and the squashed contributions), which is the view for release notes;
  `git bisect start --first-parent` bisects over pull requests.
- Milestone branches must keep clean, reviewed commits (they are published as they are): the
  commit rules of CONTRIBUTING.md apply to every commit, not only to the pull request title.
- The plan's "squash merges only" (Sections 8.9 and 10.1) is superseded by this ADR.
