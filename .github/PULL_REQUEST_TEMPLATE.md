<!--
Thank you for contributing to dynG! Please read CONTRIBUTING.md first.
- Title in Conventional-Commits style: feat(sssp): ..., fix(io): ..., perf(cycle_count): ..., docs: ...
  (contributions are squash-merged: the title becomes the commit subject on main; milestone
  work is merged with a merge commit, ADR 0019).
- Aim for at most about 500 changed lines, excluding generated files and goldens.
- Tick what applies; strike through (~~text~~) or delete what does not apply, and say why.
-->

## Summary

<!-- What does this change do, and why? -->

Closes #

## Kind of change

- [ ] Bug fix
- [ ] New feature or improvement
- [ ] New algorithm (discussed in a `new_algorithm` issue: # )
- [ ] Port of research code (discussed in a `port_research_code` issue: # )
- [ ] Performance (kernel or hot path)
- [ ] Public API change (the `api-change` label is set)
- [ ] Framework or operators (internal-stable)
- [ ] Documentation only
- [ ] Build, CI or tooling

## How was it tested?

<!-- Commands you ran (ci/check.sh, ctest labels, presets, parity), and on which hardware
     (CPU only, or which GPU). -->

## Checklist

- [ ] **Tests:** unit tests added or updated (and the conformance kit, once available); a bug
      fix adds a test that fails without it; `ci/check.sh` passes locally.
- [ ] **Parity:** for a ported algorithm, the parity replay is green (`ci/check.sh --parity`;
      a maintainer can run it on the GPU machine).
- [ ] **Docs:** Doxygen on every new public entity (`ci/docs.sh` passes); the algorithm page
      and the user documentation are updated; the examples still run.
- [ ] **CHANGELOG:** an entry in the `Unreleased` section of `CHANGELOG.md` (for user-visible
      changes).
- [ ] **DCO:** every commit is signed off (`git commit -s`; required for external contributors).
- [ ] **Benchmarks:** for kernel or hot-path changes, a before/after comparison is attached
      (`ci:bench`, or `parity/perf_ab.py` with the number of runs and the median).
- [ ] **Library rules:** no printing, no `exit()` / `abort()`, no new global state in library
      code; errors are `dyng::error` exceptions.
- [ ] **Headers:** SPDX header on every new file; `// Derived from <repo>@<commit>:<path>` on
      every ported file.

### Also required for some kinds of change (CONTRIBUTING.md, "Pull requests and review")

- [ ] Kernel or hot path: benchmark table; `--resource-usage` diff for fused kernels.
- [ ] Ported algorithm: `parity` run on the GPU machine.
- [ ] Public API: `api-change` label, CHANGELOG entry, docs, an ADR if significant, the API
      review checklist (`docs/developer/api_review_checklist.md`).
- [ ] Framework or operators: an ADR; every in-tree algorithm and tutorial stays green.
- [ ] New algorithm: discussed `new_algorithm` issue, scaffold used, maturity declared,
      CODEOWNERS entry.
- [ ] New dependency: justification, license checked, optional where possible.
