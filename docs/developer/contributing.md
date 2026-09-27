# Contributing

The contributor guide is
[`CONTRIBUTING.md`](https://github.com/dyng-dev/dyng/blob/main/CONTRIBUTING.md) in the repository
root, where GitHub shows it next to every issue and pull request. It covers:

- where to start (the `good first issue` labels, {doc}`labels`, and the {doc}`../roadmap`);
- the development environment (`environment.yml`, `scripts/dev_env.sh`) and the presets;
- the tests (labels `cpu`, `gpu`, `parity`) and the local gate `ci/check.sh`;
- parity with the original research codes ({doc}`parity`) and performance measurements;
- style: clang-format, the naming rules of ADR 0004, SPDX headers and provenance headers of
  ported files;
- Doxygen conventions and the documentation build ({doc}`documentation`);
- commit messages, the Developer Certificate of Origin (DCO) sign-off for external contributors;
- the pull-request process and review rules (with the {doc}`api_review_checklist` for public API
  changes), and how to add an algorithm or port research code (with the provenance rules,
  {doc}`provenance`).

The community files next to it: the
[Code of Conduct](https://github.com/dyng-dev/dyng/blob/main/CODE_OF_CONDUCT.md) (Contributor
Covenant 3.0), [SECURITY.md](https://github.com/dyng-dev/dyng/blob/main/SECURITY.md) (private
vulnerability reporting), [SUPPORT.md](https://github.com/dyng-dev/dyng/blob/main/SUPPORT.md),
[GOVERNANCE.md](https://github.com/dyng-dev/dyng/blob/main/GOVERNANCE.md) (including the log of
approval checkpoints) and
[MAINTAINERS.md](https://github.com/dyng-dev/dyng/blob/main/MAINTAINERS.md).

The shortest path from a clone to a green build:

```bash
conda env create -f environment.yml
source scripts/dev_env.sh
pre-commit install
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
ci/check.sh
```
