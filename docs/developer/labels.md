# Labels

Issues and pull requests are labelled by **kind**, **status**, **area** and **backend**, and
contributors find work through the **good-first-issue** labels. The labels
are defined in `.github/labels.yml`, the single source of truth: the `labels` workflow
(`.github/workflows/labels.yml`) applies that file to the repository on every push to `main`
that changes it, and shows the changes as a dry run on pull requests. To add, change or rename
a label, edit the file (`from_name:` renames a label and keeps it on its issues) and open a pull
request; changes made in the GitHub web interface are overwritten. Labels that are not in the
file are left alone. `ci/github_meta_check.py` (a pre-commit hook) checks the file and that
every label an issue form applies is defined.

## Kind

Every issue has at least one kind, and the issue forms set it. Most issues have exactly one;
`port`, `parity` and `api-change` say *what* is affected and may come on top of another kind
(the *Parity regression* form sets `bug` and `parity`; a maintainer may add `api-change` to an
`enhancement`).

| Label | Meaning | Set by |
|---|---|---|
| `bug` | wrong result, crash, build failure or another defect | *Bug report*, *Parity regression* forms |
| `enhancement` | a new feature or an improvement | *Feature request* form |
| `documentation` | documentation, examples, docstrings | *Documentation* form |
| `performance` | slower than expected, or a performance regression | *Performance regression* form |
| `question` | a question (better asked in Discussions; usually converted) | maintainers |
| `new-algorithm` | a new dynamic algorithm | *New algorithm* form |
| `port` | porting research code into dynG | *Port research code* form |
| `parity` | parity with the original research codes | *Parity regression* form, maintainers |
| `api-change` | changes the public C++ or Python API; required on such pull requests, with the {doc}`api_review_checklist` | *API change* form, maintainers |
| `security` | hardening work (vulnerabilities are reported privately, see SECURITY.md) | maintainers |
| `dependencies`, `github_actions` | dependency updates | dependabot |

## Status

| Label | Meaning |
|---|---|
| `triage` | new, waiting for a first look; every issue form sets it, and a maintainer removes it after triage |
| `needs-info` | waiting for more information from the reporter |
| `needs-repro` | not reproduced yet; a minimal reproducer is needed |
| `blocked` | waiting on another issue, a decision or an external party |
| `release-blocker` | must be fixed before the next release (the release checklist looks for it) |
| `duplicate`, `invalid`, `wontfix` | closed without a change; a comment says why |

## Contributors: good first issues

A good first issue carries `good first issue` **and** one category label. GitHub lists issues
labelled `good first issue` and `help wanted` on the repository's `/contribute` page, which is
why those two keep GitHub's standard names.

| Label | Category | Typical task |
|---|---|---|
| `good first issue` | any | small, well-described, with a pointer to the files to change |
| `good-first-issue:op` | a new hypergraph pattern Op (from 0.2) | a StatHyper-style pattern for `count_local_patterns` |
| `good-first-issue:reader` | a file format | a reader or writer in `cpp/src/io/` with round-trip tests |
| `good-first-issue:backend` | a backend | an OpenMP backend for a sequential-only algorithm |
| `good-first-issue:docs` | documentation | a how-to guide, an example, a clarified page |
| `help wanted` | larger tasks | where outside help is welcome; see the roadmap |

At every release the maintainers curate a backlog of good first issues. The seeds:
StatHyper-style Ops; a CPU incremental `triad_count`; an OpenMP `label_propagation`;
time-window cycle updates; the temporal and incident-vertex triads; a dynamic k-core or
connected-components algorithm as a new example of the fixed-point template.

## CI requests (maintainers only)

| Label | Effect |
|---|---|
| `ci:gpu` | run the GPU tests for this pull request (`ci/gpu_local.sh` on the lab machine; later the `gpu-test` workflow) |
| `ci:bench` | run the benchmark comparison against `main` and attach the table (required for kernel and hot-path changes) |

Only maintainers add these labels: they start work on the maintainers' hardware, which must
never run code from a fork without a review.

## Area and backend

`area:*` says which part of the library an issue concerns, so that the right maintainer sees
it: `area:core`, `area:framework`, `area:graph`, `area:hypergraph`, `area:io`, one label per
algorithm (`area:sssp`, `area:mosp`, `area:cycle_count`, `area:triad_count`,
`area:label_propagation`, `area:hyper_sssp`), `area:python`, `area:cli`, `area:build` and
`area:ci`. A new algorithm adds its `area:<algo>` label in the same pull request that adds it.
`backend:sequential`, `backend:openmp` and `backend:cuda` say which backend is affected. An
issue may carry several area and backend labels.
