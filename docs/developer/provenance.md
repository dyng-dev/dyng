# Provenance and copyright holders

Where the code of dynG comes from, who holds copyright in it, and on what basis it is published
under Apache-2.0. The file headers say it per file; this page says it per source. The license
decision is ADR 0002 ({doc}`../adr/0002-license-and-credit`); the approvals are logged in
[GOVERNANCE.md](https://github.com/dyng-dev/dyng/blob/main/GOVERNANCE.md).

## Basis for publishing ported code

None of the six original research repositories carries a license, so publishing code derived
from them under Apache-2.0 is a relicensing by their copyright holders. At the approval of the
plan (2026-09-27) the lead maintainer recorded that:

- the university's IP office asked for the code to be made public, so IP clearance is done
  (decision O26);
- written consent e-mails from co-authors are not needed, because the code was written by the
  lead maintainer (the table below), and ports may be published as soon as they pass their gates
  (decision A5);
- the originals' git histories are not imported (decision O22); provenance is kept by the
  per-file headers `// Derived from <repo>@<commit>:<path>` (checked by
  `ci/provenance_check.py`) and by `parity/references.toml`.

## The original repositories

Counts are of the commits reachable from the pinned commit (`git log <commit>` in the original,
2026-09-27). Every original is public under `https://github.com/SMShovan/<repository>`.

| Repository @ pinned commit | Algorithm in dynG | Commits (first - last) | Commit identities | AI-assisted commits |
|---|---|---|---|---|
| MOSP-OpenMP @ c352151 | `sssp` (CPU), `mosp` | 59 (2023-09 - 2026-09) | S M Shovan (as "S M Shovan" and "SMShovan"): 59 | 31 |
| MOSP-CUDA @ e220ee2 | `sssp` (CUDA), `mosp` | 35 (2025-08 - 2026-09) | S M Shovan: 34; placeholder `CUDA <user@example.com>`: 1 | 33 |
| CycleEnumeration-GPU @ 0a976ad | `cycle_count` | 110 (2026-05 - 2026-09) | S M Shovan: 110 | 35 |
| ESCHER-GPU @ abcf9b2 | hypergraph store, `triad_count` | 80 (2024-06 - 2026-09) | S M Shovan (as "S M Shovan", "SMShovan" and a university cluster account): 66; placeholder: 14 | 34 |
| MOSP_ESCHER @ 4b86159 | `hyper_sssp`, hypergraph store | 51 (2026-04 - 2026-09) | S M Shovan (as "S M Shovan" and "S. M. Shovan"): 51 | 50 |
| LabelPropagation-CUDA @ a276a3a | `label_propagation` | 63 (2024-07 - 2026-09) | S M Shovan (as "S M Shovan", "SMShovan" and a university cluster account): 56; placeholder: 7 | 37 |

- **Placeholder identity.** 22 commits (14 in ESCHER-GPU, 7 in LabelPropagation-CUDA, 1 in
  MOSP-CUDA) carry the unconfigured identity `CUDA <user@example.com>`. The lead maintainer
  confirmed (2026-09-27) that they were made by the author or by an AI coding assistant working
  on the author's behalf; they are credited to S M Shovan (`AUTHORS.md`), and the placeholder is not a person.
- **AI-assisted commits** carry a `Co-Authored-By` trailer naming the AI coding assistant (the
  fixes of 2026 before the port). The IP office's answer covers the code as it is.
- **Code written outside git.** The papers list co-authors (`AUTHORS.md`, `CITATION.cff`). For
  TruCy / DynTruCy (CycleEnumeration-GPU), Arindam Khanda is a co-first author; whether any of
  the code was written by a co-author outside the recorded commits is to be confirmed by the
  lead maintainer. Until then, co-authors are credited in `AUTHORS.md` as research
  collaborators, not as authors of the code.
- The cluster account's e-mail address and the host names of the originals' histories are not
  repeated here (the repository never records host names or personal paths).

Which dynG files derive from each original: `git grep -l "Derived from <repository>@"`.

## Third-party code

No third-party code is copied into the repository. Build-time dependencies (GoogleTest, CPM.cmake
and, with CUDA, CCCL from the toolkit) are fetched or found at a pinned version and keep their
own licenses. The Code of Conduct is the Contributor Covenant 3.0 (CC BY-SA 4.0; `REUSE.toml`,
`LICENSES/CC-BY-SA-4.0.txt`).

If code whose copyright is held outside The dynG Authors ever has to be copied in, it goes to
`third_party/<name>/` with its own license text, keeps its original copyright and license
notices (an extra `SPDX-FileCopyrightText` line or a `REUSE.toml` annotation, and the license
in `LICENSES/`), gets a `NOTICE` line only if its license requires one, and gets a row here:

| Directory | Origin (URL @ commit) | Copyright holders | License | Why it is copied |
|---|---|---|---|---|
| (none) | | | | |
