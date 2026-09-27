# ADR 0002: License and credit

- **Status:** Accepted
- **Date:** 2026-09-27
- **Deciders:** S M Shovan (lead maintainer)

## Context

The author wants the library to be open source while requiring credit. No OSI-approved license
can require academic citation, so credit has a legally required part (attribution in
redistributions) and a requested part (citation in publications). None of the six original
repositories has a license, so copying their code into a licensed repository is a relicensing:
it needs the agreement of the copyright holders. Essentially every commit in the originals is
by the author; the remaining gate is the university (and any funding terms). Some commits used
a placeholder identity, and parts of the fixed codes were written with an AI assistant.

## Decision

- **Apache-2.0**, verbatim in `LICENSE` (also `LICENSES/Apache-2.0.txt` for REUSE). Plain
  Apache-2.0, not "WITH LLVM-exception", because that exception waives attribution for compiled
  template code.
- **Required credit** through Apache-2.0 section 4(d): a short `NOTICE` (Missouri University of
  Science and Technology; portions derived from the DynaMOSP, DynLP, ESCHER/ESCHER+, H-SOSP and
  TruCy/DynTruCy research codes, Copyright 2023-2026 their authors).
- **Requested credit** through `CITATION.cff` (CFF 1.2.0; the software is the preferred
  citation until the dissertation has a stable identifier; one reference per algorithm paper),
  `docs/references.bib` (the single BibTeX source, compiled into `dyng::citation()`), a README
  "How to cite" section, a "Cite this" box on each algorithm page, and a Zenodo DOI per release.
<!-- REUSE-IgnoreStart -->
- **SPDX headers** in every file (`SPDX-FileCopyrightText: 2026 The dynG Authors`,
  `SPDX-License-Identifier: Apache-2.0`), checked with `reuse lint`; `AUTHORS.md` defines
  "The dynG Authors". Ported files add `// Derived from <repo>@<commit>:<path>`.
<!-- REUSE-IgnoreEnd -->
- **IP and consent (Appendix E of the plan):** the university IP office has asked for the code
  to be public, so IP clearance is done; the author decided that co-author consent e-mails are
  not needed. Ports may be published as soon as they pass their gates. The decisions are logged
  in `GOVERNANCE.md`.
- **Fresh history (O22):** the originals' git histories are not imported (they contain cluster
  host names and a placeholder identity). Provenance is kept through file headers and
  `parity/references.toml`.
- **Contributions:** DCO 1.1 sign-off, no CLA (Apache-2.0 section 5: inbound = outbound).
- **Third-party code** is not vendored; build-time dependencies are fetched by CPM. Datasets,
  papers, cluster scripts and personal paths are never committed.

## Consequences

- Anyone may use, modify and redistribute dynG, including commercially, provided they keep
  `LICENSE` and `NOTICE`.
- Funding acknowledgements go into the README and docs "Acknowledgements", not into `NOTICE`.
- If the university later asks for a CLA or different terms, this ADR is superseded.
