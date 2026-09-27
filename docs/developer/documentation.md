# Writing documentation

The site follows the Diataxis structure. Put a page where its reader looks for it:

| Directory | Kind | A page here... |
|---|---|---|
| `docs/getting_started/` | first steps | installs and runs a first result |
| `docs/tutorials/` | tutorials | teaches by doing a complete task, step by step |
| `docs/how_to/` | how-to guides | solves one task for a reader who knows the basics |
| `docs/concepts/` | explanation | explains a design and the reasons for it |
| `docs/algorithms/` | reference | one page per algorithm, with the nine required sections (PLAN Section 9.5) |
| `docs/api/` | reference | the C++ API (generated), file formats |
| `docs/developer/` | developer | the plan, design pages, guides for maintainers, retrospectives |
| `docs/adr/` | decisions | one ADR per decision (`NNNN-title.md`) |

## Pages

- Pages are **MyST Markdown** (`.md`); no reStructuredText is needed. Directives use fenced
  blocks, for example a note:

  ````md
  :::{note}
  Text.
  :::
  ````

- Every page must be in a toctree (the section's `index.md`); the build fails otherwise. ADRs,
  retrospectives and algorithm pages are picked up by glob patterns, so a new file there needs
  no toctree edit (the ADR table in `docs/adr/README.md` still does).
- Link pages with `` {doc}`../concepts/update_model` `` or a relative Markdown link
  (`[text](../concepts/update_model.md)`); headings up to level 3 have anchors
  (`page.md#heading-text`). Files outside `docs/` are linked by their GitHub URL. Do not
  write links as raw HTML: the build checks Markdown links and directives, not raw HTML.
- Quote code from the repository with `literalinclude` and `:start-at:` / `:end-at:` text
  anchors rather than line numbers: a change that removes the anchor fails the build instead of
  silently quoting the wrong lines.

## The C++ API reference

`docs/api/cpp/` has one page per Doxygen group (`@defgroup` / `@ingroup`), rendered with
`doxygengroup`. A new public header must belong to a group; a new group needs a page there
(the build warns about a group without a page). The documentation of every entity comes from the
header's Doxygen comments, written to the conventions of PLAN Section 9.1 (a one-line `@brief`,
every parameter and exception, `@sync` / `@async`, and `@backends`, `@determinism`, `@paper` on
`compute()` and `update()`); Doxygen runs with warnings as errors and
`ci/doxygen_coverage.py` checks the rest.

`docs/_ext/dyng_breathe.py` works around three Breathe limitations (the scope of free functions
on group pages, links to groups and headers, a duplicated `constexpr` in Doxygen 1.18's XML);
its docstring explains each. Names without a page by design (the standard library, the CUDA
runtime, `dyng::detail`, namespace prefixes) are listed in `nitpick_ignore_regex` in
`docs/conf.py`; everything else must resolve.

## Citations

`docs/references.bib` is the single source of citations: `dyng::citation()` embeds it, and the
site renders it on {doc}`../citing`. Keep `CITATION.cff` in sync. Comments in the file must not
contain an at sign, which BibTeX parsers read as the start of an entry.

## Tools and versions

Sphinx, MyST, pydata-sphinx-theme, Breathe, sphinx-copybutton, sphinx-design,
sphinxcontrib-bibtex and Doxygen are pinned in `environment.yml`. The local build
({doc}`../how_to/build_the_docs`), `.github/workflows/docs.yml` and `.readthedocs.yaml` all
install from that file. When you change a pin, update the environment
(`conda env update -f environment.yml`) and run `ci/docs.sh`.
