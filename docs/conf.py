# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration of the dynG documentation (PLAN Section 9.2).

Doxygen XML (docs/Doxyfile, run by ci/docs.sh) -> Breathe -> Sphinx, with MyST Markdown pages
and the pydata-sphinx-theme. The documented build command is `ci/docs.sh` (Doxygen, the
convention check, this site with warnings as errors, and the internal link check). The tool
versions are pinned in environment.yml; .github/workflows/docs.yml and .readthedocs.yaml use the
same file.

Environment variables:
  DYNG_DOXYGEN_OUTPUT       Doxygen output directory (default build/doxygen, relative to the
                            repository root); its xml/ subdirectory feeds Breathe.
  DYNG_LINKCHECK_EXTERNAL   1 = ci/docs.sh also runs the linkcheck builder on the external
                            http(s) links (the weekly job of docs.yml); off by default, so the
                            gate never depends on the network.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from pybtex.plugin import register_plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from sphinx.errors import ConfigError

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(DOCS_DIR / "_ext"))

from dyng_breathe import namespaces  # noqa: E402 (docs/_ext is on sys.path only now)

# -- Project ---------------------------------------------------------------------------------------

project = "dynG"
author = "The dynG Authors"
copyright = "2026, The dynG Authors"  # noqa: A001 (Sphinx's name)
release = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
version = ".".join(release.split(".")[:2])

# -- General ---------------------------------------------------------------------------------------

extensions = [
    "myst_parser",
    "breathe",
    "dyng_breathe",  # docs/_ext: qualify the free functions of group pages
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]
source_suffix = {".md": "markdown"}
root_doc = "index"
# adr/README.md is the ADR index as GitHub shows it; the site includes it from adr/index.md.
exclude_patterns = ["_build", "adr/README.md"]
templates_path: list[str] = []


# -- MyST ------------------------------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist", "linkify", "attrs_inline"]
myst_heading_anchors = 3
myst_linkify_fuzzy_links = False

# -- Breathe (C++ API reference from Doxygen XML) --------------------------------------------------

doxygen_output = Path(os.environ.get("DYNG_DOXYGEN_OUTPUT", "build/doxygen"))
if not doxygen_output.is_absolute():
    doxygen_output = REPO_ROOT / doxygen_output
doxygen_xml = doxygen_output / "xml"
if not (doxygen_xml / "index.xml").is_file():
    raise ConfigError(
        f"no Doxygen XML in {doxygen_xml}: build the docs with ci/docs.sh, or run "
        "`ci/docs.sh --doxygen-only` first (it needs `cmake --preset cpu-only`)"
    )
breathe_projects = {"dyng": str(doxygen_xml)}

# Every cross-reference must resolve (with -W, a missing target fails the build). The exceptions
# are names that have no page by design: the standard library and the CUDA runtime (no
# inventory is loaded, so the build never needs the network), the private namespace
# `dyng::detail` (PLAN Section 9.1), and the namespaces themselves, which Breathe renders as the
# prefix of qualified names although group pages do not declare them.
nitpicky = True
_namespace_names = {
    "::".join(parts[i:])
    for name in namespaces(doxygen_xml)
    for parts in [name.split("::")]
    for i in range(len(parts))
}
nitpick_ignore_regex = [
    (r"cpp:identifier", r"(std|size_t|u?int\d+_t)(::.*)?"),
    (r"cpp:identifier", r"CUstream_st"),
    (r"cpp:identifier", r"(dyng::)?detail(::.*)?"),
    (r"cpp:identifier", "|".join(sorted(re.escape(n) for n in _namespace_names))),
]
breathe_default_project = "dyng"
breathe_default_members = ("members",)
breathe_domain_by_extension = {"hpp": "cpp", "cuh": "cpp"}
breathe_show_include = True
primary_domain = "cpp"
highlight_language = "cpp"

# -- Citations (docs/references.bib, the single source of citations) -------------------------------


class _DyngBibStyle(UnsrtStyle):
    """The unsrt style, with @software entries (the dynG entry) formatted as @misc."""

    def format_software(self, context):  # noqa: ANN001, ANN201 (pybtex's interface)
        return self.format_misc(context)


register_plugin("pybtex.style.formatting", "dyng", _DyngBibStyle)
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "dyng"
bibtex_reference_style = "author_year"

# -- HTML ------------------------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"dynG {release}"
html_theme_options = {
    "github_url": "https://github.com/dyng-dev/dyng",
    "use_edit_page_button": True,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_context = {
    "github_user": "dyng-dev",
    "github_repo": "dyng",
    "github_version": "main",
    "doc_path": "docs",
}
html_show_sourcelink = False
html_copy_source = False

# Pages that only include another file (changelog.md -> CHANGELOG.md, developer/parity.md ->
# parity/README.md, adr/index.md -> adr/README.md): "Edit this page" opens the included file,
# which holds the text, instead of the two-line stub.
_INCLUDE_RE = re.compile(r"^(?:`{3,}|:{3,})\{include\}\s+(\S+)\s*$", re.MULTILINE)


def _edit_included_source(app, pagename, templatename, context, doctree):  # noqa: ANN001, ANN201
    source = DOCS_DIR / f"{pagename}.md"
    if doctree is None or not source.is_file():
        return
    includes = _INCLUDE_RE.findall(source.read_text(encoding="utf-8"))
    if len(includes) != 1:
        return
    target = (source.parent / includes[0]).resolve().relative_to(REPO_ROOT).as_posix()
    url = f"https://github.com/dyng-dev/dyng/edit/{html_context['github_version']}/{target}"
    context["get_edit_provider_and_url"] = lambda: ("GitHub", url)


def setup(app):  # noqa: ANN001, ANN201 (Sphinx's interface)
    # After the theme's own html-page-context handler (priority 500), which sets the default.
    app.connect("html-page-context", _edit_included_source, priority=900)


# -- Link check ------------------------------------------------------------------------------------

# The linkcheck builder runs only with DYNG_LINKCHECK_EXTERNAL=1 (ci/docs.sh; the weekly job of
# docs.yml) and checks the external links. Links within the site are checked by the -W -n build
# (MyST reference resolution), links to repository files and the site's anchors by
# ci/docs_links.py. Never checked: pages that exist only for a logged-in owner or return 404 to
# anonymous clients (settings, "new" forms, package management), and the documentation host that
# is connected only at checkpoint A4; they are click paths, not references. Links to repository
# files on main (blob/tree/edit/raw) are also left to ci/docs_links.py, which checks each path
# against the checked-out tree on every build: fetching them adds dozens of requests to GitHub's
# anonymous rate limit (HTTP 429 made the weekly job flaky) and they would 404 for files a branch
# adds before it reaches main.
if os.environ.get("DYNG_LINKCHECK_EXTERNAL", "0") != "1":
    linkcheck_ignore = [r"https?://.*"]
else:
    linkcheck_ignore = [
        r"https://github\.com/(organizations/)?dyng-dev(/dyng)?/settings.*",
        r"https://github\.com/dyng-dev/dyng/(issues|discussions)/new.*",
        r"https://github\.com/dyng-dev/dyng/security/advisories/new.*",
        r"https://github\.com/dyng-dev/dyng/(blob|tree|edit|raw)/main/.*",
        r"https://(test\.)?pypi\.org/manage/.*",
        r"https://app\.readthedocs\.org.*",
        r"https://dyng\.readthedocs\.io.*",
    ]
linkcheck_timeout = 30
linkcheck_retries = 2
linkcheck_workers = 8
# GitHub rate-limits anonymous clients; wait at most a minute per host instead of the default
# five, so the weekly job stays well inside its timeout.
linkcheck_rate_limit_timeout = 60.0
linkcheck_anchors_ignore_for_url = [r"https://github\.com/.*"]
