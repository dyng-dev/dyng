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
  DYNG_LINKCHECK_EXTERNAL   1 = the linkcheck builder also checks http(s) links (off by default:
                            CI checks internal links only, so the gate never depends on the
                            network).
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

# -- Link check ------------------------------------------------------------------------------------

# Internal links (documents, anchors, local files) are always checked. External links only with
# DYNG_LINKCHECK_EXTERNAL=1, because the gate must not depend on the network or on rate limits.
if os.environ.get("DYNG_LINKCHECK_EXTERNAL", "0") != "1":
    linkcheck_ignore = [r"https?://.*"]
linkcheck_timeout = 30
linkcheck_retries = 2
linkcheck_anchors_ignore_for_url = [r"https://github\.com/.*"]
