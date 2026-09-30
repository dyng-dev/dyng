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
    "sphinx.ext.napoleon",  # the Google-style docstrings of python/dyng
    "autoapi.extension",  # the Python API reference (below)
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}  # .rst: the autoapi pages
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

# -- sphinx-autoapi (the Python API reference) -----------------------------------------------------

# PLAN Section 9.2: the Python reference is generated without compiling or importing the
# extension. sphinx-autoapi parses the typed layer python/dyng/*.py and the committed stubs of the
# native module (python/dyng/_core.pyi, checked by `scripts/regen.py --stubs --check`) statically,
# so the site builds on CPU-only runners and on Read the Docs. The public API is what `import dyng`
# exposes: the package page documents the names of `dyng.__all__` (re-exported from the private
# implementation modules), and the algorithm and helper namespaces (dyng.sssp, dyng.cycle_count,
# dyng.io, dyng.generators, dyng.testing) get a page each. Private modules (a leading underscore),
# the implementation modules whose names the package re-exports, and the command line (documented
# in api/cli.md) are left out.
autoapi_type = "python"
autoapi_dirs = [str(REPO_ROOT / "python" / "dyng")]
autoapi_root = "api/python/reference"
autoapi_file_patterns = ["*.pyi", "*.py"]
autoapi_ignore = ["*/cli/*", "*/__main__.py"]
# Parsed (so the package's re-exports resolve) but given no page of their own.
_HIDDEN_MODULES = {
    f"dyng.{name}"
    for name in ("array", "batch", "config", "errors", "graph", "profiler", "resources")
}
autoapi_options = ["members", "show-inheritance", "show-module-summary", "imported-members"]
autoapi_member_order = "groupwise"
autoapi_add_toctree_entry = False  # api/python/index.md links the generated pages
autoapi_keep_files = False
autoapi_python_class_content = "both"
autodoc_typehints = "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
# `dyng._backend.native` is the native module chosen at import time (dyng._core, or a CUDA plugin
# from 0.1.x), so it cannot be resolved statically; the stubs document dyng._core.
suppress_warnings = ["autoapi.python_import_resolution"]
# Names without a page: NumPy and the standard library (no inventory is loaded, so the build never
# needs the network), and the string-literal aliases of the typed layer (their values are listed
# in each docstring).
nitpick_ignore_regex += [
    (r"py:(class|obj|data)", r"(numpy|np|collections\.abc|typing|os|pathlib|enum)\..*"),
    (r"py:(class|obj|data)", r"(PathLike|[A-Z][A-Za-z]*Name|Any|ArrayLike)"),
]

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


def _python_domain_for_autoapi(app, docname, source):  # noqa: ANN001, ANN201
    # The site's primary domain is C++ (the curated Breathe pages); the generated Python pages
    # resolve the bare roles of the docstrings (:class:`Graph`, :func:`update`) in Python's.
    if docname.startswith(autoapi_root + "/"):
        source[0] = ".. default-domain:: py\n\n" + source[0]


def _resolve_reexported(app, env, node, contnode):  # noqa: ANN001, ANN201
    # Signatures name a class by its defining module (dyng.graph.Graph), but the reference
    # documents it where users import it (dyng.Graph): resolve the former to the latter.
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget", "")
    parts = target.split(".")
    if len(parts) < 3 or parts[0] != "dyng" or ".".join(parts[:2]) not in _HIDDEN_MODULES:
        return None
    py = env.get_domain("py")
    node = node.deepcopy()
    node["reftarget"] = "dyng." + ".".join(parts[2:])
    return py.resolve_xref(
        env, node["refdoc"], app.builder, node["reftype"], node["reftarget"], node, contnode
    )


def _skip_hidden_modules(app, what, name, obj, skip, options):  # noqa: ANN001, ANN201, PLR0913
    private = name.split(".")[-1].startswith("_")
    if what in ("module", "package") and (name in _HIDDEN_MODULES or private):
        return True
    if name == "dyng.__version__":  # a special name, documented all the same
        return False
    return None


def setup(app):  # noqa: ANN001, ANN201 (Sphinx's interface)
    # After the theme's own html-page-context handler (priority 500), which sets the default.
    app.connect("html-page-context", _edit_included_source, priority=900)
    app.connect("source-read", _python_domain_for_autoapi)
    app.connect("autoapi-skip-member", _skip_hidden_modules)
    app.connect("missing-reference", _resolve_reexported)


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
