# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Breathe support for the curated group pages of the C++ API reference.

Two fixes, both needed because the reference is written as one `doxygengroup` page per
Doxygen group (PLAN Section 9.2) instead of one page per file or namespace:

1. Scope of free functions (below).
3. Doxygen 1.18 writes `constexpr` both as an attribute and as the `<type>` of a constexpr
   constructor; Breathe emits both, and Sphinx cannot parse `constexpr constexpr array_view()`.
   The duplicate keyword is dropped.
2. Links to groups and headers. Doxygen links `@ref sssp` to the group and a header name in a
   comment to the header's file page; neither exists on a group page, so Sphinx cannot resolve
   `group__sssp` or `error_8hpp`. They are resolved here to the page that renders the group (for
   a header: the group it belongs to through its `@file ... @ingroup`). A Doxygen group without a
   page is reported as a warning, so a new group cannot silently go missing from the reference.

Scope of free functions:

`doxygengroup` renders the classes of a group with their qualified names (`dyng::resources`)
but free functions, enumerations, type aliases and variables with their bare names
(`to_vector`, `backend`): Breathe skips the group compound when it builds the scope and never
sees the namespace. Those entities then land in the global C++ scope, and every reference to
them, or from them to a type in `dyng`, fails to resolve (with `nitpicky`, a build error).

This extension qualifies such a name with the scope in Doxygen's own `<qualifiedname>` (for
example `dyng::` or `dyng::sssp::`). Names that Breathe already qualifies (class members) are
left alone. Tested with the Breathe version pinned in environment.yml; if a later Breathe
qualifies group members itself, the patch becomes a no-op.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from breathe import parser
from breathe.renderer import sphinxrenderer
from docutils import nodes
from sphinx.util import logging
from sphinx.util.nodes import make_refnode

logger = logging.getLogger(__name__)
_DUPLICATE_CONSTEXPR = re.compile(r"\bconstexpr(\s+constexpr)+\b")
_GROUP_DIRECTIVE = re.compile(r"^```\{doxygengroup\}\s+(\S+)", re.MULTILINE)


def _scope_of(node: Any) -> list[str]:
    """The enclosing scope of a member, from its Doxygen qualified name (dyng::io::f: dyng, io)."""
    if not isinstance(node, parser.Node_memberdefType):
        return []
    qualified = getattr(node, "qualifiedname", None)
    if not qualified or not qualified.endswith("::" + node.name):
        return []
    return qualified[: -len(node.name) - 2].split("::")


def doxygen_groups(xml_dir: Path) -> dict[str, str]:
    """Map each group refid and each header refid of the Doxygen XML to its group name."""
    targets: dict[str, str] = {}
    index = ET.parse(xml_dir / "index.xml").getroot()
    for compound in index.findall("compound[@kind='group']"):
        refid = compound.get("refid", "")
        definition = ET.parse(xml_dir / f"{refid}.xml").getroot().find("compounddef")
        if definition is None:
            continue
        name = definition.findtext("compoundname", "")
        targets[refid] = name
        for innerfile in definition.findall("innerfile"):
            targets[innerfile.get("refid", "")] = name
    return targets


def namespaces(xml_dir: Path) -> list[str]:
    """The C++ namespaces of the Doxygen XML (dyng, dyng::io, ...)."""
    index = ET.parse(xml_dir / "index.xml").getroot()
    return [c.findtext("name", "") for c in index.findall("compound[@kind='namespace']")]


def _index_group_pages(app: Any) -> None:
    xml_dir = Path(app.config.breathe_projects[app.config.breathe_default_project])
    app.env.dyng_group_targets = doxygen_groups(xml_dir)
    pages: dict[str, str] = {}
    srcdir = Path(app.srcdir)
    for source in sorted(srcdir.rglob("*.md")):
        for group in _GROUP_DIRECTIVE.findall(source.read_text(encoding="utf-8")):
            pages[group] = source.relative_to(srcdir).with_suffix("").as_posix()
    app.env.dyng_group_pages = pages
    for group in sorted(set(app.env.dyng_group_targets.values()) - set(pages)):
        logger.warning(
            "Doxygen group '%s' has no page: add a doxygengroup page under docs/api/cpp/", group
        )


def _resolve_group_ref(app: Any, env: Any, node: Any, contnode: Any) -> Any:
    if node.get("refdomain") != "std" or node.get("reftype") != "ref":
        return None
    group = getattr(env, "dyng_group_targets", {}).get(node.get("reftarget", ""))
    page = getattr(env, "dyng_group_pages", {}).get(group) if group else None
    if page is None:
        return None
    child = contnode if contnode is not None else nodes.Text(group)
    return make_refnode(app.builder, node["refdoc"], page, "", child)


def setup(app: Any) -> dict[str, Any]:
    app.setup_extension("breathe")
    app.connect("builder-inited", _index_group_pages)
    app.connect("missing-reference", _resolve_group_ref)
    original = sphinxrenderer.SphinxRenderer.get_qualification
    if not getattr(original, "_dyng_patched", False):

        def get_qualification(self: Any) -> list[str]:
            names = original(self)
            if names or self.nesting_level > 0 or not self.qualification_stack:
                return names
            return _scope_of(self.qualification_stack[0])

        get_qualification._dyng_patched = True  # type: ignore[attr-defined]
        sphinxrenderer.SphinxRenderer.get_qualification = get_qualification

    original_declaration = sphinxrenderer.SphinxRenderer.handle_declaration
    if not getattr(original_declaration, "_dyng_patched", False):

        def handle_declaration(self: Any, node: Any, obj_type: str, declaration: str, **kwargs):  # noqa: ANN202
            declaration = _DUPLICATE_CONSTEXPR.sub("constexpr", declaration)
            return original_declaration(self, node, obj_type, declaration, **kwargs)

        handle_declaration._dyng_patched = True  # type: ignore[attr-defined]
        sphinxrenderer.SphinxRenderer.handle_declaration = handle_declaration
    return {"parallel_read_safe": True, "parallel_write_safe": True}
