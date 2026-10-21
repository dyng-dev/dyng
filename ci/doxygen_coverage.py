#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Check the Doxygen conventions that Doxygen itself does not enforce (PLAN Section 9.1).

Doxygen (run by ci/docs.sh with WARN_AS_ERROR) already fails on undocumented entities and on
missing @param / @return. This script reads its XML output and additionally requires:

  * every public header has an @file comment with a @brief;
  * every class, struct, function, enumeration, typedef, variable and macro at namespace (or file)
    scope belongs to a group (@ingroup / @defgroup); members of a class inherit the group of their
    class;
  * every documented entity has a one-line @brief (a non-empty brief description);
  * compute() and update() of every algorithm namespace (one per manifest under
    cpp/src/algorithms/) carry @backends, @determinism, @paper and @guarantee (the exception
    guarantee, PLAN Section 4.7.3; the 0.1 API review, ADR 0023), and so do the other functions
    that mutate a container or a result: graph::apply(), dyng::update() and dyng::update_each()
    (@guarantee only), and every public member function of the container, batch and result
    classes (GUARANTEE_CLASSES) that can change the object or consume an argument: non-const,
    not noexcept, not a constructor, destructor or assignment (a noexcept function is no-throw by
    its signature), plus the static factories that take an rvalue reference (they consume it);
  * every CUDA-capable function carries @sync or @async and documents its exceptions (at least
    one @throws, or `noexcept`; destructors are exempt from the latter): every function that
    takes a `resources`, a `stream_ref` or a `memory_resource_ref`, and the members that do
    stream-ordered work with the stream they were built with (STREAM_ORDERED_MEMBERS, the move
    assignment of `buffer`). The copy and move operations of `resources` itself are exempt;
  * the frozen hypergraph headers (dyng/hypergraph/, M8b; ADR 0047) and the experimental extension
    headers (dyng/experimental/, M9b; ADR 0058: their group `experimental`, and every one's @file
    brief starts with the banner "EXPERIMENTAL PUBLIC; may change in any minor release") are held
    to the freeze's rules: every entity they declare is in the group `hypergraph`
    (`experimental`); every function they declare (free or member, deleted special members and
    destructors exempt) documents its exceptions (at least one @throws) or is `noexcept`; and
    every one that documents a cuda_error (device work, also without a `resources` argument: the
    views) carries @sync or @async.

Usage: ci/doxygen_coverage.py <doxygen-xml-dir>
"""

from __future__ import annotations

import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REQUIRED_ALGO_PARS = ("Backends:", "Determinism:", "Paper:", "Exception safety:")
# Functions outside the algorithm namespaces that change a container or a result: each states
# its exception guarantee (strong / basic) with @guarantee.
GUARANTEED_FUNCTIONS = {"dyng::graph::apply", "dyng::update", "dyng::update_each"}
# The container, batch and result classes of the public API: their mutating members state their
# guarantee (see the module docstring).
GUARANTEE_CLASSES = {
    "dyng::graph",
    "dyng::edge_batch",
    "dyng::edge_list",
    "dyng::hypergraph",
    "dyng::hyperedge_batch",
    "dyng::hyperedge_list",
    "dyng::sssp::result",
    "dyng::cycle_count::result",
    "dyng::triad_count::result",
}


def needs_guarantee(class_name: str, member: ET.Element) -> bool:
    """Whether a member function of a GUARANTEE_CLASSES class must carry @guarantee."""
    if class_name not in GUARANTEE_CLASSES or member.get("kind") != "function":
        return False
    mname = text_of(member.find("name"))
    short = class_name.rsplit("::", 1)[1]
    if mname in (short, "~" + short, "operator="):
        return False
    args = text_of(member.find("argsstring"))
    if "noexcept" in args:
        return False
    if member.get("static") == "yes":
        return any(text_of(p.find("type")).endswith("&&") for p in member.findall("param"))
    return member.get("const") != "yes"


def algorithm_namespaces(root: Path = REPO) -> set[str]:
    """dyng::<name> of every algorithm manifest (cpp/src/algorithms/<name>/manifest.toml; the
    scaffold's _template is not an algorithm)."""
    names = set()
    for manifest in sorted((root / "cpp" / "src" / "algorithms").glob("*/manifest.toml")):
        if manifest.parent.name.startswith("_"):
            continue
        names.add("dyng::" + tomllib.loads(manifest.read_text())["name"])
    return names


ALGORITHM_NAMESPACES = algorithm_namespaces()
IGNORED_NAMESPACES = {"std"}


ALIAS_OF = {"Exception safety:": "@guarantee"}


def text_of(node: ET.Element | None) -> str:
    return "" if node is None else "".join(node.itertext()).strip()


STREAM_TYPES = (
    "stream_ref",
    "dyng::stream_ref",
    "memory_resource_ref",
    "dyng::memory_resource_ref",
)

# Members that enqueue or release stream-ordered work without taking a stream, a memory resource
# or resources as a parameter (they use the ones they were built with).
STREAM_ORDERED_MEMBERS = {
    ("dyng::buffer", "~buffer"),
    ("dyng::buffer", "resize"),
    ("dyng::resources", "release_workspaces"),
    ("dyng::resources", "synchronize"),
    ("dyng::resources", "warm_up"),
    ("dyng::stream_ref", "synchronize"),
} | {
    # The synchronous allocation members of every memory resource (M1b review): CUDA-capable
    # (pinned and device memory) without a stream parameter.
    (resource, member)
    for resource in (
        "dyng::memory_resource_ref",
        "dyng::host_memory_resource",
        "dyng::cuda_async_memory_resource",
        "dyng::pinned_host_memory_resource",
    )
    for member in ("allocate_sync", "deallocate_sync")
}


def takes_resources(member: ET.Element) -> bool:
    """Whether a function does backend or stream-ordered work: it has a parameter of type (const)
    dyng::resources&, stream_ref or memory_resource_ref."""
    for param in member.findall("param"):
        ptype = text_of(param.find("type")).replace(" ", "")
        if ptype in ("constresources&", "resources&", "constdyng::resources&", "dyng::resources&"):
            return True
        if ptype in STREAM_TYPES:
            return True
    return False


def is_move_assignment(member: ET.Element) -> bool:
    """Whether a function is a move assignment operator (it releases the target's memory)."""
    params = member.findall("param")
    return (
        text_of(member.find("name")) == "operator="
        and len(params) == 1
        and text_of(params[0].find("type")).replace(" ", "").endswith("&&")
    )


# The frozen hypergraph headers (M8b, ADR 0047) and the experimental extension headers (M9b,
# ADR 0058: PLAN 9.1's group `experimental`), and the group their entities belong to.
FROZEN_GROUP_HEADERS = {
    "dyng/hypergraph/": "hypergraph",
    "dyng/experimental/": "experimental",
    "dyng/triad_count.hpp": "triad_count",
}
# The banner every experimental header's @file brief starts with (PLAN 5.9).
EXPERIMENTAL_BANNER = "EXPERIMENTAL PUBLIC; may change in any minor release"


def experimental_banners(compounds: dict[str, ET.Element], problems: list[str]) -> None:
    """Every header under dyng/experimental/ carries the experimental banner in its @file brief."""
    for c in compounds.values():
        if c.get("kind") != "file":
            continue
        loc = c.find("location")
        file = loc.get("file", "") if loc is not None else ""
        if "dyng/experimental/" not in file:
            continue
        brief = " ".join(text_of(c.find("briefdescription")).split())
        if not brief.startswith(EXPERIMENTAL_BANNER):
            problems.append(f"{file}: the @file brief does not start with '{EXPERIMENTAL_BANNER}'")


def frozen_group_rules(compounds: dict[str, ET.Element], problems: list[str]) -> None:
    """The freeze's rules for FROZEN_GROUP_HEADERS (see the module docstring)."""

    def header_of(node: ET.Element) -> tuple[str, str] | None:
        loc = node.find("location")
        file = loc.get("file", "") if loc is not None else ""
        for prefix, group in FROZEN_GROUP_HEADERS.items():
            if prefix in file:
                return group, f"{file}:{loc.get('line') if loc is not None else '?'}"
        return None

    group_of: dict[str, str] = {}
    for c in compounds.values():
        if c.get("kind") == "group":
            g = text_of(c.find("compoundname"))
            for inner in c.findall("innerclass"):
                group_of[inner.get("refid", "")] = g
            for m in c.iter("memberdef"):
                group_of[m.get("id", "")] = g
    class_names = {
        text_of(c.find("compoundname"))
        for c in compounds.values()
        if c.get("kind") in ("class", "struct", "union")
    }
    for refid, c in compounds.items():
        kind = c.get("kind")
        if kind in ("class", "struct", "union"):
            where = header_of(c)
            name = text_of(c.find("compoundname"))
            if where and name.rsplit("::", 1)[0] not in class_names:
                if group_of.get(refid) != where[0]:
                    problems.append(f"{where[1]}: {kind} {name} is not in the group {where[0]}")
        for m in c.iter("memberdef"):
            where = header_of(m)
            if where is None or m.get("prot") not in (None, "public"):
                continue
            mname = text_of(m.find("qualifiedname")) or text_of(m.find("name"))
            if kind in ("group", "namespace") and group_of.get(m.get("id", "")) != where[0]:
                problems.append(f"{where[1]}: {mname} is not in the group {where[0]}")
            if m.get("kind") != "function":
                continue
            args = text_of(m.find("argsstring"))
            if args.replace(" ", "").endswith("=delete") or text_of(m.find("name")).startswith("~"):
                continue
            throws = [
                text_of(n)
                for pl in m.iter("parameterlist")
                if pl.get("kind") == "exception"
                for n in pl.iter("parametername")
            ]
            if not throws and "noexcept" not in args:
                problems.append(f"{where[1]}: {mname} documents no @throws (and is not noexcept)")
            titles = {text_of(t) for t in m.iter("title")}
            if any("cuda_error" in t for t in throws) and "Synchronization:" not in titles:
                problems.append(
                    f"{where[1]}: {mname} may run device work but has no @sync / @async"
                )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    xml_dir = Path(argv[1])
    index = ET.parse(xml_dir / "index.xml").getroot()
    problems: list[str] = []

    grouped_classes: set[str] = set()
    compounds: dict[str, ET.Element] = {}
    for entry in index.findall("compound"):
        refid = entry.get("refid")
        root = ET.parse(xml_dir / f"{refid}.xml").getroot().find("compounddef")
        assert root is not None
        compounds[refid] = root
        if entry.get("kind") == "group":
            grouped_classes.update(c.get("refid") for c in root.findall("innerclass"))

    private_classes = {
        inner.get("refid")
        for c in compounds.values()
        for inner in c.findall("innerclass")
        if inner.get("prot") != "public"
    }
    class_names = {
        text_of(c.find("compoundname"))
        for c in compounds.values()
        if c.get("kind") in ("class", "struct", "union")
    }

    for refid, cdef in compounds.items():
        kind = cdef.get("kind")
        name = text_of(cdef.find("compoundname"))
        where = cdef.find("location")
        loc = where.get("file") if where is not None else refid
        if kind == "file":
            if not text_of(cdef.find("briefdescription")):
                problems.append(f"{loc}: header has no @file / @brief")
            # A grouped macro is listed with its group; one left in the file compound is not.
            for member in cdef.iter("memberdef"):
                if member.get("kind") == "define":
                    mloc = member.find("location")
                    problems.append(
                        f"{mloc.get('file')}:{mloc.get('line')}: macro "
                        f"{text_of(member.find('name'))} is not in any group (@ingroup)"
                    )
            continue
        if kind == "namespace":
            if name in IGNORED_NAMESPACES:
                continue
            for member in cdef.iter("memberdef"):
                problems.append(
                    f"{member.find('location').get('file')}:{member.find('location').get('line')}:"
                    f" {name}::{text_of(member.find('name'))} is not in any group (@ingroup)"
                )
            continue
        if refid in private_classes:
            continue
        if kind in ("class", "struct", "union"):
            outer = name.rsplit("::", 1)[0]
            if refid not in grouped_classes and outer not in class_names:
                problems.append(f"{loc}: {kind} {name} is not in any group (@ingroup)")
            if not text_of(cdef.find("briefdescription")):
                problems.append(f"{loc}: {kind} {name} has no @brief")
        for member in cdef.iter("memberdef"):
            if member.get("prot") not in (None, "public"):
                continue
            mname = text_of(member.find("name"))
            mloc = member.find("location")
            at = f"{mloc.get('file')}:{mloc.get('line')}" if mloc is not None else loc
            if member.get("kind") == "friend":
                continue
            if text_of(member.find("argsstring")).replace(" ", "").endswith("=delete"):
                continue  # deleted special members (non-copyable, non-movable) need no text
            if not text_of(member.find("briefdescription")):
                problems.append(f"{at}: {name}::{mname} has no @brief")
            cuda_capable = (
                takes_resources(member)
                or (name, mname) in STREAM_ORDERED_MEMBERS
                or (name == "dyng::buffer" and is_move_assignment(member))
            )
            if (
                member.get("kind") == "function"
                and cuda_capable
                and not (name == "dyng::resources" and mname in ("resources", "operator="))
            ):
                titles = {text_of(t) for t in member.iter("title")}
                qualified = text_of(member.find("qualifiedname")) or f"{name}::{mname}"
                if "Synchronization:" not in titles:
                    problems.append(f"{at}: {qualified} is CUDA-capable but has no @sync / @async")
                throws = [
                    pl for pl in member.iter("parameterlist") if pl.get("kind") == "exception"
                ]
                noexcept = "noexcept" in text_of(member.find("argsstring"))
                if not throws and not noexcept and not mname.startswith("~"):
                    problems.append(
                        f"{at}: {qualified} is CUDA-capable but documents no @throws "
                        "(and is not noexcept)"
                    )
            if (
                kind == "group"
                and member.get("kind") == "function"
                and mname
                in (
                    "compute",
                    "update",
                )
            ):
                qualified = text_of(member.find("qualifiedname"))
                namespace = qualified.rsplit("::", 1)[0]
                if namespace in ALGORITHM_NAMESPACES:
                    titles = {text_of(t) for t in member.iter("title")}
                    for required in REQUIRED_ALGO_PARS:
                        if required not in titles:
                            tag = ALIAS_OF.get(required, "@" + required.rstrip(":").lower())
                            problems.append(f"{at}: {qualified} has no {tag}")
            if member.get("kind") == "function":
                qualified = text_of(member.find("qualifiedname")) or f"{name}::{mname}"
                if qualified in GUARANTEED_FUNCTIONS or needs_guarantee(name, member):
                    titles = {text_of(t) for t in member.iter("title")}
                    if "Exception safety:" not in titles:
                        problems.append(f"{at}: {qualified} has no @guarantee")

    frozen_group_rules(compounds, problems)
    experimental_banners(compounds, problems)

    for problem in sorted(set(problems)):
        print(f"doxygen-coverage: {problem}")
    if problems:
        print(f"doxygen-coverage: {len(set(problems))} problem(s)")
        return 1
    print(f"doxygen-coverage: OK ({len(compounds)} compounds checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
