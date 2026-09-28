#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Check the links that the Sphinx build cannot check (step 3 of ci/docs.sh).

The Sphinx build (step 2, ``-W -n``) already fails on a link to a missing page, heading or
local file, through MyST's reference resolution. Two kinds of link are invisible to it:

* **Links to repository files by their GitHub URL.** Pages link files outside ``docs/`` as
  ``https://github.com/dyng-dev/dyng/blob/main/<path>`` (docs/developer/documentation.md), and
  the community files, issue forms and workflows do the same. Sphinx treats them as external
  links, so a typo'd path would pass. This check maps every such URL (``blob``, ``tree``,
  ``edit`` or ``raw`` on ``main``) in the tracked text files to ``<repository>/<path>`` and
  fails if the file (``blob``, ``edit``, ``raw``) or directory (``tree``) does not exist.
* **In-page anchors of the built site.** With ``--html DIR``, every relative ``href`` of every
  built page must name an existing file, and its ``#fragment`` an existing ``id`` in that file
  (for example the C++ reference anchors ``#_CPPv4...`` that Breathe and Doxygen generate).

External links (other hosts) are checked only by the weekly job of docs.yml
(``DYNG_LINKCHECK_EXTERNAL=1 ci/docs.sh``, the Sphinx linkcheck builder).

Usage::

    python3 ci/docs_links.py                         # repository URLs only
    python3 ci/docs_links.py --html build/docs/html  # ... and the built site's anchors
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent

REPO_URL_RE = re.compile(
    r"https://github\.com/dyng-dev/dyng/(blob|tree|edit|raw)/main/([^\s)\]<>\"'`#?|,;]*)"
)
TEXT_SUFFIXES = {".md", ".yml", ".yaml", ".py", ".sh", ".toml", ".cff", ".txt", ".hpp", ".cpp"}


def tracked_text_files(root: Path) -> list[Path]:
    """Return the tracked text files of the repository (a directory walk outside git)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        files = [root / name for name in out.split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        files = [p for p in root.rglob("*") if ".git" not in p.parts and "build" not in p.parts]
    return sorted(p for p in files if p.suffix in TEXT_SUFFIXES and p.is_file())


def check_repo_urls(root: Path, files: list[Path]) -> list[str]:
    """Return an error for every GitHub URL of this repository that names a missing path."""
    errors: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), start=1):
            for kind, target in REPO_URL_RE.findall(line):
                target = unquote(target).rstrip(".:")
                if not target:
                    continue  # a placeholder such as blob/main/<path>
                where = root / target
                ok = where.is_dir() if kind == "tree" else where.is_file()
                if not ok:
                    what = "directory" if kind == "tree" else "file"
                    errors.append(
                        f"{path.relative_to(root)}:{n}: {kind}/main/{target}: no such {what} "
                        "in the repository"
                    )
    return errors


class _Page(HTMLParser):
    """Collect the ids and the hrefs of one HTML page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name == "id" or (tag == "a" and name == "name"):
                self.ids.add(value)
            elif name == "href" and tag in ("a", "link"):
                self.hrefs.append(value)


def check_html(html_dir: Path) -> list[str]:
    """Return an error for every relative link of the built site to a missing file or anchor."""
    pages: dict[Path, _Page] = {}
    for path in sorted(html_dir.rglob("*.html")):
        if "_static" in path.relative_to(html_dir).parts:
            continue  # theme templates (Jinja source), not pages
        page = _Page()
        page.feed(path.read_text(encoding="utf-8", errors="replace"))
        pages[path.resolve()] = page
    if not pages:
        return [f"{html_dir}: no HTML pages (build the site first)"]
    errors: list[str] = []
    for path, page in pages.items():
        for href in page.hrefs:
            parts = urlsplit(href)
            if parts.scheme or parts.netloc or href.startswith(("javascript:", "data:")):
                continue
            target = path if not parts.path else (path.parent / unquote(parts.path)).resolve()
            if parts.path.endswith("/") or target.is_dir():
                target = target / "index.html"
            where = f"{path.relative_to(html_dir.resolve())}: {href}"
            if not target.exists():
                errors.append(f"{where}: no such file")
                continue
            fragment = unquote(parts.fragment)
            if fragment and target.suffix == ".html":
                ids = pages[target].ids if target in pages else set()
                if fragment not in ids:
                    errors.append(f"{where}: no anchor '#{fragment}' in {target.name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--html", type=Path, help="the built site (ci/docs.sh: build/docs/html)")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    args = parser.parse_args()
    root = args.root.resolve()
    files = tracked_text_files(root)
    errors = check_repo_urls(root, files)
    checked = f"repository URLs in {len(files)} files"
    if args.html:
        errors += check_html(args.html)
        checked += f", relative links and anchors of {args.html}"
    for error in errors:
        print(f"docs_links: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"docs_links: OK ({checked})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
