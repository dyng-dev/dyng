# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of ci/docs_links.py: good links pass, each kind of broken link is reported."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "docs_links", Path(__file__).resolve().parent.parent / "docs_links.py"
)
docs_links = importlib.util.module_from_spec(_SPEC)
sys.modules["docs_links"] = docs_links
_SPEC.loader.exec_module(docs_links)

URL = "https://github.com/dyng-dev/dyng"


def _repo(tmp_path: Path, text: str) -> list[str]:
    (tmp_path / "docs").mkdir()
    (tmp_path / "CONTRIBUTING.md").write_text("x\n")
    page = tmp_path / "docs" / "page.md"
    page.write_text(text)
    return docs_links.check_repo_urls(tmp_path, [page])


def test_repository_urls_that_exist_pass(tmp_path: Path) -> None:
    text = (
        f"[a]({URL}/blob/main/CONTRIBUTING.md#dco), <{URL}/tree/main/docs>, "
        f"{URL}/edit/main/docs/page.md. Also {URL}/issues/new, {URL}/blob/v0.1.0/GONE.md and "
        f"`{URL}/blob/main/<path>`.\n"
    )
    assert _repo(tmp_path, text) == []


def test_missing_file_and_directory_are_reported(tmp_path: Path) -> None:
    text = f"[a]({URL}/blob/main/CONTRIBUTNG.md)\n[b]({URL}/tree/main/doc)\n"
    errors = _repo(tmp_path, text)
    assert len(errors) == 2
    assert "docs/page.md:1: blob/main/CONTRIBUTNG.md: no such file" in errors[0]
    assert "docs/page.md:2: tree/main/doc: no such directory" in errors[1]


def test_blob_of_a_directory_is_reported(tmp_path: Path) -> None:
    assert _repo(tmp_path, f"{URL}/blob/main/docs\n")


def _site(tmp_path: Path, index: str) -> list[str]:
    site = tmp_path / "html"
    (site / "sub").mkdir(parents=True)
    (site / "_static").mkdir()
    (site / "_static" / "t.html").write_text('<a href="{{ pathto(x) }}">jinja</a>')
    (site / "_static" / "s.css").write_text("")
    (site / "sub" / "b.html").write_text('<h1 id="top">B</h1><a name="old"></a>')
    (site / "sub" / "index.html").write_text("<p>index</p>")
    (site / "index.html").write_text(index)
    return docs_links.check_html(site)


def test_site_links_that_exist_pass(tmp_path: Path) -> None:
    index = (
        '<p id="here"></p><a href="#here">1</a><a href="sub/b.html#top">2</a>'
        '<a href="sub/b.html#old">3</a><a href="sub/">4</a><a href="#">5</a>'
        '<link href="_static/s.css"><a href="https://example.org/x#y">6</a>'
        '<a href="mailto:a@example.org">7</a>'
    )
    assert _site(tmp_path, index) == []


def test_missing_page_and_anchors_are_reported(tmp_path: Path) -> None:
    index = '<a href="#nowhere">1</a><a href="sub/c.html">2</a><a href="sub/b.html#gone">3</a>'
    errors = _site(tmp_path, index)
    assert len(errors) == 3
    assert "#nowhere: no anchor '#nowhere' in index.html" in errors[0]
    assert "sub/c.html: no such file" in errors[1]
    assert "no anchor '#gone' in b.html" in errors[2]


def test_empty_site_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "html").mkdir()
    assert docs_links.check_html(tmp_path / "html")
