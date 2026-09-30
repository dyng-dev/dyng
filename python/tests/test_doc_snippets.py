# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The Python snippets of the READMEs and the documentation run and print what they say.

PLAN Section 9.6: the README quickstarts are executed, so they cannot drift. The snippets are the
code blocks marked `<!-- snippet: name -->` (format and rules: ci/doc_snippets.py); a snippet with
an output block must print exactly that block. Skipped where the repository files are not
present (an installed sdist or wheel tested without the checkout).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_SCRIPT = REPO / "ci" / "doc_snippets.py"

if not _SCRIPT.is_file() or not (REPO / "README.md").is_file():
    pytest.skip("the repository's README and ci/ are not present", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("dyng_doc_snippets", _SCRIPT)
assert _spec and _spec.loader
doc_snippets = importlib.util.module_from_spec(_spec)
sys.modules["dyng_doc_snippets"] = doc_snippets
_spec.loader.exec_module(doc_snippets)

SNIPPETS = [s for s in doc_snippets.all_snippets(REPO) if s.lang == "python"]


def test_the_quickstarts_exist_and_keep_their_size() -> None:
    assert doc_snippets.check(REPO) == []
    names = {s.name for s in SNIPPETS if s.path == REPO / "README.md"}
    assert "quickstart-python" in names


@pytest.mark.parametrize(
    "snippet", SNIPPETS, ids=[f"{s.path.relative_to(REPO)}:{s.name}" for s in SNIPPETS]
)
def test_snippet_runs(snippet) -> None:  # noqa: ANN001 (a doc_snippets.Snippet)
    out = doc_snippets.run_python(snippet, python=sys.executable)
    if snippet.expected is not None:
        assert out == snippet.expected
