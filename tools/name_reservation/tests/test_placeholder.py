# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of the name-reservation package."""

import pathlib
import tomllib

import dyng


def test_version_matches_pyproject():
    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        assert dyng.__version__ == tomllib.load(f)["project"]["version"]


def test_status_mentions_development():
    assert "under development" in dyng.status()
    assert "github.com/dyng-dev/dyng" in dyng.status()
