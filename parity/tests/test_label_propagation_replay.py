# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/label_propagation_replay.py (M10a step 2, ADRs 0067 and 0068) that need no
GPU, build, original or golden: the comparison rule of the oracle replay (1e-9 absolute or 1e-6
relative), the stream digest, the corpora, the --check argv, the dispatch of compare.py
label_propagation, the committed subset's names against the C++ model's fields, and the label
oracle's format strings of the recorder."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))

import label_propagation_goldens as g  # noqa: E402
import label_propagation_replay as r  # noqa: E402


def test_oracle_tolerance_is_absolute_or_relative():
    assert r.close(1.0e-3, 1.0e-3 + 9e-10)
    assert not r.close(1.0e-3, 1.0e-3 + 2e-9)
    assert r.close(10.0, 10.0 + 9e-6)  # relative 1e-6 on a large value
    assert not r.close(10.0, 10.0 + 2e-5)
    assert (r.ABS_TOL, r.REL_TOL) == (1e-9, 1e-6)


def test_stream_digest_ignores_key_order_but_not_values():
    a = r.stream_digest({"row_ptr": "aa", "b0.inserted": "bb"})
    assert a == r.stream_digest({"b0.inserted": "bb", "row_ptr": "aa"})
    assert a != r.stream_digest({"row_ptr": "aa", "b0.inserted": "bc"})


def test_corpora_c_and_l():
    assert [c.name for c in r.C_CASES] == [
        "5k5_d1e-6_s42",
        "5k5_d1e-6_s42_inplace",
        "50k_d1e-6_s42",
        "50k_d1e-6_s42_inplace",
    ]
    assert [c.name for c in r.L_CASES] == [
        "500k_d1e-4_s42",
        "500k_d1e-4_s42_inplace",
        "5m_d1e-4_s42",
        "5m_d1e-4_s42_inplace",
    ]
    with pytest.raises(g.GoldenError):
        r.select(r.C_CASES, ["nope"])
    assert r.select(r.C_CASES, None) == r.C_CASES


def test_check_argv_validates_at_5k5_only():
    a = r.check_case_args(r.C_CASES[0])
    assert "--validate" in a and "--noValidate" not in a and "--output" not in a
    assert a[a.index("--delta") + 1] == "0.000001"
    b = r.check_case_args(r.L_CASES[2])
    assert "--noValidate" in b and b[b.index("--totalNodes") + 1] == "5000000"


def test_compare_main_dispatches_the_replays(capsys):
    assert set(g.REPLAY_COMMANDS) == {
        "generator",
        "model",
        "check-runs",
        "oracle",
        "margin",
        "subset",
    }
    assert g.compare_main([]) == 2
    assert "--verify" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        g.compare_main(["model"])  # --exe is required: argparse exits


def test_model_fields_match_the_cpp_replay():
    # The names the subset writes are the fields lp_replay.cpp's model_fields() predicts.
    src = (REPO / "cpp/tests/parity/label_propagation/lp_replay.cpp").read_text()
    cpp = set(re.findall(r'f\["([a-z_]+\.[a-z0-9_]+)"\]', src))
    py = {f"{tag}.{f}" for tag, names in r.MODEL_FIELDS.items() for f in names}
    assert cpp == py | {"append_edges.row_ptr"}


def test_committed_subset_names():
    model = (r.SUBSET_DIR / "model_5k5_d1e-4_s42.sha256").read_text().splitlines()
    names = [ln.split()[1] for ln in model if ln and not ln.startswith("#")]
    assert len(names) == len(set(names))
    assert "b0.append_edges.row_ptr" in names
    assert all(re.fullmatch(r"b[0-4]\.[a-z_]+\.[a-z0-9_]+", n) for n in names)
    gen = (r.SUBSET_DIR / "generator_50k_d1e-4_s42.sha256").read_text()
    for f in g.CSR_FILES:
        assert re.search(rf"^[0-9a-f]{{64}}  {re.escape(f)}$", gen, re.M)
    assert re.search(r"^[0-9a-f]{64}  b9\.edge_weight$", gen, re.M)


def test_recorder_formats_are_the_checkers():
    # The two formats of DynLPChecker::checkLabels (src/checker.cu at a276a3a), as the recorder
    # matches them; the checker's text is quoted by the extract when it is present.
    src = (REPO / "parity/exporters/dynlp/recorder_dynlp.cpp").read_text()
    joined = re.sub(r'"\s*\n\s*"', "", src)
    checked = re.search(r'k_checked =\s*"([^"]+)"', joined).group(1)
    skipped = re.search(r'k_skipped =\s*"([^"]+)"', joined).group(1)
    assert checked.replace("%%", "").count("%") == 9 and "mean |F-F*| %.2e" in checked
    assert skipped.endswith("labels not compared")
    extract = g.scratch() / "plan-m10" / "lp" / "src" / "checker.cu"
    if not extract.is_file():
        pytest.skip("the read-only extract of LabelPropagation-CUDA is not on this machine")
    text = re.sub(r'"\s*\n\s*"', "", extract.read_text())
    assert f'"{checked}"' in text and f'"{skipped}"' in text
