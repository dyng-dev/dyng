# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of ci/api_snapshot.py: the listing of a small Doxygen XML tree, and the check against a
baseline (a signature change fails with a diff, a documentation change does not)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "api_snapshot", Path(__file__).resolve().parent.parent / "api_snapshot.py"
)
api_snapshot = importlib.util.module_from_spec(_SPEC)
sys.modules["api_snapshot"] = api_snapshot
_SPEC.loader.exec_module(api_snapshot)

INDEX = """<?xml version='1.0' encoding='UTF-8'?>
<doxygenindex>
  <compound refid="group__core" kind="group"><name>core</name></compound>
  <compound refid="structdyng_1_1options" kind="struct"><name>dyng::options</name></compound>
  <compound refid="types_8hpp" kind="file"><name>types.hpp</name></compound>
  <compound refid="io_8hpp" kind="file"><name>io.hpp</name></compound>
</doxygenindex>
"""

GROUP = """<?xml version='1.0' encoding='UTF-8'?>
<doxygen><compounddef id="group__core" kind="group">
  <compoundname>core</compoundname>
  <sectiondef kind="enum">
    <memberdef kind="enum" id="e1" prot="public" static="no" strong="yes">
      <type>std::uint8_t</type><name>engine</name><qualifiedname>dyng::engine</qualifiedname>
      <enumvalue id="e1a" prot="public"><name>automatic</name></enumvalue>
      <enumvalue id="e1b" prot="public"><name>fused</name></enumvalue>
      <briefdescription><para>{doc}</para></briefdescription>
      <location file="dyng/core/types.hpp" line="10" column="1"/>
    </memberdef>
  </sectiondef>
  <sectiondef kind="func">
    <memberdef kind="function" id="f1" prot="public" static="no" nodiscard="yes" const="no"
               explicit="no" inline="no" noexcept="yes" virt="non-virtual">
      <templateparamlist><param><type>typename id_t</type></param></templateparamlist>
      <type>id_t</type><definition>id_t dyng::invalid_id</definition>
      <argsstring>{args}</argsstring><name>invalid_id</name>
      <qualifiedname>dyng::invalid_id</qualifiedname>
      <location file="dyng/core/types.hpp" line="20" column="1"/>
    </memberdef>
    <memberdef kind="function" id="f2" prot="public" static="no" const="no" explicit="no"
               inline="no" virt="non-virtual">
      <type>void</type><definition>void dyng::io::write</definition>
      <argsstring>(const std::string &amp;path)</argsstring><name>write</name>
      <qualifiedname>dyng::io::write</qualifiedname>
      <location file="dyng/io/io.hpp" line="5" column="1"/>
    </memberdef>
  </sectiondef>
</compounddef></doxygen>
"""

STRUCT = """<?xml version='1.0' encoding='UTF-8'?>
<doxygen><compounddef id="structdyng_1_1options" kind="struct" prot="public">
  <compoundname>dyng::options</compoundname>
  <basecompoundref prot="public" virt="non-virtual">dyng::update_stats</basecompoundref>
  <sectiondef kind="public-attrib">
    <memberdef kind="variable" id="v1" prot="public" static="no" mutable="no">
      <type>std::int64_t</type><definition>std::int64_t dyng::options::delta</definition>
      <argsstring></argsstring><name>delta</name><initializer>= {default}</initializer>
      <location file="dyng/core/types.hpp" line="31" column="3"/>
    </memberdef>
  </sectiondef>
  <sectiondef kind="private-attrib">
    <memberdef kind="variable" id="v2" prot="private" static="no" mutable="no">
      <type>int</type><definition>int dyng::options::hidden_</definition>
      <argsstring></argsstring><name>hidden_</name>
      <location file="dyng/core/types.hpp" line="32" column="3"/>
    </memberdef>
  </sectiondef>
  <location file="dyng/core/types.hpp" line="30" column="1"/>
</compounddef></doxygen>
"""


def _file(refid: str, path: str) -> str:
    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        f'<doxygen><compounddef id="{refid}" kind="file"><compoundname>{Path(path).name}'
        f'</compoundname><location file="{path}"/></compounddef></doxygen>\n'
    )


def _xml(tmp_path: Path, doc: str = "d", args: str = "() noexcept", default: str = "0") -> Path:
    xml = tmp_path / "xml"
    xml.mkdir(exist_ok=True)
    (xml / "index.xml").write_text(INDEX)
    (xml / "group__core.xml").write_text(GROUP.format(doc=doc, args=args))
    (xml / "structdyng_1_1options.xml").write_text(STRUCT.format(default=default))
    (xml / "types_8hpp.xml").write_text(_file("types_8hpp", "dyng/core/types.hpp"))
    (xml / "io_8hpp.xml").write_text(_file("io_8hpp", "dyng/io/io.hpp"))
    return xml


def test_the_listing_names_every_public_declaration(tmp_path: Path) -> None:
    text = api_snapshot.listing(_xml(tmp_path))
    assert "[dyng/core/types.hpp] frozen\n" in text
    assert "[dyng/io/io.hpp] tracked\n" in text
    assert "  enum class dyng::engine : std::uint8_t { automatic, fused }\n" in text
    assert "  template <typename id_t> [[nodiscard]] id_t dyng::invalid_id() noexcept\n" in text
    assert "  struct dyng::options : public dyng::update_stats\n" in text
    assert "      std::int64_t delta = 0\n" in text
    assert "hidden_" not in text  # private members are not API
    assert "void dyng::io::write(const std::string &path)" in text
    # declaration order within a header: the enum (line 10), the function (20), the struct (30)
    assert text.index("dyng::engine") < text.index("invalid_id") < text.index("dyng::options")


def test_a_documentation_edit_keeps_the_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "public_api.txt"
    xml = _xml(tmp_path)
    assert api_snapshot.main(["--xml", str(xml), "--baseline", str(baseline), "--update"]) == 0
    _xml(tmp_path, doc="a longer description of the engines")
    assert api_snapshot.main(["--xml", str(xml), "--baseline", str(baseline)]) == 0


def test_a_signature_or_default_change_fails_with_a_diff(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "public_api.txt"
    xml = _xml(tmp_path)
    assert api_snapshot.main(["--xml", str(xml), "--baseline", str(baseline), "--update"]) == 0
    _xml(tmp_path, args="(int base) noexcept")
    assert api_snapshot.main(["--xml", str(xml), "--baseline", str(baseline)]) == 1
    captured = capsys.readouterr()
    assert "-  template <typename id_t> [[nodiscard]] id_t dyng::invalid_id() noexcept" in (
        captured.out
    )
    assert "+  template <typename id_t> [[nodiscard]] id_t dyng::invalid_id(int base) noexcept" in (
        captured.out
    )
    assert "ci/docs.sh --update-api" in captured.err  # a command that can succeed
    _xml(tmp_path, default="1")  # a changed default value is a change of the API (PLAN 5.9)
    assert api_snapshot.main(["--xml", str(xml), "--baseline", str(baseline)]) == 1


def test_missing_xml_is_an_error(tmp_path: Path) -> None:
    assert api_snapshot.main(["--xml", str(tmp_path / "none")]) == 2


def test_the_committed_baseline_is_well_formed() -> None:
    text = api_snapshot.BASELINE.read_text()
    assert text.startswith(api_snapshot.HEADER)
    for header in (
        "dyng/sssp.hpp",
        "dyng/cycle_count.hpp",
        "dyng/mosp.hpp",  # stable from 0.2.0 (ADR 0035)
        "dyng/update.hpp",
        "dyng/graph/graph.hpp",
        "dyng/core/resources.hpp",
    ):
        assert f"[{header}] frozen\n" in text
    # The tutorial algorithms are listed and checked, but not frozen (ADRs 0033, 0035).
    for header in ("dyng/dynamic_bfs.hpp", "dyng/triangle_delta.hpp"):
        assert f"[{header}] tracked\n" in text


def test_an_algorithm_header_is_frozen_exactly_when_its_manifest_says_stable() -> None:
    manifests = [
        p
        for p in sorted(api_snapshot.ALGORITHMS_DIR.glob("*/manifest.toml"))
        if not p.parent.name.startswith("_")
    ]
    assert manifests
    seen = {}
    for manifest in manifests:
        data = api_snapshot.tomllib.loads(manifest.read_text(encoding="utf-8"))
        header = f"dyng/{data['name']}.hpp"
        assert (api_snapshot.INCLUDE_DIR / header).is_file(), header
        expected = "frozen" if data["maturity"] == "stable" else "tracked"
        assert api_snapshot.status(header) == expected, (header, data["maturity"])
        seen[data["name"]] = api_snapshot.status(header)
    assert seen["sssp"] == seen["cycle_count"] == seen["mosp"] == "frozen"
    assert seen["dynamic_bfs"] == seen["triangle_delta"] == "tracked"


def test_the_maturity_decides_the_freeze(tmp_path: Path) -> None:
    for name, maturity in (("alpha", "stable"), ("beta", "experimental"), ("gamma", "tutorial")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.toml").write_text(
            f'name = "{name}"\nmaturity = "{maturity}"\n'
        )
    (tmp_path / "_template").mkdir()
    (tmp_path / "_template" / "manifest.toml").write_text('name = "t"\nmaturity = "stable"\n')
    assert api_snapshot.stable_algorithm_headers(tmp_path) == ["dyng/alpha.hpp"]


MOSP_INDEX = """<?xml version='1.0' encoding='UTF-8'?>
<doxygenindex>
  <compound refid="group__mosp" kind="group"><name>mosp</name></compound>
  <compound refid="mosp_8hpp" kind="file"><name>mosp.hpp</name></compound>
</doxygenindex>
"""

MOSP_GROUP = """<?xml version='1.0' encoding='UTF-8'?>
<doxygen><compounddef id="group__mosp" kind="group">
  <compoundname>mosp</compoundname>
  <sectiondef kind="func">
    <memberdef kind="function" id="m1" prot="public" static="no" nodiscard="yes" const="no"
               explicit="no" inline="yes" virt="non-virtual">
      <templateparamlist><param><type>typename vertex_t</type></param></templateparamlist>
      <type>result&lt; vertex_t &gt;</type><definition>result&lt; vertex_t &gt; dyng::mosp::compute
      </definition>
      <argsstring>{args}</argsstring><name>compute</name>
      <qualifiedname>dyng::mosp::compute</qualifiedname>
      <location file="dyng/mosp.hpp" line="40" column="1"/>
    </memberdef>
  </sectiondef>
</compounddef></doxygen>
"""


def test_a_mosp_signature_change_fails_in_the_frozen_mosp_section(tmp_path: Path, capsys) -> None:
    """The freeze of mosp (ADR 0035): a deliberate change of a mosp signature fails the check."""
    xml = tmp_path / "xml"
    xml.mkdir()
    (xml / "index.xml").write_text(MOSP_INDEX)
    (xml / "mosp_8hpp.xml").write_text(_file("mosp_8hpp", "dyng/mosp.hpp"))
    before = "(const resources &amp;res, vertex_t source, const options &amp;opt={})"
    after = "(const resources &amp;res, vertex_t source, int k, const options &amp;opt={})"
    (xml / "group__mosp.xml").write_text(MOSP_GROUP.format(args=before))
    baseline = tmp_path / "public_api.txt"
    assert api_snapshot.main(["--xml", str(xml), "--baseline", str(baseline), "--update"]) == 0
    assert "[dyng/mosp.hpp] frozen\n" in baseline.read_text()
    (xml / "group__mosp.xml").write_text(MOSP_GROUP.format(args=after))
    assert api_snapshot.main(["--xml", str(xml), "--baseline", str(baseline)]) == 1
    out = capsys.readouterr().out
    assert (
        "-  template <typename vertex_t> [[nodiscard]] result< vertex_t > dyng::mosp::compute("
        in out
    )
    assert "vertex_t source, int k, const options &opt={})" in out


def test_the_detail_contract_is_read_from_the_source() -> None:
    source = """
namespace dyng::detail {
/// A comment { with a brace.
template <typename v_t>
inline constexpr bool x_supported_v = std::is_same_v<v_t, int>;
template <>
struct update_traits<x::result> {
  using stats_type = x::stats;  ///< the stats
  template <typename c_t> static int make_participant(int r) { return r; }
};
struct unrelated { using type = int; };
}  // namespace dyng::detail
"""
    lines = api_snapshot.detail_contract(source)
    assert lines == [
        "template <typename v_t> inline constexpr bool x_supported_v = std::is_same_v<v_t, int>;",
        "template <> struct update_traits<x::result> { using stats_type = x::stats; }",
    ]
    changed = source.replace("x::stats;", "x::other_stats;")
    assert api_snapshot.detail_contract(changed) != lines


def test_the_umbrella_includes_are_listed() -> None:
    source = "#pragma once\n#include <dyng/a.hpp>\n#include <dyng/b.hpp>\n"
    assert api_snapshot.umbrella_includes(source) == [
        "#include <dyng/a.hpp>",
        "#include <dyng/b.hpp>",
    ]


# --- The frozen API against the base branch (--against; R020) ------------------------------------

BASELINE_TEXT = api_snapshot.BASELINE.read_text()


def _edit(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def test_an_unchanged_baseline_has_no_frozen_change() -> None:
    assert api_snapshot.frozen_changes(BASELINE_TEXT, BASELINE_TEXT) == ([], [])


def test_a_changed_mosp_declaration_is_a_breaking_change() -> None:
    # A deliberate change of a mosp signature (the frozen header of 0.2, ADR 0035): the default
    # of compute_path_costs and the limit max_objectives.
    changed = _edit(
        BASELINE_TEXT, "bool compute_path_costs = true", "bool compute_path_costs = false"
    )
    changed = _edit(changed, "max_objectives = 64", "max_objectives = 32")
    breaking, _ = api_snapshot.frozen_changes(BASELINE_TEXT, changed)
    assert breaking and all(b.startswith("[dyng/mosp.hpp]") for b in breaking)
    assert any("compute_path_costs = false" in b for b in breaking)


def test_appended_lines_and_tracked_sections_are_not_breaking() -> None:
    lines = BASELINE_TEXT.splitlines(keepends=True)
    at = next(i for i, line in enumerate(lines) if line.startswith("[dyng/mosp.hpp]"))
    added = "".join(
        [*lines[: at + 1], "  inline constexpr int dyng::mosp::new_limit = 1\n", *lines[at + 1 :]]
    )
    breaking, appended = api_snapshot.frozen_changes(BASELINE_TEXT, added)
    assert breaking == [] and appended == [
        "[dyng/mosp.hpp] + inline constexpr int dyng::mosp::new_limit = 1"
    ]
    # dynamic_bfs is a tutorial algorithm: tracked, not frozen (ADR 0035).
    at = next(i for i, line in enumerate(lines) if line.startswith("[dyng/dynamic_bfs.hpp]"))
    tracked = "".join([*lines[: at + 1], *lines[at + 2 :]])
    assert api_snapshot.frozen_changes(BASELINE_TEXT, tracked) == ([], [])


def test_unfreezing_or_removing_a_frozen_header_is_breaking() -> None:
    unfrozen = _edit(BASELINE_TEXT, "[dyng/mosp.hpp] frozen", "[dyng/mosp.hpp] tracked")
    assert api_snapshot.frozen_changes(BASELINE_TEXT, unfrozen)[0] == [
        "[dyng/mosp.hpp] no longer frozen"
    ]
    secs = api_snapshot.sections(BASELINE_TEXT)
    assert secs["dyng/mosp.hpp"][0] == "frozen" and secs["dyng/sssp.hpp"][0] == "frozen"
    removed = "\n".join(
        line for line in BASELINE_TEXT.splitlines() if not line.startswith("[dyng/sssp.hpp]")
    )
    breaking, _ = api_snapshot.frozen_changes(BASELINE_TEXT, removed)
    assert breaking  # its lines now belong to the section above, so they are no longer sssp's


def test_against_fails_without_the_label(tmp_path, monkeypatch, capsys) -> None:
    changed = tmp_path / "public_api.txt"
    changed.write_text(_edit(BASELINE_TEXT, "max_objectives = 64", "max_objectives = 32"))
    monkeypatch.delenv("DYNG_API_CHANGE", raising=False)
    assert api_snapshot.check_against("HEAD", changed) == 1
    assert "breaking change of the frozen API: [dyng/mosp.hpp]" in capsys.readouterr().out
    # With the label, a CHANGELOG entry is still required (CHANGELOG.md equals HEAD's here,
    # unless the working tree changed it).
    monkeypatch.setenv("DYNG_API_CHANGE", "1")
    import subprocess

    clean = (
        subprocess.run(
            ["git", "-C", str(api_snapshot.REPO), "diff", "--quiet", "HEAD", "--", "CHANGELOG.md"]
        ).returncode
        == 0
    )
    assert api_snapshot.check_against("HEAD", changed) == (1 if clean else 0)
    head = subprocess.run(
        [
            "git",
            "-C",
            str(api_snapshot.REPO),
            "show",
            "HEAD:cpp/tests/api/api_snapshot/public_api.txt",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    unchanged = tmp_path / "same.txt"
    unchanged.write_text(head)
    monkeypatch.delenv("DYNG_API_CHANGE")
    assert api_snapshot.check_against("HEAD", unchanged) == 0
