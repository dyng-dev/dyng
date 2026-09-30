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
        "dyng/update.hpp",
        "dyng/graph/graph.hpp",
        "dyng/core/resources.hpp",
    ):
        assert f"[{header}] frozen\n" in text


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
