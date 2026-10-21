// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_bins.cpp
 * @brief Triad bins as a JSON document (dyng::io::write_triad_bins_json(),
 *        dyng::io::read_triad_bins_json()): a strict reader of RFC 8259 JSON restricted to what
 *        the format needs (one object; strings, integers, arrays, nested values skipped when a
 *        key is unknown), with io_error at path:line:column.
 */
#include "util/allocation.hpp"
#include "util/parser.hpp"
#include "util/text_writer.hpp"

#include <dyng/core/error.hpp>
#include <dyng/io/result_io.hpp>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng::io {

namespace {

constexpr std::string_view format_name = "dyng.triad_bins";
constexpr std::int64_t format_version = 1;
constexpr int max_depth = 64;

/// The patterns of dyng::triad_count by their names (triad_count::to_string(); the io module does
/// not depend on the algorithm, and dyng_io_tests checks the two lists equal) and bin counts.
constexpr std::pair<std::string_view, int> kinds[] = {
    {"h_motif", 30}, {"stathyper_type1", 1}, {"stathyper_type2", 1}, {"stathyper_type3", 1}};

/// The bin count of a known name, or 0.
int bins_of(std::string_view kind) noexcept {
  for (const auto& [name, bins] : kinds) {
    if (name == kind) {
      return bins;
    }
  }
  return 0;
}

/// The names, for messages: "h_motif", "stathyper_type1", ...
std::string known_kinds() {
  std::string out;
  for (const auto& kind : kinds) {
    if (!out.empty()) {
      out += ", ";
    }
    out += '"';
    out += kind.first;
    out += '"';
  }
  return out;
}

/// A parsed JSON value: its kind, its text (a string's decoded characters or a number's lexeme),
/// its elements (array) or members (object), and where it starts.
struct json_value {
  enum class kind { null, boolean, number, string, array, object };
  kind type = kind::null;
  std::string text;
  std::vector<json_value> items;
  std::vector<std::pair<std::string, json_value>> members;
  std::size_t offset = 0;
};

/// A recursive-descent parser of one JSON document (RFC 8259) over a whole file's text.
class json_parser {
 public:
  json_parser(const std::string& path, std::string_view text) : path_(path), s_(text) {}

  /// The document: one value and nothing after it but whitespace.
  json_value document() {
    skip_space();
    json_value v = value(0);
    skip_space();
    if (pos_ != s_.size()) {
      fail(pos_, "unexpected text after the JSON document");
    }
    return v;
  }

  /// Throw io_error at a byte offset of the text (its 1-based line and column).
  [[noreturn]] void fail(std::size_t offset, const std::string& message) const {
    std::int64_t line = 1;
    std::size_t line_start = 0;
    for (std::size_t i = 0; i < offset && i < s_.size(); ++i) {
      if (s_[i] == '\n') {
        ++line;
        line_start = i + 1;
      }
    }
    detail::throw_io_error(path_, line, static_cast<std::int64_t>(offset - line_start) + 1,
                           message);
  }

 private:
  void skip_space() noexcept {
    while (pos_ < s_.size() &&
           (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) {
      ++pos_;
    }
  }

  [[nodiscard]] bool at(char c) const noexcept {
    return pos_ < s_.size() && s_[pos_] == c;
  }

  void expect(char c, const char* what) {
    if (!at(c)) {
      fail(pos_, std::string("expected ") + what);
    }
    ++pos_;
  }

  json_value value(int depth) {
    if (depth >= max_depth) {
      fail(pos_, "JSON nested deeper than " + std::to_string(max_depth) + " levels");
    }
    if (pos_ >= s_.size()) {
      fail(pos_, "unexpected end of the file (expected a JSON value)");
    }
    json_value v;
    v.offset = pos_;
    const char c = s_[pos_];
    if (c == '{') {
      v.type = json_value::kind::object;
      ++pos_;
      skip_space();
      if (at('}')) {
        ++pos_;
        return v;
      }
      for (;;) {
        skip_space();
        if (!at('"')) {
          fail(pos_, "expected a string key");
        }
        const std::size_t key_at = pos_;
        std::string key = string_text();
        for (const auto& m : v.members) {
          if (m.first == key) {
            fail(key_at, "repeated key \"" + key + "\"");
          }
        }
        skip_space();
        expect(':', "':' after a key");
        skip_space();
        json_value member = value(depth + 1);
        v.members.emplace_back(std::move(key), std::move(member));
        skip_space();
        if (at(',')) {
          ++pos_;
          continue;
        }
        expect('}', "',' or '}' in an object");
        return v;
      }
    }
    if (c == '[') {
      v.type = json_value::kind::array;
      ++pos_;
      skip_space();
      if (at(']')) {
        ++pos_;
        return v;
      }
      for (;;) {
        skip_space();
        v.items.push_back(value(depth + 1));
        skip_space();
        if (at(',')) {
          ++pos_;
          continue;
        }
        expect(']', "',' or ']' in an array");
        return v;
      }
    }
    if (c == '"') {
      v.type = json_value::kind::string;
      v.text = string_text();
      return v;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
      v.type = json_value::kind::number;
      v.text = number_text();
      return v;
    }
    for (const std::string_view word : {std::string_view("true"), std::string_view("false"),
                                        std::string_view("null")}) {
      if (s_.substr(pos_, word.size()) == word) {
        pos_ += word.size();
        v.type = word == "null" ? json_value::kind::null : json_value::kind::boolean;
        return v;
      }
    }
    fail(pos_, "expected a JSON value");
  }

  /// The lexeme of a number (RFC 8259's grammar: -? int frac? exp?).
  std::string number_text() {
    const std::size_t begin = pos_;
    if (at('-')) {
      ++pos_;
    }
    if (at('0')) {
      ++pos_;
    } else if (pos_ < s_.size() && s_[pos_] >= '1' && s_[pos_] <= '9') {
      digits();
    } else {
      fail(pos_, "malformed number");
    }
    if (at('.')) {
      ++pos_;
      if (!digits()) {
        fail(pos_, "malformed number (no digit after '.')");
      }
    }
    if (at('e') || at('E')) {
      ++pos_;
      if (at('+') || at('-')) {
        ++pos_;
      }
      if (!digits()) {
        fail(pos_, "malformed number (no digit in the exponent)");
      }
    }
    return std::string(s_.substr(begin, pos_ - begin));
  }

  bool digits() noexcept {
    const std::size_t begin = pos_;
    while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') {
      ++pos_;
    }
    return pos_ > begin;
  }

  unsigned hex4() {
    if (s_.size() - pos_ < 4) {
      fail(pos_, "malformed \\u escape");
    }
    unsigned value = 0;
    for (int i = 0; i < 4; ++i) {
      const char h = s_[pos_++];
      value <<= 4U;
      if (h >= '0' && h <= '9') {
        value |= static_cast<unsigned>(h - '0');
      } else if (h >= 'a' && h <= 'f') {
        value |= static_cast<unsigned>(h - 'a' + 10);
      } else if (h >= 'A' && h <= 'F') {
        value |= static_cast<unsigned>(h - 'A' + 10);
      } else {
        fail(pos_ - 1, "malformed \\u escape");
      }
    }
    return value;
  }

  static void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80U) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800U) {
      out.push_back(static_cast<char>(0xC0U | (cp >> 6U)));
      out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else if (cp < 0x10000U) {
      out.push_back(static_cast<char>(0xE0U | (cp >> 12U)));
      out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
      out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else {
      out.push_back(static_cast<char>(0xF0U | (cp >> 18U)));
      out.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
      out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
      out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    }
  }

  /// A string's decoded characters (pos_ at its opening quote).
  std::string string_text() {
    ++pos_;  // the opening quote
    std::string out;
    for (;;) {
      if (pos_ >= s_.size()) {
        fail(pos_, "unterminated string");
      }
      const char c = s_[pos_];
      if (c == '"') {
        ++pos_;
        return out;
      }
      if (static_cast<unsigned char>(c) < 0x20U) {
        fail(pos_, "a control character in a string");
      }
      if (c != '\\') {
        out.push_back(c);
        ++pos_;
        continue;
      }
      ++pos_;
      if (pos_ >= s_.size()) {
        fail(pos_, "unterminated string");
      }
      const char e = s_[pos_++];
      switch (e) {
        case '"':
        case '\\':
        case '/':
          out.push_back(e);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          unsigned cp = hex4();
          if (cp >= 0xD800U && cp <= 0xDBFFU) {
            if (s_.substr(pos_, 2) != "\\u") {
              fail(pos_, "an unpaired surrogate in a \\u escape");
            }
            pos_ += 2;
            const unsigned low = hex4();
            if (low < 0xDC00U || low > 0xDFFFU) {
              fail(pos_ - 4, "an unpaired surrogate in a \\u escape");
            }
            cp = 0x10000U + ((cp - 0xD800U) << 10U) + (low - 0xDC00U);
          } else if (cp >= 0xDC00U && cp <= 0xDFFFU) {
            fail(pos_ - 4, "an unpaired surrogate in a \\u escape");
          }
          append_utf8(out, cp);
          break;
        }
        default:
          fail(pos_ - 1, std::string("unknown escape \\") + e);
      }
    }
  }

  const std::string& path_;
  std::string_view s_;
  std::size_t pos_ = 0;
};

/// An integer field: a number without fraction or exponent in [minimum, maximum].
template <typename value_t>
value_t integer_of(const json_parser& p, const json_value& v, const std::string& what) {
  if (v.type != json_value::kind::number) {
    p.fail(v.offset, what + " must be an integer");
  }
  if (v.text.find_first_of(".eE") != std::string::npos) {
    p.fail(v.offset, what + " must be an integer (no fraction or exponent): " + v.text);
  }
  value_t out{};
  const char* first = v.text.data();
  const char* last = first + v.text.size();
  if constexpr (std::is_unsigned_v<value_t>) {
    if (v.text[0] == '-') {
      p.fail(v.offset, what + " must not be negative: " + v.text);
    }
  }
  const auto r = std::from_chars(first, last, out);
  if (r.ec != std::errc() || r.ptr != last) {
    p.fail(v.offset, what + " is out of range: " + v.text);
  }
  return out;
}

const json_value& array_of(const json_parser& p, const json_value& v, const std::string& what) {
  if (v.type != json_value::kind::array) {
    p.fail(v.offset, what + " must be an array");
  }
  return v;
}

/// The text of the document (the writers' shared part).
std::string triad_bins_text(const triad_bins& bins) {
  const int n = bins_of(bins.kind);
  DYNG_EXPECTS(n > 0, "io::write_triad_bins_json: unknown kind \"", bins.kind, "\" (one of ",
               known_kinds(), ")");
  DYNG_EXPECTS(bins.counts.size() == static_cast<std::size_t>(n),
               "io::write_triad_bins_json: ", bins.counts.size(), " counts for kind ", bins.kind,
               ", which has ", n, " bins");
  for (std::size_t b = 0; b < bins.deltas.size(); ++b) {
    DYNG_EXPECTS(bins.deltas[b].size() == static_cast<std::size_t>(n),
                 "io::write_triad_bins_json: delta row ", b, " has ", bins.deltas[b].size(),
                 " values; kind ", bins.kind, " has ", n, " bins");
  }
  std::string out = "{\n  \"format\": \"";
  out += format_name;
  out += "\",\n  \"version\": ";
  out += std::to_string(format_version);
  out += ",\n  \"kind\": \"";
  out += bins.kind;  // a known name: no character needs an escape
  out += "\",\n  \"num_bins\": ";
  out += std::to_string(n);
  out += ",\n  \"graph_version\": ";
  out += std::to_string(bins.graph_version);
  out += ",\n  \"counts\": [";
  for (std::size_t i = 0; i < bins.counts.size(); ++i) {
    out += i == 0 ? "" : ", ";
    out += std::to_string(bins.counts[i]);
  }
  out += ']';
  if (!bins.deltas.empty()) {
    out += ",\n  \"deltas\": [";
    for (std::size_t b = 0; b < bins.deltas.size(); ++b) {
      out += b == 0 ? "\n    [" : ",\n    [";
      for (std::size_t i = 0; i < bins.deltas[b].size(); ++i) {
        out += i == 0 ? "" : ", ";
        out += std::to_string(bins.deltas[b][i]);
      }
      out += ']';
    }
    out += "\n  ]";
  }
  out += "\n}\n";
  return out;
}

}  // namespace

int triad_bins_size(const std::string& kind) {
  const int n = bins_of(kind);
  DYNG_EXPECTS(n > 0, "io::triad_bins_size: unknown kind \"", kind, "\" (one of ", known_kinds(),
               ")");
  return n;
}

void write_triad_bins_json(std::ostream& out, const triad_bins& bins) try {
  const std::string text = triad_bins_text(bins);
  out << text;
  if (!out) {
    throw io_error("dyng: io::write_triad_bins_json: writing to the stream failed");
  }
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_triad_bins_json (", bins.counts.size(), " bins, ",
                                  bins.deltas.size(), " delta rows)")

void write_triad_bins_json(const std::string& path, const triad_bins& bins) try {
  const std::string text = triad_bins_text(bins);
  detail::text_writer out(path);
  out.put(text);
  out.close();
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::write_triad_bins_json (", path, ")")

triad_bins read_triad_bins_json(const std::string& path) try {
  const std::string text = detail::read_file(path);
  json_parser p(path, text);
  const json_value doc = p.document();
  if (doc.type != json_value::kind::object) {
    p.fail(doc.offset, "a triad-bins file is one JSON object");
  }
  const json_value* format = nullptr;
  const json_value* version = nullptr;
  const json_value* kind = nullptr;
  const json_value* num_bins = nullptr;
  const json_value* graph_version = nullptr;
  const json_value* counts = nullptr;
  const json_value* deltas = nullptr;
  for (const auto& [key, value] : doc.members) {
    const json_value** slot = key == "format"          ? &format
                              : key == "version"       ? &version
                              : key == "kind"          ? &kind
                              : key == "num_bins"      ? &num_bins
                              : key == "graph_version" ? &graph_version
                              : key == "counts"        ? &counts
                              : key == "deltas"        ? &deltas
                                                       : nullptr;
    if (slot != nullptr) {
      *slot = &value;  // other keys are ignored (documented)
    }
  }
  const auto require = [&](const json_value* v, const char* key) -> const json_value& {
    if (v == nullptr) {
      p.fail(doc.offset, std::string("missing key \"") + key + "\"");
    }
    return *v;
  };
  const json_value& f = require(format, "format");
  if (f.type != json_value::kind::string || f.text != format_name) {
    p.fail(f.offset, "\"format\" must be \"" + std::string(format_name) + "\"");
  }
  const json_value& ver = require(version, "version");
  const auto ver_value = integer_of<std::int64_t>(p, ver, "\"version\"");
  if (ver_value != format_version) {
    p.fail(ver.offset, "unsupported triad-bins format version " + std::to_string(ver_value) +
                           " (this library reads version " + std::to_string(format_version) +
                           ")");
  }
  const json_value& k = require(kind, "kind");
  if (k.type != json_value::kind::string || bins_of(k.text) == 0) {
    p.fail(k.offset, "\"kind\" must be one of " + known_kinds());
  }
  const int n = bins_of(k.text);
  const json_value& nb = require(num_bins, "num_bins");
  if (integer_of<std::int64_t>(p, nb, "\"num_bins\"") != n) {
    p.fail(nb.offset, "\"num_bins\" must be " + std::to_string(n) + " for kind " + k.text);
  }
  triad_bins out;
  out.kind = k.text;
  if (graph_version != nullptr) {
    out.graph_version = integer_of<std::uint64_t>(p, *graph_version, "\"graph_version\"");
  }
  const json_value& c = array_of(p, require(counts, "counts"), "\"counts\"");
  if (c.items.size() != static_cast<std::size_t>(n)) {
    p.fail(c.offset, "\"counts\" has " + std::to_string(c.items.size()) + " values; kind " +
                         k.text + " has " + std::to_string(n) + " bins");
  }
  out.counts.reserve(c.items.size());
  for (const json_value& x : c.items) {
    out.counts.push_back(integer_of<std::uint64_t>(p, x, "a count"));
  }
  if (deltas != nullptr) {
    const json_value& d = array_of(p, *deltas, "\"deltas\"");
    out.deltas.reserve(d.items.size());
    for (const json_value& row : d.items) {
      const json_value& r = array_of(p, row, "a row of \"deltas\"");
      if (r.items.size() != static_cast<std::size_t>(n)) {
        p.fail(r.offset, "a row of \"deltas\" has " + std::to_string(r.items.size()) +
                             " values; kind " + k.text + " has " + std::to_string(n) + " bins");
      }
      std::vector<std::int64_t> values;
      values.reserve(r.items.size());
      for (const json_value& x : r.items) {
        values.push_back(integer_of<std::int64_t>(p, x, "a delta"));
      }
      out.deltas.push_back(std::move(values));
    }
  }
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("io::read_triad_bins_json (", path, ")")

}  // namespace dyng::io
