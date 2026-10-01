// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief dyng::algorithms() (the registry of the built algorithms) and dyng::citation().
 */
#include "common.hpp"

#include <dyng/citation.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/types.hpp>

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

namespace dyng::python {

void bind_registry(nb::module_& m) {
  m.def(
      "algorithms",
      [] {
        nb::list out;
        for (const algorithm_info& a : algorithms()) {
          nb::dict d;
          d["name"] = std::string(a.name);
          d["title"] = std::string(a.title);
          d["family"] = std::string(to_string(a.family));
          d["container"] = std::string(to_string(a.container));
          d["maturity"] = std::string(to_string(a.maturity));
          d["determinism"] = std::string(to_string(a.determinism_level));
          d["oracle"] = std::string(to_string(a.oracle));
          nb::list backends;
          for (backend b : a.backends) {
            backends.append(std::string(to_string(b)));
          }
          d["backends"] = nb::tuple(backends);
          nb::list cite;
          for (std::string_view key : a.cite) {
            cite.append(std::string(key));
          }
          d["cite"] = nb::tuple(cite);
          out.append(d);
        }
        return out;
      },
      "The registry: one dict per built algorithm.");
  m.def(
      "citation", [](const std::string& what) { return citation(what); }, nb::arg("what"),
      "BibTeX for dynG, an algorithm name or a references.bib key.");
  m.def(
      "citation_keys", [](const std::string& what) { return citation_keys(what); }, nb::arg("what"),
      "The references.bib keys citation(what) returns.");
}

}  // namespace dyng::python
