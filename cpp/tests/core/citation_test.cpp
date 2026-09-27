// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
#include <dyng/citation.hpp>
#include <dyng/core/error.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

TEST(Citation, LibraryOnly) {
  const std::string bib = dyng::citation();
  EXPECT_EQ(bib.rfind("@software{dyng2027,", 0), 0U) << bib;
  EXPECT_NE(bib.find("https://github.com/dyng-dev/dyng"), std::string::npos);
  EXPECT_EQ(bib.find("@inproceedings"), std::string::npos);
  EXPECT_EQ(dyng::citation_keys(), std::vector<std::string>{"dyng2027"});
}

TEST(Citation, AlgorithmsAddTheirPapers) {
  const std::string bib = dyng::citation("sssp");
  EXPECT_NE(bib.find("@software{dyng2027,"), std::string::npos);
  EXPECT_NE(bib.find("@inproceedings{dynamosp2025,"), std::string::npos);
  EXPECT_NE(bib.find("@article{dynamosptpds2025,"), std::string::npos);
  EXPECT_EQ(bib.find("escher2026"), std::string::npos);
  EXPECT_EQ(dyng::citation_keys("triad_count"),
            (std::vector<std::string>{"dyng2027", "escher2026", "escherplus2026"}));
}

TEST(Citation, EveryKnownNameResolves) {
  for (const char* name : {"dyng", "sssp", "mosp", "cycle_count", "triad_count", "hypergraph",
                           "label_propagation", "hyper_sssp"}) {
    const std::string bib = dyng::citation(name);
    for (const auto& key : dyng::citation_keys(name)) {
      EXPECT_NE(bib.find("{" + key + ","), std::string::npos) << name << " " << key;
    }
    // every entry is complete: as many closing lines as entries
    std::size_t entries = 0;
    for (std::size_t p = bib.find('@'); p != std::string::npos; p = bib.find("\n@", p + 1)) {
      ++entries;
    }
    EXPECT_EQ(entries, dyng::citation_keys(name).size()) << name;
  }
}

TEST(Citation, UnknownNameThrows) {
  EXPECT_THROW((void)dyng::citation("pagerank"), dyng::invalid_argument_error);
}
