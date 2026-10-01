// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file module.cpp
 * @brief The extension module dyng._core: the private native layer of the Python package.
 *
 * Users import `dyng`, whose typed layer (python/dyng (the .py files)) owns every public name; the names
 * here may change in any release (PLAN Section 5.4).
 */
#include "common.hpp"

#include <algorithm>

namespace dyng::python {

std::string suffixed(const char* base, std::initializer_list<const char*> codes, const char* sep) {
  std::string out(base);
  for (const char* code : codes) {
    out += sep;
    out += code;
  }
  return out;
}

call_gate& native_call_mutex() {
  static call_gate m;
  return m;
}

void lock_set::lock() {
  std::sort(entries_.begin(), entries_.end(), [](const entry& a, const entry& b) {
    return std::less<const std::shared_mutex*>()(a.mutex, b.mutex);
  });
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    // An object listed more than once is locked once, in the strongest of its modes.
    if (i > 0 && entries_[i].mutex == entries_[i - 1].mutex) {
      continue;
    }
    bool exclusive = entries_[i].exclusive;
    for (std::size_t j = i + 1; j < entries_.size() && entries_[j].mutex == entries_[i].mutex;
         ++j) {
      exclusive = exclusive || entries_[j].exclusive;
    }
    if (exclusive) {
      entries_[i].mutex->lock();
    } else {
      entries_[i].mutex->lock_shared();
    }
    taken_.push_back({entries_[i].mutex, exclusive});
  }
}

lock_set::~lock_set() {
  for (auto it = taken_.rbegin(); it != taken_.rend(); ++it) {
    if (it->exclusive) {
      it->mutex->unlock();
    } else {
      it->mutex->unlock_shared();
    }
  }
}

}  // namespace dyng::python

NB_MODULE(_core, m) {
  m.doc() =
      "dynG's native module (private: use the `dyng` package; these names may change in any "
      "release).";
  using namespace dyng::python;
  bind_errors(m);
  bind_resources(m);
  bind_profiler(m);
  bind_graph(m);
  bind_batch(m);
  bind_sssp(m);
  bind_cycle_count(m);
  bind_update(m);
  bind_io(m);
  bind_generators(m);
  bind_registry(m);
  bind_testing(m);
}
