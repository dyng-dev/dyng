// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file workspace.cpp
 * @brief The workspace pool (ADR 0015).
 */
#include "framework/workspace.hpp"

#include <algorithm>
#include <utility>

namespace dyng::detail {

std::unique_ptr<pooled_workspace> workspace_pool::take(std::type_index key) {
  const std::lock_guard<std::mutex> lock(mutex_);
  ++leases_;
  ++leased_;
  // The most recently returned workspace of the type: its pages are the warmest.
  for (auto it = idle_.rbegin(); it != idle_.rend(); ++it) {
    if (it->key == key) {
      std::unique_ptr<pooled_workspace> taken = std::move(it->workspace);
      idle_.erase(std::next(it).base());
      return taken;
    }
  }
  return nullptr;
}

void workspace_pool::give_back(std::type_index key, std::unique_ptr<pooled_workspace> workspace,
                               bool discard) noexcept {
  std::unique_ptr<pooled_workspace> dropped;  // destroyed after the lock is released
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    --leased_;
    if (workspace == nullptr) {
      return;
    }
    if (discard) {
      ++discarded_;
      dropped = std::move(workspace);
    } else {
      try {
        idle_.push_back(entry{key, std::move(workspace)});
      } catch (...) {
        // Out of memory for the list entry: drop the workspace instead of leaking it; the next
        // lease creates a new one.
        ++discarded_;
        dropped = std::move(workspace);
      }
    }
  }
}

void workspace_pool::note_created() noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  ++created_;
}

void workspace_pool::release_idle() noexcept {
  std::vector<entry> dropped;  // destroyed after the lock is released
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    dropped.swap(idle_);
  }
}

workspace_pool_statistics workspace_pool::statistics() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  workspace_pool_statistics s;
  s.idle = idle_.size();
  s.leased = leased_;
  s.created = created_;
  s.discarded = discarded_;
  s.leases = leases_;
  for (const entry& e : idle_) {
    s.idle_bytes += e.workspace->bytes();
  }
  return s;
}

}  // namespace dyng::detail
