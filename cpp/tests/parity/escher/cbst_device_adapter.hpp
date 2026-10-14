// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_device_adapter.hpp
 * @brief The dynG CBST store (dyng::detail::escher::cbst_store, cpp/src/hypergraph/escher) as a
 *        store of the trace replay comparator (M8a step 3, ADR 0038).
 *
 * Every tree of a trace becomes one cbst_store on the device. After every operation the adapter
 * reads the store's state back to the host and describes it exactly as the trace recorder of
 * ADR 0036 describes the originals' state: the keys by an in-order walk of the node pointers from
 * the root, every key's record and its row read along the chain from its first segment (at most
 * as many steps as the used payload has slots), subtree_avail by heap position. It also runs the
 * store's integrity checker (ESCHER-GPU's checkTreeRows, ported) on the rows as read, which the
 * comparator compares with the original's verdict (`check.rows`). Every construct is built both
 * from row counts and by the scan, and the two node arrays must be equal byte for byte.
 *
 * CUDA builds only (library dyng_cbst_device).
 */
#pragma once

#include "parity/escher/cbst_compare.hpp"

#include <dyng/core/resources.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace dyng::detail::escher {
struct cbst_store;
}  // namespace dyng::detail::escher

namespace dyng::test::cbst_compare {

/// The dynG device store as a store of the comparator.
class device_store_adapter final : public store_adapter {
 public:
  /**
   * @brief Every tree runs under the named store policy.
   * @param[in] policy A preset of store_policy::from_name() ("escher_gpu", "mosp_escher",
   *            "dyng_default").
   * @param[in] allow_compact Run compact() under escher_gpu() too (ESCHER-GPU had no compact(), so
   *            a replay of a trace under it reports the operation as unsupported by default).
   * @throws cbst_trace::trace_error for an unknown policy.
   */
  explicit device_store_adapter(std::string policy, bool allow_compact = false);
  ~device_store_adapter() override;
  device_store_adapter(const device_store_adapter&) = delete;
  device_store_adapter& operator=(const device_store_adapter&) = delete;
  [[nodiscard]] std::string name() const override;
  void create_tree(std::uint32_t tree, const std::string& dataset, std::int64_t capacity,
                   std::int64_t alignment) override;
  void destroy_tree(std::uint32_t tree) override;
  op_outcome run(std::uint32_t tree, const op_request& op) override;
  [[nodiscard]] tree_digest scalars(std::uint32_t tree) override;
  [[nodiscard]] tree_digest digest(std::uint32_t tree) override;

  /// The degree-bin launch counts of every operation run so far (fill, reuse, unfill x thread,
  /// warp, block), summed over the trees.
  [[nodiscard]] const std::array<std::int64_t, 9>& total_bins() const noexcept {
    return total_bins_;
  }

  /// The constructs built both ways (the caller's row counts and the scan) and compared byte for
  /// byte (M8a step 4): every construct of more than 0 records.
  [[nodiscard]] std::int64_t twin_constructs() const noexcept {
    return twin_constructs_;
  }

  /// The twin constructs of every adapter of the process so far (the replay tool reports them).
  [[nodiscard]] static std::int64_t total_twin_constructs() noexcept;

  /// A 64-bit hash of every digest the adapter returned, in order (two replays of a trace that
  /// leave identical states give identical hashes).
  [[nodiscard]] std::uint64_t digest_hash() const noexcept {
    return digest_hash_;
  }

  /// The store of a tree (nullptr if there is none): the scenario tests read it back with the
  /// store's own integrity checker against their expected rows.
  [[nodiscard]] const dyng::detail::escher::cbst_store* store(std::uint32_t tree) const;

  struct tree;  ///< one store and its name

 private:
  std::string policy_;
  bool allow_compact_ = false;
  resources res_;  ///< every tree's store runs on these (resources::cuda())
  std::map<std::uint32_t, std::unique_ptr<tree>> trees_;
  std::array<std::int64_t, 9> total_bins_{};
  std::int64_t twin_constructs_ = 0;
  std::uint64_t digest_hash_ = 0xcbf29ce484222325ULL;
};

/// The dynG device store under `policy` (empty: the preset of each trace's original, M8a step 5).
store_factory device_store_factory(const std::string& policy);

/// Whether a CUDA device is visible (the device replays skip otherwise).
bool device_available();

}  // namespace dyng::test::cbst_compare
