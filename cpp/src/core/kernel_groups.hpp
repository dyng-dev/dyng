// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernel_groups.hpp
 * @brief Kernel groups: each algorithm's CUDA kernels (and the hypergraph containers') are loaded
 *        only when the process first uses them (ADR 0109 part B; the author's decisions of
 *        2026-11-03 and 2026-11-05).
 *
 * Every `.cu` file of the library registers its fat binary with the CUDA runtime from a static
 * constructor, so every module of a linked library is registered when the library is loaded, and
 * under `CUDA_MODULE_LOADING=EAGER` (the timing protocol of every gate) the context creation of
 * every CUDA process loads every kernel of the library. A kernel group changes that for the
 * objects of one OBJECT library (cmake/dyng_kernel_groups.cmake): its compiler launcher renames,
 * in each object, the CUDA runtime's registration entry points (`__cudaRegisterFatBinary` and
 * the rest) and the runtime functions that take a kernel or a device symbol
 * (`cudaFuncGetAttributes`, the occupancy queries, the kernel handle of a launch, ...) to
 * functions of a generated translation unit of the group, which call the kernel_group below.
 * The group records the registrations at load time instead of passing them on, and replays them
 * (then has the runtime load every kernel of the group) the first time:
 *
 * - a program asks for it (kernel_groups::load(), the warm-ups below), or
 * - any object of the group calls one of the renamed runtime functions: the first launch of one
 *   of its kernels, an occupancy query, cudaFuncGetAttributes (the safety net: a group can never
 *   be used unloaded).
 *
 * The device code, the host code and every kernel's binding are unchanged; only the moment the
 * modules are created moves. Builds without CUDA, and builds configured with
 * `-DDYNG_DEFER_KERNEL_GROUPS=OFF`, have no group: every function here is then a no-op over an
 * empty list.
 *
 * Host-only (no CUDA headers); the definitions live in kernel_groups.cpp, whose CUDA part is
 * compiled only with DYNG_HAS_CUDA.
 */
#pragma once

#include <dyng/core/resources.hpp>

#include <atomic>
#include <cstddef>
#include <initializer_list>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace dyng::detail {

/**
 * @brief The deferred registrations of one kernel group and their replay.
 *
 * Created once per group (kernel_groups::get(), during static initialisation) and never
 * destroyed: the CUDA runtime unregisters the fat binaries from `atexit` handlers that may run
 * after the static destructors of this library.
 */
class kernel_group {
 public:
  /**
   * @brief An empty group.
   * @param[in] name The group's name (the algorithm's, or `hypergraph`).
   */
  explicit kernel_group(std::string name);

  kernel_group(const kernel_group&) = delete;             ///< not copyable
  kernel_group& operator=(const kernel_group&) = delete;  ///< not copyable
  kernel_group(kernel_group&&) = delete;                  ///< not movable
  kernel_group& operator=(kernel_group&&) = delete;       ///< not movable
  /// @brief Destructor (never run: the groups are never destroyed).
  ~kernel_group();

  /// @return The group's name.
  [[nodiscard]] const std::string& name() const noexcept {
    return name_;
  }

  /// @return True once the group's registrations were passed to the CUDA runtime.
  [[nodiscard]] bool loaded() const noexcept {
    return loaded_.load(std::memory_order_acquire);
  }

  /// @return The fat binaries (translation units) the group recorded.
  [[nodiscard]] std::size_t fatbins() const;

  /// @return The kernels the group recorded.
  [[nodiscard]] std::size_t kernels() const;

  /**
   * @brief Register the group's fat binaries with the CUDA runtime (once; thread-safe) and make
   *        the runtime create their modules on the current device.
   * @return 0 (cudaSuccess) or the CUDA runtime's error code of the module creation.
   */
  int ensure() noexcept;

  // The recorders, called by the group's generated translation unit (the renamed registration
  // calls of the objects' static constructors and of their atexit handlers). `handle` is the value
  // record_fatbin() returned for the object.

  /// @brief Record a fat binary; returns the handle its object passes to the calls below.
  void** record_fatbin(void* fatbin) noexcept;
  /// @brief Record a kernel of a recorded fat binary.
  void record_function(void** handle, const char* host_function, char* device_function,
                       const char* device_name, int thread_limit) noexcept;
  /// @brief Record a device variable of a recorded fat binary.
  void record_variable(void** handle, char* host_variable, char* device_address,
                       const char* device_name, int ext, std::size_t size, int constant,
                       int global) noexcept;
  /// @brief Record a managed variable of a recorded fat binary.
  void record_managed_variable(void** handle, void** host_variable_address, char* device_address,
                               const char* device_name, int ext, std::size_t size, int constant,
                               int global) noexcept;
  /// @brief The end of a fat binary's registration (nothing to record).
  void record_end(void** handle) noexcept;
  /// @brief The object's atexit handler: unregister the fat binary if it was registered.
  void unregister(void** handle) noexcept;
  /// @brief The runtime handle of a recorded fat binary (the group is loaded first).
  void** runtime_handle(void** handle) noexcept;

 private:
  struct function_record {
    const char* host_function;
    char* device_function;
    const char* device_name;
    int thread_limit;
  };
  struct variable_record {
    char* host_variable;
    void** host_variable_address;  // managed variables
    char* device_address;
    const char* device_name;
    int ext;
    std::size_t size;
    int constant;
    int global;
  };
  struct fatbin_record {
    void* fatbin = nullptr;
    void** runtime = nullptr;  // the runtime's handle once registered
    std::vector<function_record> functions;
    std::vector<variable_record> variables;
  };

  std::string name_;
  mutable std::mutex mutex_;
  std::atomic<bool> loaded_{false};
  int status_ = 0;  // the module creation's error code (under mutex_)
  std::vector<fatbin_record*> fatbins_;
};

/**
 * @brief A snapshot of one group, for tests and diagnostics.
 */
struct kernel_group_info {
  std::string name;         ///< the group's name
  bool loaded = false;      ///< registered with the CUDA runtime
  std::size_t fatbins = 0;  ///< translation units
  std::size_t kernels = 0;  ///< kernels
};

namespace kernel_groups {

/**
 * @brief The group of a name, created on first use (the generated translation units call it from
 *        the objects' static constructors).
 * @param[in] name The group's name (a string literal).
 * @return The group (never destroyed).
 */
kernel_group& get(const char* name) noexcept;

/**
 * @brief The group of a name, if the library has one.
 * @param[in] name The group's name.
 * @return The group, or nullptr.
 */
[[nodiscard]] kernel_group* find(std::string_view name) noexcept;

/**
 * @brief Every group of the library, in creation order.
 * @return The snapshots.
 */
[[nodiscard]] std::vector<kernel_group_info> list();

/**
 * @brief Load the named groups (on the current device; names the library has no group for are
 *        ignored, so the call is the same in builds without groups).
 * @param[in] names The groups' names.
 * @throws cuda_error if the CUDA runtime reports an error while creating the modules.
 */
void load(std::initializer_list<std::string_view> names);

/**
 * @brief Load every group (resources::warm_up(): every kernel of the library).
 * @throws cuda_error if the CUDA runtime reports an error while creating the modules.
 */
void load_all();

}  // namespace kernel_groups

/**
 * @brief resources::warm_up() restricted to the named kernel groups: the CUDA context, the
 *        library's kernels outside every group, the kernels of the named groups (and of the
 *        groups already loaded), the stream and the memory resource; the other groups stay
 *        unloaded until they are used. For programs that know the algorithms they run (the
 *        compatibility tools), so that their start-up loads only those kernels. A no-op for host
 *        backends.
 * @param[in] res    The resources.
 * @param[in] groups The groups' names (algorithm names, `hypergraph`).
 * @return The number of kernels warmed up.
 * @throws cuda_error if the CUDA runtime reports an error.
 */
std::size_t warm_up_kernel_groups(const resources& res,
                                  std::initializer_list<std::string_view> groups);

}  // namespace dyng::detail
