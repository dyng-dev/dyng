// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernel_groups.cpp
 * @brief Kernel groups: the deferred registration of an algorithm's CUDA kernels (ADR 0109 part
 *        B). The replay is compiled only with DYNG_HAS_CUDA; without it no group can record
 *        anything (no object of the library is compiled by nvcc).
 */
#include "core/kernel_groups.hpp"

#include "core/cuda_runtime.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#include <exception>
#include <memory>
#include <utility>

#if DYNG_HAS_CUDA

#include "util/cuda_check.hpp"

#include <dyng/core/logging.hpp>

#include <cuda_runtime_api.h>

#include <string>

// The CUDA runtime's registration entry points (crt/host_runtime.h declares them for nvcc's
// generated host code only). The same declarations; called here with the arguments the group
// recorded.
extern "C" {
void** __cudaRegisterFatBinary(void* fatbin);    // NOLINT(bugprone-reserved-identifier)
void __cudaRegisterFatBinaryEnd(void** handle);  // NOLINT(bugprone-reserved-identifier)
void __cudaUnregisterFatBinary(void** handle);   // NOLINT(bugprone-reserved-identifier)
// NOLINTNEXTLINE(bugprone-reserved-identifier)
void __cudaRegisterFunction(void** handle, const char* host_function, char* device_function,
                            const char* device_name, int thread_limit, uint3* tid, uint3* bid,
                            dim3* block_dim, dim3* grid_dim, int* warp_size);
// NOLINTNEXTLINE(bugprone-reserved-identifier)
void __cudaRegisterVar(void** handle, char* host_variable, char* device_address,
                       const char* device_name, int ext, std::size_t size, int constant,
                       int global);
// NOLINTNEXTLINE(bugprone-reserved-identifier)
void __cudaRegisterManagedVar(void** handle, void** host_variable_address, char* device_address,
                              const char* device_name, int ext, std::size_t size, int constant,
                              int global);
}

#endif  // DYNG_HAS_CUDA

namespace dyng::detail {

namespace {

struct group_list {
  std::mutex mutex;
  std::vector<kernel_group*> groups;
};

// Never destroyed (see kernel_group): created by the first static constructor that records.
group_list& all_groups() noexcept {
  static auto* list = new group_list;
  return *list;
}

}  // namespace

kernel_group::kernel_group(std::string name) : name_(std::move(name)) {}

kernel_group::~kernel_group() {
  for (fatbin_record* record : fatbins_) {
    delete record;
  }
}

std::size_t kernel_group::fatbins() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return fatbins_.size();
}

std::size_t kernel_group::kernels() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  std::size_t count = 0;
  for (const fatbin_record* record : fatbins_) {
    count += record->functions.size();
  }
  return count;
}

void** kernel_group::record_fatbin(void* fatbin) noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  try {
    auto record = std::make_unique<fatbin_record>();
    record->fatbin = fatbin;
    fatbins_.push_back(record.get());
    // The object keeps this pointer as its "handle" and passes it back to the recorders.
    return reinterpret_cast<void**>(record.release());
  } catch (...) {
    // Out of memory while the library loads: nothing can be registered later either.
    std::terminate();
  }
}

namespace {

template <typename record_t>
record_t* as_record(void** handle) noexcept {
  return reinterpret_cast<record_t*>(handle);
}

}  // namespace

void kernel_group::record_function(void** handle, const char* host_function, char* device_function,
                                   const char* device_name, int thread_limit) noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  try {
    as_record<fatbin_record>(handle)->functions.push_back(
        function_record{host_function, device_function, device_name, thread_limit});
  } catch (...) {
    std::terminate();
  }
}

void kernel_group::record_variable(void** handle, char* host_variable, char* device_address,
                                   const char* device_name, int ext, std::size_t size, int constant,
                                   int global) noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  try {
    as_record<fatbin_record>(handle)->variables.push_back(variable_record{
        host_variable, nullptr, device_address, device_name, ext, size, constant, global});
  } catch (...) {
    std::terminate();
  }
}

void kernel_group::record_managed_variable(void** handle, void** host_variable_address,
                                           char* device_address, const char* device_name, int ext,
                                           std::size_t size, int constant, int global) noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  try {
    as_record<fatbin_record>(handle)->variables.push_back(variable_record{
        nullptr, host_variable_address, device_address, device_name, ext, size, constant, global});
  } catch (...) {
    std::terminate();
  }
}

void kernel_group::record_end(void** /*handle*/) noexcept {}

#if DYNG_HAS_CUDA

int kernel_group::ensure() noexcept {
  if (loaded_.load(std::memory_order_acquire)) {
    return status_;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  if (loaded_.load(std::memory_order_relaxed)) {
    return status_;
  }
  std::size_t kernels = 0;
  for (fatbin_record* record : fatbins_) {
    record->runtime = __cudaRegisterFatBinary(record->fatbin);
    for (const function_record& f : record->functions) {
      __cudaRegisterFunction(record->runtime, f.host_function, f.device_function, f.device_name,
                             f.thread_limit, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    for (const variable_record& v : record->variables) {
      if (v.host_variable_address != nullptr) {
        __cudaRegisterManagedVar(record->runtime, v.host_variable_address, v.device_address,
                                 v.device_name, v.ext, v.size, v.constant, v.global);
      } else {
        __cudaRegisterVar(record->runtime, v.host_variable, v.device_address, v.device_name, v.ext,
                          v.size, v.constant, v.global);
      }
    }
    __cudaRegisterFatBinaryEnd(record->runtime);
    kernels += record->functions.size();
  }
  // cudaFree(nullptr) creates the current device's context if there is none yet. The runtime
  // does not create the modules of fat binaries registered after the context exists at that call,
  // not even under CUDA_MODULE_LOADING=EAGER: it loads each kernel when it is first used (the
  // first update of label_propagation's published flow paid about 7.6 ms for that at 50k; M10b
  // step 4's final re-read). Ask for every kernel's attributes here, which loads it, so that the
  // call that loads the group pays for its modules, as the context creation does for the
  // modules registered at load time, and the next (timed) call does not. A kernel without an
  // image for this device (its first launch reports that) is skipped, and its error is cleared
  // so that no later check takes it for its own.
  status_ = static_cast<int>(cudaFree(nullptr));
  if (status_ == 0) {
    for (const fatbin_record* record : fatbins_) {
      for (const function_record& f : record->functions) {
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(&attributes, f.host_function) != cudaSuccess) {
          (void)cudaGetLastError();
        }
      }
    }
  }
  loaded_.store(true, std::memory_order_release);
  if (log_enabled(log_level::debug)) {
    log_message(log_level::debug, "dyng: kernel group " + name_ + " loaded (" +
                                      std::to_string(fatbins_.size()) + " translation units, " +
                                      std::to_string(kernels) + " kernels)");
  }
  return status_;
}

void kernel_group::unregister(void** handle) noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  fatbin_record* record = as_record<fatbin_record>(handle);
  if (record->runtime != nullptr) {
    __cudaUnregisterFatBinary(record->runtime);
    record->runtime = nullptr;
  }
}

void** kernel_group::runtime_handle(void** handle) noexcept {
  (void)ensure();
  return as_record<fatbin_record>(handle)->runtime;
}

#else  // !DYNG_HAS_CUDA

int kernel_group::ensure() noexcept {
  // Without CUDA there is nothing to register: the group is loaded, with no error.
  const std::lock_guard<std::mutex> lock(mutex_);
  status_ = 0;
  loaded_.store(true, std::memory_order_release);
  return status_;
}

void kernel_group::unregister(void** /*handle*/) noexcept {}

void** kernel_group::runtime_handle(void** handle) noexcept {
  return handle;
}

#endif  // DYNG_HAS_CUDA

namespace kernel_groups {

kernel_group& get(const char* name) noexcept {
  group_list& list = all_groups();
  const std::lock_guard<std::mutex> lock(list.mutex);
  for (kernel_group* group : list.groups) {
    if (group->name() == name) {
      return *group;
    }
  }
  try {
    auto* group = new kernel_group(name);
    list.groups.push_back(group);
    return *group;
  } catch (...) {
    std::terminate();
  }
}

kernel_group* find(std::string_view name) noexcept {
  group_list& list = all_groups();
  const std::lock_guard<std::mutex> lock(list.mutex);
  for (kernel_group* group : list.groups) {
    if (group->name() == name) {
      return group;
    }
  }
  return nullptr;
}

std::vector<kernel_group_info> list() {
  std::vector<kernel_group*> groups;
  {
    group_list& all = all_groups();
    const std::lock_guard<std::mutex> lock(all.mutex);
    groups = all.groups;
  }
  std::vector<kernel_group_info> out;
  out.reserve(groups.size());
  for (const kernel_group* group : groups) {
    out.push_back(
        kernel_group_info{group->name(), group->loaded(), group->fatbins(), group->kernels()});
  }
  return out;
}

namespace {

void load_group(kernel_group& group) {
#if DYNG_HAS_CUDA
  const int status = group.ensure();
  if (status != 0) {
    throw_cuda_error(static_cast<cudaError_t>(status), ("kernel group " + group.name()).c_str(),
                     __FILE__, __LINE__);
  }
#else
  (void)group.ensure();
#endif
}

}  // namespace

void load(std::initializer_list<std::string_view> names) {
  for (const std::string_view name : names) {
    if (kernel_group* group = find(name); group != nullptr) {
      load_group(*group);
    }
  }
}

void load_all() {
  std::vector<kernel_group*> groups;
  {
    group_list& all = all_groups();
    const std::lock_guard<std::mutex> lock(all.mutex);
    groups = all.groups;
  }
  for (kernel_group* group : groups) {
    load_group(*group);
  }
}

}  // namespace kernel_groups

std::size_t warm_up_kernel_groups(const resources& res,
                                  std::initializer_list<std::string_view> groups) {
  if (res.get_backend() != backend::cuda) {
    return 0;
  }
  return cuda_warm_up_groups(res.device(), res.stream(), res.memory(), groups);
}

}  // namespace dyng::detail
