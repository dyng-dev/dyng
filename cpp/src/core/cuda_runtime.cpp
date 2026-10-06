// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_runtime.cpp
 * @brief Host-side helpers of the CUDA backend (device queries, device guard, synchronization,
 *        warm-up, copies). The body is compiled only with DYNG_HAS_CUDA.
 */
#include "core/cuda_runtime.hpp"

#include "core/budget_counters.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#if DYNG_HAS_CUDA

#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/logging.hpp>

#include <cuda_runtime_api.h>

#include <string>

namespace dyng::detail {

namespace {

cudaStream_t native(stream_ref stream) noexcept {
  return static_cast<cudaStream_t>(stream.get());
}

int device_attribute(cudaDeviceAttr attribute, int device) {
  int value = 0;
  DYNG_CUDA_TRY(cudaDeviceGetAttribute(&value, attribute, device));
  return value;
}

/// The remedy for cudaErrorInitializationError, which a forked child of a CUDA process gets.
constexpr const char* fork_hint =
    "; CUDA cannot be initialized in a process forked after its parent initialized CUDA: start "
    "worker processes with the 'spawn' or 'forkserver' method (Python: "
    "multiprocessing.get_context('spawn')), or use CUDA in the parent only after the fork";

/// What a kernel without code for the current device means, with the device's compute capability.
std::string no_kernel_image_hint() {
  int device = -1;
  int major = 0;
  int minor = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    (void)cudaGetLastError();
    return "; this build of dynG has no code for the current device";
  }
  return concat_message("; this build of dynG has no code for device ", device,
                        " (compute capability ", major, ".", minor,
                        "): dynG's release builds need compute capability 7.5 or newer");
}

/// The CUDA error message, with a remedy for the errors that have a known common cause.
std::string cuda_error_message(cudaError_t status, const char* call, const char* file, int line) {
  std::string message = concat_message("dyng: CUDA error ", cudaGetErrorName(status), " (",
                                       static_cast<int>(status), "): ", cudaGetErrorString(status),
                                       " in ", call, " (", source_basename(file), ":", line, ")");
  if (status == cudaErrorInitializationError) {
    message += fork_hint;
  } else if (status == cudaErrorNoKernelImageForDevice) {
    message += no_kernel_image_hint();
  }
  return message;
}

}  // namespace

void throw_no_cuda_device() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  (void)cudaGetLastError();
  if (status == cudaErrorInitializationError) {
    throw not_supported_error(concat_message(
        "dyng: the cuda backend is built but the CUDA runtime cannot be initialized in this "
        "process (cudaErrorInitializationError)",
        fork_hint));
  }
  throw not_supported_error(concat_message(
      "dyng: the cuda backend is built but no CUDA device is visible (check the driver and "
      "CUDA_VISIBLE_DEVICES; ",
      cudaGetErrorName(status), ")"));
}

void throw_cuda_error(cudaError_t status, const char* call, const char* file, int line) {
  // Reset the runtime's last error, so the next DYNG_CHECK_KERNEL does not report this one again
  // (a sticky error, which corrupts the context, stays: every later call reports it).
  (void)cudaGetLastError();
  std::string message = cuda_error_message(status, call, file, line);
  if (status == cudaErrorMemoryAllocation) {
    throw out_of_memory_error(message);
  }
  throw cuda_error(message, static_cast<int>(status));
}

void log_cuda_error(cudaError_t status, const char* call, const char* file, int line) noexcept {
  (void)cudaGetLastError();
  if (status == cudaErrorCudartUnloading) {
    return;  // process teardown: the runtime already released everything
  }
  try {
    log_message(log_level::error,
                concat_message("dyng: CUDA error ", cudaGetErrorName(status), " (",
                               static_cast<int>(status), "): ", cudaGetErrorString(status), " in ",
                               call, " (", source_basename(file), ":", line, ")"));
  } catch (...) {
    // A sink that throws, or no memory for the message: nothing more can be done on this path.
  }
}

int cuda_device_count() noexcept {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();  // no driver or no device: not an error of later calls
    return 0;
  }
  return count;
}

cuda_device_properties query_cuda_device(int device) {
  const int count = cuda_device_count();
  if (count == 0) {
    throw_no_cuda_device();
  }
  DYNG_EXPECTS(device >= 0 && device < count, "CUDA device ", device, " does not exist; ", count,
               " device(s) visible");
  cudaDeviceProp prop{};
  DYNG_CUDA_TRY(cudaGetDeviceProperties(&prop, device));
  cuda_device_properties out;
  out.ordinal = device;
  out.name = prop.name;
  out.major = prop.major;
  out.minor = prop.minor;
  out.multiprocessor_count = prop.multiProcessorCount;
  out.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
  out.max_threads_per_block = prop.maxThreadsPerBlock;
  out.max_shared_memory_per_block_optin = static_cast<int>(prop.sharedMemPerBlockOptin);
  out.total_global_memory = prop.totalGlobalMem;
  out.cooperative_launch = device_attribute(cudaDevAttrCooperativeLaunch, device) != 0;
  out.memory_pools = device_attribute(cudaDevAttrMemoryPoolsSupported, device) != 0;
  return out;
}

scoped_device::scoped_device(int device) {
  DYNG_CUDA_TRY(cudaGetDevice(&previous_));
  if (previous_ != device) {
    DYNG_CUDA_TRY(cudaSetDevice(device));
  } else {
    previous_ = -1;  // nothing to restore
  }
}

scoped_device::~scoped_device() {
  if (previous_ >= 0) {
    DYNG_CUDA_TRY_NO_THROW(cudaSetDevice(previous_));
  }
}

void cuda_synchronize(int device, stream_ref stream) {
  note_host_sync();  // invariant I9 (counts only with DYNG_DEBUG_BUDGETS)
  if (device < 0) {
    DYNG_CUDA_TRY(cudaStreamSynchronize(native(stream)));
    return;
  }
  const scoped_device guard(device);
  DYNG_CUDA_TRY(cudaStreamSynchronize(native(stream)));
}

std::size_t cuda_warm_up(int device, stream_ref stream, memory_resource_ref memory) {
  const scoped_device guard(device);
  // The primary context (cudaSetDevice creates it since CUDA 12; cudaFree(nullptr) is the
  // traditional way to force it on every version).
  DYNG_CUDA_TRY(cudaFree(nullptr));
  // cudaFuncGetAttributes loads a kernel's module under lazy loading.
  std::size_t loaded = 0;
  for (const kernel_entry& kernel : registered_kernels()) {
    cudaFuncAttributes attributes{};
    const cudaError_t status = cudaFuncGetAttributes(&attributes, kernel.function);
    if (status != cudaSuccess) {
      throw_cuda_error(status, kernel.name, __FILE__, __LINE__);
    }
    ++loaded;
  }
  // Prime the memory resource (a pool reserves its first block) and the stream (the per-thread
  // default stream is created on first use).
  void* probe = memory.allocate(stream, 256, 256);
  memory.deallocate(stream, probe, 256, 256);
  DYNG_CUDA_TRY(cudaStreamSynchronize(native(stream)));
  return loaded;
}

void cuda_copy_bytes(void* dst, const void* src, std::size_t bytes, stream_ref stream, int device) {
  if (device < 0) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, native(stream)));
    return;
  }
  const scoped_device guard(device);
  DYNG_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, native(stream)));
}

cuda_event_timer::cuda_event_timer(int device, stream_ref stream)
    : device_(device), stream_(stream) {
  const scoped_device guard(device);
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  DYNG_CUDA_TRY(cudaEventCreate(&start));
  start_ = start;
  DYNG_CUDA_TRY(cudaEventCreate(&stop));
  stop_ = stop;
  DYNG_CUDA_TRY(cudaEventRecord(start, native(stream)));
}

cuda_event_timer::~cuda_event_timer() {
  if (start_ != nullptr) {
    DYNG_CUDA_TRY_NO_THROW(cudaEventDestroy(static_cast<cudaEvent_t>(start_)));
  }
  if (stop_ != nullptr) {
    DYNG_CUDA_TRY_NO_THROW(cudaEventDestroy(static_cast<cudaEvent_t>(stop_)));
  }
}

double cuda_event_timer::stop() {
  const scoped_device guard(device_);
  const auto start = static_cast<cudaEvent_t>(start_);
  const auto stop = static_cast<cudaEvent_t>(stop_);
  DYNG_CUDA_TRY(cudaEventRecord(stop, native(stream_)));
  DYNG_CUDA_TRY(cudaEventSynchronize(stop));
  float ms = 0.0F;
  DYNG_CUDA_TRY(cudaEventElapsedTime(&ms, start, stop));
  return static_cast<double>(ms);
}

cuda_stream_fence::~cuda_stream_fence() {
  if (event_ != nullptr) {
    DYNG_CUDA_TRY_NO_THROW(cudaEventDestroy(static_cast<cudaEvent_t>(event_)));
  }
}

void cuda_stream_fence::record(int device, stream_ref stream) {
  const scoped_device guard(device);
  if (event_ == nullptr || device_ != device) {
    if (event_ != nullptr) {
      DYNG_CUDA_TRY_NO_THROW(cudaEventDestroy(static_cast<cudaEvent_t>(event_)));
      event_ = nullptr;
    }
    recorded_ = false;
    cudaEvent_t event = nullptr;
    DYNG_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    event_ = event;
    device_ = device;
  }
  recorded_ = false;
  DYNG_CUDA_TRY(cudaEventRecord(static_cast<cudaEvent_t>(event_), native(stream)));
  stream_ = stream;
  thread_ = std::this_thread::get_id();
  recorded_ = true;
}

void cuda_stream_fence::wait(int device, stream_ref stream) const {
  if (!recorded_ || (stream == stream_ &&
                     (!stream.is_per_thread_default() || thread_ == std::this_thread::get_id()))) {
    return;
  }
  const scoped_device guard(device);
  DYNG_CUDA_TRY(cudaStreamWaitEvent(native(stream), static_cast<cudaEvent_t>(event_), 0));
}

void cuda_stream_fence::synchronize() const noexcept {
  if (recorded_) {
    DYNG_CUDA_TRY_NO_THROW(cudaEventSynchronize(static_cast<cudaEvent_t>(event_)));
  }
}

}  // namespace dyng::detail

#else  // !DYNG_HAS_CUDA

namespace dyng::detail {

namespace {

[[noreturn]] void cuda_not_built() {
  throw not_supported_error(
      "dyng: the cuda backend is not built (configure with "
      "DYNG_ENABLE_CUDA=ON)");
}

}  // namespace

int cuda_device_count() noexcept {
  return 0;
}

void throw_no_cuda_device() {
  cuda_not_built();
}

cuda_device_properties query_cuda_device(int /*device*/) {
  cuda_not_built();
}

scoped_device::scoped_device(int /*device*/) {
  cuda_not_built();
}

scoped_device::~scoped_device() = default;

void cuda_synchronize(int /*device*/, stream_ref /*stream*/) {}

std::size_t cuda_warm_up(int /*device*/, stream_ref /*stream*/, memory_resource_ref /*memory*/) {
  cuda_not_built();
}

void cuda_copy_bytes(void* /*dst*/, const void* /*src*/, std::size_t /*bytes*/,
                     stream_ref /*stream*/, int /*device*/) {
  cuda_not_built();
}

cuda_event_timer::cuda_event_timer(int /*device*/, stream_ref /*stream*/) {
  cuda_not_built();
}

cuda_event_timer::~cuda_event_timer() = default;

double cuda_event_timer::stop() {
  cuda_not_built();
}

cuda_stream_fence::~cuda_stream_fence() = default;

void cuda_stream_fence::record(int /*device*/, stream_ref /*stream*/) {
  cuda_not_built();
}

void cuda_stream_fence::wait(int /*device*/, stream_ref /*stream*/) const {}

void cuda_stream_fence::synchronize() const noexcept {}

}  // namespace dyng::detail

#endif  // DYNG_HAS_CUDA
