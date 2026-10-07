// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file arrays.cpp
 * @brief Arrays in device memory from Python (PLAN Section 5.4, rule 4; ADR 0031): the metadata of
 *        an array without touching its elements, host copies ordered on a resources handle's
 *        stream, and the stream ordering of a DLPack export.
 *
 * dyng.Array (python/dyng/array.py) uses these for the result arrays of the CUDA backend
 * (`shape`, `dtype`, `to_numpy()`, `__cuda_array_interface__`, `__dlpack__(stream=...)`), and the
 * typed layer for inputs in device memory (CuPy, PyTorch, any DLPack producer), which are copied
 * to the host once. In a module without CUDA every array is host memory and the functions that
 * need a device raise NotSupportedError.
 */
#include "common.hpp"
#include "core/cuda_runtime.hpp"  // detail::cuda_stream_fence (cpp/src, private)

#include <dyng/core/array_view.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stream.hpp>

#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace dyng::python {
namespace {

/// Any array nanobind can import (DLPack, the buffer protocol, a DLPack capsule); read-only.
using any_array = nb::ndarray<nb::ro>;

/// The DLPack device types of CUDA memory: kDLCUDA (2) and kDLCUDAManaged (13).
bool is_cuda(int device_type) {
  return device_type == nb::device::cuda::value || device_type == nb::device::cuda_managed::value;
}

/// The memory space of a DLPack device type.
memory_space space_of(int device_type) {
  if (device_type == nb::device::cuda::value) {
    return memory_space::device;
  }
  if (device_type == nb::device::cuda_managed::value) {
    return memory_space::managed;
  }
  if (device_type == nb::device::cuda_host::value) {
    return memory_space::pinned_host;
  }
  return memory_space::host;
}

/// The array-interface type string of a DLPack dtype ("<i8", "|b1", ...).
std::string typestr_of(nb::dlpack::dtype dt) {
  char kind = 0;
  switch (static_cast<nb::dlpack::dtype_code>(dt.code)) {
    case nb::dlpack::dtype_code::Int:
      kind = 'i';
      break;
    case nb::dlpack::dtype_code::UInt:
      kind = 'u';
      break;
    case nb::dlpack::dtype_code::Float:
      kind = 'f';
      break;
    case nb::dlpack::dtype_code::Bool:
      kind = 'b';
      break;
    case nb::dlpack::dtype_code::Complex:
      kind = 'c';
      break;
    default:
      throw not_supported_error("dyng: arrays of DLPack type code " + std::to_string(dt.code) +
                                " are not supported");
  }
  DYNG_EXPECTS(dt.lanes == 1 && dt.bits % 8 == 0, "dyng: arrays of ", dt.bits, "-bit elements in ",
               dt.lanes, " lanes are not supported");
  const std::size_t bytes = dt.bits / 8;
  return std::string(bytes == 1 ? "|" : "<") + kind + std::to_string(bytes);
}

/// Whether `a` is C-contiguous (strides in elements; size-1 dimensions are free).
bool c_contiguous(const any_array& a) {
  std::int64_t expected = 1;
  for (std::size_t i = a.ndim(); i-- > 0;) {
    if (a.shape(i) != 1 && a.stride(i) != expected) {
      return false;
    }
    expected *= static_cast<std::int64_t>(a.shape(i));
  }
  return true;
}

}  // namespace

void bind_arrays(nb::module_& m) {
  m.def(
      "array_info",
      [](const any_array& a) {
        nb::list shape;
        for (std::size_t i = 0; i < a.ndim(); ++i) {
          shape.append(static_cast<std::int64_t>(a.shape(i)));
        }
        return nb::make_tuple(reinterpret_cast<std::uintptr_t>(a.data()), nb::tuple(shape),
                              typestr_of(a.dtype()), a.device_type(), a.device_id(),
                              c_contiguous(a));
      },
      nb::arg("array"),
      "(data pointer, shape, type string, DLPack device type, device id, C-contiguous) of an "
      "array, without reading its elements.");
  m.def(
      "array_to_host",
      [](const resources& res, const any_array& a) {
        DYNG_EXPECTS(
            c_contiguous(a),
            "dyng: an array in device memory must be C-contiguous to be copied to the "
            "host (make a contiguous copy first, e.g. .contiguous() / cupy.ascontiguousarray)");
        const bool device = is_cuda(a.device_type());
        DYNG_EXPECTS(!device || res.get_backend() == backend::cuda,
                     "dyng: copying device memory to the host needs CUDA resources");
        const std::size_t nbytes = a.nbytes();
        auto* heap = new std::vector<std::byte>(nbytes);
        nb::capsule owner(heap,
                          [](void* p) noexcept { delete static_cast<std::vector<std::byte>*>(p); });
        if (nbytes > 0) {
          const auto* src = static_cast<const std::byte*>(a.data());
          if (device) {
            const array_view<const std::byte> from(src, nbytes, space_of(a.device_type()),
                                                   static_cast<int>(a.device_id()));
            without_gil([&] {
              copy<std::byte>(res, from, host_view(heap->data(), heap->size()));
              res.synchronize();
            });
          } else {
            std::memcpy(heap->data(), src, nbytes);
          }
        }
        std::vector<std::size_t> shape(a.ndim());
        for (std::size_t i = 0; i < a.ndim(); ++i) {
          shape[i] = a.shape(i);
        }
        return nb::ndarray<nb::numpy>(heap->data(), a.ndim(), shape.data(), owner, nullptr,
                                      a.dtype(), nb::device::cpu::value, 0);
      },
      nb::arg("resources"), nb::arg("array"),
      "A host (NumPy) copy of a C-contiguous array; device memory is copied on the stream of "
      "`resources` (CUDA), which is then synchronized.");
  m.def(
      "order_stream",
      [](const resources& res, std::uintptr_t consumer) {
        DYNG_EXPECTS(res.get_backend() == backend::cuda,
                     "dyng: stream ordering needs CUDA resources");
        without_gil([&] {
          detail::cuda_stream_fence fence;
          fence.record(res.device(), res.stream());
          fence.wait(res.device(), stream_ref(reinterpret_cast<cuda_stream_handle>(consumer)));
        });
      },
      nb::arg("resources"), nb::arg("consumer"),
      "Order the later work of the stream `consumer` (a cudaStream_t as an integer; 1 = the "
      "legacy default stream, 2 = the per-thread default stream) after the work enqueued so far "
      "on the stream of `resources` (an event; the host does not wait).");
}

}  // namespace dyng::python
