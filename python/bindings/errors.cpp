// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file errors.cpp
 * @brief Translation of the C++ exceptions (dyng::error and its subclasses) to the Python
 *        hierarchy of PLAN Section 5.4, which python/dyng/errors.py defines and registers here.
 *
 * | C++                        | Python (dyng.)                                |
 * |----------------------------|-----------------------------------------------|
 * | stale_result_error         | StaleResultError (InvalidArgumentError)       |
 * | invalid_argument_error     | InvalidArgumentError (Error, ValueError)      |
 * | io_error                   | FileFormatError (Error, OSError), .path/.line |
 * | capacity_error             | CapacityError (Error, MemoryError)            |
 * | not_supported_error        | NotSupportedError (Error, NotImplementedError)|
 * | convergence_error          | ConvergenceError (Error, RuntimeError)        |
 * | cuda_error                 | CudaError (Error, RuntimeError), .code        |
 * | out_of_memory_error        | OutOfMemoryError (Error, MemoryError)         |
 * | internal_error             | InternalError (Error, RuntimeError)           |
 * | error                      | Error                                         |
 *
 * Until the typed layer has registered its classes (it does so on import), the builtin base
 * classes are raised instead.
 */
#include "common.hpp"

#include <dyng/core/error.hpp>

#include <nanobind/stl/string.h>

#include <array>
#include <cstring>

namespace dyng::python {
namespace {

enum class kind : std::size_t {
  error,
  invalid_argument,
  stale_result,
  file_format,
  capacity,
  not_supported,
  convergence,
  cuda,
  out_of_memory,
  internal,
  count,
};

constexpr std::array<const char*, static_cast<std::size_t>(kind::count)> kind_names = {
    "Error",
    "InvalidArgumentError",
    "StaleResultError",
    "FileFormatError",
    "CapacityError",
    "NotSupportedError",
    "ConvergenceError",
    "CudaError",
    "OutOfMemoryError",
    "InternalError"};

/// The registered classes (strong references, kept for the life of the process).
std::array<PyObject*, static_cast<std::size_t>(kind::count)>& registered() {
  static std::array<PyObject*, static_cast<std::size_t>(kind::count)> types{};
  return types;
}

/// The class raised for `k`: the registered one, else a builtin base.
PyObject* type_of(kind k) {
  PyObject* t = registered()[static_cast<std::size_t>(k)];
  if (t != nullptr) {
    return t;
  }
  switch (k) {
    case kind::invalid_argument:
    case kind::stale_result:
      return PyExc_ValueError;
    case kind::file_format:
      return PyExc_OSError;
    case kind::capacity:
    case kind::out_of_memory:
      return PyExc_MemoryError;
    case kind::not_supported:
      return PyExc_NotImplementedError;
    default:
      return PyExc_RuntimeError;
  }
}

void raise(kind k, const char* message) {
  PyErr_SetString(type_of(k), message);
}

/// Raise an instance built by calling the class with `args` (for the classes with attributes).
void raise_instance(kind k, const char* message, nb::object instance) {
  if (!instance.is_valid()) {
    raise(k, message);
    return;
  }
  PyErr_SetObject(type_of(k), instance.ptr());
}

void set_error_types(const nb::dict& types) {
  for (std::size_t i = 0; i < kind_names.size(); ++i) {
    if (!types.contains(kind_names[i])) {
      throw nb::key_error(kind_names[i]);
    }
    nb::object t = types[kind_names[i]];
    if (!PyType_Check(t.ptr()) || PyObject_IsSubclass(t.ptr(), PyExc_Exception) != 1) {
      throw nb::type_error("_set_error_types: every value must be an exception class");
    }
    PyObject*& slot = registered()[i];
    Py_XDECREF(slot);
    slot = t.release().ptr();
  }
}

void translate(const std::exception_ptr& p, void* /*payload*/) {
  try {
    std::rethrow_exception(p);
  } catch (const stale_result_error& e) {
    raise(kind::stale_result, e.what());
  } catch (const invalid_argument_error& e) {
    raise(kind::invalid_argument, e.what());
  } catch (const io_error& e) {
    nb::object instance;
    if (registered()[static_cast<std::size_t>(kind::file_format)] != nullptr) {
      try {
        instance = nb::borrow(type_of(kind::file_format))(
            e.what(), e.path().empty() ? nb::object(nb::none()) : nb::cast(e.path()),
            e.line() > 0 ? nb::cast(e.line()) : nb::object(nb::none()),
            e.column() > 0 ? nb::cast(e.column()) : nb::object(nb::none()));
      } catch (const nb::python_error&) {
        instance = nb::object();
      }
    }
    raise_instance(kind::file_format, e.what(), instance);
  } catch (const capacity_error& e) {
    raise(kind::capacity, e.what());
  } catch (const not_supported_error& e) {
    raise(kind::not_supported, e.what());
  } catch (const convergence_error& e) {
    raise(kind::convergence, e.what());
  } catch (const cuda_error& e) {
    nb::object instance;
    if (registered()[static_cast<std::size_t>(kind::cuda)] != nullptr) {
      try {
        instance = nb::borrow(type_of(kind::cuda))(e.what(), e.code());
      } catch (const nb::python_error&) {
        instance = nb::object();
      }
    }
    raise_instance(kind::cuda, e.what(), instance);
  } catch (const out_of_memory_error& e) {
    raise(kind::out_of_memory, e.what());
  } catch (const internal_error& e) {
    raise(kind::internal, e.what());
  } catch (const error& e) {
    raise(kind::error, e.what());
  }
  // Anything else (std::bad_alloc, nanobind's own exceptions, ...) is left to the next translator.
}

}  // namespace

void bind_errors(nb::module_& m) {
  nb::register_exception_translator(&translate);
  m.def("_set_error_types", &set_error_types, nb::arg("types"),
        "Register the Python exception classes (called by dyng.errors on import).");
  m.def(
      "_raise_for_test",
      [](const std::string& which, const std::string& message) {
        if (which == "stale_result") {
          throw stale_result_error(message);
        }
        if (which == "invalid_argument") {
          throw invalid_argument_error(message);
        }
        if (which == "io") {
          throw io_error(message, "some/file.txt", 7, 3);
        }
        if (which == "capacity") {
          throw capacity_error(message);
        }
        if (which == "not_supported") {
          throw not_supported_error(message);
        }
        if (which == "convergence") {
          throw convergence_error(message);
        }
        if (which == "cuda") {
          throw cuda_error(message, 2);
        }
        if (which == "out_of_memory") {
          throw out_of_memory_error(message);
        }
        if (which == "internal") {
          throw internal_error(message);
        }
        throw error(message);
      },
      nb::arg("which"), nb::arg("message"),
      "Throw a C++ exception of the given kind (for the tests of the translation).");
}

}  // namespace dyng::python
