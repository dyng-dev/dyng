# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Kernel groups (ADR 0109 part B; the author's decisions of 2026-11-03 and 2026-11-05): the CUDA
# kernels of one OBJECT library are registered with the CUDA runtime, and so loaded, only when the
# process first uses them, instead of when libdyng is loaded (which, under
# CUDA_MODULE_LOADING=EAGER, made every CUDA process of the library load every kernel of it).
#
# dyng_defer_kernel_group(<target> <group>)
#   For the CUDA sources of <target> (an OBJECT library of libdyng):
#   - the compiler launcher cmake/kernel_group_launcher.sh compiles each .cu file as before, then
#     renames, in the object, the CUDA runtime's registration entry points (__cudaRegisterFatBinary
#     and the rest) and the runtime functions that take a kernel or a device symbol to functions of
#     the group (objcopy --redefine-syms with the generated <group>.syms), and refuses an object
#     that still calls such a function unrenamed (nm);
#   - the generated translation unit <group>.cpp (from cmake/kernel_group.cpp.in) defines those
#     functions: they record the registrations (dyng::detail::kernel_group, core/kernel_groups.hpp)
#     and load the group before passing a kernel or symbol to the runtime;
#   - DYNG_KERNEL_GROUP="<group>" tags the kernels the sources register for resources::warm_up().
#   Nothing changes in the device code, in the host code the compiler generates, or in builds
#   without CUDA. DYNG_DEFER_KERNEL_GROUPS=OFF turns the mechanism off (every module is registered
#   when libdyng is loaded, as before 0.4).

# The runtime entry points the launcher renames, as "<runtime name> <suffix>": the group's function
# is dyng_kg_<group>_<suffix>. kernel_group.cpp.in defines one function per line, with the
# runtime's signature (CUDA 12.4 and later).
set(DYNG_KERNEL_GROUP_SYMBOLS
    "__cudaRegisterFatBinary register_fatbin"
    "__cudaRegisterFatBinaryEnd register_fatbin_end"
    "__cudaUnregisterFatBinary unregister_fatbin"
    "__cudaRegisterFunction register_function"
    "__cudaRegisterVar register_var"
    "__cudaRegisterManagedVar register_managed_var"
    "__cudaInitModule init_module"
    "__cudaGetKernel get_kernel"
    "cudaFuncGetAttributes func_get_attributes"
    "cudaFuncSetAttribute func_set_attribute"
    "cudaFuncSetCacheConfig func_set_cache_config"
    "cudaLaunchKernel launch_kernel"
    "cudaLaunchCooperativeKernel launch_cooperative_kernel"
    "cudaLaunchKernelExC launch_kernel_ex_c"
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor occupancy_max_active_blocks"
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags occupancy_max_active_blocks_with_flags"
    "cudaOccupancyAvailableDynamicSMemPerBlock occupancy_available_dynamic_smem"
    "cudaGetSymbolAddress get_symbol_address"
    "cudaGetSymbolSize get_symbol_size"
    "cudaMemcpyToSymbol memcpy_to_symbol"
    "cudaMemcpyToSymbolAsync memcpy_to_symbol_async"
    "cudaMemcpyFromSymbol memcpy_from_symbol"
    "cudaMemcpyFromSymbolAsync memcpy_from_symbol_async"
    "cudaGetFuncBySymbol get_func_by_symbol"
)

function(dyng_defer_kernel_group target group)
  if(NOT DYNG_ENABLE_CUDA OR NOT DYNG_DEFER_KERNEL_GROUPS)
    return()
  endif()
  if(NOT CMAKE_OBJCOPY OR NOT CMAKE_NM)
    message(FATAL_ERROR "dynG: the kernel groups need objcopy and nm (binutils); none was found "
                        "(configure with -DDYNG_DEFER_KERNEL_GROUPS=OFF to build without them)")
  endif()
  if(NOT group MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "dynG: kernel group name '${group}' is not a lower-case identifier")
  endif()
  set(_dir "${PROJECT_BINARY_DIR}/cpp/generated/kernel_groups")
  set(_syms_text "")
  foreach(_entry IN LISTS DYNG_KERNEL_GROUP_SYMBOLS)
    string(REPLACE " " ";" _pair "${_entry}")
    list(GET _pair 0 _runtime)
    list(GET _pair 1 _suffix)
    string(APPEND _syms_text "${_runtime} dyng_kg_${group}_${_suffix}\n")
  endforeach()
  file(MAKE_DIRECTORY "${_dir}")
  file(CONFIGURE OUTPUT "${_dir}/${group}.syms" CONTENT "${_syms_text}")
  set(DYNG_KERNEL_GROUP_NAME "${group}")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/kernel_group.cpp.in" "${_dir}/${group}.cpp" @ONLY)
  target_sources(${target} PRIVATE "${_dir}/${group}.cpp")
  set_property(
    TARGET ${target}
    PROPERTY CUDA_COMPILER_LAUNCHER sh "${PROJECT_SOURCE_DIR}/cmake/kernel_group_launcher.sh"
             "${CMAKE_OBJCOPY}" "${CMAKE_NM}" "${_dir}/${group}.syms" ${CMAKE_CUDA_COMPILER_LAUNCHER}
  )
  target_compile_definitions(
    ${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:DYNG_KERNEL_GROUP=\"${group}\">"
  )
  # A launcher or symbol map that changes must recompile the group's objects.
  get_target_property(_sources ${target} SOURCES)
  get_target_property(_source_dir ${target} SOURCE_DIR)
  foreach(_source IN LISTS _sources)
    if(_source MATCHES "\\.cu$")
      cmake_path(ABSOLUTE_PATH _source BASE_DIRECTORY "${_source_dir}")
      set_property(
        SOURCE "${_source}" TARGET_DIRECTORY ${target} APPEND
        PROPERTY OBJECT_DEPENDS "${PROJECT_SOURCE_DIR}/cmake/kernel_group_launcher.sh"
                                "${_dir}/${group}.syms"
      )
    endif()
  endforeach()
  set_property(GLOBAL APPEND PROPERTY DYNG_KERNEL_GROUPS ${group})
endfunction()
