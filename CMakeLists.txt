if("x${CMAKE_SOURCE_DIR}" STREQUAL "x${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "\
In-source build is not a good practice.
Please use:
  mkdir build
  cd build
  cmake ..
to build this project"
  )
endif()

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# see https://cmake.org/cmake/help/latest/policy/CMP0111.html
#
# This is to suppress the warnings for importing PyTorch.
if(POLICY CMP0111)
  cmake_policy(SET CMP0111 OLD)
endif()
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(languages CXX)
set(_K2_WITH_CUDA ON)

find_program(K2_HAS_NVCC nvcc)
if(NOT K2_HAS_NVCC AND "$ENV{CUDACXX}" STREQUAL "")
  message(STATUS "No NVCC detected. Disable CUDA support")
  set(_K2_WITH_CUDA OFF)
endif()

if(APPLE OR (DEFINED K2_WITH_CUDA AND NOT K2_WITH_CUDA))
  if(_K2_WITH_CUDA)
    message(STATUS "Disable CUDA support")
    set(_K2_WITH_CUDA OFF)
  endif()
endif()

if(_K2_WITH_CUDA)
  set(languages ${languages} CUDA)
endif()

message(STATUS "Enabled languages: ${languages}")

project(k2 ${languages})

set(K2_VERSION "1.11")

# ----------------- Supported build types for K2 project -----------------
set(ALLOWABLE_BUILD_TYPES Debug Release RelWithDebInfo MinSizeRel)
set(DEFAULT_BUILD_TYPE "Debug")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${ALLOWABLE_BUILD_TYPES}")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # CMAKE_CONFIGURATION_TYPES: with config type values from other generators (IDE).
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Debug")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}")
elseif(NOT CMAKE_BUILD_TYPE IN_LIST ALLOWABLE_BUILD_TYPES)
  message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}, \
    choose one from ${ALLOWABLE_BUILD_TYPES}")
endif()

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPERCASE)
if("${CMAKE_BUILD_TYPE_UPPERCASE}" STREQUAL "DEBUG")
  # refer to https://docs.nvidia.com/cuda/cuda-memcheck/index.html#compilation-options
  # The two options are to make cuda-memcheck's stack backtrace feature more useful.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -rdynamic --compiler-options -lineinfo")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Whether to build shared or static lib" ON)
option(K2_USE_PYTORCH "Whether to build with PyTorch" ON)
option(K2_ENABLE_BENCHMARK "Whether to enable benchmark" ON)
option(K2_WITH_CUDA "Whether to build k2 with CUDA" ${_K2_WITH_CUDA})

# If K2_WITH_CUDA is ON, then K2_ENABLE_NVTX has a default value ON
# If K2_WITH_CUDA is OFF, then K2_ENABLE_NVTX is set to OFF
include(CMakeDependentOption)
cmake_dependent_option(K2_ENABLE_NVTX "Whether to build with the NVTX library" ON K2_WITH_CUDA OFF)

if(NOT K2_USE_PYTORCH)
  message(FATAL_ERROR "\
    Please set K2_USE_PYTORCH to ON.
    Support for other frameworks will be added later")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(WIN32 AND BUILD_SHARED_LIBS)
  message(STATUS "Set BUILD_SHARED_LIBS to OFF for Windows")
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_RPATH "$ORIGIN")

if(UNIX AND NOT APPLE)
  execute_process(COMMAND
    lsb_release -sd
    OUTPUT_VARIABLE K2_OS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
elseif(APPLE)
  execute_process(COMMAND
    sw_vers -productName
    OUTPUT_VARIABLE _product_name
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(COMMAND
    sw_vers -productVersion
    OUTPUT_VARIABLE _product_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(COMMAND
    sw_vers -buildVersion
    OUTPUT_VARIABLE _build_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(K2_OS "${_product_name} ${_product_version} ${_build_version}")
elseif(WIN32)
  execute_process(COMMAND
    wmic os get caption,version
    OUTPUT_VARIABLE K2_OS_TWO_LINES
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  # Now K2_OS_TWO_LINES contains something like
  #  Caption                          Version
  #  Microsoft Windows 10 Pro         10.0.18362
  string(REPLACE "\n" ";" K2_OS_LIST ${K2_OS_TWO_LINES})
  list(GET K2_OS_LIST 1 K2_OS)
else()
  set(K2_OS "Unknown")
endif()

string(REGEX REPLACE "^\"+|\"+$" "" K2_OS "${K2_OS}")
message(STATUS "K2_OS: ${K2_OS}")

find_package(Git REQUIRED)
execute_process(COMMAND
  "${GIT_EXECUTABLE}" describe --always --abbrev=40
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE K2_GIT_SHA1
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(COMMAND
  "${GIT_EXECUTABLE}" log -1 --format=%ad --date=local
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE K2_GIT_DATE
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h K2_HAVE_CXXABI_H)
check_include_file_cxx(execinfo.h K2_HAVE_EXECINFO_H)

include(CheckCXXCompilerFlag)
if(NOT WIN32)
  check_cxx_compiler_flag("-std=c++14" K2_COMPILER_SUPPORTS_CXX14)
else()
  # windows x86 or x86_64
  check_cxx_compiler_flag("/std:c++14" K2_COMPILER_SUPPORTS_CXX14)
endif()
if(NOT K2_COMPILER_SUPPORTS_CXX14)
  message(FATAL_ERROR "
    k2 requires a compiler supporting at least C++14.
    If you are using GCC, please upgrade it to at least version 5.0.
    If you are using Clang, please upgrade it to at least version 3.4.")
endif()

# ========= Settings for CUB begin =========
# the following settings are modified from cub/CMakeLists.txt
set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")

if(K2_WITH_CUDA)
  # Force CUDA C++ standard to be the same as the C++ standard used.
  #
  # Now, CMake is unaligned with reality on standard versions: https://gitlab.kitware.com/cmake/cmake/issues/18597
  # which means that using standard CMake methods, it's impossible to actually sync the CXX and CUDA versions for pre-11
  # versions of C++; CUDA accepts 98 but translates that to 03, while CXX doesn't accept 03 (and doesn't translate that to 03).
  # In case this gives You, dear user, any trouble, please escalate the above CMake bug, so we can support reality properly.
  if(DEFINED CMAKE_CUDA_STANDARD)
    message(WARNING "You've set CMAKE_CUDA_STANDARD; please note that this variable is ignored, and CMAKE_CXX_STANDARD"
      " is used as the C++ standard version for both C++ and CUDA.")
  endif()


  unset(CMAKE_CUDA_STANDARD CACHE)
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

  include(cmake/select_compute_arch.cmake)
  cuda_select_nvcc_arch_flags(K2_COMPUTE_ARCH_FLAGS)
  message(STATUS "K2_COMPUTE_ARCH_FLAGS: ${K2_COMPUTE_ARCH_FLAGS}")

  # set(K2_COMPUTE_ARCHS 30 32 35 50 52 53 60 61 62 70 72)
  # message(WARNING "arch 62/72 are not supported for now")

  # see https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  # https://www.myzhar.com/blog/tutorials/tutorial-nvidia-gpu-cuda-compute-capability/
  set(K2_COMPUTE_ARCH_CANDIDATES 35 50 60 61 70 75)
  if(CUDA_VERSION VERSION_GREATER "11.0")
    list(APPEND K2_COMPUTE_ARCH_CANDIDATES 80 86)
  endif()
  message(STATUS "K2_COMPUTE_ARCH_CANDIDATES ${K2_COMPUTE_ARCH_CANDIDATES}")

  set(K2_COMPUTE_ARCHS)

  foreach(COMPUTE_ARCH IN LISTS K2_COMPUTE_ARCH_CANDIDATES)
    if("${K2_COMPUTE_ARCH_FLAGS}" MATCHES ${COMPUTE_ARCH})
      message(STATUS "Adding arch ${COMPUTE_ARCH}")
      list(APPEND K2_COMPUTE_ARCHS ${COMPUTE_ARCH})
    else()
      message(STATUS "Skipping arch ${COMPUTE_ARCH}")
    endif()
  endforeach()

  if(NOT K2_COMPUTE_ARCHS)
    set(K2_COMPUTE_ARCHS ${K2_COMPUTE_ARCH_CANDIDATES})
  endif()

  message(STATUS "K2_COMPUTE_ARCHS: ${K2_COMPUTE_ARCHS}")

  foreach(COMPUTE_ARCH IN LISTS K2_COMPUTE_ARCHS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -gencode arch=compute_${COMPUTE_ARCH},code=sm_${COMPUTE_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES "${COMPUTE_ARCH}-real;${COMPUTE_ARCH}-virtual;${CMAKE_CUDA_ARCHITECTURES}")
  endforeach()
# ========= Settings for CUB end=========
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(Valgrind)
if(Valgrind_FOUND)
  find_program(MEMORYCHECK_COMMAND NAMES ${Valgrind_EXECUTABLE})
  set(MEMORYCHECK_COMMAND_OPTIONS "--suppressions=${CMAKE_SOURCE_DIR}/scripts/valgrind.supp --leak-check=full")
  include(Dart)
  message(STATUS "To check memory, run `ctest -R <NAME> -D ExperimentalMemCheck`")
endif()
enable_testing()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(pybind11)
if(K2_USE_PYTORCH)
  add_definitions(-DK2_USE_PYTORCH)
  add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)
  include(torch)
endif()

if(K2_WITH_CUDA)
  add_definitions(-DK2_WITH_CUDA)
endif()

if(K2_WITH_CUDA AND CUDA_VERSION VERSION_LESS 11.0)
  # CUB is included in CUDA toolkit 11.0 and above
  include(cub)
endif()

if(K2_WITH_CUDA)
  include(moderngpu)
endif()

include(googletest)

if(K2_WITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -Wall --compiler-options -Wno-unknown-pragmas --compiler-options -Wno-strict-overflow")
  message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()

if(NOT K2_WITH_CUDA AND NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
endif()

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-overflow")
endif()

if(WIN32)
  # disable various warnings for MSVC
  # NOTE: Most of the warnings are from PyTorch C++ APIs
  # 4068: unknown pragma "unroll"
  # 4996: "getenv": This function is unsafe
  # 4224: conversion from 'int64_t' to 'int32_t', possible loss of data
  # 4099: type name first seen using 'class' now seen using 'struct'
  # 4267: conversion from 'size_t' to 'I', possible loss of data
  # 4305: truncation from 'int' to 'bool'
  # 4244: conversion from 'const M' to 'const FloatType'
  # 4624: destructor was implicitly defined as deleted
  # 4551: function call missing argument list
  # 4067: unexpected tokens following preprocessor directive
  # 4819: The file contains a character that cannot be presented in the current code page.
  # 4005: macro redefinition
  # 4722: destructor never returns
  # 4018: signed/unsigned mismatch
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068 /wd4996 /wd4224 /wd4099 /wd4267 /wd4305 /wd4244 /wd4624 /wd4551 /wd4067 /wd4819 /wd4005 /wd4722 /wd4018")
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

add_subdirectory(k2)
