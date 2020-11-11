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

project(k2 CUDA CXX)

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

# Add -O0 to remove optimizations for debugging.
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Whether to build shared or static lib" ON)
option(USE_PYTORCH "Whether to build with PyTorch" ON)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(WIN32 AND BUILD_SHARED_LIBS)
  message(STATUS "Set BUILD_SHARED_LIBS to OFF for Windows")
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
  set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
  set(CUDAToolkit_LIBRARY_DIR "" CACHE STRING "The cudatoolkit library directory.")
  set(CUDAToolkit_INCLUDE_DIRS "" CACHE STRING "The cudatoolkit include directory.")

  find_package(CUDAToolkit REQUIRED)
  if(CUDAToolkit_FOUND)
    message(STATUS "found CUDAToolkit " ${CUDAToolkit_LIBRARY_DIR})
    message(STATUS "CUDAToolkit_INCLUDE_DIRS " ${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "CUDAToolkit_LIBRARY_DIR " ${CUDAToolkit_LIBRARY_DIR})

    enable_language(CUDA)

    # With many architectures set here, the nvcc build time would be much longer.
    # Thus, "61" is put here for speed and compatibility.
    # @ToDo Need to cover more architectures, before release these code.
    set(CMAKE_CUDA_ARCHITECTURES 61)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  endif()
else()
  # the following settings are modified from cub/CMakeLists.txt
  set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
  set(CMAKE_CXX_EXTENSIONS OFF)

  message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")

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

  # set(K2_COMPUTE_ARCHS 30 32 35 50 52 53 60 61 62 70 72)
  message(WARNING "arch 62/72 are not supported for now")
  set(K2_COMPUTE_ARCHS 35 50 60 61 70 75)
  foreach(COMPUTE_ARCH IN LISTS K2_COMPUTE_ARCHS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -gencode arch=compute_${COMPUTE_ARCH},code=sm_${COMPUTE_ARCH}")
  endforeach()
endif()

enable_testing()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(pybind11)
if(USE_PYTORCH)
  add_definitions(-DK2_USE_PYTORCH)
  include(torch)
endif()
include(cub)
include(moderngpu)
include(googletest)

add_subdirectory(k2)
