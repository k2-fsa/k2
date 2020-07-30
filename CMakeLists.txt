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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Whether to build shared or static lib" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Set the build type. Available values are: Debug Release RelWithDebInfo MinSizeRel"
    FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel
  )
endif()

if(WIN32 AND BUILD_SHARED_LIBS)
  message(STATUS "Set BUILD_SHARED_LIBS to OFF for Windows")
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()

# the following settings are modified from cub/CMakeLists.txt
#[[ start settings for CUB ]]

set(CMAKE_CXX_STANDARD 11 CACHE STRING "The C++ version to be used.")
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

set(K2_COMPUTE_ARCHS 30 32 35 50 52 53 60 61 62 70 72)
foreach(COMPUTE_ARCH IN LISTS K2_COMPUTE_ARCHS)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${COMPUTE_ARCH},code=sm_${COMPUTE_ARCH}")
endforeach()

#[[ end settings for CUB ]]

enable_testing()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(cpplint)
include(glog)
include(googletest)
include(pybind11)
include(cub)

add_subdirectory(k2)
