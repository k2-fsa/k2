if (CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

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

message(STATUS "CMAKE_VERSION: ${CMAKE_VERSION}")

# see https://cmake.org/cmake/help/latest/policy/CMP0111.html
#
# This is to suppress the warnings for importing PyTorch.
if(POLICY CMP0111)
  cmake_policy(SET CMP0111 OLD)
endif()
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_DISABLE_FIND_PACKAGE_MKL TRUE)

# Build with HIP for AMD GPUs (ROCm). When ON, the CUDA toolchain is not used;
# the .cu sources are compiled as HIP (see the K2_WITH_HIP blocks below) and the
# two non-portable third-party deps (moderngpu, the vendored CUDPP segmented
# scan) are replaced by hipCUB/rocThrust. See k2/csrc/cuda_to_hip.h.
option(K2_WITH_HIP "Build k2 with HIP for AMD GPUs" OFF)

set(languages CXX)
set(_K2_WITH_CUDA ON)

if(K2_WITH_HIP)
  # On the HIP build the CUDA toolchain is irrelevant.
  set(_K2_WITH_CUDA OFF)
else()
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
endif()

if(_K2_WITH_CUDA)
  set(languages ${languages} CUDA)
endif()

if(K2_WITH_HIP)
  # Add HIP to the project() language list (mirrors how CUDA is added) so the
  # compiler is enabled in the initial project() call; enabling it afterwards via
  # enable_language() forces a mid-configure re-run that drops cache values. The
  # project() probe then honors a user-supplied -DCMAKE_HIP_ARCHITECTURES or, when
  # unset, auto-detects the host GPU; it is not pinned to any specific arch.
  set(languages ${languages} HIP)
endif()

message(STATUS "Enabled languages: ${languages}")

project(k2 ${languages})

if(K2_WITH_HIP)
  message(STATUS "CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
  find_package(hip REQUIRED)
  find_package(hipcub REQUIRED)
  find_package(rocprim REQUIRED)
  find_package(hiprand REQUIRED)
  # These must be found before find_package(Torch) so that Caffe2Targets.cmake
  # can resolve all ROCm interface targets it imports (hiprtc::hiprtc, roc::hipblas, etc.).
  find_package(hiprtc QUIET)
  find_package(hipblas QUIET)
  find_package(hipblaslt QUIET)
  find_package(hipsolver QUIET)
  find_package(hipsparse QUIET)
  find_package(rocblas QUIET)
  find_package(rocsolver QUIET)
  find_package(hipfft QUIET)
endif()

set(K2_VERSION "1.24.4")

# ----------------- Supported build types for K2 project -----------------
set(K2_ALLOWABLE_BUILD_TYPES Debug Release RelWithDebInfo MinSizeRel)
set(K2_DEFAULT_BUILD_TYPE "Release")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${K2_ALLOWABLE_BUILD_TYPES}")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # CMAKE_CONFIGURATION_TYPES: with config type values from other generators (IDE).
  message(STATUS "No CMAKE_BUILD_TYPE given, default to ${K2_DEFAULT_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${K2_DEFAULT_BUILD_TYPE}")
elseif(NOT CMAKE_BUILD_TYPE IN_LIST K2_ALLOWABLE_BUILD_TYPES)
  message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}, \
    choose one from ${K2_ALLOWABLE_BUILD_TYPES}")
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
option(K2_ENABLE_NVTX "Whether to build k2 with the NVTX library" ON)
option(K2_ENABLE_TESTS "Whether to build tests" ON)

# You have to enable this option if you will run k2 on a machine different from
# the one you used to build k2 and the two machines have different types of GPUs
#
# We set it to OFF by default since most users will use the same machine to build
# and run k2. If you enable it, it will take much longer to build k2 and the
# resulting shared library also becomes larger in filesize.
option(K2_BUILD_FOR_ALL_ARCHS "Whether to build k2 for all GPU architectures" OFF)

if(NOT K2_WITH_CUDA)
  message(STATUS "Set K2_ENABLE_NVTX to OFF since K2_WITH_CUDA is OFF")
  set(K2_ENABLE_NVTX OFF CACHE BOOL "" FORCE)
endif()

if(NOT K2_USE_PYTORCH)
  message(FATAL_ERROR "\
    Please set K2_USE_PYTORCH to ON.
    Support for other frameworks will be added later")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(BUILD_SHARED_LIBS AND (MSVC OR (WIN32 AND K2_WITH_HIP)))
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(k2_rpath_origin "$ORIGIN")
else()
  set(k2_rpath_origin "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${k2_rpath_origin})
set(CMAKE_BUILD_RPATH ${k2_rpath_origin})

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
  # Try PowerShell first to get OS name + version
  execute_process(
    COMMAND powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).Caption + ' ' + (Get-CimInstance Win32_OperatingSystem).Version"
    OUTPUT_VARIABLE K2_ONNX_OS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  if(NOT K2_ONNX_OS)
    message(WARNING "PowerShell not available, falling back to cmd /c ver")
    # Fallback: cmd.exe /c ver (only version info, less detailed)
    execute_process(
      COMMAND cmd /c ver
      OUTPUT_VARIABLE _cmd_out
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    string(REPLACE "\r" "" _cmd_out "${_cmd_out}")
    if(_cmd_out)
      set(K2_ONNX_OS "Windows ${_cmd_out}")
    else()
      set(K2_ONNX_OS "Windows (version unknown)")
    endif()
  endif()
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

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
endif()
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")

include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h K2_HAVE_CXXABI_H)
check_include_file_cxx(execinfo.h K2_HAVE_EXECINFO_H)

include(CheckCXXCompilerFlag)
if(NOT WIN32)
  check_cxx_compiler_flag("-std=c++${CMAKE_CXX_STANDARD}" K2_COMPILER_SUPPORTS_CXX${CMAKE_CXX_STANDARD})
else()
  # windows x86 or x86_64
  check_cxx_compiler_flag("/std:c++${CMAKE_CXX_STANDARD}" K2_COMPILER_SUPPORTS_CXX${CMAKE_CXX_STANDARD})
endif()
if(NOT K2_COMPILER_SUPPORTS_CXX${CMAKE_CXX_STANDARD})
  message(FATAL_ERROR "
    k2 requires a compiler supporting at least C++${CMAKE_CXX_STANDARD}.
    If you are using GCC, please upgrade it to at least version 7.0.
    If you are using Clang, please upgrade it to at least version 3.4.")
endif()

# ========= Settings for CUB begin =========
# the following settings are modified from cub/CMakeLists.txt

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
  message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")

  # set(K2_COMPUTE_ARCHS 30 32 35 50 52 53 60 61 62 70 72)
  # message(WARNING "arch 62/72 are not supported for now")

  # see https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  # https://www.myzhar.com/blog/tutorials/tutorial-nvidia-gpu-cuda-compute-capability/
  set(K2_COMPUTE_ARCH_CANDIDATES 35 50 60 61 70 75)
  if(CUDA_VERSION VERSION_GREATER "11.0")
      list(APPEND K2_COMPUTE_ARCH_CANDIDATES 80 86)
      if(WIN32)
        # To fix the following warning from PyTorch:
        # c10/util/TypeCast.h(39): warning : calling a constexpr __host__ function from a
        # __host__ __device__ function is not allowed. The experimental flag '--expt-relaxed-constexpr'
        # can be used to allow this
        string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr ")
      endif()

      string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets ")
  endif()

  if(CUDA_VERSION VERSION_GREATER_EQUAL "11.8")
    # https://docs.nvidia.com/cuda/archive/11.8.0/pdf/CUDA_Toolkit_Release_Notes.pdf
    # added support for Hopper (90)
    list(APPEND K2_COMPUTE_ARCH_CANDIDATES 90)
  endif()

  if(CUDA_VERSION VERSION_GREATER_EQUAL "12.0")
    # https://docs.nvidia.com/cuda/archive/12.0.0/pdf/CUDA_Toolkit_Release_Notes.pdf
    # support for 35, 37 compute capabilities removed in CUDA 12.0
    list(REMOVE_ITEM K2_COMPUTE_ARCH_CANDIDATES 35 37)
  endif()

  if(CUDA_VERSION VERSION_GREATER_EQUAL "12.6")
    message(WARNING "Use NVTX3 for ${CUDA_VERSION}")
    add_definitions(-DK2_USE_NVTX3)
  else()
    message(WARNING "Use NVTX2 for ${CUDA_VERSION}")
  endif()

  if(CUDA_VERSION VERSION_GREATER_EQUAL "13.0")
    list(REMOVE_ITEM K2_COMPUTE_ARCH_CANDIDATES 50 60 61 70)

    list(APPEND K2_COMPUTE_ARCH_CANDIDATES 100 120 121)
  endif()

  message(STATUS "K2_COMPUTE_ARCH_CANDIDATES ${K2_COMPUTE_ARCH_CANDIDATES}")

  if(NOT K2_BUILD_FOR_ALL_ARCHS)
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
  else()
      message(STATUS "Build k2 for all GPU architectures")
      set(K2_COMPUTE_ARCHS ${K2_COMPUTE_ARCH_CANDIDATES})
  endif()

  message(STATUS "K2_COMPUTE_ARCHS: ${K2_COMPUTE_ARCHS}")

  # Flags for moderngpu
  set(K2_MODERN_GPU_FLAGS " -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w ")

  foreach(COMPUTE_ARCH IN LISTS K2_COMPUTE_ARCHS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${K2_MODERN_GPU_FLAGS} --expt-extended-lambda -gencode arch=compute_${COMPUTE_ARCH},code=sm_${COMPUTE_ARCH}")
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

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)


include(pybind11)
if(K2_USE_PYTORCH)
  add_definitions(-DK2_USE_PYTORCH)
  include(torch)
  configure_file(
    ${PROJECT_SOURCE_DIR}/k2/python/k2/torch_version.py.in
    ${PROJECT_SOURCE_DIR}/k2/python/k2/torch_version.py @ONLY
  )
  message(STATUS "Generated ${PROJECT_BINARY_DIR}/torch_version.py")
endif()

if(K2_WITH_CUDA)
  add_definitions(-DK2_WITH_CUDA)
endif()

if(K2_WITH_HIP)
  # K2_WITH_HIP/USE_HIP gate the genuinely divergent code; K2_WITH_CUDA is ALSO
  # defined as a compile macro so the many `#ifdef K2_WITH_CUDA` blocks that hold
  # shared GPU/driver/torch code (c10::cuda bridge, cub.h, rand.cu, eval.h CG)
  # compile -- cuda_to_hip.h retargets the cuda*/curand*/cub spellings to HIP.
  # The CMake OPTION K2_WITH_CUDA stays OFF (no nvcc arch gencode, no moderngpu
  # fetch, NVTX auto-off). (AutoDock-GPU lesson.)
  add_definitions(-DK2_WITH_HIP)
  add_definitions(-DUSE_HIP)
  add_definitions(-DK2_WITH_CUDA)

  # Force-include the single compat header on every HIP TU so its aliases
  # precede all other includes regardless of per-file include order.
  string(APPEND CMAKE_HIP_FLAGS
    " -include ${CMAKE_SOURCE_DIR}/k2/csrc/cuda_to_hip.h")

  # Vendored ROCm/libhipcxx supplies <cuda/std/*> (ROCm ships no cuda/std), used
  # by cub.h and utils_inl.h (cuda::std::plus). Point at it if present.
  if(DEFINED K2_LIBHIPCXX_INCLUDE_DIR AND
     EXISTS "${K2_LIBHIPCXX_INCLUDE_DIR}/cuda/std/functional")
    message(STATUS "libhipcxx include dir: ${K2_LIBHIPCXX_INCLUDE_DIR}")
    include_directories(SYSTEM ${K2_LIBHIPCXX_INCLUDE_DIR})
  else()
    message(FATAL_ERROR
      "K2_WITH_HIP=ON requires ROCm/libhipcxx (provides <cuda/std/*>). Pass "
      "-DK2_LIBHIPCXX_INCLUDE_DIR=/path/to/libhipcxx/include")
  endif()
endif()

if(WIN32)
  add_definitions(-DNOMINMAX) # Otherwise, std::max() and std::min() won't work
endif()

if(K2_WITH_CUDA AND CUDA_VERSION VERSION_LESS 11.0)
  # CUB is included in CUDA toolkit 11.0 and above
  include(cub)
endif()

if(K2_WITH_CUDA)
  include(moderngpu)
endif()

if(K2_ENABLE_TESTS)
  enable_testing()
  include(googletest)
endif()


if(K2_WITH_CUDA AND NOT WIN32)
  string(APPEND CMAKE_CUDA_FLAGS " --compiler-options -Wall ")
  string(APPEND CMAKE_CUDA_FLAGS " --compiler-options -Wno-strict-overflow ")
  string(APPEND CMAKE_CUDA_FLAGS " --compiler-options -Wno-unknown-pragmas ")
  message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()

if(K2_WITH_HIP)
  # hipcc is clang; these are clang flags (no nvcc --compiler-options wrapper).
  string(APPEND CMAKE_HIP_FLAGS " -Wno-strict-overflow ")
  string(APPEND CMAKE_HIP_FLAGS " -Wno-unknown-pragmas ")
  string(APPEND CMAKE_HIP_FLAGS " -Wno-unused-variable ")
  message(STATUS "CMAKE_HIP_FLAGS: ${CMAKE_HIP_FLAGS}")
endif()


if(NOT WIN32)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-variable ")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-strict-overflow ")
endif()

if(WIN32 AND NOT K2_WITH_HIP)
  # disable various warnings for MSVC (not applicable to the clang HIP build)
  # NOTE: Most of the warnings are from PyTorch C++ APIs
  # 4005: macro redefinition
  # 4018: signed/unsigned mismatch
  # 4067: unexpected tokens following preprocessor directive
  # 4068: unknown pragma "unroll"
  # 4099: type name first seen using 'class' now seen using 'struct'
  # 4101: 'identifier' : unreferenced local variable
  # 4190: 'identifier1' has C-linkage specified, but returns UDT 'identifier2' which is incompatible with C
  # 4224: conversion from 'int64_t' to 'int32_t', possible loss of data
  # 4244: conversion from 'const M' to 'const FloatType'
  # 4251: 'type' : class 'type1' needs to have dll-interface to be used by clients of class 'type2'
  # 4267: conversion from 'size_t' to 'I', possible loss of data
  # 4275: non - DLL-interface class 'class_1' used as base for DLL-interface class 'class_2'
  # 4305: truncation from 'int' to 'bool'
  # 4522: 'class' : multiple assignment operators specified
  # 4551: function call missing argument list
  # 4624: destructor was implicitly defined as deleted
  # 4700: uninitialized local variable 'device' used
  # 4722: destructor never returns
  # 4805: '|': unsafe mix of type 'uintptr_t' and type 'bool' in operation
  # 4819: The file contains a character that cannot be presented in the current code page.
  # 4838: conversion from 'type_1' to 'type_2' requires a narrowing conversion
  # 4996: "getenv": This function is unsafe
  set(disabled_warnings
      /wd4005
      /wd4018
      /wd4067
      /wd4068
      /wd4099
      /wd4101
      /wd4190
      /wd4224
      /wd4244
      /wd4251
      /wd4267
      /wd4275
      /wd4305
      /wd4522
      /wd4551
      /wd4624
      /wd4700
      /wd4722
      /wd4805
      /wd4819
      /wd4838
      /wd4996
  )
  message(STATUS "Disabled warnings: ${disabled_warnings}")
  foreach(w IN LISTS disabled_warnings)
    string(APPEND CMAKE_CXX_FLAGS " ${w} ")
    string(APPEND CMAKE_CUDA_FLAGS " --compiler-options ${w} ")
  endforeach()
  string(APPEND CMAKE_CXX_FLAGS " /bigobj ")
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

include(GNUInstallDirs)

add_subdirectory(k2)

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/k2ConfigVersion.cmake.in
  ${PROJECT_BINARY_DIR}/k2ConfigVersion.cmake
  @ONLY
)

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/k2Config.cmake.in
  ${PROJECT_BINARY_DIR}/k2Config.cmake
  @ONLY
)

install(FILES
  ${PROJECT_BINARY_DIR}/k2ConfigVersion.cmake
  ${PROJECT_BINARY_DIR}/k2Config.cmake
  DESTINATION share/cmake/k2
)

install(FILES
  ${PROJECT_SOURCE_DIR}/k2/python/k2/torch_version.py
  DESTINATION ./
)
