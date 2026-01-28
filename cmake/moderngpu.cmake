# Copyright      2020  Mobvoi AI Lab, Beijing, China (authors: Fangjun Kuang)
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function(download_moderngpu)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  # this is the latest commit of modern gpu as of 2022-04-03
  set(moderngpu_URL  "https://github.com/moderngpu/moderngpu/archive/8ec9ac0de8672de7217d014917eedec5317f75f3.zip")
  set(moderngpu_URL2 "https://hub.nuaa.cf/moderngpu/moderngpu/archive/8ec9ac0de8672de7217d014917eedec5317f75f3.zip")
  set(moderngpu_HASH "SHA256=1c20ffbb81d6f7bbe6107aaa5ee6d37392677c8a5fc7894935149c3ef0a3c2fb")

  # If you don't have access to the Internet,
  # please pre-download moderngpu
  set(possible_file_locations
    $ENV{HOME}/Downloads/moderngpu-8ec9ac0de8672de7217d014917eedec5317f75f3.zip
    ${CMAKE_SOURCE_DIR}/moderngpu-8ec9ac0de8672de7217d014917eedec5317f75f3.zip
    ${CMAKE_BINARY_DIR}/moderngpu-8ec9ac0de8672de7217d014917eedec5317f75f3.zip
    /tmp/moderngpu-8ec9ac0de8672de7217d014917eedec5317f75f3.zip
    /star-fj/fangjun/download/github/moderngpu-8ec9ac0de8672de7217d014917eedec5317f75f3.zip
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(moderngpu_URL  "file://${f}")
      set(moderngpu_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(moderngpu
    URL
      ${moderngpu_URL}
      ${moderngpu_URL2}
    URL_HASH          ${moderngpu_HASH}
  )

  FetchContent_GetProperties(moderngpu)
  if(NOT moderngpu)
    message(STATUS "Downloading moderngpu from ${moderngpu_URL}")
    FetchContent_Populate(moderngpu)
  endif()
  message(STATUS "moderngpu is downloaded to ${moderngpu_SOURCE_DIR}")

  # Patch moderngpu for CUDA 13 where cudaDeviceProp no longer has clockRate/memoryClockRate.
  set(mgpu_context_path "${moderngpu_SOURCE_DIR}/src/moderngpu/context.hxx")
  if(EXISTS "${mgpu_context_path}")
    file(READ "${mgpu_context_path}" mgpu_context_contents)
    if(mgpu_context_contents MATCHES "memoryClockRate" AND NOT mgpu_context_contents MATCHES "cudaDeviceGetAttribute")
      string(REPLACE "#include <exception>"
                     "#include <exception>\n#include <cuda_runtime_api.h>"
                     mgpu_context_contents "${mgpu_context_contents}")
      string(REPLACE
        "  double memBandwidth = (prop.memoryClockRate * 1000.0) *\n    (prop.memoryBusWidth / 8 * 2) / 1.0e9;\n"
        "  int clock_khz = 0;\n  int mem_clock_khz = 0;\n#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000\n  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, ordinal);\n  cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, ordinal);\n#else\n  clock_khz = prop.clockRate;\n  mem_clock_khz = prop.memoryClockRate;\n#endif\n\n  double memBandwidth = (mem_clock_khz * 1000.0) *\n    (prop.memoryBusWidth / 8 * 2) / 1.0e9;\n"
        mgpu_context_contents "${mgpu_context_contents}")
      string(REPLACE
        "    prop.name, prop.clockRate / 1000.0, ordinal,\n"
        "    prop.name, clock_khz / 1000.0, ordinal,\n"
        mgpu_context_contents "${mgpu_context_contents}")
      string(REPLACE
        "    prop.memoryClockRate / 1000.0, prop.memoryBusWidth, memBandwidth,\n"
        "    mem_clock_khz / 1000.0, prop.memoryBusWidth, memBandwidth,\n"
        mgpu_context_contents "${mgpu_context_contents}")
      file(WRITE "${mgpu_context_path}" "${mgpu_context_contents}")
      message(STATUS "Patched moderngpu context.hxx for CUDA 13 clock attributes")
    endif()
  endif()

  add_library(moderngpu INTERFACE)
  target_include_directories(moderngpu INTERFACE ${moderngpu_SOURCE_DIR}/src)
endfunction()

download_moderngpu()
