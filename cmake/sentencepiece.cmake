# Copyright     2021 Fangjun Kuang (csukuangfj@gmail.com)
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

function(download_sentencepiece)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    # FetchContent is available since 3.11,
    # we've copied it to ${CMAKE_SOURCE_DIR}/cmake/Modules
    # so that it can be used in lower CMake versions.
    message(STATUS "Use FetchContent provided by k2")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(sentencepiece_URL "https://github.com/google/sentencepiece/archive/refs/tags/v0.1.96.tar.gz")
  set(sentencepiece_HASH "SHA256=5198f31c3bb25e685e9e68355a3bf67a1db23c9e8bdccc33dc015f496a44df7a")

  set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(sentencepiece
    URL               ${sentencepiece_URL}
    URL_HASH          ${sentencepiece_HASH}
  )

  FetchContent_GetProperties(sentencepiece)
  if(NOT sentencepiece_POPULATED)
    message(STATUS "Downloading sentencepiece from ${sentencepiece_URL}")
    FetchContent_Populate(sentencepiece)
  endif()
  message(STATUS "sentencepiece is downloaded to ${sentencepiece_SOURCE_DIR}")
  message(STATUS "sentencepiece's binary dir is ${sentencepiece_BINARY_DIR}")

  add_subdirectory(${sentencepiece_SOURCE_DIR} ${sentencepiece_BINARY_DIR} EXCLUDE_FROM_ALL)

  # Link to sentencepiece statically
  target_include_directories(sentencepiece-static
    INTERFACE
      ${sentencepiece_SOURCE_DIR}
      ${sentencepiece_SOURCE_DIR}/src
      ${sentencepiece_BINARY_DIR}
  )
endfunction()

download_sentencepiece()
