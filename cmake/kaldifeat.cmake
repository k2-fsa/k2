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

function(download_kaldifeat)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    # FetchContent is available since 3.11,
    # we've copied it to ${CMAKE_SOURCE_DIR}/cmake/Modules
    # so that it can be used in lower CMake versions.
    message(STATUS "Use FetchContent provided by k2")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(kaldifeat_URL "https://github.com/csukuangfj/kaldifeat/archive/refs/tags/v1.20.tar.gz")
  set(kaldifeat_HASH "SHA256=509110abbb4bf510831a9abbf1f3e7a0768f9e505d7f25defeaf6545566e1aaf")

  # If you don't have access to the Internet, please download the file to your
  # local drive and use the line below (you need to change it accordingly.
  # I am placing it in /star-fj/fangjun/download/github, but you can place it
  # anywhere you like)
  if(EXISTS "/star-fj/fangjun/download/github/kaldifeat-1.20.tar.gz")
    set(kaldifeat_URL "file:///star-fj/fangjun/download/github/kaldifeat-1.20.tar.gz")
  elseif(EXISTS "/tmp/kaldifeat-1.20.tar.gz")
    set(kaldifeat_URL "file:///tmp/kaldifeat-1.20.tar.gz")
  endif()

  set(kaldifeat_BUILD_TESTS OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(kaldifeat
    URL               ${kaldifeat_URL}
    URL_HASH          ${kaldifeat_HASH}
  )

  FetchContent_GetProperties(kaldifeat)
  if(NOT kaldifeat_POPULATED)
    message(STATUS "Downloading kaldifeat from ${kaldifeat_URL}")
    FetchContent_Populate(kaldifeat)
  endif()
  message(STATUS "kaldifeat is downloaded to ${kaldifeat_SOURCE_DIR}")
  message(STATUS "kaldifeat's binary dir is ${kaldifeat_BINARY_DIR}")

  set(KALDIFEAT_TORCH_VERSION_MAJOR ${K2_TORCH_VERSION_MAJOR})
  set(KALDIFEAT_TORCH_VERSION_MINOR ${K2_TORCH_VERSION_MINOR})
  add_subdirectory(${kaldifeat_SOURCE_DIR} ${kaldifeat_BINARY_DIR} EXCLUDE_FROM_ALL)

  target_include_directories(kaldifeat_core PUBLIC ${kaldifeat_SOURCE_DIR})
endfunction()

download_kaldifeat()
