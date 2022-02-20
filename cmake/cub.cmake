# Copyright     2020 Fangjun Kuang (csukuangfj@gmail.com)
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

function(download_cub)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(cub_URL  "https://github.com/NVlabs/cub/archive/1.15.0.tar.gz")
  set(cub_HASH "SHA256=1781ee5eb7f00acfee5bff88e3acfc67378f6b3c24281335e18ae19e1f2ff685")


  FetchContent_Declare(cub
    URL               ${cub_URL}
    URL_HASH          ${cub_HASH}
  )

  FetchContent_GetProperties(cub)
  if(NOT cub)
    message(STATUS "Downloading cub")
    FetchContent_Populate(cub)
  endif()
  message(STATUS "cub is downloaded to ${cub_SOURCE_DIR}")
  add_library(cub INTERFACE)
  target_include_directories(cub INTERFACE ${cub_SOURCE_DIR})
endfunction()

download_cub()
