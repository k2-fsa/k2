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
  set(moderngpu_HASH "SHA256=1c20ffbb81d6f7bbe6107aaa5ee6d37392677c8a5fc7894935149c3ef0a3c2fb")

  FetchContent_Declare(moderngpu
    URL               ${moderngpu_URL}
    URL_HASH          ${moderngpu_HASH}
  )

  FetchContent_GetProperties(moderngpu)
  if(NOT moderngpu)
    message(STATUS "Downloading moderngpu")
    FetchContent_Populate(moderngpu)
  endif()
  message(STATUS "moderngpu is downloaded to ${moderngpu_SOURCE_DIR}")
  add_library(moderngpu INTERFACE)
  target_include_directories(moderngpu INTERFACE ${moderngpu_SOURCE_DIR}/src)
  target_compile_options(moderngpu INTERFACE -lineinfo --expt-extended-lambda -use_fast_math -Xptxas=-w)
endfunction()

download_moderngpu()
