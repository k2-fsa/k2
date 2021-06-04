# Copyright (c)  2020  Mobvoi AI Lab, Beijing, China (authors: Fangjun Kuang)
# See ../LICENSE for clarification regarding multiple authors

function(download_moderngpu)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  # this is the latest commit of modern gpu as of 2020-09-26
  set(moderngpu_URL  "https://github.com/moderngpu/moderngpu/archive/2b3985541c8e88a133769598c406c33ddde9d0a5.zip")
  set(moderngpu_HASH "SHA256=191546af18cd5fb858ecb561316f3af67537ab16f610fc8f1a5febbffc27755a")

  FetchContent_Declare(moderngpu
    URL               ${moderngpu_URL}
    URL_HASH          ${moderngpu_HASH}
    PATCH_COMMAND     git init && git apply ${CMAKE_CURRENT_LIST_DIR}/moderngpu.patch
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
