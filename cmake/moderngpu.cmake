# Copyright (c)  2020  Mobvoi AI Lab, Beijing, China (authors: Fangjun Kuang)
# See ../LICENSE for clarification regarding multiple authors

function(download_moderngpu)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(moderngpu_URL  "https://github.com/moderngpu/moderngpu/archive/v2.12_june_8_2016.tar.gz")
  set(moderngpu_HASH "SHA256=dff7010c6ffa1081c93ba450d3c39e5942e6ccecf09e9abd644c002eeac0735a")

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
  target_compile_options(moderngpu INTERFACE -lineinfo --expt-extended-lambda -use_fast_math)
endfunction()

download_moderngpu()
