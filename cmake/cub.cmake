# Copyright (c) 2020 Fangjun Kuang (csukuangfj@gmail.com)
# See ../LICENSE for clarification regarding multiple authors

function(download_cub)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(cub_URL  "https://github.com/NVlabs/cub/archive/1.9.10.tar.gz")
  set(cub_HASH "SHA256=2bd7077a3d9741f0689e6c1eb58c6278fc96eccc27d964168bc8be1bc3a9040f")

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
