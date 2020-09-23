# Copyright (c) 2020 Fangjun Kuang (csukuangfj@gmail.com)
# See ../LICENSE for clarification regarding multiple authors

function(download_pybind11)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(pybind11_URL  "https://github.com/pybind/pybind11/archive/v2.5.0.tar.gz")
  set(pybind11_HASH "SHA256=97504db65640570f32d3fdf701c25a340c8643037c3b69aec469c10c93dc8504")

  set(double_quotes "\"")
  set(dollar "\$")
  set(semicolon "\;")
  FetchContent_Declare(pybind11
    URL               ${pybind11_URL}
    URL_HASH          ${pybind11_HASH}
    PATCH_COMMAND
      sed -i s/\\${double_quotes}-flto\\\\${dollar}/\\${double_quotes}-Xcompiler=-flto${dollar}/g "tools/pybind11Tools.cmake" &&
      sed -i s/${seimcolon}-fno-fat-lto-objects/${seimcolon}-Xcompiler=-fno-fat-lto-objects/g "tools/pybind11Tools.cmake"
  )

  FetchContent_GetProperties(pybind11)
  if(NOT pybind11_POPULATED)
    message(STATUS "Downloading pybind11")
    FetchContent_Populate(pybind11)
  endif()
  message(STATUS "pybind11 is downloaded to ${pybind11_SOURCE_DIR}")
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_pybind11()
