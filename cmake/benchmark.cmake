# Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
# See ../LICENSE for clarification regarding multiple authors

function(download_benchmark)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    # FetchContent is available since 3.11,
    # we've copied it to ${CMAKE_SOURCE_DIR}/cmake/Modules
    # so that it can be used in lower CMake versions.
    message(STATUS "Use FetchContent provided by k2")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(benchmark_URL  "https://github.com/google/benchmark/archive/v1.5.2.tar.gz")
  set(benchmark_HASH "SHA256=dccbdab796baa1043f04982147e67bb6e118fe610da2c65f88912d73987e700c")

  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(benchmark
    URL               ${benchmark_URL}
    URL_HASH          ${benchmark_HASH}
  )

  FetchContent_GetProperties(benchmark)
  if(NOT benchmark_POPULATED)
    message(STATUS "Downloading benchmark")
    FetchContent_Populate(benchmark)
  endif()
  message(STATUS "benchmark is downloaded to ${benchmark_SOURCE_DIR}")
  message(STATUS "benchmark's binary dir is ${benchmark_BINARY_DIR}")

  add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_benchmark()
