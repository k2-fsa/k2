# Copyright (c) 2020 Fangjun Kuang (csukuangfj@gmail.com)
# See ../LICENSE for clarification regarding multiple authors

function(download_glog)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(glog_URL  "https://github.com/google/glog/archive/v0.4.0.tar.gz")
  set(glog_HASH "SHA256=f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c")

  set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(glog_glog
    URL               ${glog_URL}
    URL_HASH          ${glog_HASH}
    PATCH_COMMAND     sed -i.bak "/include \(CTest\)/d" CMakeLists.txt
  )

  FetchContent_GetProperties(glog_glog)
  if(NOT glog_glog_POPULATED)
    message(STATUS "Downloading glog")
    FetchContent_Populate(glog_glog)
  endif()
  message(STATUS "glog is downloaded to ${glog_SOURCE_DIR}")
  message(STATUS "glog's binary dir is ${glog_BINARY_DIR}")

  add_subdirectory(${glog_glog_SOURCE_DIR} ${glog_glog_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_glog()
