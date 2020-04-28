# Copyright (c) 2020 Fangjun Kuang (csukuangfj@gmail.com)
# See ../LICENSE for clarification regarding multiple authors

function(download_glog)
  include(ExternalProject)

  set(glog_URL  "https://github.com/google/glog/archive/v0.4.0.tar.gz")
  set(glog_HASH "sha256=f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c")

  set(glog_DIR  "${k2_THIRD_PARTY_DIR}/glog")
  set(glog_INSTALL_DIR  "${glog_DIR}/install")

  if(WIN32)
    set(glog_libglog "${glog_INSTALL_DIR}/lib/glog.lib")
  else()
    set(glog_libglog "${glog_INSTALL_DIR}/lib/libglog.a")
  endif()

  set(glog_INCLUDE_DIR "${glog_INSTALL_DIR}/include")
  file(MAKE_DIRECTORY ${glog_INCLUDE_DIR})

  ExternalProject_Add(
    glog_glog
    URL               ${glog_URL}
    PREFIX            ${glog_DIR}
    INSTALL_DIR       ${glog_DIR/install}
    CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=${glog_INSTALL_DIR}
                      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                      -DWITH_GFLAGS=OFF
                      -DWITH_THREADS=ON
                      -DWITH_TLS=ON
                      -DBUILD_SHARED_LIBS=OFF
    LOG_DOWNLOAD      ON
    LOG_CONFIGURE     ON
  )

  add_library(glog STATIC IMPORTED GLOBAL)
  set_property(TARGET glog PROPERTY IMPORTED_LOCATION ${glog_libglog})
  target_include_directories(glog INTERFACE ${glog_INCLUDE_DIR})
  add_dependencies(glog glog_glog)
endfunction()

download_glog()
