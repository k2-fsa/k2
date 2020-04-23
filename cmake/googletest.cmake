# Copyright (c) 2020 Fangjun Kuang (csukuangfj@gmail.com)
# See ../LICENSE for clarification regarding multiple authors

function(download_googltest)
  include(ExternalProject)

  set(googletest_URL  "https://github.com/google/googletest/archive/release-1.10.0.tar.gz")
  set(googletest_HASH "sha256=9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb")
  set(googletest_DIR  "${k2_THIRD_PARTY_DIR}/googletest")
  set(googletest_INSTALL_DIR  "${googletest_DIR}/install")

  if(WIN32)
    set(googletest_libgtest "${googletest_INSTALL_DIR}/lib/gtest.lib")
    set(googletest_libgtest_main "${googletest_INSTALL_DIR}/lib/gtest_main.lib")
  else()
    set(googletest_libgtest "${googletest_INSTALL_DIR}/lib/libgtest.a")
    set(googletest_libgtest_main "${googletest_INSTALL_DIR}/lib/libgtest_main.a")
  endif()

  set(googletest_INCLUDE_DIR "${googletest_INSTALL_DIR}/include")

  # NOTE(fangjun): since "googletest_INCLUDE_DIR" is created during build time,
  # we create it manually so that it can be accessed during configuration time.
  file(MAKE_DIRECTORY ${googletest_INCLUDE_DIR})

  ExternalProject_Add(
    googletest
    URL               ${googletest_URL}
    PREFIX            ${googletest_DIR}
    INSTALL_DIR       ${googletest_DIR/install}
    CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=${googletest_INSTALL_DIR}
                      -DBUILD_GMOCK=ON
                      -Dgtest_disable_pthreads=ON
                      -Dgtest_force_shared_crt=ON
    LOG_DOWNLOAD      ON
    LOG_CONFIGURE     ON
  )
  add_library(gtest STATIC IMPORTED GLOBAL)
  set_property(TARGET gtest PROPERTY IMPORTED_LOCATION ${googletest_libgtest})
  target_include_directories(gtest INTERFACE ${googletest_INCLUDE_DIR})

  add_library(gtest_main STATIC IMPORTED GLOBAL)
  set_property(TARGET gtest_main PROPERTY IMPORTED_LOCATION ${googletest_libgtest_main})
  target_include_directories(gtest_main INTERFACE ${googletest_INCLUDE_DIR})

  add_dependencies(gtest googletest)
  add_dependencies(gtest_main googletest)
endfunction()

download_googltest()
