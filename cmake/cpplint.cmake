# Copyright (c) 2020 Fangjun Kuang (csukuangfj@gmail.com)
# See ../LICENSE for clarification regarding multiple authors

function(download_cpplint)
  include(ExternalProject)

  set(cpplint_URL "https://github.com/cpplint/cpplint/archive/1.4.5.tar.gz")
  set(cpplint_HASH "sha256= 96db293564624543a2fd3b1a0d23f663b8054c79853a5918523655721a9f6b53")

  set(cpplint_DIR "${k2_THIRD_PARTY_DIR}/cpplint")

  ExternalProject_Add(
    cpplint_py
    URL                 ${cpplint_URL}
    TIMEOUT             10
    PREFIX              ${cpplint_DIR}
    CONFIGURE_COMMAND   ""
    BUILD_COMMAND       ""
    INSTALL_COMMAND     ""
    TEST_COMMAND        ""
    LOG_DOWNLOAD        ON
    LOG_CONFIGURE       ON
  )
endfunction()

download_cpplint()

add_custom_target(check_style
  WORKING_DIRECTORY
    ${CMAKE_SOURCE_DIR}
  COMMAND
    ${CMAKE_SOURCE_DIR}/scripts/check_style_cpplint.sh ${CMAKE_BINARY_DIR} 1
)
add_dependencies(check_style cpplint_py)
