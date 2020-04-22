# Copyright 2020 Fangjun Kuang (csukuangfj@gmail.com)
# See ../COPYING for clarification regarding multiple authors

function(download_cpplint)
  include(ExternalProject)

  set(cpplint_URL "https://raw.githubusercontent.com/cpplint/cpplint/master/cpplint.py")
  set(cpplint_DIR "${k2_THIRD_PARTY_DIR}/cpplint")

  ExternalProject_Add(
    cpplint
    URL                 ${cpplint_URL}
    DOWNLOAD_NO_EXTRACT NO
    TIMEOUT             10
    PREFIX              ${cpplint_DIR}
    CONFIGURE_COMMAND   ""
    BUILD_COMMAND       ""
    INSTALL_COMMAND     ""
    TEST_COMMAND        ""
    LOG_DOWNLOAD        ON
    LOG_CONFIGURE       ON
  )

  ExternalProject_Get_Property(cpplint SOURCE_DIR)
  set(cpplint_SOURCE_DIR ${SOURCE_DIR} PARENT_SCOPE)
endfunction()

download_cpplint()

add_custom_target(check_style
  WORKING_DIRECTORY
    ${CMAKE_SOURCE_DIR}
  COMMAND
    ${CMAKE_SOURCE_DIR}/scripts/check_style_cpplint.sh ${CMAKE_BINARY_DIR} 1
)
add_dependencies(check_style cpplint)
