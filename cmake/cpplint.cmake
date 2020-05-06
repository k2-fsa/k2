# Copyright (c) 2020 Fangjun Kuang (csukuangfj@gmail.com)
# See ../LICENSE for clarification regarding multiple authors

function(download_cpplint)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(cpplint_URL "https://github.com/cpplint/cpplint/archive/1.4.5.tar.gz")
  set(cpplint_HASH "SHA256=96db293564624543a2fd3b1a0d23f663b8054c79853a5918523655721a9f6b53")

  FetchContent_Declare(cpplint
    URL               ${cpplint_URL}
    URL_HASH          ${cpplint_HASH}
  )

  FetchContent_GetProperties(cpplint)
  if(NOT cpplint_POPULATED)
    message(STATUS "Downloading cpplint")
    FetchContent_Populate(cpplint)
  endif()
  message(STATUS "cpplint is downloaded to ${cpplint_SOURCE_DIR}")

  add_custom_target(check_style
    WORKING_DIRECTORY
      ${CMAKE_SOURCE_DIR}
    COMMAND
      ${CMAKE_SOURCE_DIR}/scripts/check_style_cpplint.sh ${CMAKE_BINARY_DIR} 1
  )
endfunction()

download_cpplint()
