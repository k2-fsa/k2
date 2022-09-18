cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(test_k2_torch_api)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(pybind11)
include(torch)
include(k2)

enable_testing()
include(googletest)

add_executable(torch_api_test torch_api_test.cc)
target_link_libraries(torch_api_test
  PRIVATE
      gtest
      gtest_main
      ${TORCH_LIBRARIES}
      ${K2_LIBRARIES}
)

# NOTE: We set the working directory here so that
# it works also on windows. The reason is that
# the required DLLs are inside ${TORCH_DIR}/lib
# and they can be found by the exe if the current
# working directory is ${TORCH_DIR}\lib
add_test(NAME "Test.torch_api_test"
  COMMAND
  $<TARGET_FILE:torch_api_test>
  WORKING_DIRECTORY ${TORCH_DIR}/lib
)
