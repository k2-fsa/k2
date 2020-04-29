add_library(properties properties.cc)
target_include_directories(properties PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(properties PUBLIC cxx_std_11)
target_link_libraries(properties PUBLIC fsa_algo)

add_library(fsa_equivalent fsa_equivalent.cc)
target_include_directories(fsa_equivalent PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(fsa_equivalent PUBLIC cxx_std_11)
target_link_libraries(fsa_equivalent PUBLIC properties)

add_library(fsa_util fsa_util.cc)
target_include_directories(fsa_util PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(fsa_util PUBLIC cxx_std_11)

add_library(fsa_renderer fsa_renderer.cc)
target_include_directories(fsa_renderer PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(fsa_renderer PUBLIC cxx_std_11)

add_library(fsa_algo fsa_algo.cc)
target_include_directories(fsa_algo PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(fsa_algo PUBLIC cxx_std_11)
target_link_libraries(fsa_algo PRIVATE glog)

add_executable(properties_test properties_test.cc)

target_link_libraries(properties_test
  PRIVATE
    properties
    fsa_algo
    gtest
    gtest_main
)

add_test(NAME Test.properties_test
  COMMAND
    $<TARGET_FILE:properties_test>
)

add_executable(fsa_equivalent_test fsa_equivalent_test.cc)

target_link_libraries(fsa_equivalent_test
  PRIVATE
    fsa_equivalent
    gtest
    gtest_main
)

add_test(NAME Test.fsa_equivalent_test
  COMMAND
    $<TARGET_FILE:fsa_equivalent_test>
)

add_executable(fsa_util_test fsa_util_test.cc)

target_link_libraries(fsa_util_test
  PRIVATE
    fsa_util
    gtest
    gtest_main
)

add_test(NAME Test.fsa_util_test
  COMMAND
    $<TARGET_FILE:fsa_util_test>
)

add_executable(fsa_renderer_test fsa_renderer_test.cc)

target_link_libraries(fsa_renderer_test
  PRIVATE
    fsa_renderer
    gtest
    gtest_main
)

add_test(NAME Test.fsa_renderer_test
  COMMAND
    $<TARGET_FILE:fsa_renderer_test>
)

add_executable(fsa_algo_test fsa_algo_test.cc)

target_link_libraries(fsa_algo_test
  PRIVATE
    fsa_algo
    gtest
    gtest_main
)

add_test(NAME Test.fsa_algo_test
  COMMAND
    $<TARGET_FILE:fsa_algo_test>
)

# TODO(fangjun): write some helper functions to create targets.
