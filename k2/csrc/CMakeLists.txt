add_library(properties properties.cc)
target_include_directories(properties PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(properties PUBLIC cxx_std_11)

add_library(fsa_util fsa_util.cc)
target_include_directories(fsa_util PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(fsa_util PUBLIC cxx_std_11)

add_executable(properties_test properties_test.cc)

target_link_libraries(properties_test
  PRIVATE
    properties
    gtest
    gtest_main
)

add_test(NAME Test.properties_test
  COMMAND
    $<TARGET_FILE:properties_test>
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

# TODO(fangjun): write some helper functions to create targets.
