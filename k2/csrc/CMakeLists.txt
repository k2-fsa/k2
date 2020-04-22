add_library(properties properties.cc)
target_include_directories(properties PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_features(properties PUBLIC cxx_std_11)

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
