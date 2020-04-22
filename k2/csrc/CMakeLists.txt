cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
project(fsa_library)

add_library(fsa_library SHARED
   properties.cc 
)

#TODO: replace with Google Testing Framework
add_executable(test_properties
    test/properties_test.cc
)

target_link_libraries(test_properties
    PRIVATE
        fsa_library
)
