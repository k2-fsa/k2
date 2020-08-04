add_subdirectory(util)

# please sort the source files alphabetically
set(FSA_SOURCES
    arcsort.cc
    aux_labels.cc
    connect.cc
    determinize.cc
    fsa.cc
    fsa_equivalent.cc
    fsa_renderer.cc
    fsa_util.cc
    intersect.cc
    properties.cc
    rmepsilon.cc
    topsort.cc
    util.cc
    weights.cc)

add_library(fsa ${FSA_SOURCES})

target_include_directories(fsa PUBLIC ${CMAKE_SOURCE_DIR})

if(K2_USE_GLOG)
  target_compile_definitions(fsa PUBLIC K2_USE_GLOG)
  target_link_libraries(fsa PUBLIC glog)
else()
  target_link_libraries(fsa PUBLIC non-glog)
endif()

function(k2_add_fsa_test name)
  add_executable(${name} "${name}.cc")
  target_link_libraries(${name}
    PRIVATE
      fsa
      gtest
      gtest_main
  )
  add_test(NAME "Test.${name}"
    COMMAND
      $<TARGET_FILE:${name}>
  )
endfunction()

# please sort the source files alphabetically
set(fsa_tests
  arcsort_test
  array_test
  aux_labels_test
  connect_test
  determinize_test
  fsa_equivalent_test
  fsa_renderer_test
  fsa_test
  fsa_util_test
  intersect_test
  properties_test
  rmepsilon_test
  topsort_test
  weights_test
)

foreach(name IN LISTS fsa_tests)
  k2_add_fsa_test(${name})
endforeach()
