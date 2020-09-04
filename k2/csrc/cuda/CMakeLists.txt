add_library(context 
  context.cu
  tensor.cu
  dtype.cc
)
target_include_directories(context PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(context PUBLIC cub)
target_link_libraries(context PUBLIC glog)

function(k2_add_cuda_test name)
  # TODO(haowen): add prefix `cu` for now to avoid name conflicts
  # with files in k2/csrc/, will remove this finally.
  set(target_name "cu_${name}")
  add_executable(${target_name} "${name}.cu")
  target_link_libraries(${target_name}
    PRIVATE
      context
      gtest
      gtest_main
  )
  add_test(NAME "Test.Cuda.${target_name}"
    COMMAND
      $<TARGET_FILE:${target_name}>
  )
endfunction()

# please sort the source files alphabetically
set(cuda_tests
  array_test
  ops_test
  tensor_test
  utils_test
  debug_test
)

foreach(name IN LISTS cuda_tests)
  k2_add_cuda_test(${name})
endforeach()
