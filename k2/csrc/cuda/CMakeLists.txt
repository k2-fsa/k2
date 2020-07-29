
add_library(context context.cu)
target_include_directories(context PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(context PUBLIC cub)

function(k2_add_cuda_test name)
  add_executable(${name} "${name}.cu")
  target_link_libraries(${name}
    PRIVATE
      context
      gtest
      gtest_main
  )
  add_test(NAME "Test.Cuda.${name}"
    COMMAND
      $<TARGET_FILE:${name}>
  )
endfunction()

# please sort the source files alphabetically
set(cuda_tests
  utils_test
)

foreach(name IN LISTS cuda_tests)
  k2_add_cuda_test(${name})
endforeach()
