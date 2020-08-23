include_directories(${CMAKE_SOURCE_DIR})

function(k2_add_Cuda_test name)
  add_executable(${name} "${name}.cu")
  set_target_properties(${name} PROPERTIES
      CXX_STANDARD 14
      CUDA_ARCHITECTURES 61
      )
  target_compile_features(${name} PUBLIC cuda_std_14)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-extended-lambda
      >)
  target_link_libraries(${name} PRIVATE
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
    extended_lambda_test
    )

foreach (name IN LISTS cuda_tests)
  k2_add_Cuda_test(${name})
endforeach ()
