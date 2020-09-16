# Make project root dir as the INCLUDE_DIRECTORIES of all tagets in csrc/.
# The header should be included as `#include "k2/csrc/.."`, to avoid conflicts.
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(host)

#---------------------------- Build K2 CUDA sources ----------------------------

# the target
add_library(context STATIC
    array_ops.cu
#    compose.cu
    context.cu
    dtype.cu
    fsa.cu
#    fsa_algo.cu
    math.cu
    ragged.cu
    tensor.cu
    utils.cu
    )
set_target_properties(context PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# lib deps
target_link_libraries(context PUBLIC cub)

#---------------------------- Test K2 CUDA sources ----------------------------

# please sort the source files alphabetically
set(cuda_tests
    array_ops_test
    array_test
    log_test
    ragged_shape_test
    ragged_test
    tensor_test
    utils_test
    )

# utility function to add gtest
function(k2_add_cuda_test name)
  # TODO(haowen): add prefix `cu` for now to avoid name conflicts
  # with files in k2/csrc/, will remove this finally.
  set(target_name "cu_${name}")
  add_executable(${target_name} "${name}.cu")
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
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

foreach (name IN LISTS cuda_tests)
  k2_add_cuda_test(${name})
endforeach ()
