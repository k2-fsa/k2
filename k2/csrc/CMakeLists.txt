# Make project root dir as the INCLUDE_DIRECTORIES of all tagets in csrc/.
# The header should be included as `#include "k2/csrc/.."`, to avoid conflicts.
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(host)

#---------------------------- Build K2 CUDA sources ----------------------------

# please keep it sorted
set(context_srcs
  algorithms.cu
  array_ops.cu
  intersect.cu
  top_sort.cu
  context.cu
  dtype.cu
  fsa.cu
  fsa_algo.cu
  fsa_utils.cu
  host_shim.cu
  math.cu
  moderngpu_allocator.cu
  ragged.cu
  ragged_ops.cu
  tensor.cu
  tensor_ops.cu
  utils.cu
)

if(USE_PYTORCH)
  list(APPEND context_srcs pytorch_context.cu)
else()
  list(APPEND context_srcs default_context.cu)
endif()

# the target
add_library(context SHARED ${context_srcs})
set_target_properties(context PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(context PROPERTIES OUTPUT_NAME "k2context")


# lib deps
target_link_libraries(context PUBLIC cub)
target_link_libraries(context PUBLIC fsa)
target_link_libraries(context PUBLIC moderngpu)
if(USE_PYTORCH)
  target_link_libraries(context PUBLIC ${TORCH_LIBRARIES})
endif()

#---------------------------- Test K2 CUDA sources ----------------------------

add_library(test_utils SHARED test_utils.cu)
target_link_libraries(test_utils PUBLIC context gtest)

# please sort the source files alphabetically
set(cuda_tests
  algorithms_test
  array_ops_test
  array_test
  fsa_algo_test
  fsa_test
  fsa_utils_test
  host_shim_test
  intersect_test
  log_test
  ragged_shape_test
  ragged_test
  tensor_test
  top_sort_test
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
    fsa  # for code in k2/csrc/host
    gtest
    gtest_main
    test_utils
  )
  add_test(NAME "Test.Cuda.${target_name}"
    COMMAND
    $<TARGET_FILE:${target_name}>
  )
endfunction()

foreach(name IN LISTS cuda_tests)
  k2_add_cuda_test(${name})
endforeach()
