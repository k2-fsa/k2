# it is located in k2/csrc/cmake/transform.cmake
include(transform)

# please keep the list sorted
set(torch_srcs
  fsa_class.cu
  ops.cu
  ragged_any.cu
  torch_utils.cu
)

if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE torch_srcs SRCS ${torch_srcs})
endif()

add_library(k2_torch ${torch_srcs})
target_link_libraries(k2_torch PUBLIC context)
target_link_libraries(k2_torch PUBLIC fsa)
target_include_directories(k2_torch PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(k2_torch PUBLIC ${CMAKE_BINARY_DIR})


#---------------------------- Test torch CUDA sources ----------------------------

# please sort the source files alphabetically
set(torch_cuda_test_srcs
  fsa_class_test.cu
)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE torch_cuda_test_srcs SRCS ${torch_cuda_test_srcs})
endif()

# utility function to add gtest
function(torch_add_cuda_test source)
  get_filename_component(target_name ${source} NAME_WE)
  add_executable(${target_name} "${source}")
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(${target_name}
    PRIVATE
    k2_torch
    gtest
  )

  # NOTE: We set the working directory here so that
  # it works also on windows. The reason is that
  # the required DLLs are inside ${TORCH_DIR}/lib
  # and they can be found by the exe if the current
  # working directory is ${TORCH_DIR}\lib
  add_test(NAME "Test.Cuda.${target_name}"
    COMMAND
    $<TARGET_FILE:${target_name}>
    WORKING_DIRECTORY ${TORCH_DIR}/lib
  )
endfunction()

foreach(source IN LISTS torch_cuda_test_srcs)
  torch_add_cuda_test(${source})
endforeach()

