# it is located in k2/csrc/cmake/transform.cmake
include(transform)

# please keep the list sorted
set(k2_srcs
  k2.cu
  torch.cu
  version.cu
)

if(K2_USE_PYTORCH)
  add_subdirectory(torch)
  set(k2_srcs ${k2_srcs} ${torch_srcs})
else()
  message(FATAL_ERROR "Please select a framework.")
endif()

if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE k2_srcs SRCS ${k2_srcs})
endif()

pybind11_add_module(_k2 ${k2_srcs} SHARED)
target_link_libraries(_k2 PRIVATE context)
target_link_libraries(_k2 PRIVATE fsa)
target_include_directories(_k2 PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(_k2 PRIVATE ${CMAKE_BINARY_DIR})
set_property(TARGET _k2 PROPERTY CXX_VISIBILITY_PRESET "default")

#---------------------------- Test torch CUDA sources ----------------------------

# please sort the source files alphabetically
set(torch_cuda_test_srcs
  torch/v2/ragged_arc_test.cu
  torch/v2/pybind_utils_test.cu
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
    _k2
    context
    fsa
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

