# it is located in k2/csrc/cmake/transform.cmake
include(transform)

# please keep the list sorted
set(torch_pybind_srcs
  any.cu
  arc.cu
  discounted_cum_sum.cu
  doc/doc.cu
  fsa.cu
  k2.cu
  nbest.cu
  ops.cu
  pybind_utils.cu
  ragged_shape.cu
  torch.cu
  version.cu
)

if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE torch_pybind_srcs SRCS ${torch_pybind_srcs})
endif()

pybind11_add_module(_k2 ${torch_pybind_srcs} SHARED)
target_link_libraries(_k2 PRIVATE k2_torch)
target_include_directories(_k2 PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(_k2 PRIVATE ${CMAKE_BINARY_DIR})
set_property(TARGET _k2 PROPERTY CXX_VISIBILITY_PRESET "default")

#---------------------------- Test torch CUDA sources ----------------------------

# please sort the source files alphabetically
set(torch_pybind_test_srcs
  pybind_utils_test.cu
)

if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE torch_pybind_test_srcs SRCS ${torch_pybind_test_srcs})
endif()

# utility function to add gtest
function(torch_add_pybind_test source)
  get_filename_component(target_name ${source} NAME_WE)
  add_executable(${target_name} "${source}")
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # TODO: Find more elegant way to choose pybind11
  target_include_directories(${target_name} PRIVATE "${CMAKE_BINARY_DIR}/_deps/pybind11-src/include")
  if (APPLE)
    target_link_libraries(${target_name} PRIVATE "${TORCH_DIR}/lib/libtorch_python.dylib")
  endif()
  target_link_libraries(${target_name}
    PRIVATE
    k2_torch
    _k2
    pybind11::embed
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

foreach(source IN LISTS torch_pybind_test_srcs)
  torch_add_pybind_test(${source})
endforeach()

