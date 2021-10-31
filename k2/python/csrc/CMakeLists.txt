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
