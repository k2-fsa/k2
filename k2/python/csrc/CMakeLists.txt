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

pybind11_add_module(_k2 ${k2_srcs})
target_link_libraries(_k2 PRIVATE context)
target_link_libraries(_k2 PRIVATE fsa)
target_include_directories(_k2 PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(_k2 PRIVATE ${CMAKE_BINARY_DIR})
