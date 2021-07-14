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

if(MSVC AND K2_WITH_CUDA)
  # The `/bigobj` option is passed to NVCC and results in the following error:
  # nvcc fatal   : A single input file is required for a non-link phase when an outputfile is specified
  #
  # We remove it here
  get_property(old_flags TARGET pybind11::windows_extras PROPERTY INTERFACE_COMPILE_OPTIONS)
  string(REPLACE "/bigobj" "" new_flags ${old_flags})
  set_property(TARGET pybind11::windows_extras PROPERTY INTERFACE_COMPILE_OPTIONS ${new_flags})
endif()

pybind11_add_module(_k2 ${k2_srcs} SHARED)
target_link_libraries(_k2 PRIVATE context)
target_link_libraries(_k2 PRIVATE fsa)
target_include_directories(_k2 PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(_k2 PRIVATE ${CMAKE_BINARY_DIR})

if(K2_WITH_CUDA)
  set_target_properties(_k2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

