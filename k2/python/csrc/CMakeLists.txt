# please keep the list sorted
set(k2_srcs
  k2.cu
  torch.cu
)

if(USE_PYTORCH)
  add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)
  add_subdirectory(torch)
  set(k2_srcs ${k2_srcs} ${torch_srcs})
  set(k2_deps
    ${TORCH_LIBRARIES}
    ${TORCH_DIR}/lib/libtorch_python.so
  )
else()
  message(FATAL_ERROR "Please select a framework.")
endif()

pybind11_add_module(_k2 ${k2_srcs})
target_link_libraries(_k2 PRIVATE ${k2_deps})
target_link_libraries(_k2 PRIVATE context)
target_link_libraries(_k2 PRIVATE fsa)
target_include_directories(_k2 PRIVATE ${CMAKE_SOURCE_DIR})
