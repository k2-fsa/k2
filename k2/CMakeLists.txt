add_subdirectory(csrc)
add_subdirectory(python)

if(K2_USE_PYTORCH)
  # We use K2_TORCH_VERSION instead of TORCH_VERSION
  # since TORCH_VERSION may contain something like "+cpu", "+cu113"
  if(K2_TORCH_VERSION VERSION_GREATER_EQUAL 1.8 OR NOT K2_WITH_CUDA)
    message(STATUS "Including k2/torch. K2_TORCH_VERSION is ${K2_TORCH_VERSION}")
    include(kaldifeat)
    add_subdirectory(torch)
  else()
    message(WARNING "Please use at least torch 1.8.0 when CUDA \
    is enabled - skipping compiling k2/torch. Current torch version: ${TORCH_VERSION}")
  endif()
endif()
