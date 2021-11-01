add_subdirectory(csrc)
add_subdirectory(python)

if(K2_USE_PYTORCH)
  if(TORCH_VERSION VERSION_GREATER_EQUAL 1.8 OR NOT K2_WITH_CUDA)
    add_subdirectory(torch)
  else()
    message(WARNING "Please use at least torch 1.8.0 when CUDA \
    is enabled - skipping compiling k2/torch")
  endif()
endif()
