add_subdirectory(csrc)

if(K2_BUILD_TORCH_BIN)
  add_subdirectory(bin)
endif()
