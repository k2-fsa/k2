add_executable(decode decode.cc)
set_property(TARGET decode PROPERTY CXX_STANDARD 14)
target_link_libraries(decode
  ${TORCH_LIBRARIES} # see cmake/torch.cmake
  context
  k2_torch
  sentencepiece # see cmake/sentencepiece.cmake
)
