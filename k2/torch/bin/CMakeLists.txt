# it is located in k2/csrc/cmake/transform.cmake
include(transform)

set(decode_srcs decode.cu)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE decode_srcs SRCS ${decode_srcs})
endif()

add_executable(decode ${decode_srcs})

set_property(TARGET decode PROPERTY CXX_STANDARD 14)
target_link_libraries(decode
  ${TORCH_LIBRARIES} # see cmake/torch.cmake
  context
  k2_torch
  sentencepiece-static # see cmake/sentencepiece.cmake
)
