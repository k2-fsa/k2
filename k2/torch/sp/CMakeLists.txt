# it is located in k2/csrc/cmake/transform.cmake
include(transform)

set(bin_dep_libs
  ${TORCH_LIBRARIES}
  k2_torch
  sentencepiece-static # see cmake/sentencepiece.cmake
)

#----------------------------------------
#       CTC decoding
#----------------------------------------
set(sp_ctc_decode_srcs sp_ctc_decode.cu)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE sp_ctc_decode_srcs SRCS ${sp_ctc_decode_srcs})
endif()
add_executable(sp_ctc_decode ${sp_ctc_decode_srcs})
set_property(TARGET sp_ctc_decode PROPERTY CXX_STANDARD 14)
target_link_libraries(sp_ctc_decode ${bin_dep_libs})
