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
  k2_torch
  sentencepiece-static # see cmake/sentencepiece.cmake
)

set(bin_dep_libs
  ${TORCH_LIBRARIES}
  k2_torch
  sentencepiece-static # see cmake/sentencepiece.cmake
)

#----------------------------------------
#       CTC decoding
#----------------------------------------
set(ctc_decode_srcs ctc_decode.cu)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE ctc_decode_srcs SRCS ${ctc_decode_srcs})
endif()
add_executable(ctc_decode ${ctc_decode_srcs})
set_property(TARGET ctc_decode PROPERTY CXX_STANDARD 14)
target_link_libraries(ctc_decode ${bin_dep_libs})

#----------------------------------------
#       HLG decoding
#----------------------------------------
set(hlg_decode_srcs hlg_decode.cu)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE hlg_decode_srcs SRCS ${hlg_decode_srcs})
endif()
add_executable(hlg_decode ${hlg_decode_srcs})
set_property(TARGET hlg_decode PROPERTY CXX_STANDARD 14)
target_link_libraries(hlg_decode ${bin_dep_libs})
