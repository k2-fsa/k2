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

#-------------------------------------------
#       HLG decoding + n-gram LM rescoring
#-------------------------------------------
set(ngram_lm_rescore_srcs ngram_lm_rescore.cu)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE ngram_lm_rescore_srcs SRCS ${ngram_lm_rescore_srcs})
endif()
add_executable(ngram_lm_rescore ${ngram_lm_rescore_srcs})
set_property(TARGET ngram_lm_rescore PROPERTY CXX_STANDARD 14)
target_link_libraries(ngram_lm_rescore ${bin_dep_libs})
