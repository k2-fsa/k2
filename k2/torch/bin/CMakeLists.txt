# it is located in k2/csrc/cmake/transform.cmake
include(transform)

set(bin_dep_libs
  ${TORCH_LIBRARIES}
  k2_torch
  k2_fbank
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

#---------------------------------------------------------------
#       HLG decoding + n-gram LM rescoring + attenion rescoring
#---------------------------------------------------------------
set(attention_rescore_srcs attention_rescore.cu)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE attention_rescore_srcs SRCS ${attention_rescore_srcs})
endif()
add_executable(attention_rescore ${attention_rescore_srcs})
set_property(TARGET attention_rescore PROPERTY CXX_STANDARD 14)
target_link_libraries(attention_rescore ${bin_dep_libs})


#-------------------------------------------
#       online decoding
#-------------------------------------------
set(online_decode_srcs online_decode.cu)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE online_decode_srcs SRCS ${online_decode_srcs})
endif()

add_executable(online_decode ${online_decode_srcs})
set_property(TARGET online_decode PROPERTY CXX_STANDARD 14)
target_link_libraries(online_decode ${bin_dep_libs})

#-------------------------------------------
#      rnnt demo
#-------------------------------------------
set(rnnt_demo_srcs rnnt_demo.cu)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE rnnt_demo_srcs SRCS ${rnnt_demo_srcs})
endif()

add_executable(rnnt_demo ${rnnt_demo_srcs})
set_property(TARGET rnnt_demo PROPERTY CXX_STANDARD 14)
target_link_libraries(rnnt_demo ${bin_dep_libs})

#-------------------------------------------
#      pruned stateless transducer
#-------------------------------------------
set(pruned_stateless_transducer_srcs pruned_stateless_transducer.cu)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE pruned_stateless_transducer_srcs SRCS ${pruned_stateless_transducer_srcs})
endif()

add_executable(pruned_stateless_transducer ${pruned_stateless_transducer_srcs})
set_property(TARGET pruned_stateless_transducer PROPERTY CXX_STANDARD 14)
target_link_libraries(pruned_stateless_transducer ${bin_dep_libs})
