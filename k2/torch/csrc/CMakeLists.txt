include_directories(${CMAKE_SOURCE_DIR})

# it is located in k2/csrc/cmake/transform.cmake
include(transform)
set(k2_torch_srcs
  beam_search.cu
  decode.cu
  dense_fsa_vec.cu
  deserialization.cu
  fsa_algo.cu
  fsa_class.cu
  hypothesis.cu
  nbest.cu
  parse_options.cu
  symbol_table.cu
  utils.cu
  wave_reader.cu
)

if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE k2_torch_srcs SRCS ${k2_torch_srcs})
endif()

add_library(k2_torch ${k2_torch_srcs})
target_link_libraries(k2_torch PUBLIC ${TORCH_LIBRARIES} context)

add_library(k2_fbank features.cc)
target_link_libraries(k2_fbank PUBLIC ${TORCH_LIBRARIES} kaldifeat_core)

if(K2_ENABLE_TESTS)
  # Please sort files alphabetically
  set(k2_torch_test_srcs
    dense_fsa_vec_test.cu
    deserialization_test.cu
    fsa_class_test.cu
    hypothesis_test.cu
    parse_options_test.cu
    wave_reader_test.cu
  )

  if(NOT K2_WITH_CUDA)
    transform(OUTPUT_VARIABLE k2_torch_test_srcs SRCS ${k2_torch_test_srcs})
  endif()

  function(k2_add_torch_test source)
    get_filename_component(name ${source} NAME_WE)
    set(target_name "cu_k2_torch_${name}")
    add_executable(${target_name} "${source}")
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${target_name} k2_torch gtest gtest_main)

    # NOTE: We set the working directory here so that
    # it works also on windows. The reason is that
    # the required DLLs are inside ${TORCH_DIR}/lib
    # and they can be found by the exe if the current
    # working directory is ${TORCH_DIR}\lib
    add_test(NAME "Test.Cuda.${target_name}"
      COMMAND
      $<TARGET_FILE:${target_name}>
      WORKING_DIRECTORY ${TORCH_DIR}/lib
    )
  endfunction()

  foreach(source IN LISTS k2_torch_test_srcs)
    k2_add_torch_test(${source})
  endforeach()
endif()

set(k2_torch_api_srcs torch_api.cu)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE k2_torch_api_srcs SRCS ${k2_torch_api_srcs})
endif()
add_library(k2_torch_api ${k2_torch_api_srcs})
target_link_libraries(k2_torch_api PUBLIC k2_torch)

if(K2_ENABLE_TESTS)
  add_executable(torch_api_test torch_api_test.cc)
  target_link_libraries(torch_api_test PRIVATE k2_torch_api gtest gtest_main)

  # NOTE: We set the working directory here so that
  # it works also on windows. The reason is that
  # the required DLLs are inside ${TORCH_DIR}/lib
  # and they can be found by the exe if the current
  # working directory is ${TORCH_DIR}\lib
  add_test(NAME "Test.torch_api_test"
    COMMAND
    $<TARGET_FILE:torch_api_test>
    WORKING_DIRECTORY ${TORCH_DIR}/lib
  )
endif()

file(MAKE_DIRECTORY
    ${PROJECT_BINARY_DIR}/include/k2
)

file(COPY
  torch_api.h
  DESTINATION
    ${PROJECT_BINARY_DIR}/include/k2
)

install(TARGETS k2_torch_api k2_torch
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES torch_api.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/k2
)
