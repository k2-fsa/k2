# Sources here are waited to be deleted. As K2 is supoosed to run on GPUs.

#---------------------------- Build K2 host sources ----------------------------

# the target
# please sort the source files alphabetically
add_library(fsa
    arcsort.cc
    aux_labels.cc
    connect.cc
    determinize.cc
    determinize_pruned.cc
    fsa.cc
    fsa_equivalent.cc
    fsa_renderer.cc
    fsa_util.cc
    intersect.cc
    properties.cc
    rmepsilon_pruned.cc
    topsort.cc
    util.cc
    weights.cc
    )
set_target_properties(fsa PROPERTIES OUTPUT_NAME "k2fsa")
target_link_libraries(fsa PUBLIC k2_log)
target_link_libraries(fsa PUBLIC k2_nvtx)
target_include_directories(fsa PUBLIC ${CUDA_TOOLKIT_INCLUDE})
if(K2_ENABLE_NVTX)
  if(NOT WIN32)
    target_link_libraries(fsa
      PUBLIC
        -L${CUDA_TOOLKIT_ROOT_DIR}/lib64  # for /usr/local/cuda
        -L${CUDA_TOOLKIT_ROOT_DIR}/lib  # for conda
      nvToolsExt)
  else()
    target_link_directories(fsa PUBLIC
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64  # for /usr/local/cuda
      ${CUDA_TOOLKIT_ROOT_DIR}/lib  # for conda
      "C:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64/"
    )
    target_link_libraries(fsa PUBLIC NvToolsExt64_1)
  endif()
endif()

if(K2_ENABLE_TESTS)
  #---------------------------- Test K2 host sources ----------------------------

  # please sort the source files alphabetically
  set(fsa_tests
      arcsort_test
      array_test
      aux_labels_test
      connect_test
      determinize_test
      fsa_equivalent_test
      fsa_renderer_test
      fsa_test
      fsa_util_test
      intersect_test
      properties_test
      rmepsilon_test
      topsort_test
      weights_test
      )

  function(k2_add_fsa_test name)
    add_executable(${name} "${name}.cc")
    target_link_libraries(${name}
        PRIVATE
        fsa
        gtest
        gtest_main
        )
    add_test(NAME "Test.${name}"
        COMMAND
        $<TARGET_FILE:${name}>
        )
  endfunction()

  foreach (name IN LISTS fsa_tests)
    k2_add_fsa_test(${name})
  endforeach ()
endif()

install(TARGETS fsa
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
