# Make project root dir as the INCLUDE_DIRECTORIES of all tagets in csrc/.
# The header should be included as `#include "k2/csrc/.."`, to avoid conflicts.
include_directories(${CMAKE_SOURCE_DIR})

# it is located in k2/csrc/cmake/transform.cmake
include(transform)

#---------------------------- Build K2 CUDA sources ----------------------------

configure_file(version.h.in ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)
message(STATUS "Generated ${CMAKE_CURRENT_BINARY_DIR}/version.h")

set(log_srcs log.cu)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE log_srcs SRCS ${log_srcs})
endif()
add_library(k2_log ${log_srcs})
if(UNIX AND NOT APPLE AND NOT K2_WITH_CUDA)
  # Without linking to libpthread.so, it throws
  # in std::call_once
  #
  # It happens only on Linux, that's why
  # we use the above guard to link against -pthread
  #
  # See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60662
  target_link_libraries(k2_log -pthread)
endif()

if(K2_HAVE_EXECINFO_H)
  target_compile_definitions(k2_log PRIVATE K2_HAVE_EXECINFO_H=1)
endif()

if(K2_HAVE_CXXABI_H)
  target_compile_definitions(k2_log PRIVATE K2_HAVE_CXXABI_H=1)
endif()

add_library(k2_nvtx INTERFACE)
target_include_directories(k2_nvtx INTERFACE ${CMAKE_SOURCE_DIR})
if(K2_ENABLE_NVTX)
  target_compile_definitions(k2_nvtx INTERFACE K2_ENABLE_NVTX=1)
  if(WIN32)
    target_include_directories(k2_nvtx INTERFACE
      ${CUDA_TOOLKIT_ROOT_DIR}/include/nvtx3
      "C:/Program Files/NVIDIA Corporation/NvToolsExt/include"
    )
  endif()
endif()

add_subdirectory(host)

# please keep it sorted
set(context_srcs
  algorithms.cu
  array_of_ragged.cu
  array_ops.cu
  connect.cu
  context.cu
  dtype.cu
  fsa.cu
  fsa_algo.cu
  fsa_utils.cu
  hash.cu
  host_shim.cu
  intersect.cu
  intersect_dense.cu
  intersect_dense_pruned.cu
  math.cu
  moderngpu_allocator.cu
  pinned_context.cu
  ragged.cu
  ragged_ops.cu
  ragged_utils.cu
  rand.cu
  reverse.cu
  rm_epsilon.cu
  rnnt_decode.cu
  tensor.cu
  tensor_ops.cu
  thread_pool.cu
  timer.cu
  top_sort.cu
  torch_util.cu
  utils.cu
  nbest.cu
)


if(K2_USE_PYTORCH)
  list(APPEND context_srcs pytorch_context.cu)
else()
  list(APPEND context_srcs default_context.cu)
endif()

if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE context_srcs SRCS ${context_srcs})
else()
  list(APPEND context_srcs cudpp/cudpp.cu)
endif()

# the target
add_library(context ${context_srcs})
target_compile_definitions(context PUBLIC K2_TORCH_VERSION_MAJOR=${K2_TORCH_VERSION_MAJOR})
target_compile_definitions(context PUBLIC K2_TORCH_VERSION_MINOR=${K2_TORCH_VERSION_MINOR})

# see https://github.com/NVIDIA/thrust/issues/1401
# and https://github.com/k2-fsa/k2/pull/917
target_compile_definitions(context PUBLIC CUB_WRAPPED_NAMESPACE=k2)
target_compile_definitions(context PUBLIC THRUST_NS_QUALIFIER=thrust)

set_target_properties(context PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(context PROPERTIES OUTPUT_NAME "k2context")

# lib deps
if(K2_WITH_CUDA AND CUDA_VERSION VERSION_LESS 11.0)
  target_link_libraries(context PUBLIC cub)
endif()

if(K2_WITH_CUDA)
  target_link_libraries(context PUBLIC moderngpu)
endif()

target_link_libraries(context PUBLIC fsa)
target_link_libraries(context PUBLIC k2_log)
target_link_libraries(context PUBLIC k2_nvtx)
if(K2_USE_PYTORCH)
  if(NOT WIN32)
    target_link_libraries(context PUBLIC ${TORCH_LIBRARIES})
  else()
    # see https://discuss.pytorch.org/t/nvcc-fatal-a-single-input-file-is-required-for-a-non-link-phase-when-an-outputfile-is-specified/142843/6
    # Depending on ${TORCH_LIBRARIES} will introduce a compile time option "/bigobj",
    # which causes the error in the above link.
    #
    # It would be ideal to remove /bigobj so that we can use ${TORCH_LIBRARIES}.
    # To make life simpler, we use the following approach.
    #
    message(STATUS "TORCH_DIR: ${TORCH_DIR}") # TORCH_DIR is defined in cmake/torch.cmake
    # target_link_libraries(context PUBLIC D:/software/anaconda3/envs/py38/Lib/site-packages/torch/lib/*.lib)
    target_link_libraries(context PUBLIC ${TORCH_DIR}/lib/*.lib)
    target_include_directories(context PUBLIC ${TORCH_DIR}/include)
    target_include_directories(context PUBLIC ${TORCH_DIR}/include/torch/csrc/api/include)
  endif()

  if(DEFINED ENV{CONDA_PREFIX} AND APPLE)
    target_link_libraries(context PUBLIC "-L $ENV{CONDA_PREFIX}/lib")
  endif()
endif()

target_include_directories(context PUBLIC ${PYTHON_INCLUDE_DIRS})

if(K2_ENABLE_TESTS OR K2_ENABLE_BENCHMARK)
  set(test_utils_srcs test_utils.cu)
  if(NOT K2_WITH_CUDA)
    transform(OUTPUT_VARIABLE test_utils_srcs SRCS ${test_utils_srcs})
  endif()

  add_library(test_utils ${test_utils_srcs})
  target_link_libraries(test_utils PUBLIC context gtest)
endif()

if(K2_ENABLE_TESTS)
  #---------------------------- Test K2 CUDA sources ----------------------------
  # please sort the source files alphabetically
  set(cuda_test_srcs
    algorithms_test.cu
    array_of_ragged_test.cu
    array_ops_test.cu
    array_test.cu
    connect_test.cu
    dtype_test.cu
    fsa_algo_test.cu
    fsa_test.cu
    fsa_utils_test.cu
    hash_test.cu
    host_shim_test.cu
    intersect_test.cu
    log_test.cu
    macros_test.cu
    math_test.cu
    nbest_test.cu
    nvtx_test.cu
    pinned_context_test.cu
    ragged_shape_test.cu
    ragged_test.cu
    ragged_utils_test.cu
    rand_test.cu
    reverse_test.cu
    rm_epsilon_test.cu
    rnnt_decode_test.cu
    tensor_ops_test.cu
    tensor_test.cu
    thread_pool_test.cu
    top_sort_test.cu
    utils_test.cu
  )
  if(NOT K2_WITH_CUDA)
    transform(OUTPUT_VARIABLE cuda_test_srcs SRCS ${cuda_test_srcs})
  endif()

  # utility function to add gtest
  function(k2_add_cuda_test source)
    # TODO(haowen): add prefix `cu` for now to avoid name conflicts
    # with files in k2/csrc/, will remove this finally.
    get_filename_component(name ${source} NAME_WE)
    set(target_name "cu_${name}")
    add_executable(${target_name} "${source}")
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${target_name}
      PRIVATE
      context
      fsa  # for code in k2/csrc/host
      gtest
      gtest_main
      test_utils
    )

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

  foreach(source IN LISTS cuda_test_srcs)
    k2_add_cuda_test(${source})
  endforeach()
endif()

if(K2_ENABLE_BENCHMARK)
  add_subdirectory(benchmark)
endif()

install(TARGETS k2_log context
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
