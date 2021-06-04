
# PYTHON_EXECUTABLE is set by pybind11.cmake
message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_DIR
)

list(APPEND CMAKE_PREFIX_PATH "${TORCH_DIR}")
find_package(Torch REQUIRED)

# set the global CMAKE_CXX_FLAGS so that
# k2 uses the same abi flag as PyTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(K2_WITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")
endif()


execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(torch.__version__.split('.')[0])"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE K2_TORCH_VERSION_MAJOR
)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(torch.__version__.split('.')[1])"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE K2_TORCH_VERSION_MINOR
)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(torch.__version__)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_VERSION
)

message(STATUS "PyTorch version: ${TORCH_VERSION}")

if(K2_WITH_CUDA)

  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(torch.version.cuda)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CUDA_VERSION
  )

  message(STATUS "PyTorch cuda version: ${TORCH_CUDA_VERSION}")

  if(NOT CUDA_VERSION VERSION_EQUAL TORCH_CUDA_VERSION)
    message(FATAL_ERROR
      "PyTorch ${TORCH_VERSION} is compiled with CUDA ${TORCH_CUDA_VERSION}.\n"
      "But you are using CUDA ${CUDA_VERSION} to compile k2.\n"
      "Please try to use the same CUDA version for PyTorch and k2.\n"
      "**You can remove this check if you are sure this will not cause "
      "problems**\n"
    )
  endif()

  # See
  #  - https://forums.developer.nvidia.com/t/invalid-command-option-for-nvcc-in-pytorch/145588
  #  - https://github.com/pytorch/vision/issues/2677
  #  - https://github.com/pytorch/vision/pull/2754/files
  function(k2_cuda_convert_flags EXISTING_TARGET)
    # This function is copied from
    # https://github.com/pytorch/vision/pull/2754/files
    get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
    if(NOT "${old_flags}" STREQUAL "")
      string(REPLACE ";" "," CUDA_flags "${old_flags}")
      set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
        "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
      )
    endif()
  endfunction()

  k2_cuda_convert_flags(torch_cuda)
  k2_cuda_convert_flags(torch_cpu)

  if(WIN32)
    k2_cuda_convert_flags(torch_cuda_cu)
    k2_cuda_convert_flags(torch_cuda_cpp)
  endif()
endif()
