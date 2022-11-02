
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

set(K2_TORCH_VERSION "${K2_TORCH_VERSION_MAJOR}.${K2_TORCH_VERSION_MINOR}")
message(STATUS "K2_TORCH_VERSION: ${K2_TORCH_VERSION}")

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

# Solve the following error for NVCC:
#   unknown option `-Wall`
#
# It contains only some -Wno-* flags, so it is OK
# to set them to empty
  set_property(TARGET torch_cuda
    PROPERTY
      INTERFACE_COMPILE_OPTIONS ""
  )
  set_property(TARGET torch_cpu
    PROPERTY
      INTERFACE_COMPILE_OPTIONS ""
  )
endif()

