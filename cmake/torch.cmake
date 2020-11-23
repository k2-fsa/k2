
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

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(torch.__version__)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_VERSION
)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(torch.version.cuda)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_CUDA_VERSION
)

message(STATUS "PyTorch version: ${TORCH_VERSION}")
message(STATUS "PyTorch cuda version: ${TORCH_CUDA_VERSION}")

if(NOT CUDA_VERSION VERSION_EQUAL TORCH_CUDA_VERSION)
  message(FATAL_ERROR
    "PyTorch ${TORCH_VERSION} is compiled with ${TORCH_CUDA_VERSION}.\n"
    "But you are using ${CUDA_VERSION} to compile k2.\n"
    "Please try to use the same cuda version for PyTorch and k2.\n"
  )
endif()
