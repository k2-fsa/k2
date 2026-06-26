
# PYTHON_EXECUTABLE is set by pybind11.cmake
message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_DIR
)

list(APPEND CMAKE_PREFIX_PATH "${TORCH_DIR}")
include_directories(${TORCH_DIR}/include/torch/csrc/api/include)
include_directories(${TORCH_DIR}/include)

if(NOT DEFINED TORCH_LIBRARY)
  find_package(Torch REQUIRED)
endif()

# set the global CMAKE_CXX_FLAGS so that
# k2 uses the same abi flag as PyTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(K2_WITH_CUDA)
  if(CUDA_VERSION VERSION_GREATER_EQUAL "12.0")
    string(REPLACE " " ";" MY_LIST ${CMAKE_CUDA_FLAGS})
    set(TEMP_LIST)
    foreach(f IN LISTS MY_LIST)
      if(f STREQUAL arch=compute_35,code=sm_35)
        list(REMOVE_AT TEMP_LIST -1)
        continue()
      endif()
      list(APPEND TEMP_LIST ${f})
    endforeach()

    string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${TEMP_LIST}")

    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
    if(CUDA_VERSION VERSION_GREATER_EQUAL "13.0")
      string(REPLACE "-gencode arch=compute_50,code=sm_50" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
      string(REPLACE "-gencode arch=compute_100a,code=sm_100a" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
      string(REPLACE "-gencode arch=compute_101a,code=sm_101a" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    endif()
    message(STATUS "Final CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")


  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")
  message(WARNING " CMAKE_CUDA_FLAGS is ${CMAKE_CUDA_FLAGS}")
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

if(K2_WITH_HIP)
  # torch's source hipify (torch/utils/hipify) has two generations that disagree
  # on the c10 device namespace.  Generation 1 RENAMED the device classes, so the
  # hip-spelled symbols (c10::hip::*) are the only public ones.  Generation 2
  # (pytorch#174087, version.py bumped 1.0.0 -> 2.0.0) STOPPED renaming: the CUDA
  # spellings stay public as the masquerading API (c10::cuda::* on a ROCm build)
  # while c10::hip::* survive only as thin wrappers.  k2 drives the .cu through
  # CMake/USE_HIP and never runs torch source-hipify, so it must detect the
  # generation itself and select the matching namespace.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from packaging.version import Version; import torch.utils.hipify as h; print(1 if Version(getattr(h, '__version__', '1.0.0')) >= Version('2.0.0') else 0)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE K2_TORCH_HIPIFY_V2
    RESULT_VARIABLE _k2_hipify_probe_rc
  )
  if(NOT _k2_hipify_probe_rc EQUAL 0)
    set(K2_TORCH_HIPIFY_V2 0)
  endif()
  message(STATUS "torch hipify generation v2 (masquerading c10::cuda): ${K2_TORCH_HIPIFY_V2}")
endif()

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

if(K2_WITH_HIP)
  # On a ROCm torch the GPU target is torch_hip; clear its (and torch_cpu's)
  # interface compile options for the same reason as the CUDA path above.
  foreach(_t torch_hip torch_cpu)
    if(TARGET ${_t})
      set_property(TARGET ${_t} PROPERTY INTERFACE_COMPILE_OPTIONS "")
    endif()
  endforeach()

  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(torch.version.hip)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_HIP_VERSION
  )
  message(STATUS "PyTorch HIP version: ${TORCH_HIP_VERSION}")
endif()

