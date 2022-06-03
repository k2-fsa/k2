# it is located in k2/csrc/cmake/transform.cmake
include(transform)

# please keep the list sorted
set(k2_srcs
  k2.cu
  torch.cu
  version.cu
)

if(K2_USE_PYTORCH)
  add_subdirectory(torch)
  set(k2_srcs ${k2_srcs} ${torch_srcs})
else()
  message(FATAL_ERROR "Please select a framework.")
endif()

if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE k2_srcs SRCS ${k2_srcs})
endif()

if(WIN32)
  # It throws the following error on Windows
  #  nvcc fatal   : A single input file is required for a non-link phase when an outputfile is specified
  # because there is an option "/bigobj" in pybind11::windows_extra that cannot be recognized  by NVCC.
  #
  # We clear it below.
  set_property(TARGET pybind11::windows_extras PROPERTY INTERFACE_COMPILE_OPTIONS "")
endif()


pybind11_add_module(_k2 ${k2_srcs})
target_link_libraries(_k2 PRIVATE context)
target_link_libraries(_k2 PRIVATE fsa)
target_include_directories(_k2 PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(_k2 PRIVATE ${CMAKE_BINARY_DIR})
set_target_properties(_k2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(APPLE)
  # See https://github.com/k2-fsa/k2/issues/836
  # This block adds the directory of _k2.cpython-*.so to runpath
  # so that there is no need to use DYLD_LIBRARY_PATH on macOS
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  target_link_libraries(_k2 PRIVATE "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}")
endif()
