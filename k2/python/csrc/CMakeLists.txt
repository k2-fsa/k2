# it is located in k2/csrc/cmake/transform.cmake
include(transform)

# please keep the list sorted
set(k2_srcs
  k2.cu
  torch.cu
  version.cu
)

if(K2_USE_PYTORCH)
  add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)
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

if(APPLE)
  # To fix the following error:
  # ImportError: /xxx/lib/_k2.cpython-38-x86_64-linux-gnu.so: undefined symbol: THPDtypeType
  target_link_libraries(_k2 PRIVATE ${TORCH_DIR}/lib/libtorch_python.dylib)
elseif(UNIX)
  target_link_libraries(_k2 PRIVATE ${TORCH_DIR}/lib/libtorch_python.so)
endif()

target_include_directories(_k2 PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(_k2 PRIVATE ${CMAKE_BINARY_DIR})
set_target_properties(_k2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(NOT WIN32)
  target_link_libraries(_k2 PRIVATE "-Wl,-rpath,${k2_rpath_origin}/k2/${CMAKE_INSTALL_LIBDIR}")
endif()

if(UNIX AND NOT APPLE)
  # It causes errors on macOS
  target_link_libraries(_k2 PRIVATE ${TORCH_DIR}/lib/libtorch_python.so)
  # CAUTION: It is PYTHON_LIBRARY on unix
  target_link_libraries(_k2 PRIVATE ${PYTHON_LIBRARY})
  message(STATUS "PYTHON_LIBRARY: ${PYTHON_LIBRARY}")
elseif(WIN32)
  target_link_libraries(_k2 PRIVATE ${TORCH_DIR}/lib/torch_python.lib)
  # CAUTION: It is PYTHON_LIBRARIES on Windows
  target_link_libraries(_k2 PRIVATE ${PYTHON_LIBRARIES})
  message(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
endif()

install(TARGETS _k2
  DESTINATION ../
)
