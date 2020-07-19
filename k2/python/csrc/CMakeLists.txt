# please sort the files alphabetically
pybind11_add_module(_k2
  array.cc
  fsa.cc
  fsa_util.cc
  k2.cc
  tensor.cc
)

target_include_directories(_k2 PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(_k2 PRIVATE fsa)
