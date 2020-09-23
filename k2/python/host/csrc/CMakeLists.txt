# please sort the files alphabetically
pybind11_add_module(_k2host
  array.cc
  aux_labels.cc
  fsa.cc
  fsa_algo.cc
  fsa_equivalent.cc
  fsa_util.cc
  k2.cc
  properties.cc
  tensor.cc
  weights.cc
)

target_include_directories(_k2host PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(_k2host PRIVATE fsa)
