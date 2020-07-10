# sort the files alphabetically
pybind11_add_module(k2
  #fsa.cc 
  fsa_renderer.cc
  fsa_util.cc
  k2.cc
)

target_include_directories(k2 PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(k2 PRIVATE fsa)
