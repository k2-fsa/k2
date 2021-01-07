function(k2_add_benchmark source)
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE benchmark)
endfunction()

add_library(benchmark benchmark.cu)
target_link_libraries(benchmark PUBLIC context)
target_link_libraries(benchmark PUBLIC test_utils)
target_include_directories(benchmark PUBLIC ${CMAKE_BINARY_DIR})

# Please keep the source files sorted
set(benchmark_sources
  array_ops_benchmark.cu
  ragged_ops_benchmark.cu
  tensor_ops_benchmark.cu
)

foreach(source IN LISTS benchmark_sources)
  k2_add_benchmark(${source})
endforeach()
