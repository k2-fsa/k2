function(k2_add_benchmark source)
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} ${source})
  set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(${name} PRIVATE benchmark)
endfunction()

set(benchmark_main_srcs benchmark.cu)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE benchmark_main_srcs SRCS ${benchmark_main_srcs})
endif()

add_library(benchmark ${benchmark_main_srcs})
target_link_libraries(benchmark PUBLIC context)
target_link_libraries(benchmark PUBLIC test_utils)
target_include_directories(benchmark PUBLIC ${CMAKE_BINARY_DIR})

# Please keep the source files sorted
set(benchmark_sources
  array_ops_benchmark.cu
  ragged_ops_benchmark.cu
  tensor_ops_benchmark.cu
)
if(NOT K2_WITH_CUDA)
  transform(OUTPUT_VARIABLE benchmark_sources SRCS ${benchmark_sources})
endif()

foreach(source IN LISTS benchmark_sources)
  k2_add_benchmark(${source})
endforeach()
