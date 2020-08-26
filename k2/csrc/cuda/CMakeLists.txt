include_directories(${CMAKE_SOURCE_DIR})

set(CUDA_SOURCE
    context.cu
)

add_library(k2_cuda ${CUDA_SOURCE})
target_include_directories(k2_cuda PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(k2_cuda PUBLIC cub)
target_link_libraries(k2_cuda PUBLIC glog)

k2_add_gtests(Test.Cuda
    SOURCES debug_test.cu utils_test.cu #ops_test.cu
    DEPENDS k2_cuda cub glog
)
