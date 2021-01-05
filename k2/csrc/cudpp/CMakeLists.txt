
include_directories(${CMAKE_CURRENT_LIST_DIR})
add_library(cudpp
  cudpp_plan.cu
  cudpp.cu
  multisplit_app.cu
)
target_link_libraries(cudpp cub)

add_executable(cudpp_test cudpp_test.cu)
target_link_libraries(cudpp_test cudpp context)
