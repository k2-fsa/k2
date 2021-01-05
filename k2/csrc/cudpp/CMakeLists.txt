add_library(cudpp
  cudpp_plan.cu
  multisplit_app.cu
)
target_link_libraries(cudpp cub)
