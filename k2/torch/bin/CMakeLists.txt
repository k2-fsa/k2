add_executable(decode decode.cc)
set_property(TARGET decode PROPERTY CXX_STANDARD 14)
target_link_libraries(decode PRIVATE ${TORCH_LIBRARIES})