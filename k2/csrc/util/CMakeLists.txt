
#--------------------------------- non-glog (loguru) -------------------------------
if(NOT K2_USE_GLOG)
  add_library(non-glog
              ./logging.cc
              ./loguru/loguru.cc)

  target_include_directories(non-glog PUBLIC ${CMAKE_SOURCE_DIR})

  find_package(Threads)
  target_link_libraries(non-glog ${CMAKE_THREAD_LIBS_INIT}) # For pthreads
  if(NOT WIN32)
    target_link_libraries(non-glog dl) # For ldl
  endif()
endif()
