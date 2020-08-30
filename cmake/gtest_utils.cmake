enable_testing()

function(k2_add_gtest)
  cmake_parse_arguments(
      ARGS                  # prefix of output variables
      "EXCLUDE_FROM_ALL"    # list of names of the boolean arguments (only defined ones will be true)
      ""                    # list of names of mono-valued arguments
      "SOURCES;DEPENDS"     # list of names of multi-valued arguments (output variables are lists)
      ${ARGN}               # arguments of the function to parse, here we take the all original ones
  ) # remaining unparsed arguments can be found in ARGS_UNPARSED_ARGUMENTS

  set(dependencies ${ARGS_DEPENDS})

  list(LENGTH ARGS_UNPARSED_ARGUMENTS other_args_size)
  if(other_args_size EQUAL 1)
    set(name ${ARGS_UNPARSED_ARGUMENTS})
  else()
    message(FATAL_ERROR "too many or not enough unnamed arguments")
  endif()

  if(ARGS_SOURCES)
    set(sources ${ARGS_SOURCES})
  else()
    message(FATAL_ERROR "you must specify at least one source for the test")
  endif()

  if(ARGS_EXCLUDE_FROM_ALL)
    set(EXCLUDE_FROM_ALL "EXCLUDE_FROM_ALL")
  endif()

  string(REGEX REPLACE "^.*\\." "" tgt ${name})
  add_executable(${tgt} ${sources} ${EXCLUDE_FROM_ALL})

  target_link_libraries(${tgt} PUBLIC ${dependencies} gtest)
  target_link_libraries(${tgt} PRIVATE gtest_main)

  add_test(NAME ${name} COMMAND $<TARGET_FILE:${tgt}>)

  set(${CMAKE_PROJECT_NAME}_all_tests "${${CMAKE_PROJECT_NAME}_all_tests};${tgt}" CACHE STRING "" FORCE)
endfunction()

# directly give the suite name source files,
function(k2_add_gtests test_set_name)
  cmake_parse_arguments(
      ARGS                      # prefix of output variables
      "EXCLUDE_FROM_ALL"        # list of names of the boolean arguments (only defined ones will be true)
      ""                        # list of names of mono-valued arguments
      "SOURCES;DEPENDS"         # each source share the same args in this api, and each's a sub-gtest
      ${ARGN}                   # arguments of the function to parse, here we take the all original ones
  ) # remaining unparsed arguments can be found in ARGS_UNPARSED_ARGUMENTS

  list(LENGTH ARGS_UNPARSED_ARGUMENTS other_args_size)
  if(NOT other_args_size EQUAL 0)
    message(FATAL_ERROR "too many or not enough unnamed arguments")
  endif()

  foreach(source ${ARGS_SOURCES})
    string(REGEX REPLACE "\\..*$" "" name ${source})
    k2_add_gtest("${test_set_name}.${name}"
        SOURCES ${source}
        DEPENDS ${ARGS_DEPENDS}
    )
  endforeach()
endfunction()
