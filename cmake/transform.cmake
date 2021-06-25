# Copyright     2021 Fangjun Kuang (csukuangfj@gmail.com)
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This function is used to copy foo.cu to foo.cc
# Usage:
#
#   transform(OUTPUT_VARIABLE output_variable_name  SRCS foo.cu bar.cu)
#
function(transform)
  set(optional_args "") # there are no optional arguments
  set(one_value_arg OUTPUT_VARIABLE)
  set(multi_value_args SRCS)

  cmake_parse_arguments(MY "${optional_args}" "${one_value_arg}" "${multi_value_args}" ${ARGN})
  foreach(src IN LISTS MY_SRCS)
    get_filename_component(src_name ${src} NAME_WE)
    get_filename_component(src_dir ${src} DIRECTORY)
    set(dst ${CMAKE_CURRENT_BINARY_DIR}/${src_dir}/${src_name}.cc)

    list(APPEND ans ${dst})
    message(STATUS "Renaming ${CMAKE_CURRENT_SOURCE_DIR}/${src} to ${dst}")
    configure_file(${src} ${dst})
  endforeach()
  set(${MY_OUTPUT_VARIABLE} ${ans} PARENT_SCOPE)
endfunction()
