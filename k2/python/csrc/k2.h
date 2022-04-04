/**
 * @brief python wrappers for k2.
 *
 * @copyright
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef K2_PYTHON_CSRC_K2_H_
#define K2_PYTHON_CSRC_K2_H_

#ifndef _MSC_VER
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#else
// Otherwise, it will use pybind11 header from PyTorch, which causes CI errors
#include "pybind11-src/include/pybind11/pybind11.h"
#include "pybind11-src/include/pybind11/stl.h"
#endif

namespace py = pybind11;

#endif  // K2_PYTHON_CSRC_K2_H_
