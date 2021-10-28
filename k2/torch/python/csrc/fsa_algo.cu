/**
 * @copyright
 *
 * Copyright      2021  Xiaomi Corp.        (authors: Wei Kang)
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

#include "k2/torch/csrc/fsa_class.h"
#include "k2/torch/csrc/torch_utils.h"
#include "k2/torch/python/csrc/fsa_algo.h"

namespace k2 {

void PybindFsaAlgorithms(py::module &m) {
  m.def(
      "arc_sort",
      [](FsaClass &src) -> FsaClass {
        DeviceGuard guard(src.fsa.Context());
        return src.ArcSort();
      },
      py::arg("src"));

  m.def(
      "connect",
      [](FsaClass &src) -> FsaClass {
        DeviceGuard guard(src.fsa.Context());
        return src.Connect();
      },
      py::arg("src"));

  m.def(
      "top_sort",
      [](FsaClass &src) -> FsaClass {
        DeviceGuard guard(src.fsa.Context());
        return src.TopSort();
      },
      py::arg("src"));

  m.def(
      "add_epsilon_self_loops",
      [](FsaClass &src) -> FsaClass {
        DeviceGuard guard(src.fsa.Context());
        return src.AddEpsilonSelfLoops();
      },
      py::arg("src"));
}

}  // namespace k2
