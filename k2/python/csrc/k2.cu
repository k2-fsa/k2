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

#include "k2/python/csrc/k2.h"
#include "k2/python/csrc/torch.h"
#include "k2/python/csrc/version.h"

PYBIND11_MODULE(_k2, m) {
  m.doc() = "pybind11 binding of k2";
  PybindVersion(m);
  PybindTorch(m);
}
