/**
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Fangjun Kuang)
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

#include "k2/python/csrc/torch/v2/doc/doc.h"

namespace k2 {

void SetMethodDoc(py::object *cls, const char *name, const char *doc) {
  py::function method = cls->attr(name);
  if (!method) {
    K2_LOG(FATAL) << "No such method: " << name;
  }

  if (!method.is_cpp_function()) {
    K2_LOG(FATAL) << "name: " << name << " is not a method";
  }

  py::handle h = method.cpp_function();
  auto f = reinterpret_cast<PyCFunctionObject *>(h.ptr());

  // Don't assume "doc" is statically allocated.
  // So we duplicate it here.
  f->m_ml->ml_doc = strdup(doc);
}

}  // namespace k2
