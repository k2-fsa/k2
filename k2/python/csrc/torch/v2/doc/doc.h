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

#ifndef K2_PYTHON_CSRC_TORCH_V2_DOC_DOC_H_
#define K2_PYTHON_CSRC_TORCH_V2_DOC_DOC_H_
#include "k2/python/csrc/torch.h"

namespace k2 {

/* Set the documentation of a method inside a class.

   @param cls The class whose method's doc we are going to set
   @param name  The name of the method
   @param doc  The doc string to set.
 */
void SetMethodDoc(py::object *cls, const char *name, const char *doc);

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_DOC_DOC_H_
