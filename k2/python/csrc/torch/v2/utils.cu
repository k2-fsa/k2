/**
 * @copyright
 * Copyright        2021  Xiaomi Corp.       (author: Wei Kang)
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

#include "k2/python/csrc/torch/torch_util.h"
#include "k2/python/csrc/torch/v2/utils.h"

namespace k2 {

RaggedAny ToRaggedAny(torch::IValue ivalue) {
  torch::intrusive_ptr<RaggedAnyHolder> ragged_any_holder =
      ivalue.toCustomClass<RaggedAnyHolder>();
  return *(ragged_any_holder->ragged);
}

torch::IValue ToIValue(RaggedAny &any) {
  return torch::make_custom_class<RaggedAnyHolder>(
      std::make_shared<RaggedAny>(any));
}

}  // namespace k2
