/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
 *
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

#ifndef K2_TORCH_CSRC_DESERIALIZATION_H_
#define K2_TORCH_CSRC_DESERIALIZATION_H_

#include <string>

#include "k2/csrc/fsa.h"
#include "k2/torch/csrc/utils.h"

namespace k2 {

void RegisterRaggedInt();

// A helper class to construct a Ragged<int32_t> from an archive
// TODO(fangjun): Make it a template
struct RaggedIntHelper : public Ragged<int32_t>,
                         public torch::CustomClassHolder {
  using k2::Ragged<int32_t>::Ragged;
  RaggedIntHelper(Ragged<int32_t> ragged) : Ragged<int32_t>(ragged) {}
};

struct RaggedRegister {
  RaggedRegister() {
    static std::once_flag register_ragged_int_flag;
    std::call_once(register_ragged_int_flag, RegisterRaggedInt);
  }
};

/**
  Load a file saved in Python by

    torch.save(fsa.as_dict(), filename, _use_new_zipfile_serialization=True)

  Note: `_use_new_zipfile_serialization` is True by default

  @param filename Path to the filename produced in Python by `torch.save()`.
  @param ragged_aux_labels If it is not NULL and the file contains aux_labels as
            ragged tensors, then return it via this parameter.
  @return Return the FSA contained in the filename.
 */
k2::FsaOrVec LoadFsa(const std::string &filename,
                     Ragged<int32_t> *ragged_aux_labels = nullptr);

}  // namespace k2

#endif  // K2_TORCH_CSRC_DESERIALIZATION_H_
