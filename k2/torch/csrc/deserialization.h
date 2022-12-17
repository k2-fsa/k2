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
#include "k2/torch/csrc/fsa_class.h"
#include "torch/script.h"

namespace k2 {

/** Read a file saved in Python by `torch.save()`.

  Unlike torch::jit::pickle_load(), this function can also handle
  k2.ragged.RaggedTensor.

  Caution: If you save a dict of tensors in `filename`, the dict MUST
  have at least two items. Otherwise, it will throw. See
  https://github.com/pytorch/pytorch/issues/67902 for more details.

  @param filename  Path to the file to be loaded.
  @param map_location  It has the same meaning as the one in `torch.load()`.
                       The loaded IValue is moved to this device
                       before returning.
  @return Return an IValue containing the content in the given file.
 */
torch::IValue Load(
    const std::string &filename,
    torch::optional<torch::Device> map_location = torch::nullopt);

/**
  Load a file saved in Python by

    torch.save(fsa.as_dict(), filename, _use_new_zipfile_serialization=True)

  Note: `_use_new_zipfile_serialization` is True by default

  @param filename Path to the filename produced in Python by `torch.save()`.
  @param map_location  It has the same meaning as the one in `torch.load()`.
                       The loaded FSA is moved to this device
                       before returning.
  @return Return the FSA contained in the filename.
 */
k2::FsaClass LoadFsa(
    const std::string &filename,
    torch::optional<torch::Device> map_location = torch::nullopt);

}  // namespace k2

#endif  // K2_TORCH_CSRC_DESERIALIZATION_H_
