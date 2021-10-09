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

#ifndef K2_TORCH_CSRC_DENSE_DESERIALIZATION_H_
#define K2_TORCH_CSRC_DENSE_DESERIALIZATION_H_

#include <string>

#include "k2/csrc/fsa.h"

namespace k2 {

/**
  Load a file saved in Python by

    torch.save(fsa.as_dict(), filename, _use_new_zipfile_serialization=True)

  Note: `_use_new_zipfile_serialization` is True by default

  @param filename Path to the filename produced in Python by `torch.save()`.
  @return Return the FSA contained in the filename.
 */
k2::FsaOrVec LoadFsa(const std::string &filename);

}  // namespace k2

#endif  // K2_TORCH_CSRC_DENSE_DESERIALIZATION_H_
