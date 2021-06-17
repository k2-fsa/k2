/**
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
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

#include <string>

#include "k2/csrc/host/fsa.h"

#ifndef K2_CSRC_HOST_FSA_RENDERER_H_
#define K2_CSRC_HOST_FSA_RENDERER_H_

namespace k2host {

// Get a GraphViz representation of an fsa.
class FsaRenderer {
 public:
  explicit FsaRenderer(const Fsa &fsa) : fsa_(fsa) {}

  // Return a GraphViz representation of the fsa
  std::string Render() const;

 private:
  const Fsa &fsa_;
};

}  // namespace k2host

#endif  // K2_CSRC_HOST_FSA_RENDERER_H_
