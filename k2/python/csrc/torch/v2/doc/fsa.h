/**
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Wei Kang)
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

#ifndef K2_PYTHON_CSRC_TORCH_V2_DOC_FSA_H_
#define K2_PYTHON_CSRC_TORCH_V2_DOC_FSA_H_

namespace k2 {

static constexpr const char *kFsaAlgoConnectDoc = R"doc(
Connect current FSA.
Removes states that are neither accessible nor co-accessible.

Note:
  A state is not accessible if it is not reachable from the start state.
  A state is not co-accessible if it cannot reach the final state.

Caution:
  If current FSA is already connected, it is returned directly.
  Otherwise, a new connected FSA is returned.

Returns:
  An FSA that is connected.
)doc";

static constexpr const char *kFsaAlgoArcSortDoc = R"doc(
Sort arcs of every state.

Note:
  Arcs are sorted by labels first, and then by dest states.

Caution:
  If the current `fsa` is already arc sorted, we return it directly.
  Otherwise, a new sorted fsa is returned.

Returns:
  Return the sorted FSA.
)doc";

static constexpr const char *kFsaAlgoTopSortDoc = R"doc(
Sort an FSA topologically.

Note:
  An FSA is top-sorted if all its arcs satisfy that the source state number
  is less than the dest state number.

Caution:
  If the current `fsa` is already top sorted, we return it directly.
  Otherwise, a new sorted fsa is returned.

Returns:
  Return the top sorted FSA.
)doc";

static constexpr const char *kFsaAlgoAddEpsilonSelfLoopsDoc = R"doc(
Add epsilon self-loops to current FSA.

This is required when composing using a composition method that does not
treat epsilons specially, if the other FSA has epsilons in it.

Returns:
  Return an instance of :class:`Fsa` that has an
  epsilon self-loop on every non-final state.
)doc";

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_DOC_FSA_H_
