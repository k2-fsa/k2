/**
 * @brief Wrap k2 graphs
 *
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

#ifndef K2_PYTHON_CSRC_TORCH_V2_GRAPH_H_
#define K2_PYTHON_CSRC_TORCH_V2_GRAPH_H_

#include <vector>

#include "k2/python/csrc/torch/v2/ragged_arc.h"

namespace k2 {
/**Create a CTC topology.

  A token which appears once on the right side (i.e. olabels) may
  appear multiple times on the left side (ilabels), possibly with
  epsilons in between.
  When 0 appears on the left side, it represents the blank symbol;
  when it appears on the right side, it indicates an epsilon. That
  is, 0 has two meanings here.

  A standard CTC topology is the conventional one, where there
  is a mandatory blank between two repeated neighboring symbols.
  A non-standard, i.e., modified CTC topology, imposes no such constraint.

  See https://github.com/k2-fsa/k2/issues/746#issuecomment-856421616
  and https://github.com/k2-fsa/snowfall/pull/209
  for more details.

  @param [in] max_token The maximum token ID (inclusive). We assume that token
                        IDs are contiguous (from 1 to `max_token`).
                        0 represents blank.
  @param [in] modified If False, create a standard CTC topology. Otherwise,
                       create a modified CTC topology.
  @param [in,optional] device It is a torch.device. If it is not set, then the
                              returned FSA is on CPU.
  @return Return either a standard or a modified CTC topology as an FSA
          depending on whether `standard` is True or False.
 */
RaggedArc CtcTopo(int32_t max_token, torch::optional<torch::Device> device = {},
                  bool modified = false);

/**Construct ctc graphs from symbols.

  Note:
    The scores of arcs in the returned FSA are all 0.

  @param [in] symbols It is a RaggedAny with dtype equals to int32_t, and MUST
                      have `num_axes == 2`.

  @param [in] modified Option to specify the type of CTC topology: "standard"
                       or "modified", where the "standard" one makes the blank
                       mandatory between a pair of identical symbols.
                       Default false.
  @return Return an FsaVec containing the returned ctc graphs, with `Dim0()`
          the same as `symbols.any.Dim0()` and on the same device as symbols.
 */
RaggedArc CtcGraphs(RaggedAny &symbols, bool modified = false);

}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_GRAPH_H_
