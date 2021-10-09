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
/*
  Create a CTC topology.

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
RaggedArc CtcTopo(int32_t max_token, bool modified = false,
                  torch::optional<torch::Device> device = {});

/*
  Construct ctc graphs from symbols.

  Note:
    The scores of arcs in the returned FSA are all 0.

    @param [in] symbols It is a RaggedAny with dtype equals to int32_t, and MUST
                        have `num_axes == 2`.

    @param [in] modified Option to specify the type of CTC topology: "standard"
                         or "modified", where the "standard" one makes the blank
                         mandatory between a pair of identical symbols.
                         Default false.
    @return Return an Fsa containing the returned ctc graphs, with `Dim0()`
            the same as `symbols.any.Dim0()` and on the same device as symbols.
 */
RaggedArc CtcGraphs(RaggedAny &symbols, bool modified = false);

/*
  Construct ctc graphs from symbols.

  Note:
    The scores of arcs in the returned FSA are all 0.

    @param [in] symbols It is a two dimensions int32_t array containing the
                        symbol ids (must not contain kFinalSymbol == -1).
    @param [in] modified Option to specify the type of CTC topology: "standard"
                         or "modified", where the "standard" one makes the blank
                         mandatory between a pair of identical symbols.
                         Default false.
    @param [in] device It is a torch.device. If it is not set, then the
                       returned FSA is on CPU.
    @return Return an Fsa containing the returned ctc graphs, with `Dim0()`
            the same as `symbols.size()`. If device is given the returned Fsa
            will be on the given device or it will be on CPU.
 */
RaggedArc CtcGraphs(const std::vector<std::vector<int32_t>> &symbols,
                    bool modified = false,
                    torch::optional<torch::Device> device = {});

/*
  Create a linear FSA from a sequence of symbols

  Note:
    If `symbols.size() == n`, the returned FSA will have n+1 arcs
    (including the final-arc) and n+2 states.

    @param [in] symbols  Input symbol sequence (must not contain
                kFinalSymbol == -1).
    @param [in] device It is a torch.device. If it is not set, then the
                       returned FSA is on CPU.
    @return  Returns an FSA that accepts only this symbol sequence, with zero
             score.  If device is given the returned Fsa will be on the given
             device or it will be on CPU.
*/
RaggedArc LinearFsa(const std::vector<int32_t> &symbols,
                    torch::optional<torch::Device> device = {});

/*
  Create linear FSAs, given a list of sequences of symbols.

  Note:
    The returned FSA has `num_axes() == 3` and with zero score, if the i'th row
    of `symbols` has n elements, the i'th element along axis 0 of the returned
    FSA will have n+1 arcs (including the final-arc) and n+2 states.

    @param [in] symbols  It is a two dimension array containing input symbol
                        sequences (must not contain kFinalSymbol == -1).
    @param [in] device It is a torch.device. If it is not set, then the
                       returned FSA is on CPU.
    @return Returns an Fsa having `num_axes == 3` and with
            `ans.Dim0() == symbols.size()`. If device is given the returned Fsa
            will be on the given device or it will be on CPU.
 */
RaggedArc LinearFsa(const std::vector<std::vector<int32_t>> &symbols,
                    torch::optional<torch::Device> device = {});

/*
  Create linear FSAs, given a list of sequences of symbols.

  Note:
    The returned FSA has `num_axes() == 3` and with zero score, if the i'th row
    of `symbols` has n elements, the i'th element along axis 0 of the returned
    FSA will have n+1 arcs (including the final-arc) and n+2 states.

    @param [in] symbols  It is a RaggedAny tensor containing input symbol
                        sequences (must not contain kFinalSymbol == -1).
                        It Must have `num_axes == 2`.
    @return Returns an Fsa having `num_axes == 3` and with
            `ans.Dim0() == symbols.any.Dim0()`. The returned Fsa will be on the
            same device as symbols.
 */
RaggedArc LinearFsa(RaggedAny &symbols);

/*
  Create an FSA containing levenshtein graph FSAs, given a list of sequences
  of symbols. See https://github.com/k2-fsa/k2/pull/828 for more details about
  the levenshtein graph.

    @param [in] symbols Input symbol sequences (must not contain
                        kFinalSymbol == -1 and blank == 0). Its num_axes is 2.
    @param [in] ins_del_score Specify the score of the self loops in the
                              graphs, the main idea of this score is to set
                              insertion and deletion penalty, which will
                              affect the shortest path searching produre.
    @return Returns an FSA with `num_axes() == 3` and
            `ans.Dim0() == symbols.any.Dim0()` containing the levenshte graphs.
            The returned FSA will have the attribute `aux_labels` which contains
            the aux_labels of the graphs. It will also have an attribute named
            `__ins_del_score_offset_internal_attr_` which contains the score
            offset of arcs. For self loop arcs, the offset value will be
            `ins_del_score - (-0.5)`, for other arcs, it will be zeros.
            The purpose of this score_offsets is to calculate the levenshtein
            distance.
            The returned FSA will be on the same device as symbols.
 */
RaggedArc LevenshteinGraphs(RaggedAny &symbols, float ins_del_score = -0.501);

/*
  Create an FSA containing levenshtein graph FSAs, given a list of sequences
  of symbols. See https://github.com/k2-fsa/k2/pull/828 for more details about
  the levenshtein graph.

    @param [in] symbols Input symbol sequences (must not contain
                        kFinalSymbol == -1 and blank == 0).
    @param [in] ins_del_score Specify the score of the self loops in the
                              graphs, the main idea of this score is to set
                              insertion and deletion penalty, which will
                              affect the shortest path searching produre.
    @param [in] device It is a torch.device. If it is not set, then the
                       returned FSA is on CPU.
    @return Returns an FSA with `num_axes() == 3` and
            `ans.Dim0() == symbols.any.Dim0()` containing the levenshte graphs.
            The returned FSA will have the attribute `aux_labels` which contains
            the aux_labels of the graphs. It will also have an attribute named
            `__ins_del_score_offset_internal_attr_` which contains the score
            offset of arcs. For self loop arcs, the offset value will be
            `ins_del_score - (-0.5)`, for other arcs, it will be zeros.
            The purpose of this score_offsets is to calculate the levenshtein
            distance.
            If device is given the returned FSA will be on the given device or
            it will be on CPU.
 */
RaggedArc LevenshteinGraphs(const std::vector<std::vector<int32_t>> &symbols,
                            float ins_del_score = -0.501,
                            torch::optional<torch::Device> deivce = {});
}  // namespace k2

#endif  // K2_PYTHON_CSRC_TORCH_V2_GRAPH_H_
