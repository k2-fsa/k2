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

#ifndef K2_TORCH_PYTHON_CSRC_DOC_GRAPH_H_
#define K2_TORCH_PYTHON_CSRC_DOC_GRAPH_H_

namespace k2 {

static constexpr const char *kFsaGraphCtcTopoDoc = R"doc(
Create a CTC topology.

Hint:
  A token which appears once on the right side (i.e. olabels) may
  appear multiple times on the left side (ilabels), possibly with
  epsilons in between.
  When 0 appears on the left side, it represents the blank symbol;
  when it appears on the right side, it indicates an epsilon. That
  is, 0 has two meanings here.

Note:
  A standard CTC topology is the conventional one, where there
  is a mandatory blank between two repeated neighboring symbols.
  A non-standard, i.e., modified CTC topology, imposes no such constraint.
  
  See https://github.com/k2-fsa/k2/issues/746#issuecomment-856421616
  and https://github.com/k2-fsa/snowfall/pull/209
  for more details.

Note:
  The scores of arcs in the returned FSA are all 0.

Args:
  max_token:
    The maximum token ID (inclusive). We assume that token IDs
    are contiguous (from 1 to `max_token`). 0 represents blank.
  modified:
    If False, create a standard CTC topology. Otherwise, create a
    modified CTC topology.
  device:
    Optional. It can be either a string (e.g., 'cpu',
    'cuda:0') or a torch.device.
    If it is None, then the returned FSA is on CPU.
Returns:
  Return either a standard or a modified CTC topology as an FSA
  depending on whether `standard` is True or False.
)doc";

static constexpr const char *kFsaGraphCtcGraphDoc = R"doc(
Construct ctc graphs from symbols.

Note:
  The scores of arcs in the returned FSA are all 0.

Args:
  symbols:
    It can be one of the following types:

        - A list of list-of-integers, e..g, `[ [1, 2], [1, 2, 3] ]`
        - An instance of :class:`k2.RaggedTensor`.
          Must have `num_axes == 2`.

  modified:
    Option to specify the type of CTC topology: "standard" or "modified",
    where the "standard" one makes the blank mandatory between a pair of
    identical symbols. Default False.
  device:
    Optional. It can be either a string (e.g., 'cpu', 'cuda:0') or a
    torch.device.
    By default, the returned FSA is on CPU.
    If `symbols` is an instance of :class:`k2.RaggedTensor`, the returned
    FSA will on the same device as `k2.RaggedTensor`.

Returns:
    An FsaVec containing the returned ctc graphs, with "Dim0()" the same as
    "len(symbols)"(List[List[int]]) or "dim0"(k2.RaggedTensor)
)doc";

static constexpr const char *kFsaGraphLinearFsaDoc = R"doc(
Construct an linear FSA from labels.

Note:
  The scores of arcs in the returned FSA are all 0.

Args:
  labels:
    It can be one of the following types:

        - A list of integers, e.g., `[1, 2, 3]`
        - A list of list-of-integers, e..g, `[ [1, 2], [1, 2, 3] ]`
        - An instance of :class:`k2.RaggedTensor`.
          Must have `num_axes == 2`.
  device:
    Optional. It can be either a string (e.g., 'cpu', 'cuda:0') or a
    torch.device.
    If it is ``None``, then the returned FSA is on CPU. It has to be None
    if ``labels`` is an instance of :class:`k2.RaggedTensor`.

Returns:
  - If ``labels`` is a list of integers, return an FSA
  - If ``labels`` is a list of list-of-integers, return an FsaVec
  - If ``labels`` is an instance of :class:`k2.RaggedTensor`, return
    an FsaVec
)doc";

static constexpr const char *kFsaGraphLevenshteinGraphDoc = R"doc(
Construct levenshtein graphs from symbols.

See https://github.com/k2-fsa/k2/pull/828 for more details about levenshtein
graph.

Args:
  symbols:
    It can be one of the following types:

        - A list of list-of-integers, e..g, `[ [1, 2], [1, 2, 3] ]`
        - An instance of :class:`k2.RaggedTensor`.
          Must have `num_axes == 2` and with dtype `torch.int32`.

  ins_del_score:
    The score on the self loops arcs in the graphs, the main idea of this
    score is to set insertion and deletion penalty, which will affect the
    shortest path searching produre.
  device:
    Optional. It can be either a string (e.g., 'cpu', 'cuda:0') or a
    torch.device.
    By default, the returned FSA is on CPU.
    If `symbols` is an instance of :class:`k2.RaggedTensor`, the returned
    FSA will on the same device as `k2.RaggedTensor`.

Returns:
    An FsaVec containing the returned levenshtein graphs, with "Dim0()"
    the same as "len(symbols)"(List[List[int]]) or "dim0"(k2.RaggedTensor).
)doc";
}  // namespace k2

#endif  // K2_TORCH_PYTHON_CSRC_DOC_GRAPH_H_
