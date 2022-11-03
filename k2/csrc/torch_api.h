/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#ifndef K2_CSRC_TORCH_API_H_
#define K2_CSRC_TORCH_API_H_

#include <memory>
#include <string>
#include <vector>

#include "torch/script.h"

namespace k2 {

class RaggedShape;
using RaggedShapePtr = std::shared_ptr<RaggedShape>;

/** Compute the exclusive sum of "src".
 *
 * @param src A 1-D tensor of dtype torch.int32.
 * @param dst A 1-D tensor of dtype torch.int32 that should have the same
 *            number of elements as src. On return, dst[0] is always 0.
 *            dst[i] = sum_{j=0}^{i-1} src[j] for i > 0
 *            Note: src and dst can share the same address.
 */
void ExclusiveSum(torch::Tensor src, torch::Tensor *dst);

/** Create a ragged shape by specifying its row_splits and row_ids.
 *
 * Note: You have to provide at least one of them.
 *
 * @param row_splits If not empty, it is a 1-D tensor with dtype torch.int32.
 * @param row_ids If not empty, it is a 1-D tensor with dtype torch.int32
 * @param cached_tot_size If not -1, it contains the total number of elements
 *                        in the shape.
 *
 * @return Return a ragged shape with 2 axes.
 */
RaggedShapePtr RaggedShape2(torch::Tensor row_splits, torch::Tensor row_ids,
                            int32_t cached_tot_size = -1);

/** Return shape->TotSize(axis);
 *
 * Refer to the help information of RaggedShape::TotSize().
 */
int32_t TotSize(RaggedShapePtr shape, int32_t axis);

/** Return shape->RowIds(axis);
 *
 * Refer to the help information of RaggedShape::RowIds().
 */
torch::Tensor RowIds(RaggedShapePtr shape, int32_t axis);

/** Return shape->RowSplits(axis);
 *
 * Refer to the help information of RaggedShape::RowSplits().
 */
torch::Tensor RowSplits(RaggedShapePtr shape, int32_t axis);

class FsaClass;
using FsaClassPtr = std::shared_ptr<FsaClass>;

/* Create a CTC topology.

   Note:
     A standard CTC topology is the conventional one, where there
     is a mandatory blank between two repeated neighboring symbols.
     A non-standard, i.e., modified CTC topology, imposes no such constraint.

   @param max_token  The maximum token ID (inclusive). We assume that token IDs
                     are contiguous (from 1 to `max_token`). 0 represents blank.
   @param modified  If False, create a standard CTC topology. Otherwise, create
                    a modified CTC topology.
   @param device  A torch.device indicating what device the returned Fsa will
                  be. Default torch::CPU.
   @return  Return either a standard or a modified CTC topology as an FSA
            depending on whether `modified` is false or true.
 */
FsaClassPtr GetCtcTopo(int32_t max_token, bool modified = false,
                       torch::Device device = torch::kCPU);

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
FsaClassPtr LoadFsa(const std::string &filename,
                    torch::Device map_location = torch::kCPU);

/** Run CTC decode.
 * @param log_softmax_out A tensor of shape (N, T, C) containing the output
 *                        from a log_softmax layer.
 * @param log_softmax_out_lens  A tensor of shape (N,) containing the number
 *                              of valid frames in log_softmax_out before
 *                              padding.
 * @param decoding_graph  Can be either the return value of CtcTopo() or
 *                        an HLG returned from LoadFsa()
 *
 * @return Return the decoding results of size `N`. ans[i] is the result
 *         for the i-th utterance. If the decoding_graph is a CtcTopo,
 *         then the decoding result contains token IDs; if the decoding_graph
 *         is an HLG, then the decoding result contains word IDs.
 *         Note: The decoding result does not contain repeats and does not
 *         contain blanks.
 */
std::vector<std::vector<int32_t>> Decode(torch::Tensor log_softmax_out,
                                         torch::Tensor log_softmax_out_lens,
                                         FsaClassPtr decoding_graph);

}  // namespace k2

#endif  // K2_CSRC_TORCH_API_H_
