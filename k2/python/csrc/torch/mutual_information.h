/**
 * @copyright
 * Copyright      2021  Xiaomi Corporation (authors: Daniel Povey)
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

#ifndef K2_PYTHON_CSRC_TORCH_MUTUAL_INFORMATION_H_
#define K2_PYTHON_CSRC_TORCH_MUTUAL_INFORMATION_H_

#include <torch/extension.h>

#include <vector>

#include "k2/python/csrc/torch.h"

namespace k2 {
/*
  Forward of mutual_information.  See also comment of `mutual_information`
  in mutual_information.py.  This is the core recursion
  in the sequence-to-sequence mutual information computation.

    @param px  Tensor of shape [B][S][T + 1] if not modified, [B][S][T] if
        modified. `modified` can be worked out from this. In not-modified case,
        it can be thought of as the log-odds ratio of generating the next x in
        the sequence, i.e.
        xy[b][s][t] is the log of
          p(x_s | x_0..x_{s-1}, y_0..y_{t-1}) / p(x_s),
        i.e. the log-prob of generating x_s given subsequences of
        lengths (s, t), divided by the prior probability of generating x_s.
        (See mutual_information.py for more info).
    @param py  The log-odds ratio of generating the next y in the sequence.
               Shape [B][S + 1][T]
    @param p   This function writes to p[b][s][t] the mutual information between
               sub-sequences of x and y of length s and t respectively, from the
               b'th sequences in the batch.  Its shape is [B][S + 1][T + 1].
               Concretely, this function implements the following recursion,
               in the case where s_begin == t_begin == 0:

                p[b,0,0] = 0.0
               if not modified:
                 p[b,s,t] = log_add(p[b,s-1,t] + px[b,s-1,t],
                                   p[b,s,t-1] + py[b,s,t-1])
               if modified:
                 p[b,s,t] = log_add(p[b,s-1,t-1] + px[b,s-1,t-1],
                                   p[b,s,t-1] + py[b,s,t-1])
               ...     treating values with any -1 index as -infinity.
               .. if `boundary` is set, we start fom p[b,s_begin,t_begin]=0.0.
    @param boundary  If set, a tensor of shape [B][4] of type int64_t, which
                     contains, where for each batch element b, boundary[b]
                     equals [s_begin, t_begin, s_end, t_end]
                     which are the beginning and end (i.e. one-past-the-last)
                     of the x and y sequences that we should process.
                     Alternatively, may be a tensor of shape [0][0] and type
                     int64_t; the elements will default to (0, 0, S, T).
    @return A tensor `ans` of shape [B], where this function will set
               ans[b] = p[b][s_end][t_end],
               with s_end and t_end being (S, T) if `boundary` was specified,
               and (boundary[b][2], boundary[b][3]) otherwise.
               `ans` represents the mutual information between each pair of
               sequences (i.e. x[b] and y[b], although the sequences are not
               supplied directly to this function).

   The block-dim and grid-dim must both be 1-dimensional, and the block-dim must
   be at least 128.
*/
torch::Tensor MutualInformationCpu(
    torch::Tensor px,                         // [B][S][T+1]
    torch::Tensor py,                         // [B][S+1][T]
    torch::optional<torch::Tensor> boundary,  // [B][4], int64_t.
    torch::Tensor p);                         //  [B][S+1][T+1]; an output

torch::Tensor MutualInformationCuda(
    torch::Tensor px,  // [B][S][T+1] if !modified, [B][S][T] if modified.
    torch::Tensor py,  // [B][S+1][T]
    torch::optional<torch::Tensor> boundary,  // [B][4], int64_t.
    torch::Tensor p);                         //  [B][S+1][T+1]; an output

/*
  backward of mutual_information; returns (grad_px, grad_py)

  if overwrite_ans_grad == true, this function will overwrite ans_grad with a
  value that, if the computation worked correctly, should be identical to or
  very close to the value of ans_grad at entry.  This can be used
  to validate the correctness of this code.
*/
std::vector<torch::Tensor> MutualInformationBackwardCpu(
    torch::Tensor px, torch::Tensor py, torch::optional<torch::Tensor> boundary,
    torch::Tensor p, torch::Tensor ans_grad);

std::vector<torch::Tensor> MutualInformationBackwardCuda(
    torch::Tensor px, torch::Tensor py, torch::optional<torch::Tensor> boundary,
    torch::Tensor p, torch::Tensor ans_grad, bool overwrite_ans_grad);

}  // namespace k2

void PybindMutualInformation(py::module &m);

#endif  // K2_PYTHON_CSRC_TORCH_MUTUAL_INFORMATION_H_
