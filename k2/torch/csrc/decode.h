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

#ifndef K2_TORCH_CSRC_DECODE_H_
#define K2_TORCH_CSRC_DECODE_H_

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/rnnt_decode.h"
#include "k2/torch/csrc/fsa_class.h"
#include "torch/script.h"

namespace k2 {

/** Get decoding lattice from a neural network output and a decoding graph.

    @param nnet_output  A 3-D tensor with dtype torch.float32. It is usally
                        the last layer of the neural network model, e.g.,
                        the output of `log-softmax` layer. It has shape
                        `(N, T, C)`.
    @param decoding_graph  It is an FsaClass. It usually contains only one
                           graph. For instance, when using CTC decoding,
                           it contains a single CTC topo graph; when using
                           HLG decoding, it contains a single HLG graph.

   @param supervision_segments  A 2-D tensor with dtype torch.int32.
                                Please refer to `k2::CreateDenseFsaVec()`
                                for its format.
   @param search_beam  See `k2::IntersectDensePruned()` for its meaning.
   @param output_beam  See `k2::IntersectDensePruned()` for its meaning.
   @param min_activate_states  See `k2::IntersectDensePruned()` for its meaning.
   @param max_activate_states  See `k2::IntersectDensePruned()` for its meaning.
   @param subsampling_factor  The subsampling factor of the model.

   @return Return an FsaClass, which contains the intersection of decoding graph
           and the FSA constructed from `nnet_output`. All the attributes of the
           decoding_graph are propagated the returned FsaClass as well.
 */
FsaClass GetLattice(torch::Tensor nnet_output, FsaClass &decoding_graph,
                    torch::Tensor supervision_segments, float search_beam,
                    float output_beam, int32_t min_activate_states,
                    int32_t max_activate_states, int32_t subsampling_factor);

/** Get aux labels of each FSA contained in the lattice.

    @param lattice An FsaVec containing linear FSAs. It can be the return
                   value of `OneBestDecoding()`.

    @return Return a ragged array with two axes [utt][aux_label].
 */
Ragged<int32_t> GetTexts(FsaClass &lattice);

/** Rescore a lattice with an n-gram LM.

    @param G  An acceptor. It MUST be an FsaVec containing only one
              arc-sorted FSA. Also, it contains epsilon self loops
              (see AddEpsilonSelfLoops()). It contains only one tensor
              attribute: "lm_scores".
    @param ngram_lm_scale  The scale value for ngram LM scores.
    @param lattice The input/output lattice. It can be the
                   return value of `GetLattice()`.
 */
void WholeLatticeRescoring(FsaClass &G, float ngram_lm_scale,
                           FsaClass *lattice);

/** Get the best path of a given lattice.

    @param lattice  The given lattice.
    @param use_max  True to use max operation to select the hypothesis with the
                    largest log_prob when there are duplicate hypotheses; False
                    to use log-add.
    @param num_paths  Number of paths to sample when generating Nbest. Only used
                      when use_max equals to false.
    @param nbest_scale  The scale value applying to lattice.score before
                        sampling. Only used when use_max equals to false.

    @return  Return the lattice containing the best paths for each Fsa.
 */
FsaClass GetBestPaths(FsaClass &lattice, bool use_max, int32_t num_paths,
                      float nbest_scale);

/** Advance a chunk of frames for rnnt decoding.

    @param streams  The rnnt decoding streams.
    @param module  Jit script module containing "decoder_forword" and
                   "joiner_forward" methods.
    @param encoder_outs  The output of rnnt encoder which has a shape of
                         (B, T, C), B (i.e. the batch size) equals to
                         streams.NumStreams(). T is the chunk size. C is the
                         embedding dimension.

    Note: streams.TerminateAndFlushToStreams() will be invoked in this function,
          so all the decoding results will be flushed back to the individual
          streams belonging to the corresponding sequences.
 */
void DecodeOneChunk(rnnt_decoding::RnntDecodingStreams &streams,
                    torch::jit::script::Module module,
                    torch::Tensor encoder_outs);

}  // namespace k2

#endif  // K2_TORCH_CSRC_DECODE_H_
