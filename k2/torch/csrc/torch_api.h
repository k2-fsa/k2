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

#ifndef K2_TORCH_CSRC_TORCH_API_H_
#define K2_TORCH_CSRC_TORCH_API_H_

#include <memory>
#include <string>
#include <utility>
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

/*
  Create a trivial graph which has only two states. On state 0, there are
  `max_token` self loops(i.e. a loop for each symbol from 1 to max_token), and
  state 1 is the final state.

    @param [in] max_token  The maximum token ID (inclusive). We assume that
                           token IDs are contiguous (from 1 to `max_token`).
    @param device  A torch.device indicating what device the returned Fsa will
                   be. Default torch::CPU.
    @return    Returns the expected trivial graph on the given device.
               Note the returned graph does not contain arcs with label being 0.
 */
FsaClassPtr GetTrivialGraph(int32_t max_token,
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
FsaClassPtr LoadFsaClass(const std::string &filename,
                         torch::Device map_location = torch::kCPU);

/** Get the lattice of CTC decode.
 * @param log_softmax_out A tensor of shape (N, T, C) containing the output
 *                        from a log_softmax layer.
 * @param log_softmax_out_lens  A tensor of shape (N,) containing the number
 *                              of valid frames in log_softmax_out before
 *                              padding.
 * @param decoding_graph  Can be either the return value of CtcTopo() or
 *                        an HLG returned from LoadFsa()
 * @param search_beam  Decoding beam, e.g. 20.  Smaller is faster, larger is
 *                     more exact (less pruning). This is the default value;
 *                     it may be modified by `min_active_states` and
 *                     `max_active_states`.
 * @param output_beam  Beam to prune output, similar to lattice-beam in Kaldi.
 *                     Relative to best path of output.
 * @param min_active_states  Minimum number of FSA states that are allowed to
 *                             be active on any given frame for any given
 *                             intersection/composition task. This is advisory,
 *                             in that it will try not to have fewer than this
 *                             number active. Set it to zero if there is no
 *                             constraint.
 * @param max_active_states  Maximum number of FSA states that are allowed to
 *                             be active on any given frame for any given
 *                             intersection/composition task. This is advisory,
 *                             in that it will try not to exceed that but may
 *                             not always succeed. You can use a very large
 *                             number if no constraint is needed.
 * @param subsampling_factor  The subsampling factor of the model.
 *
 * @return Return the decoding lattice having 3 axes with the dim0 equaling to
 *         `N`.
 */
FsaClassPtr GetLattice(torch::Tensor log_softmax_out,
                       torch::Tensor log_softmax_out_lens,
                       FsaClassPtr decoding_graph, float search_beam = 20,
                       float output_beam = 8, int32_t min_active_states = 30,
                       int32_t max_active_states = 10000,
                       int32_t subsampling_factor = 4);

/** Get the best path of a lattice.
 * @param lattice  The decoding lattice containing an FsaVec.
 *
 * @return Return the decoding results of size `lattice.dim0`. ans[i] is the
 *         result for the i-th utterance. If the decoding_graph used to generate
 *         the lattice is a CtcTopo, then the decoding result contains token
 *         IDs; if the decoding_graph used to generate the lattice is an HLG,
 *         then the decoding result contains word IDs.
 *         Note: The decoding result does not contain repeats and does not
 *         contain blanks.
 */
std::vector<std::vector<int32_t>> BestPath(const FsaClassPtr &lattice);

/* Return the best path of a lattice.
 *
 * Different from `BestPath`, this function returns a lattice.
 */
FsaClassPtr ShortestPath(const FsaClassPtr &lattice);

/** Scale the given attribute for a Fsa.
 *
 *  Note: Support only float type attributes.
 *
 *  Caution: It requires that the Fsa has the given attribute.
 *
 *  @param fsa The given Fsa.
 *  @param  scale  The value used to scale the attribute.
 *  @param attribute The attribute name.
 */
void ScaleTensorAttribute(FsaClassPtr &fsa, float scale,
                          const std::string &attribute);

/** Get tensor type attribute of a Fsa.
 *
 * @param fsa  The given Fsa.
 * @param  attribute  The attribute name.
 *
 * @return The attribute value.
 */
torch::Tensor GetTensorAttr(FsaClassPtr &fsa, const std::string &attribute);

/** Set tensor type attribute of a Fsa.
 *
 * @param fsa  The given Fsa.
 * @param  attribute  The attribute name.
 * @param  value  The value of the given attribute.
 */
void SetTensorAttr(FsaClassPtr &fsa, const std::string &attribute,
                   torch::Tensor value);

class RnntStream;
class RnntStreams;

using RnntStreamPtr = std::shared_ptr<RnntStream>;
using RnntStreamsPtr = std::shared_ptr<RnntStreams>;

/* Create a decoding stream for one sequence.

   Every sequence(wave data) need a decoding stream, this function is expected
   to be called when a new sequence comes.

   Note: It is assumed that the decoding_graph is epsilon-free.

   @param decoding_graph  The decoding graph used in this stream.

   @return  The pointer to this decoding stream, which will be combined into
            `RnntStreamsPtr` to do decoding together with other
            sequences in parallel.
 */
RnntStreamPtr CreateRnntStream(FsaClassPtr decoding_graph);

/* Combine multiple individual `RnntStream` into one `RnntStreams`.
 *
 * @param source_streams  A list of individual `RnntStream` created by the
 *                         function `CreateRnntStream` above.
 * @param vocab_size  The vocabulary size indicating how many symbols we are
 *                    using, which equals the id of largest-symbol plus one.
 * @param context_size  The number of symbols of history the decoder takes;
 *                      will normally be one or two ("stateless decoder").
 * @param beam  `beam` imposes a limit on the score of a state, relative to the
 *               best-scoring state on the same frame.  E.g. 15.
 * @param max_contexts  `max_contexts` is a limit on the number of distinct
 *                      contexts that we allow per frame, per stream; the number
 *                      of contexts will not be allowed to exceed this limit.
 * @param max_states  `max_states` is a limit on the number of distinct states
 *                    that we allow per frame, per stream; the number of states
 *                    will not be allowed to exceed this limit.
 * @return  The pointer to this RnntStreams.
 */
RnntStreamsPtr CreateRnntStreams(
    const std::vector<RnntStreamPtr> &source_streams, int32_t vocab_size,
    int32_t context_size, float beam = 15, int32_t max_contexts = 8,
    int32_t max_states = 64);

/* Get the contexts for transducer decoder.
 *
 * Note: This function must be called prior to evaluating the joiner network
 *       for a particular frame.  It tells the calling code which contexts
 *       it must evaluate the joiner network for.
 *
 * @param rnnt_streams  The RnntStreams for this batch created from the
 *                      `CreateRnntStreams` function above.
 * @return A pair of two items:
 *   - shape  A RaggedShapePtr with 2 axes, representing [stream][context].
 *   - contexts  A tensor of shape (tot_contexts, context_size),
 *              where `tot_contexts == shape->TotSize(1)` and `context_size`
 *              represents the number of symbols in the contexts of the decoder
 *              network (assumed to be finite, equals to what we passed in to
 *              construct the `rnnt_streams`).
 *              It contains the token ids into the vocabulary
 *              (i.e. `0 <= value < vocab_size`).
 */
std::pair<RaggedShapePtr, torch::Tensor> GetRnntContexts(
    RnntStreamsPtr rnnt_streams);

/*
 * Advance rnnt streams by one frame.
 *
 * @param rnnt_streams  The given decoding streams that we are decoding with.
 * @param logprobs  A tensor of shape (tot_contexts, vocab_size),
 *                 containing log-probs of symbols given the contexts output
 *                 by `GetRnntContexts()`.
 */
void AdvanceRnntStreams(RnntStreamsPtr rnnt_streams, torch::Tensor log_probs);

/*
 * Terminate the decoding process of the given `RnntStreams` object, it
 * will update the states & scores of each individual `RnntStream`
 *
 * Note: We can not decode with this rnnt_streams anymore after calling
 * TerminateAndFlushRnntStreams().
 *
 * @param rnnt_streams The RnntStream that will be terminated.
 */
void TerminateAndFlushRnntStreams(RnntStreamsPtr rnnt_streams);

/*
 * Generate the lattice.
 *
 * @param rnnt_streams The RnntStreams that is used to generate lattice from.
 *
 * @param  num_frames  A vector containing the number of frames we want
 *           to gather for each stream (note: the frames we have
 *           ever received). Its size is num_stream.
 * @param  allow_partial  If true and there is no final state active,
 *                        we will treat all the states on the last frame
 *                        to be final state. If false, we only
 *                        care about the real final state in the decoding
 *                        graph on the last frame when generating lattice.
 * @return  The generated lattice.
 */
FsaClassPtr FormatOutput(RnntStreamsPtr rnnt_streams,
                         std::vector<int32_t> num_frames,
                         bool allow_partial = false);

}  // namespace k2

#endif  // K2_TORCH_CSRC_TORCH_API_H_
