/**
 * Copyright      2022  Xiaomi Corporation (authors:Daniel Povey, Wei kang)
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

#ifndef K2_CSRC_RNNT_DECODE_H_
#define K2_CSRC_RNNT_DECODE_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/array_of_ragged.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/log.h"
#include "k2/csrc/macros.h"

namespace k2 {
namespace rnnt_decoding {

/*
  The RNN-T decoding implemented here is for what we call "modified" RNN-T, or
  equivalently, regular RNN-T but with max_sym_per_frame set to 1.  (We can
  train with the "modified" option set with some probability, in order to ensure
  that the trained model is compatible with decoding with this setting).

  - contexts are finite symbol left-contexts, of length
  RnntDecodingConfig::decoder_history_len. Conceptually they represent a list of
  `decoder_history_len` symbols; they are represented numerically as, for
  example in the length-2-history case: symbol_{t-1} +  symbol_{t-2} *
  vocab_size.

  - frames come from the transcription network, they are derived from
  sub-sampling of acoustic frames or samples.
 */

struct RnntDecodingConfig {
  RnntDecodingConfig(int32_t vocab_size, int32_t decoder_history_len,
                     double beam, int32_t max_states, int32_t max_contexts)
      : vocab_size(vocab_size),
        decoder_history_len(decoder_history_len),
        beam(beam),
        max_states(max_states),
        max_contexts(max_contexts) {
    // num_context_states = pow(vocab_size, decoder_history_len);
  }

  // vocab_size is the largest-symbol plus one.
  int32_t vocab_size;

  // decoder_history_len is the number of symbols
  // of history the decoder takes; will normally
  // be one or two ("stateless decoder"), this
  // RNN-T decoding setup does not support
  // unlimited decoder context such as with LSTMs
  int32_t decoder_history_len;

  // num_context_states == pow(vocab_size, decoder_history_len).
  // We need an unique id for each context state, think about that if vocab_size
  // equals to 10, we need 0 ~ 9 to distinguish each context state when
  // decoder_history_len is 1, and 0 ~ 99 (10 ^ 2 ids) for decoder_history_len
  // equals to 2, 0 ~ 999 (10 ^ 3 ids) for decoder_history_len equals to 3.
  // int32_t num_context_states;

  // `beam` imposes a limit on the score of a state, relative to the
  // best-scoring state on the same frame.  E.g. 10.
  double beam;

  // `max_states` is a limit on the number of distinct states that we allow per
  // frame, per stream; the number of states will not be allowed to exceed
  // this limit.
  int32_t max_states;

  // `max_contexts` is a limit on the number of distinct contexts that we allow
  // per frame, per stream; the number of contexts will not be allowed to
  // exceed this limit.
  int32_t max_contexts;
};

struct ArcInfo {
  // The arc-index within the RnntDecodingStream::graph that corresponds to this
  // arc, or -1 if this arc is a "termination symbol" (these do not appear in
  // the graph).
  int32_t graph_arc_idx01;

  // The score on the arc; contains both the graph score (if any) and the score
  // from the RNN-T joiner.
  float score;

  // label of this arc (i.e. the label of the associated arc in the decoding
  // graph. We keep it here to make identifying final arcs easier.
  int32_t label;

  // dest_state is the state index within the array of states on the next frame;
  // it would be an (idx1 or idx2) depending whether this is part of an
  // RnntDecodingStream or RnntDecodingStreams object.
  int32_t dest_state;
};

struct RnntDecodingStream {
  // `graph` is a pointer to the FSA (decoding graph) that we are decoding this
  // stream with.  Different streams might have different graphs.  This must
  // be an Fsa, not FsaVec (i.e. 2 axes).
  std::shared_ptr<Fsa> graph;

  // The states number of the graph, equals to graph->shape.Dim0().
  int32_t num_graph_states;

  // `states` contains int64_t which represents the decoder state; this is:
  //    state_idx = context_state * num_graph_states + graph_state.
  // `states` would be indexed
  // [context_state][state], i.e. the states are grouped first
  // by context_state (they are sorted, to make this possible).
  Ragged<int64_t> states;

  // `scores` contains the forward scores of the states in `states`;
  // it has the same shape as `states`.
  Ragged<double> scores;

  // frames contains the arc information, for previously decoded
  // frames, that we can later use to create a lattice.
  // It contains Ragged<ArcInfo> with 2 axes (state, arc).
  std::vector<std::shared_ptr<Ragged<ArcInfo>>> prev_frames;
};

class RnntDecodingStreams {
 public:
  /* Constructor. Combines multiple RnntDecodingStream objects to create a
     RnntDecodingStreams object */
  RnntDecodingStreams(std::vector<std::shared_ptr<RnntDecodingStream>> &srcs,
                      const RnntDecodingConfig &config);

  /* This function must be called prior to evaluating the joiner network
     for a particular frame.  It tells the calling code which contexts
     it must evaluate the joiner network for.

       @param [out] shape  A RaggedShape with 2 axes, representing
                      [stream][context], will be written to here.
       @param [out] contexts  An array of shape
                      [tot_contexts][decoder_history_len], will be output to
                      here, where tot_contexts == shape->TotSize(1) and
                      decoder_history_len comes from the config, it represents
                      the number of symbols in the context of the decode
                      network (assumed to be finite). It contains the token ids
                      into the vocabulary(i.e. `0 <= value < vocab_size`).
  */
  void GetContexts(RaggedShape *shape, Array2<int32_t> *contexts);

  /*
    Advance decoding streams by one frame.

      @param [in] logprobs  Array of shape [tot_contexts][num_symbols],
                    containing log-probs of symbols given the contexts output
                    by `GetContexts()`. Will satisfy
                    logprobs.Dim0() == states_.TotSize(1).
   */
  void Advance(const Array2<float> &logprobs);

  /*
    Generate the lattice.

    Note: The prev_frames_ only contains decoded by current object, in order to
          generate the lattice we will first gather all the previous frames from
          individual streams.

      @param [in] num_frames  A vector containing the number of frames we want
                    to gather for each stream (note: the frames we have
                    ever received).
                    It MUST satisfy `num_frames.size() == num_streams_`, and
                    `num_frames[i] <= srcs_[i].prev_frames.size()`.
      @param [in] allow_partial If true and there is no final state active,
                                we will treat all the states on the last frame
                                to be final state. If false, we only
                                care about the real final state in the decoding
                                graph on the last frame when generating lattice.
      @param [out] ofsa  The output lattice will write to here, its num_axes
                         equals to 3, will be re-allocated.
      @param [out] out_map  It is an Array1 with Dim() equals to
                     ofsa.NumElements() containing the idx01 into the graph of
                     each individual streams, mapping current arc in ofsa to
                     original decoding graphs. It may contain -1 which means
                     this arc is a "termination symbol".
   */
  void FormatOutput(const std::vector<int32_t> &num_frames, bool allow_partial,
                    FsaVec *ofsa, Array1<int32_t> *out_map);

  /*
    Terminate the decoding process of current RnntDecodingStreams object, it
    will update the states & scores of each individual stream and split &
    append the prev_frames_ in current object to the `prev_frames` of the
    individual stream.

    Note: We can not decode with this object anymore after calling
    TerminateAndFlushToStreams().
   */
  void TerminateAndFlushToStreams();

  const ContextPtr &Context() const { return c_; }
  const Ragged<int64_t> &States() const { return states_; }
  const Ragged<double> &Scores() const { return scores_; }
  const Array1<int32_t> &NumGraphStates() const { return num_graph_states_; }
  int32_t NumStreams() const { return num_streams_; }

  // Note: The following three functions should be private members, they are not
  // expected to be called outside this class. We make it public because of the
  // extended lambda restrictions, see
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#extended-lambda-restrictions
  // for more details.

  /* Expand arcs according to states_.

     `states_` has a shape of [stream][context][state], each of its value is a
     combination of context_state and graph_state, that is:
     `state = context_state * num_graph_states + graph_state`. The graph_state
     is the idx0 of corresponding individual graph(with shape [state][arc]).
     This function will expand each of these states into several
     arcs(i.e. the out-going arcs of state idx0), so that we can get a new
     shape of [stream][context][state][arc].

     Caution: This function is intended to be used in `Advance()` only.

     @return Return the expected 4 axes shape
     (i.e.[stream][context][state][arc]).
   */
  RaggedShape ExpandArcs();

  /*
     Do initial pruning pass on the arcs (because it will be quite a large
     array), populating the `keep` array of a Renumbering object. The pruning
     rule is:
       (1) keep all epsilon transitions to the next frame, to ensure there is
           no way we can have no states surviving.
       (2) for all other arcs, keep it if the forward scores after the
           arc would be >= the max_scores_per_stream entry for this stream
           minus the beam from the config.

     Caution: This function is intended to be used in `Advance()` only.

      @param [in] unprund_arcs_shape   The RaggedShape returned by
                                       `ExpandArcs()`.
      @param [in] logprobs  Array of shape [tot_contexts][num_symbols],
                    containing log-probs of symbols given the contexts output
                    by `GetContexts()`. Will satisfy
                    logprobs.Dim0() == states_.TotSize(1).
                    (Note: states_.ToSize(1) == unprund_arcs_shape.Tosize(1)).

      @return Return the renumbering object indicating which arc will be kept.
   */
  Renumbering DoFisrtPassPruning(RaggedShape &unprund_arcs_shape,
                                 const Array2<float> &logprobs);
  /*
     Group states by contexts.

     `states` has a shape of [stream][arc], it contains the sorted values
     (per stream) which is:
     `state = context_state * num_graph_states + graph_state`, this guarantees
     the context_states of the states are sorted too. So that we can easily
     separate these states by finding the boundaries of context_states.

     Note: Actually we will group the states by contexts and states, because
           we need a shape of [stream][context][state][arc], obviously the
           sub-lists along axis -1 contains same values.

     Here is an example: suppose vocab_size=10, num_graph_states=10,
     decoder_history_len=2, we have a states like:

     [ [ 112 120 123 125 345 345 ] [ 123 124 567 568 670 ] ]

     the context_states are (context_state = state / num_graph_states):

     [ [ 11 12 12 12 34 34 ] [ 12 12 56 56 67 ] ]

     It will finally be grouped into ([stream][context][state][arc]):

     [ [ [ [ 112 ] ] [ [ 120 ] [ 123 ] [ 125 ] ] [ [ 345 345 ] ] ]
       [ [ [ 123 ] [ 124 ] ] [ [ 567 ] [ 568 ] ] [ [ 670 ] ] ] ]

     Caution: This function is intended to be used in `Advance()` only.

     @param [in] states  A two axes ragged tensor with each sub-list **sorted**.

     @return  Return RaggedShape with 4 axes (i.e.[stream][context][state][arc])
              it satisfies `ans.NumElements() == states.NumElements()` and
              `ans.Dim0() == states.Dim0()`.
   */
  RaggedShape GroupStatesByContexts(Ragged<int64_t> &states);

 private:
  /*
  Prune the incoming scores based on beam, max-states and max-contexts.
  Actually the beam part is not really necessary, as we already pruned
  with the beam, but it doesn't cost anything extra.
  Args:
     incoming_scores [in]  The ragged array of scores to be pruned, indexed
           [stream][context][state][arc].  The scores are per arc, but
           it's at the state and context level
           that we prune, based on settings in this->config, so entire
           sub-lists of arcs will be deleted.
     arcs_new2old [out]  The new2old map of the pruned arcs will be
           written to here.
   Returns: pruned array of incoming scores, indexed
      [stream][context][state][arc].
 */
  Ragged<double> PruneTwice(Ragged<double> &incoming_scores,
                            Array1<int32_t> *arcs_new2old);

  /*
    Gather all previously decoded frames until now, we need all the previous
    frames to generate lattice.

    Note: The prev_frames_ in current object only contains the frames from the
          point we created this object to the frame we called
          `TerminateAndFlushToStreams()` (i.e. prev_frames_.size() equals to the
          times we called `Advance()`.

      @param [in] num_frames  A vector containing the number of frames we want
                    to gather for each stream.
                    It MUST satisfy `num_frames.size() == num_streams_`, and
                    `num_frames[i] <= srcs_[i].prev_frames.size()`.
   */
  void GatherPrevFrames(const std::vector<int32_t> &num_frames);

  ContextPtr c_;

  bool attached_;  // A flag indicating whether this streams is still attached,
                   // initialized with true, only if the
                   // TerminateAndFlushToStreams() being called `attached_` will
                   // set to false, that means we can not do decoding any more.

  int32_t num_streams_;  // The number of RnntDecodingStream

  // RnntDecodingStream pointers.
  std::vector<std::shared_ptr<RnntDecodingStream>> srcs_;

  // The configuration object.
  const RnntDecodingConfig config_;

  // array of the individual graphs of the streams, with graphs.NumSrcs() ==
  // number of streams. All the graphs might actually be the same.
  Array1OfRagged<Arc> graphs_;

  // Number of graph states, per graph; this is used in constructing:
  //   state_idx = context_state * num_graph_states + graph_state.
  // for elements of `states`.
  Array1<int32_t> num_graph_states_;

  // `states` contains int64_t which represents the decoder state; this is:
  //   state = context_state * num_graph_states + graph_state.
  // the num_graph_states is specific to the decoding stream,
  // and would be an element of the array `num_graph_states`.
  //
  // `states` is indexed [stream][context_state][state], i.e.
  // i.e. the states are grouped first
  // by context_state (they are sorted, to make this possible).
  Ragged<int64_t> states_;

  // `scores` contains the forward scores of the states in `states`;
  // it has the same shape as `states`.
  Ragged<double> scores_;

  // frames contains the arc information for previously decoded
  // frames, to be split and appended to the prev_frames of the
  // individual streams when we are done with this RnnDecodingStreams
  // object. These arrays are indexed [stream][state][arc].
  std::vector<std::shared_ptr<Ragged<ArcInfo>>> prev_frames_;
};

/* Create a new decoding stream.

   Every sequence(wave data) need a decoding stream, this function is expected
   to be called when a new sequence comes. We support different decoding graphs
   for different streams.

   @param [in] graph  The decoding graph used in this stream.

   @return  The pointer to this decoding stream, which will be combined into
            `RnntDecodingStreams` to do decoding together with other
            sequences in parallel.
 */
std::shared_ptr<RnntDecodingStream> CreateStream(
    const std::shared_ptr<Fsa> &graph);

}  // namespace rnnt_decoding
}  // namespace k2

#endif  // K2_CSRC_RNNT_DECODE_H_
