/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
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
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/log.h"
#include "k2/csrc/macros.h"

namespace k2 {
namespace rnnt_decoding {

/*
  The RNN-T decoding implemented here is for what we call "modified" RNN-T, or equivalently,
  regular RNN-T but with max_sym_per_frame set to 1.  (We can train with the "modified"
  option set with some probability, in order to ensure that the trained model is
  compatible with decoding with this setting).
 */
struct RnntDecodingConfig {
  // vocab_size is the largest-symbol plus one.
  int32_t vocab_size;
  // decoder_history_len is the number of symbols
  // of history the decoder takes; will normally
  // be one or two ("stateless decoder"), this
  // RNN-T decoding setup does not support
  // unlimited decoder context such as with LSTMs
  int32_t decoder_history_len;

  // num_context_states == pow(vocab_size, decoder_history_len).
  int32_t num_context_states;

  //// max_symbols_per_frame is the maximum number of symbols that we allow to be decoded per frame.
  // NO: this is just 1.
  //int32_t max_symbols_per_frame;


  // `beam` imposes a limit on the score of a state, relative to the best-scoring
  // state on the same frame.  E.g. 10.
  float beam;

  // If > 0, `max_arcs` is a limit on the number of arcs that we allow per
  // sub-frame, per stream; the number of arcs will not be allowed to exceed this limit.
  //??  int32_t max_arcs;

  // `max_states` is a limit on the number of distinct states that we allow per
  // sub-frame, per stream; the number of states will not be allowed to exceed
  // this limit.
  int32_t max_states;
  // `max_contexts` is a limit on the number of distinct contexts that we allow
  // per sub-frame, per stream; the number of contexts will not be allowed to exceed
  // this limit.  (search below for "- contexts" for documentation on contexts).
  int32_t max_contexts;
};



/*
  - contexts are finite symbol left-contexts, of length RnntDecodingConfig::decoder_history_len.
    conceptuatly they represent a list of `decoder_history_len` symbols; they are represented
    numerically as, for example in the length-2-history case:
       symbol_{t-1} +  symbol_{t-2} * num_symbols.

  - frames come from the predictor network, they are derived from sub-sampling of acoustic
    frames or samples.

  - steps are steps of generating symbols within a frame, numbered
    0, 1, ... max_symbols_per_frame; on step 0, we have generated no symbols yet on this
    frame.

  - sub-frames combine frame and step.  Let us write M == max_symbols_per_frame; so there
    are M+1 sub-frames per frame, and the sub-frames are numbered:
      sub_frame = frame * (M+1) + step

  - "destination sub-frame": the "destination sub-frame" from sub-frame m means:
     sub-frame m+1 if the label is not the termination symbol; or sub-frame M*((m/M)+1),
     i.e. the first sub-frame of the next frame, if the label is the termination symbol.

  Note: on the last sub-frame of a frame, the only symbol allowed is the termination
  symbol.
*/

struct ArcInfo {
  // The arc-index within the RnntDecodingStream::graph that corresponds to this
  // arc, or -1 if this arc is a "termination symbol" (these do not appear in the
  // graph).
  int32_t graph_arc_idx01;

  // The score on the arc; contains both the graph score (if any) and the score
  // from the RNN-T joiner.
  float score;

  // ?? label on the arc, might delete.
  int32_t label;

  // dest_state is the state index within the array of states on the next frame; it
  // would be an (idx1 or idx2) depending whether this is part of an RnntDecodingStream
  // or RnntDecodingStreams object. [TODO: revisit this?]
  int32_t dest_state;
};




class RnntDecodingFrame {
  // `states` contains int64_t which represents the decoder state; this is:
  //   context_state * num_graph_states + graph_state.
  // the num_graph_states is specific to the decoding stream;
  // it is held at an outer level, in the RnntDecodingStream
  // or RnntDecodingStreams object.
  //
  // For a single RNN-T stream, `states` would be indexed
  // [context_state][state], i.e. the states are grouped first
  // by context_state (they are sorted, to make this possible).
  // If this object represents multiple RNN-T streams,
  // the `states` object is indexed [stream][context_state][state].
  Ragged<int64_t> states;

  // `scores` contains the forward scores of the states in `states`;
  // it has the same shape as `states`.
  Ragged<double> scores;

  // max_scores contains the best scores per stream, used for pruning;
  // for a single stream, will be a vector containing just one
  // element.
  Array1<double> max_scores;

  // For single RNN-T stream, `arcs` would be indexed [state][arc];
  // for multiple RNN-T streams, would be indexed [stream][state][arc].
  Ragged<ArcInfo> arcs;
};



// Decoding process:
// have Ragged<int64_t> for states (3 axes: stream, context, state)  <-- keep these for all sub-frames.
// have Ragged<ArcInfo> for arcs (4 axes: stream, context, state, arc)
//
// Decoding process:
// Start from sub-step 0, Ragged<int64_t> for states, plus Ragged<double> for forward scores of
// states.
//
// First stage of pruning: use current max_scores plus regular beam to write keep (??but always
// keep epsilon arcs), get old2new.
//
// Produce Ragged<ArcInfo>, dest_state just
// position in array.  Shape is: [stream,context,state,arc]
//
// Also write Ragged<int64_t> for state identities from this initially-pruned
// array, and Ragged<double> for forward score.   Shape is: [stream,state]
//
// Sort the Ragged<int64_t> on state identities, and get shape in terms of
// contexts.  Shape is: [stream,state,context].  Use same map on the double
// forward_score.  Store new2old map becuase we need it to map state identities
// later, [actually need to write it as an old2new, use InvertPermutation()?]
//
// Pruning on the double forward_score: 2-level pruning, keeping at most
// num_ctx contexts and at most num_states states.  This pruning can
// be done with 2 separate calls.
//
//







class RnntDecodingStream {
  // `graph` is a pointer to the FSA (decoding graph) that we are decoding this
  // stream with.  Different streams might have different graphs.  This must
  // be an Fsa, not FsaVec (i.e. 2 axes).
  std::shared_ptr<Fsa> graph;

  // The frame to which `states` corresponds, and which is the most recent
  // frame on which we have states, and have arcs leading to it.
  int32_t cur_frame;


  // `states` contains int64_t which represents the decoder state; this is:
  //    state_idx = context_state * num_graph_states + graph_state.
  // the num_graph_states is specific to the decoding stream;
  // it is held at an outer level, in the RnntDecodingStream
  // or RnntDecodingStreams object.
  //
  // For a single RNN-T stream, `states` would be indexed
  // [context_state][state], i.e. the states are grouped first
  // by context_state (they are sorted, to make this possible).
  // If this object represents multiple RNN-T streams,
  // the `states` object is indexed [stream][context_state][state].
  Ragged<int64_t> states;

  // `scores` contains the forward scores of the states in `states`;
  // it has the same shape as `states`.
  Ragged<double> scores;

  // forward_scores contains the forward scores of the states in `states.values`
  // (best score from start state to here); the shape is the same as
  // `states`
  Ragged<double> forward_scores;

  // frames contains the arc information, for previously decoded
  // frames, that we can later use to create a lattice.
  // It contains Ragged<ArcInfo> with 2 axes (state, arc).
  std::vector<std::unique_ptr<Ragged<ArcInfo> > prev_frames;
};


class RnntDecodingStreams {
 public:
  // per stream, contains the frame to which `states` corresponds, which is the most recent
  // frame on which we have states.
  std::vector<int32_t> cur_frame;

  // array of the individual graphs of the streams, with graphs.NumSrcs() == number of streams.
  // All the graphs might actually be the same.
  Array1OfRagged<Arc> graphs;

  // Number of graph states, per graph; this is used in constructing:
  //   state_idx = context_state * num_graph_states + graph_state.
  // for elements of `states`.
  Array<int32_t> num_graph_states;

  // `states` contains int64_t which represents the decoder state; this is:
  //   state = context_state * num_graph_states + graph_state.
  // the num_graph_states is specific to the decoding stream,
  // and would be an element of the array `num_graph_states`.
  //
  // `states` is indexed [stream][context_state][state], i.e.
  // i.e. the states are grouped first
  // by context_state (they are sorted, to make this possible).
  // If this object represents multiple RNN-T streams,
  // the `states` object is indexed [stream][context_state][state].
  Ragged<int64_t> states;

  // `scores` contains the forward scores of the states in `states`;
  // it has the same shape as `states`.
  Ragged<double> scores;


  // max_scores contains the best scores per stream, used for pruning.
  Array1<double> max_scores;

  // frames contains the arc information for previously decoded
  // frames, to be split and appended to the prev_frames of the
  // individual streams when we are done with this RnnDecodingStreams
  // object.
  // These arrays are indexed [stream][state][arc].
  std::vector<std::unique_ptr<Ragged<ArcInfo> > prev_frames;

  /* Constructor.  Combines multiple RnntDecodingStream objects to create a
     RnntDecodingStreams object */
  RnntDecodingStreams(RnntDecodingStream *srcs, int32_t num_srcs);


  /* This function must be called prior to evaluating the joiner network
     for a particular frame.  It tells the calling code which contexts
     it must evaluate the joiner network for.

     Args:
          shape [out]  A RaggedShape with 2 axes, representing [stream][context],
                    will be written to here.
          contexts [out]   An array of shape [tot_contexts][decoder_history_len],
                     will be output to here, where tot_contexts == shape->TotSize(1)
                     and decoder_history_len comes from the config, it represents
                     the number of symbols in the context of the decode network
                     (assumed to be finite).
  */
  void GetContexts(RaggedShape *shape, Array2<int64_t> *contexts);

  /*
    Advance decoding streams by one frame.  Args:

      logprobs [in]   Array of shape [tot_contexts][num_symbols], containing log-probs
                  of symbols given the contexts output by `GetContexts()`.  Will
                  satisfy logprobs.Dim0() == states.TotSize(1).

   */
  void Advance(Array2<float> &logprobs);


  const RnntDecodingConfig config,

  Array2<int64_t>

};





class RnntDecodingStreams {
 public:
  RnntDecodingStreams(std::vector<std::unique_ptr<RnntDecodingStream &streams);


  std::vector<int32_t> cur_sub_frame;


  Ragged<int64_t>



};


// Create a new decoding stream.
std::shared_ptr<RnntDecodingStream> CreateStream(
    std::shared_ptr<Fsa> graph,
    const RnntDecodingConfig  &config);



/*
  This is a utility function to be used in RNN-T decoding.  It performs pruning
  of 3 different types simultaneously, so the tightest of the 3 constraints will
  apply.

    @param [in] shape  A ragged shape with 2 axes; in practice, shape.Dim0()
                     will be the number of decoding streams, and shape.TotSize(1)
                     will be the total number of states over all streams.
                     Each stream must have at least one state.
    @param [in] scores   The scores we want to prune on, such that within each
                    decoding stream, the highest scores are kept.
                    Must satisfy scores.Dim() == shape.TotSize(1).
    @param [in] categories An array of with categories->Dim() == shape.TotSize(1),
                   that streams states into categories so we can impose a maximum
                   number of categories per stream, and also sort the states
                   by category  (The categories will actually correspond to
                   limited-symbol "contexts", e.g. 1 or 2 symbols of left-context).
    @param [in] beam  The pruning beam, as a difference in scores.  Within
                    a stream, we will not keep states whose score is less
                    than best_score - beam.
    @param [in]  max_per_stream   The maximum number of states to keep
                    per stream, if >0.  (If <=0, there is no such constraint).
    @param [in]  max_per_category   The maximum number of states to keep
                    per category per stream, if >0.  (If <=0, there is no such
                    constraint).  In practice the categories correspond to
                    "contexts", corresponding to the limited symbol histories
                    (1 or 2 symbols) that we feed to the decoder network.
    @param [out] renumbering  A renumbering object that represents the states
                 to keep, with num_old_elems == shape.TotSize(1).  Should be
                 constructed with default constructor, this function will
                 assign to it.
    @param [out]  kept_states  At exit, will be a ragged array with 3 axes
                 with indexes corresponding to [stream][category][state],
                 containing the numbers of the states we kept after pruning.
                 The elements of this ragged array can be interpreted as
                 indexes into `scores`.  The states will not be in the
                 same order as the original states, because they will be
                 sorted by category.
    @param [out] kept_categories  At exit, will be an array with
                 kept_categories->Dim() == kept_states->TotSize(1),
                 whose elements are categories (elements of the
                 input arary `categories`) to which the kept states belong.
   @param [out] best_scores  At exit, will contain the best
                score for any state, indexed by stream, i.e.
                best_scores.Dim() == shape.Dim0().

   Implementation note: if max_per_stream >= shape.TotSize(1) or
   max_per_category >= shape.TotSize(1), an optimization may be possible in which we
   can avoid sorting the scores.
 */
void PruneStreams(const RaggedShape &shape,
                  const Array1<double> &scores,
                  const Array1<int32_t> &categories,
                  float beam,
                  int32_t max_per_stream,
                  int32_t max_per_category,
                  Ragged<int32_t> *kept_states,
                  Array1<int32_t> *kept_categories,
                  Array1<double> *best_scores,
                  Array1<double> *score_cutoffs);


/*
  Overall decoding process:

  While any stream has un-processed nnet output:
     - Construct a DecodingStep object containing all the streams that still have
       un-processed nnet output.
     - For each step but last.
        - Prune streams' states + categories
        - Do nnet computation for those categories.
        - Process transitions with beam pruning, get next stateinfo -> hash.

           - Epsilon transitions match with an implicit epsilon self-loop on graph and
             go to next frame.
           - Non-epsilon transitions go to next step.
           - How to process new states??  Perhaps need separate categories for states
             on next-frame vs.

     - For last step:
        - No pruning needed.
        - Assume epsilon score is 0.
        - Create epsilon link from each active state, to the same state on
          next frame.
 */








}  // namespace rnnt_decoding
}  // namespace k2

#endif  // K2_CSRC_RNNT_DECODE_H_
