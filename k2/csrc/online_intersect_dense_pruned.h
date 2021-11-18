/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey, Wei Kang)
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

#include <algorithm>
#include <limits>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/hash.h"
#include "k2/csrc/ragged.h"

namespace k2 {

namespace online_intersect_pruned_internal {

/* Information associated with a state active on a particular frame..  */
struct StateInfo {
  /* abs_state_id is the state-index in a_fsas_.  Note: the ind0 in here
     won't necessarily match the ind0 within FrameInfo::state if
     a_fsas_stride_ == 0. */
  int32_t a_fsas_state_idx01;

  /* Caution: this is ACTUALLY A FLOAT that has been bit-twiddled using
     FloatToOrderedInt/OrderedIntToFloat so we can use atomic max.  It
     represents a Viterbi-style 'forward probability'.  (Viterbi, meaning: we
     use max not log-sum).  You can take the pruned lattice and rescore it if
     you want log-sum.  */
  int32_t forward_loglike;

  /* Note: this `backward_loglike` is the best score of any path from here to
     the end, minus the best path in the overall FSA, i.e. it's the backward
     score you get if, at the final-state, you set backward_loglike ==
     -forward_loglike. So backward_loglike + OrderedIntToFloat(forward_loglike)
     <= 0, and you can treat it somewhat like a posterior (except they don't sum
     to one as we're using max, not log-add).
  */
  float backward_loglike;
};

struct ArcInfo {              // for an arc that wasn't pruned away...
  int32_t a_fsas_arc_idx012;  // the arc-index in a_fsas_.
  float arc_loglike;          // loglike on this arc: equals loglike from data
  // (nnet output, == b_fsas), plus loglike from
  // the arc in a_fsas.

  union {
    // these 2 different ways of storing the index of the destination state
    // are used at different stages of the algorithm; we give them different
    // names for clarity.
    int32_t dest_a_fsas_state_idx01;  // The destination-state as an index
    // into a_fsas_.
    int32_t dest_info_state_idx1;  // The destination-state as an idx1 into the
                                   // next FrameInfo's `arcs` or `states`,
                                   // omitting the FSA-index which can be worked
                                   // out from the structure of this frame's
                                   // ArcInfo.
  } u;
  float end_loglike;  // loglike at the end of the arc just before
  // (conceptually) it joins the destination state.
};

// The information we have for each frame of the pruned-intersection (really:
// decoding) algorithm.  We keep an array of these, one for each frame, up to
// the length of the longest sequence we're decoding plus one.
struct FrameInfo {
  // States that are active at the beginning of this frame.  Indexed
  // [fsa_idx][state_idx], where fsa_idx indexes b_fsas_ (and a_fsas_, if
  // a_fsas_stride_ != 0); and state_idx just enumerates the active states
  // on this frame (as state_idx01's in a_fsas_).
  Ragged<StateInfo> states;  // 2 axes: fsa, state

  // Indexed [fsa_idx][state_idx][arc_idx].. the first 2 indexes are
  // the same as those into 'states' (the first 2 levels of the structure
  // are shared), and the last one enumerates the arcs leaving each of those
  // states.
  //
  // Note: there may be indexes [fsa_idx] that have no states (because that
  // FSA had fewer frames than the max), and indexes [fsa_idx][state_idx] that
  // have no arcs due to pruning.
  Ragged<ArcInfo> arcs;  // 3 axes: fsa, state, arc
};

}  // namespace online_intersect_pruned_internal

using namespace online_intersect_pruned_internal;  // NOLINT

/*
   Pruned intersection (a.k.a. composition) that corresponds to decoding for
   speech recognition-type tasks.  Can use either different decoding graphs (one
   per acoustic sequence) or a shared graph
*/
class OnlineIntersectDensePruned {
 public:
  /**
     Pruned intersection (a.k.a. composition) that corresponds to decoding for
     speech recognition-type tasks

       @param [in] a_fsas  The decoding graphs, one per sequence.  E.g. might
                           just be a linear sequence of phones, or might be
                           something more complicated.  Must have either the
                           same Dim0() as b_fsas, or Dim0()==1 in which
                           case the graph is shared.
       @param [in] num_seqs  The number of sequesce, i.e. the batch size of the
                             neural-net output
       @param [in] search_beam    "Default" search/decoding beam.  The actual
                           beam is dynamic and also depends on max_active and
                           min_active.
       @param [in] output_beam    Beam for pruning the output FSA, will
                                  typically be smaller than search_beam.
       @param [in] min_active  Minimum number of FSA states that are allowed to
                           be active on any given frame for any given
                           intersection/composition task. This is advisory,
                           in that it will try not to have fewer than this
                           number active.
       @param [in] max_active  Maximum number of FSA states that are allowed to
                           be active on any given frame for any given
                           intersection/composition task. This is advisory,
                           in that it will try not to exceed that but may not
                           always succeed.  This determines the hash size.
   */
  OnlineIntersectDensePruned(FsaVec &a_fsas, int32_t num_seqs,
                             float search_beam, float output_beam,
                             int32_t min_active, int32_t max_active);

  /* Does the main work of intersection/composition, but doesn't produce any
     output; the output is provided when you call FormatOutput().

     @param [in] b_fsas  The neural-net output, with each frame containing the
                         log-likes of each phone.  A series of sequences of
                         (in general) different length.
     @param [in] is_final A flag to tell that whether current DenseFsaVec is the
                          last chunk of the whole sequesce.
   */
  void Intersect(std::shared_ptr<DenseFsaVec> &b_fsas, bool is_final = false);

  void BackwardPass();

  // Return FrameInfo for 1st frame, with `states` set but `arcs` not set.
  std::unique_ptr<FrameInfo> InitialFrameInfo();

  /* Gets the partial/final decoding results.

     @param [out] ofsa The FsaVec to contain the decoding results.
     @param [out] arc_map_a  Will be set to a vector with Dim() equal to
                     the number of arcs in `ofsa`, whose elements contain
                     the corresponding arc_idx012 in a_fsas, i.e. decoding
                     graphs.
     @param [in] is_final False to get partial result, true to get final result.
                          You can only get the final results after doing
                          intersection for the last chunk (i.e. call Intersect
                          with is_final setting to true).
   */
  void FormatOutput(FsaVec *ofsa, Array1<int32_t> *arc_map_a,
                    bool is_final = false);

  /*
    Computes pruning cutoffs for this frame: these are the cutoffs for the
    arc "forward score", one per FSA.  This is a dynamic process involving
    dynamic_beams_ which are updated on each frame (they start off at
    search_beam_).

       @param [in] arc_end_scores  The "forward log-probs" (scores) at the
                    end of each arc, i.e. its contribution to the following
                    state.  Is a tensor indexed [fsa_id][state][arc]; we
                    will get rid of the [state] dim, combining it with the
                    [arc] dim, so it's just [fsa_id][arc]
                    It is conceptually unchanged by this operation but
    non-const because row-ids of its shape may need to be generated.
       @return      Returns a vector of log-likelihood cutoffs, one per FSA
    (the cutoff will be -infinity for FSAs that don't have any active
                    states).  The cutoffs will be of the form: the best
    score for any arc, minus the dynamic beam.  See the code for how the
    dynamic beam is adjusted; it will approach 'search_beam_' as long as the
    number of active states in each FSA is between min_active and
    max_active.
  */
  Array1<float> GetPruningCutoffs(Ragged<float> &arc_end_scores);

  /*
    Returns list of arcs on this frame, consisting of all arcs leaving
    the states active on 'cur_frame'.

       @param [in] t       The time-index (on which to look up log-likes),
                           t >= 0
       @param [in] cur_frame   The FrameInfo for the current frame; only its
                       'states' member is expected to be set up on entry.
   */
  Ragged<ArcInfo> GetArcs(int32_t t, FrameInfo *cur_frame);

  /*
    Does the forward-propagation (basically: the decoding step) and
    returns a newly allocated FrameInfo* object for the next frame.

      num_key_bits (template argument): either 32 (normal case) or 40: it is
            the number number of bits in `state_map_idx`.

      @param [in] t   Time-step that we are processing arcs leaving from;
                   will be called with t=0, t=1, ...
      @param [in] cur_frame  FrameInfo object for the states corresponding
    to time t; will have its 'states' member set up but not its 'arcs'
    member (this function will create that).
     @return  Returns FrameInfo object corresponding to time t+1; will have
    its 'states' member set up but not its 'arcs' member.
   */
  template <int32_t NUM_KEY_BITS>
  std::unique_ptr<FrameInfo> PropagateForward(int32_t t, FrameInfo *cur_frame);
  /*
    Sets backward_loglike fields of StateInfo to the negative of the forward
    prob if (this is the final-state or !only_final_probs), else -infinity.

    This is used in computing the backward loglikes/scores for purposes of
    pruning.  This may be done after we're finished decoding/intersecting,
    or while we are still decoding.

    Note: something similar to this (setting backward-prob == forward-prob)
    is also done in PropagateBackward() when we detect final-states.  That's
    needed because not all sequences have the same length, so some may have
    reached their final state earlier.  (Note: we only get to the
    final-state of a_fsas_ if we've reached the final frame of the input,
    because for non-final frames we always have -infinity as the log-prob
    corresponding to the symbol -1.)

    While we are still decoding, a background process will do pruning
    concurrently with the forward computation, for purposes of reducing
    memory usage (and so that most of the pruning can be made concurrent
    with the forward computation).  In this case we want to avoid pruning
    away anything that wouldn't have been pruned away if we were to have
    waited to the end; and it turns out that setting the backward probs to
    the negative of the forward probs (i.e.  for all states, not just final
    states) accomplishes this.  The issue was mentioned in the "Exact
    Lattice Generation.." paper and also in the code for Kaldi's
    lattice-faster-decoder; search for "As in [3], to save memory..."

      @param [in] cur_frame    Frame on which to set the backward probs
  */
  void SetBackwardProbsFinal(FrameInfo *cur_frame);

  /*
    Does backward propagation of log-likes, which means setting the
    backward_loglike field of the StateInfo variable (for cur_frame);
    and works out which arcs and which states are to be pruned
    on cur_frame; this information is output to Array1<char>'s which
    are supplied by the caller.

    These backward log-likes are normalized in such a way that you can add
    them with the forward log-likes to produce the log-likelihood ratio vs
    the best path (this will be non-positive).  (To do this, for the final
    state we have to set the backward log-like to the negative of the
    forward log-like; see SetBackwardProbsFinal()).

    This function also prunes arc-indexes on `cur_frame` and state-indexes
    on `next_frame`.

       @param [in] t      The time-index (on which to look up log-likes);
                          equals time index of `cur_frame`; t >= 0
       @param [in]  cur_frame The FrameInfo for the frame on which we want
    to set the forward log-like, and output pruning info for arcs and states
       @param [in]  next_frame The next frame's FrameInfo, on which to look
                           up log-likes for the next frame; the
                           `backward_loglike` values of states on
    `next_frame` are assumed to already be set, either by
                           SetBackwardProbsFinal() or a previous call to
                           PropagateBackward().
       @param [out] cur_frame_states_keep   An array, created by the caller,
                        to which we'll write 1s for elements of
    cur_frame->states which we need to keep, and 0s for others.
       @param [out] cur_frame_arcs_keep   An array, created by the caller,
                        to which we'll write 1s for elements of
    cur_frame->arcs which we need to keep (because they survived pruning),
                        and 0s for others.
  */
  void PropagateBackward(FrameInfo *cur_frame, FrameInfo *next_frame,
                         Array1<char> *cur_frame_states_keep,
                         Array1<char> *cur_frame_arcs_keep);
  /*
    This function does backward propagation and pruning of arcs and states
    for a specific time range.
        @param [in] begin_t   Lowest `t` value to call PropagateBackward()
    for and to prune its arcs and states.  Require t >= 0.
        @param [in] end_t    One-past-the-highest `t` value to call
    PropagateBackward() and to prune its arcs and states.  Require that
                            `frames_[t+1]` already be set up; this requires
    at least end_t <= T. Arcs on frames t >= end_t and states on frame t >
    end_t are ignored; the backward probs on time end_t are set by
    SetBackwardProbsFinal(), see its documentation to understand what this
    does if we haven't yet reached the end of one of the sequences.

    After this function is done, the arcs for `frames_[t]` with begin_t <= t
    < end_t and the states for `frames_[t]` with begin_t < t < end_t will
    have their numbering changed. (We don't renumber the states on begin_t
    because that would require the dest-states of the arcs on time `begin_t
    - 1` to be modified).  TODO: check this...
   */
  void PruneTimeRange(int32_t begin_t, int32_t end_t);

 private:
  ContextPtr c_;
  FsaVec &a_fsas_;         // Note: a_fsas_ has 3 axes.
  int32_t a_fsas_stride_;  // 1 if we use a different FSA per sequence
                           // (a_fsas_.Dim0() > 1), 0 if the decoding graph
                           // is shared (a_fsas_.Dim0() == 1).
  std::shared_ptr<DenseFsaVec> b_fsas_;
  int32_t T_;         // == b_fsas_.shape.MaxSize(1).
  int32_t num_seqs_;  // == b_fsas_.shape.Dim0().
  float search_beam_;
  float output_beam_;
  int32_t min_active_;
  int32_t max_active_;
  Array1<float> dynamic_beams_;  // dynamic beams (initially just
                                 // search_beam_ but change due to
                                 // max_active/min_active constraints).
  Array1<int32_t> final_t_;      // record the final frame id of each DenseFsa.
  int32_t reach_final_;

  std::unique_ptr<FrameInfo> partial_final_frame_;  // store the final frame for
                                                    // partial results

  int32_t state_map_fsa_stride_;  // state_map_fsa_stride_ is a_fsas_.TotSize(1)
                                  // if a_fsas_.Dim0() == 1, else 0.

  Hash state_map_;  // state_map_ maps from:
                    // key == (state_map_fsa_stride_*n) + a_fsas_state_idx01,
                    //    where n is the fsa_idx, i.e. the index into b_fsas_
                    // to
                    // value, where at different stages of
                    // PropagateForward(), value is an arc_idx012 (into
                    // cur_frame->arcs), and then later a state_idx01 into
                    // the next frame's `state` member.

  // The 1st dim is needed because If all the
  // streams share the same FSA in a_fsas_, we need
  // separate maps for each).  This map is used on
  // each frame to compute and store the mapping
  // from active states to the position in the
  // `states` array.  Between frames, all values
  // have -1 in them.
  std::vector<std::unique_ptr<FrameInfo>> frames_;
};

}  // namespace k2
