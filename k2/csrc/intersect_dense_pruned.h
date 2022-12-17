/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Wei Kang)
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

#ifndef K2_CSRC_INTERSECT_DENSE_PRUNED_H_
#define K2_CSRC_INTERSECT_DENSE_PRUNED_H_

#include <memory>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/ragged_ops.h"

namespace intersect_pruned_internal {

/* Information associated with a state active on a particular frame..  */
struct StateInfo {
  /* abs_state_id is the state-index in a_fsas_.  Note: the idx0 in here
     won't necessarily match the idx0 within FrameInfo::state if
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

/*
static std::ostream &operator<<(std::ostream &os, const StateInfo &s) {
  os << "StateInfo{" << s.a_fsas_state_idx01 << ","
     << OrderedIntToFloat(s.forward_loglike) << "," << s.backward_loglike
     << "}";
  return os;
}
static std::ostream &operator<<(std::ostream &os, const ArcInfo &a) {
  os << "ArcInfo{" << a.a_fsas_arc_idx012 << "," << a.arc_loglike << ","
     << a.u.dest_a_fsas_state_idx01 << "," << a.end_loglike
     << "[i=" << FloatToOrderedInt(a.end_loglike) << "]"
     << "}";
  return os;
}
*/

// The information we have for each frame of the pruned-intersection (really:
// decoding) algorithm.  We keep an array of these, one for each frame, up to
// the length of the longest sequence we're decoding plus one.
struct FrameInfo {
  // States that are active at the beginning of this frame.  Indexed
  // [fsa_idx][state_idx], where fsa_idx indexes b_fsas_ (and a_fsas_, if
  // a_fsas_stride_ != 0); and state_idx just enumerates the active states
  // on this frame (as state_idx01's in a_fsas_).
  k2::Ragged<StateInfo> states;  // 2 axes: fsa, state

  // Indexed [fsa_idx][state_idx][arc_idx].. the first 2 indexes are
  // the same as those into 'states' (the first 2 levels of the structure
  // are shared), and the last one enumerates the arcs leaving each of those
  // states.
  //
  // Note: there may be indexes [fsa_idx] that have no states (because that
  // FSA had fewer frames than the max), and indexes [fsa_idx][state_idx] that
  // have no arcs due to pruning.
  k2::Ragged<ArcInfo> arcs;  // 3 axes: fsa, state, arc
};

}  // namespace intersect_pruned_internal

namespace k2 {
class MultiGraphDenseIntersectPruned;

// DecodeStateInfo contains the history decoding states for each sequence, this
// is normally constructed from `frames_` in MultiGraphDenseIntersectPruned
// bu using `Stack` and `Unstack`.
struct DecodeStateInfo {
  // States that survived for the previously decoded frames. Indexed
  // [frame_idx][state_idx], state_idx just enumerates the active states
  // on this frame (as state_idx01's in a_fsas_).
  //
  // Note: frame_idx may be larger than the real number of frames decoded, it
  // may contain empty lists as this is normally the output of `Unstack`.
  Ragged<intersect_pruned_internal::StateInfo> states;  // 2 axes: frame, state

  // Indexed [frame_idx][state_idx][arc_idx].. the first 2 indexes are
  // the same as those into 'states' (the first 2 levels of the structure
  // are shared), and the last one enumerates the arcs leaving each of those
  // states.
  Ragged<intersect_pruned_internal::ArcInfo> arcs;  // 3 axes: frame, state, arc

  // current search beam for this sequence
  float beam;
};


/**
     Pruned intersection (a.k.a. composition) that corresponds to decoding for
     speech recognition-type tasks for online fashion.
       @param [in] a_fsas  The decoding graphs, one per sequence.  E.g. might
                           just be a linear sequence of phones, or might be
                           something more complicated.  Must have either the
                           same Dim0() as b_fsas, or Dim0()==1 in which
                           case the graph is shared.
       @param [in] num_seqs  The number of sequences to do intersection at a
                             time, i.e. batch size. The input DenseFsaVec in
                             `Intersect` function MUST have `Dim0()` equals to
                             this.
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
class OnlineDenseIntersecter {
 public:
    OnlineDenseIntersecter(FsaVec &a_fsas, int32_t num_seqs, float search_beam,
                      float output_beam, int32_t min_states,
                      int32_t max_states);

    /* Does intersection/composition for current chunk of nnet_output(given
       by a DenseFsaVec), sequences in every chunk may come from different
       sources.
         @param [in] b_fsas  The neural-net output, with each frame containing
                             the log-likes of each phone.
         @param [in,out] decode_states  History decoding states for current
                           batch of sequences, its size equals to
                           `b_fsas.Dim0()`, and each element in `decode_states`
                           belong to the fsa in `b_fsas` at the corresponding
                           position. For a new sequence(i.e. has no history
                           states), just put an empty DecodeStateInfo unique_ptr
                           to the corresponding position.
                           `decode_states` will be updated in this function,
                           so you can use them in the following chunks.
         @param [out] ofsa  An FsaVec where the output lattice would write to,
                        will be re-allocated. The output lattice has 3 axes
                        [seqs][states][arcs].
         @param [out] arc_map_a  At exit a map from arc-indexes in `ofsa` to
                        their source arc-indexes in `a_fsa_`(the decoding graph)
                        will have been assigned to this location.
     */
    void Decode(DenseFsaVec &b_fsas,
                std::vector<std::shared_ptr<DecodeStateInfo>> *decode_states,
                FsaVec *ofsa, Array1<int32_t> *arc_map_a);

    ContextPtr &Context() { return c_;}
    ~OnlineDenseIntersecter();

 private:
    ContextPtr c_;
    float search_beam_;
    MultiGraphDenseIntersectPruned* impl_;
};
};  // namespace k2

#endif  // K2_CSRC_INTERSECT_DENSE_PRUNED_H_
