/**
 * @brief
 * compose
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <limits>
#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/ragged_ops.h"

namespace k2 {

namespace intersect_internal {

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
     forward_loglike. So backward_loglike + OrderedIntToFloat(forward_loglike)
     <= 0, and you can treat it somewhat like a posterior (except they don't sum
     to one as we're using max, not log-add).
  */
  float backward_loglike;
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
     << a.u.dest_a_fsas_state_idx01 << "," << a.end_loglike << "}";
  return os;
}
*/

}  // namespace intersect_internal

using namespace intersect_internal;  // NOLINT

// Caution: this is really a .cu file.  It contains mixed host and device code.

/*
   Intersection (a.k.a. composition) that corresponds to decoding for
   speech recognition-type tasks.  This version does only forward-backward
   pruning in the backward pass; the forward pass does no pruning.

   Can use either different decoding graphs (one per acoustic sequence) or a
   shared graph.

   How to use this object:
       Construct it
       Call Intersect()
       Call FormatOutput()
*/
class MultiGraphDenseIntersect {
 public:
  /**
     Intersection (a.k.a. composition) that corresponds to decoding for
     speech recognition-type tasks

       @param [in] a_fsas  The decoding graphs, one per sequence.  E.g. might
                           just be a linear sequence of phones, or might be
                           something more complicated.  Must have either the
                           same Dim0() as b_fsas, or Dim0() == 1 in which
                           case the graph is shared.
       @param [in] b_fsas  The neural-net output, with each frame containing the
                           log-likes of each phone.  A series of sequences of
                           (in general) different length.  MUST BE SORTED BY
                           DECREASING LENGTH, or it is an error.
                           (Calling code should be able to ensure this.)
       @param [in] output_beam    Beam >0 for pruning output, i.e. arcs that are
                           not on a path within `output_beam` of the best path
                           will not be retained.
   */
  MultiGraphDenseIntersect(FsaVec &a_fsas,
                           DenseFsaVec &b_fsas,
                           float output_beam)
      : a_fsas_(a_fsas),
        b_fsas_(b_fsas),
        output_beam_(output_beam) {
    NVTX_RANGE(__func__);
    c_ = GetContext(a_fsas.shape, b_fsas.shape);

    K2_CHECK(a_fsas_.Dim0() == b_fsas_.shape.Dim0());
    num_fsas_ = a_fsas_.Dim0();
    K2_CHECK_GT(num_fsas_, 0);
    K2_CHECK(b_fsas.scores.IsContiguous());
    K2_CHECK_GT(output_beam, 0);
    // Set up carcs_
    InitCompressedArcs();

    {
      Array1<int32_t> dest_states = GetDestStates(a_fsas_, true);
      incoming_arcs_ = GetIncomingArcs(a_fsas_, dest_states);
    }

    {
      int32_t axis = 0, num_srcs = 2;
      RaggedShape *vec1[2] = { &incoming_arcs_.shape, &a_fsas_.shape };
      forward_then_backward_shape_ = Append(axis, num_srcs, vec1);
      RaggedShape *vec2[2] = { &a_fsas_.shape, &incoming_arcs_.shape };
      backward_then_forward_shape_ = Append(axis, num_srcs, vec2);
    }


    int32_t num_arcs = a_fsas_.NumElements();
    // arc_scores_ will be used for forward and backward computations
    // simultaneously.
    arc_scores_ = Array1<float>(c_, num_arcs * 2);

    // set up fsa_info_
    InitFsaInfo();

    // set up steps_, which contains a bunch of meta-information about the steps of the algorithm.
    InitSteps();

    int32_t num_seqs = b_fsas.shape.Dim0();

    { // check that b_fsas are in order of decreasing length.
      Array1<int32_t> r = b_fsas.shape.RowSplits(1).To(GetCpuContext());
      int32_t *r_data = r.Data();
      int32_t prev_t = r_data[1] - r_data[0];
      for (int32_t i = 1; i + 1 < r.Dim(); i++) {
        int32_t this_t = r_data[i+1] - r_data[i];
        if (this_t < prev_t)
          K2_LOG(FATAL) << "Sequences (DenseFsaVec) must be in sorted from greatest to least.";
        prev_t = this_t;
      }
      T_ = r_data[1] - r_data[0];  // longest first, so T_ is the length of the
                                   // longest sequence.
    }

    int32_t num_states = a_fsas_.TotSize(1);
    // this is the largest array size we'll be dealing with.
    size_t product = ((size_t)(T_+1) * (size_t)num_states);
    K2_CHECK_EQ((1+product), (size_t)(int32_t)(1+product)) <<
        "Problem size is too large for this algorithm; try reducing minibatch size.";
  }

  /* Does the main work of intersection/composition, but doesn't produce any
     output; the output is provided when you call FormatOutput(). */
  void Intersect() {
    NVTX_RANGE(__func__);
    DoStep0();
    for (int32_t t = 1; t <= T_; t++)
      DoStep(t);
  }
  /*
    Does pruning and returns a ragged array indexed [fsa][state][arc], containing
    the result of intersection.

         @param [out] arc_map_a_out  If non-NULL, the map from (arc-index of returned
                                  FsaVec) to (arc-index in a_fsas_) will be written
                                  to here.
         @param [out] arc_map_b_out  If non-NULL, the map from (arc-index of returned
                                     FsaVec) to (offset into b_fsas_.scores.Data())
                                     will be written to here.
         @return  Returns a FsaVec that is the composed result.  Note: due to roundoff,
                      it may possibly contain states and/or arcs that are not
                      accessible or not co-accessible.  It will be top-sorted,
                      and deterministic and arc-sorted if the input a_fsas_
                      had those properties.
   */
  FsaVec FormatOutput(Array1<int32_t> *arc_map_a_out,
                      Array1<int32_t> *arc_map_b_out) {
    NVTX_RANGE(__func__);
    Array1<float> score_cutoffs = GetScoreCutoffs();
    float *score_cutoffs_data = score_cutoffs.Data();
    int32_t num_states = a_fsas_.TotSize(1);
    int32_t product = ((size_t)(T_+1) * (size_t)num_states);

    // We'll do exclusive-sum on the following array, after setting its elements
    // to 1 if the corresponding state was not pruned away.  The order of
    // 'counts' is: (T+1) copies of all the states of fsa index 0, (T+1) copies
    // of all the states of FSA index 1, and so on.  In fact not all FSAs have
    // this many frames, most of them have fewer copies, but using this regular
    // structure avoids having to compute any extra row_ids vectors and the
    // like.  The out-of-range elements will be seto to zero.

    Renumbering renumber_states(c_, product);
    char *keep_state_data = renumber_states.Keep().Data();

    int32_t T = T_;
    const int32_t *a_fsas_row_splits1_data = a_fsas_.RowSplits(1).Data();
    FsaInfo *fsa_info_data = fsa_info_.Data();

    float **forward_state_scores_data = forward_state_scores_.Data(),
         **backward_state_scores_data = backward_state_scores_.Data();

    // the following lambda will set elements within `keep_state_data` to 0 or 1.
    auto lambda_set_keep = [=] __host__ __device__ (int32_t i) -> void {
      // i is actually an idx012

      // the following works because each FSA has (its num-states * T_+1) states
      // allocated to it.  However (i / (T_+1)) does not directly map to a state
      // index.
      int32_t fsa_idx0 = a_fsas_row_splits1_data[(i / (T+1))];
      FsaInfo fsa_info = fsa_info_data[fsa_idx0];
      float cutoff = score_cutoffs_data[fsa_idx0];

      int32_t idx_within_fsa = i - (T+1) * fsa_info.state_offset,
                    t = idx_within_fsa / fsa_info.num_states,
           state_idx1 = idx_within_fsa % fsa_info.num_states,
           state_idx01 = fsa_info.state_offset + state_idx1;

      char keep = (char)0;
      if (t <= fsa_info.T) {
        // This time is within the bounds for this FSA
        const float *forward_state_scores_t = forward_state_scores_data[t],
                   *backward_state_scores_t = backward_state_scores_data[t];
        if (forward_state_scores_t[state_idx01] +
            backward_state_scores_t[state_idx01] > cutoff)
          keep = (char)1;
      }
      keep_state_data[i] = keep;
    };
    Eval(c_, product, lambda_set_keep);

    Array1<int32_t> &new2old = renumber_states.New2Old();
    const int32_t *new2old_data = new2old.Data();
    int32_t ans_tot_num_states = new2old.Dim();

    // t_per_fsa will be set below to the number of time-steps that each FSA has
    // states active on; if each FSA i has scores for 0 <= t < T_i, then
    // t_per_fsa[i] will be T_i + 1, because there is also a copy of the state
    // at time T_i.
    Array1<int32_t> t_per_fsa(c_, num_fsas_ + 1);
    int32_t *t_per_fsa_data = t_per_fsa.Data();

    // lambda is too short, we may run into compiler bug, so use if/else.
    if (c_->GetDeviceType() == kCpu) {
      for (int32_t i = 0; i < num_fsas_; i++)
        t_per_fsa_data[i] = fsa_info_data[i].T + 1;
    } else {
      auto lambda_set_t_per_fsa_etc = [=] __host__ __device__ (int32_t i) -> void {
        t_per_fsa_data[i] = fsa_info_data[i].T + 1;
      };
      EvalDevice(c_, num_fsas_, lambda_set_t_per_fsa_etc);
    }
    ExclusiveSum(t_per_fsa, &t_per_fsa);

    // now t_per_fsa is the row_splits1 of the shape we'll be returning.  It allocates
    // fsa_info_data[i].T + 1 time-indexes to the i'th fsa.
    Array1<int32_t> &ans_row_splits1 = t_per_fsa;
    const int32_t *ans_row_splits1_data = ans_row_splits1.Data();
    Array1<int32_t> ans_row_ids1(c_, t_per_fsa.Back());
    RowSplitsToRowIds(ans_row_splits1, &ans_row_ids1);

    // ans_row_ids2 maps to an ans_idx01 that combines FSA-index and time-index.
    Array1<int32_t> ans_row_ids2(c_, ans_tot_num_states);
    int32_t *ans_row_ids2_data = ans_row_ids2.Data();
    // ans_num_arcs is the number of arcs potentially active for a state; we'll
    // prune out the invalid ones later on.
    Array1<int32_t> ans_num_arcs(c_, ans_tot_num_states + 1);
    int32_t *ans_num_arcs_data = ans_num_arcs.Data();

    // ans_state_idx01 contains the state_idx01 in a_fsas_ for each state in
    // the answer.
    Array1<int32_t> ans_state_idx01(c_, ans_tot_num_states);
    int32_t *ans_state_idx01_data = ans_state_idx01.Data();
    const int32_t *a_fsas_row_splits2_data = a_fsas_.RowSplits(2).Data();

    // set ans_row_ids2_data, which contains an ans_idx01 that combines
    // FSA-index and time-index.
    auto lambda_set_row_ids2 = [=] __host__ __device__ (int32_t ans_idx012) -> void {
      // old_i is the same as the index `i` into lambda_set_keep.  It is also an idx012.
      // The logic is the same as for lambda_set_keep, we keep the code but not the
      // comments.
      int32_t old_i = new2old_data[ans_idx012];
      int32_t fsa_idx0 = a_fsas_row_splits1_data[(old_i / (T+1))];
      FsaInfo fsa_info = fsa_info_data[fsa_idx0];
      int32_t idx_within_fsa = old_i - (T+1) * fsa_info.state_offset,
                           t = idx_within_fsa / fsa_info.num_states,
           a_fsas_state_idx1 = idx_within_fsa % fsa_info.num_states;
      int32_t a_fsas_state_idx01 = fsa_info.state_offset + a_fsas_state_idx1;
      int32_t ans_fsa_idx0x = ans_row_splits1_data[fsa_idx0],
                  ans_idx01 = ans_fsa_idx0x + t;
      ans_row_ids2_data[ans_idx012] = ans_idx01;
      ans_state_idx01_data[ans_idx012] = a_fsas_state_idx1;
      // note: fsa_info.state_offset == a_fsas_row_splits2_data[a_fsas_state_idx01];
      int32_t num_arcs = a_fsas_row_splits2_data[a_fsas_state_idx01 + 1] -
              a_fsas_row_splits2_data[a_fsas_state_idx01];
      if (t == fsa_info.T)  // No arcs leave copies of states on the last frame
                            // for each FSA.
        num_arcs = 0;
      K2_CHECK_EQ(0, 0); // temp
      ans_num_arcs_data[ans_idx012] = num_arcs;
    };
    Eval(c_, ans_tot_num_states, lambda_set_row_ids2);

    Array1<int32_t> &ans_row_splits3(ans_num_arcs);
    ExclusiveSum(ans_num_arcs, &ans_row_splits3);
    int32_t tot_arcs = ans_row_splits3.Back();
    Array1<int32_t> ans_row_ids3(c_, tot_arcs);
    RowSplitsToRowIds(ans_row_splits3, &ans_row_ids3);

    // Actually we'll do one more pass of pruning on 'ans' before we return it.
    Ragged<Arc> ans(
        RaggedShape4(&ans_row_splits1, &ans_row_ids1, -1,
                     nullptr, &ans_row_ids2, ans_tot_num_states,
                     &ans_row_splits3, &ans_row_ids3, -1),
        Array1<Arc>(c_, tot_arcs));
    Arc *ans_arcs_data = ans.values.Data();

    Array1<int32_t> arc_map_a(c_, tot_arcs),
        arc_map_b(c_, tot_arcs);
    int32_t *arc_map_a_data = arc_map_a.Data(),
            *arc_map_b_data = arc_map_b.Data();

    Renumbering renumber_arcs(c_, tot_arcs);
    char *keep_arc_data = renumber_arcs.Keep().Data();

    const int32_t *ans_row_ids1_data = ans_row_ids1.Data(),
                  *ans_row_ids3_data = ans_row_ids3.Data(),
               *ans_row_splits2_data = ans.shape.RowSplits(2).Data(),
               *ans_row_splits3_data = ans_row_splits3.Data(),
                *states_old2new_data = renumber_states.Old2New().Data();
    CompressedArc *carcs_data = carcs_.Data();
    int32_t scores_stride = b_fsas_.scores.ElemStride0();
    const float *scores_data = b_fsas_.scores.Data();

    auto lambda_set_arcs_and_keep = [=] __host__ __device__ (int32_t arc_idx0123) -> void {
      int32_t ans_state_idx012 = ans_row_ids3_data[arc_idx0123],
                   ans_idx012x = ans_row_splits3_data[ans_state_idx012],
                     ans_idx01 = ans_row_ids2_data[ans_state_idx012],
                      fsa_idx0 = ans_row_ids1_data[ans_idx01],
                     ans_idx0x = ans_row_splits1_data[fsa_idx0],
                    ans_idx0xx = ans_row_splits2_data[ans_idx0x],
                        t_idx1 = ans_idx01 - ans_idx0x,
                      arc_idx3 = arc_idx0123 - ans_idx012x;
      int32_t a_fsas_state_idx01 = ans_state_idx01_data[ans_state_idx012];
      FsaInfo fsa_info = fsa_info_data[fsa_idx0];
      float cutoff = score_cutoffs_data[fsa_idx0];
      int32_t a_fsas_state_idx0x = fsa_info.state_offset,
              a_fsas_state_idx1 = a_fsas_state_idx01 - a_fsas_state_idx0x;
      int32_t a_fsas_arc_idx012 = a_fsas_row_splits2_data[a_fsas_state_idx01]
                   + arc_idx3; //  arc_idx3 is an idx2 w.r.t. a_fsas.
      CompressedArc carc = carcs_data[a_fsas_arc_idx012];
      K2_CHECK_EQ(a_fsas_state_idx1, (int32_t)carc.src_state);
      int32_t a_fsas_dest_state_idx1 = carc.dest_state,
            a_fsas_dest_state_idx01 = fsa_info.state_offset + a_fsas_dest_state_idx1;
      arc_map_a_data[arc_idx0123] = a_fsas_arc_idx012;
      int32_t scores_index = fsa_info.scores_offset + (scores_stride * t_idx1) +
                       carc.label_plus_one;
      arc_map_b_data[arc_idx0123] = scores_index;

      float arc_score = carc.score + scores_data[scores_index];

      // unpruned_src_state_idx and unpruned_dest_state_idx are into
      // `renumber_states.Keep()` or `renumber_states.Old2New()`
      int32_t unpruned_src_state_idx = fsa_info.state_offset * (T+1) +
         ((t_idx1 + 1) * fsa_info.num_states) + a_fsas_state_idx1,
         unpruned_dest_state_idx = fsa_info.state_offset * (T+1) +
         ((t_idx1 + 1) * fsa_info.num_states) + a_fsas_dest_state_idx1;
      K2_CHECK_EQ(states_old2new_data[unpruned_src_state_idx], ans_state_idx012);
      K2_CHECK_LT(t_idx1, (int32_t)fsa_info.T);

      int32_t ans_dest_state_idx012 = states_old2new_data[unpruned_dest_state_idx],
         ans_dest_state_idx012_next = states_old2new_data[unpruned_dest_state_idx + 1];
      char keep_this_arc = (char)0;

      const float *forward_state_scores_t = forward_state_scores_data[t_idx1],
                  *backward_state_scores_t1 = backward_state_scores_data[t_idx1 + 1];

      if (ans_dest_state_idx012 > ans_dest_state_idx012_next) {
        // The dest-state of this arc has a number (was not pruned away)
        float arc_forward_backward_score = forward_state_scores_t[a_fsas_state_idx01] +
                                           arc_score +
                                           backward_state_scores_t1[a_fsas_dest_state_idx01];
        if (arc_forward_backward_score > cutoff) {
          keep_this_arc = (char)1;
          Arc arc;
          arc.label = static_cast<int32_t>(carc.label_plus_one) - 1;
          // the idx12 into `ans`, which includes the 't' and 'state' indexes,
          // corresponds to the state-index in the FSA we will return (the 't' index
          // will be removed).
          int32_t src_state_idx12 = ans_state_idx012 - ans_idx0xx,
                 dest_state_idx12 = ans_dest_state_idx012 - ans_idx0xx;
          arc.src_state = src_state_idx12;
          arc.dest_state = dest_state_idx12;
          arc.score = arc_score;
          ans_arcs_data[arc_idx0123] = arc;
        }
        keep_arc_data[arc_idx0123] = keep_this_arc;
      }
    };
    Eval(c_, tot_arcs, lambda_set_arcs_and_keep);

    if (arc_map_a_out)
      *arc_map_a_out = arc_map_a[renumber_arcs.New2Old()];
    if (arc_map_b_out)
      *arc_map_b_out = arc_map_b[renumber_arcs.New2Old()];
    // subsample the output shape, removing arcs that weren't kept
    RaggedShape ans_shape_subsampled = SubsampleRaggedShape(ans.shape,
                                                            renumber_arcs);
    // .. and remove the 't' axis
    return Ragged<Arc>(RemoveAxis(ans_shape_subsampled, 1),
                       ans.values[renumber_arcs.New2Old()]);
  }

  // We can't actually make the rest private for reasons relating to use of
  // __host__ __device__ lambdas, but logically the rest would be private.

  //private:


  void InitCompressedArcs() {
    K2_LOG(FATAL) << "Not implemented";
  }

  void InitFsaInfo() {
    K2_LOG(FATAL) << "Not implemented";
  }

  /*
    InitSteps() sets up steps_; it works out metadata and allocates memory, but
    does not do any of the actual computation.
   */
  void InitSteps() {
    NVTX_RANGE(__func__);
    // This vector, of length num_fsas_, tells us how many copies of (the states
    // of the i'th decoding graph) we have.  It equals (the length of the sequence
    // of log-likes in n_fsas_) + 1.  It is monotonically decreasing (thanks to
    // how we require the FSAs to be sorted).
    Array1<int32_t> num_copies_per_fsa(c_, num_fsas_);
    const int32_t *b_row_splits_data = b_fsas_.shape.RowSplits(1).Data();
    int32_t *num_copies_per_fsa_data = num_copies_per_fsa.Data();
    auto lambda_set_num_copies = [=]  __host__ __device__ (int32_t i) -> void {
      num_copies_per_fsa_data[i] = 1 + b_row_splits_data[i + 1] - b_row_splits_data[i];
    };
    Eval(c_, num_fsas_, lambda_set_num_copies);

    std::vector<int32_t> range(num_fsas_ + 1);
    // fill with num_fsas_, num_fsas_ + 1, num_fsas_ + 2, ... num_fsas_ * 2.
    std::iota(range.begin(), range.end(), num_fsas_);
    std::vector<RaggedShape> bf_shape_prefixes = GetPrefixes(backward_then_forward_shape_,
                                                             range),
                             fb_shape_prefixes = GetPrefixes(forward_then_backward_shape_,
                                                             range);


    ContextPtr c_cpu = GetCpuContext();

    // This vector, of length T_ + 1, tells us, for each frame 0 <= t <= T, how
    // many FSAs have a copy of their decoding-graph states alive on this
    // time-index.  It equals InvertMonotonicDecreasing(num_copies_per_fsa_)
    // and it is also the case that InvertMonotonicDecreasing(num_fsas_per_t_)
    // == num_copies_per_fsa_.
    Array1<int32_t> num_fsas_per_t = InvertMonotonicDecreasing(
        num_copies_per_fsa),
                num_fsas_per_t_cpu = num_fsas_per_t.To(c_cpu);

    Array1<int32_t> a_fsas_row_splits1_cpu = a_fsas_.RowSplits(1).To(c_cpu),
                  a_fsas_row_splits12_cpu = a_fsas_.RowSplits(2)[
                      a_fsas_.RowSplits(1)].To(c_cpu);
    int32_t tot_arcs = a_fsas_.NumElements(),
          tot_states = a_fsas_.TotSize(1);

    std::vector<Step> steps(T_ + 1);

    for (int32_t t = 0; t <= T_; t++) {
      Step &step = steps[t];
      step.forward_t = t;
      step.backward_t = T_ - t;
      step.forward_before_backward = (step.forward_t <= step.backward_t);
      int32_t nf = step.forward_num_fsas = num_fsas_per_t_cpu[step.forward_t],
              nb = step.backward_num_fsas = num_fsas_per_t_cpu[step.backward_t];


      int32_t num_arcs_forward = a_fsas_row_splits12_cpu[nf],
             num_arcs_backward = a_fsas_row_splits12_cpu[nb],
            num_states_forward = a_fsas_row_splits1_cpu[nf],
           num_states_backward = a_fsas_row_splits1_cpu[nb];

      if (step.forward_before_backward) {
        // fb_shape_prefixes[nb] is incoming_arcs_.shape appended with the first
        // `nb` FSAs of a_fsas_.shape.  Note: for purposes of allocation (and
        // reduction of arcs->states) we assume that for the forward pass all
        // the FSAs are active; this may not really be true, but it keeps the
        // shapes regular.
        step.arc_scores = Ragged<float>(fb_shape_prefixes[nb],
                                        arc_scores_.Arange(0, tot_arcs +
                                                           num_arcs_backward));
        step.forward_arc_scores = arc_scores_.Arange(0, tot_arcs);
        step.backward_arc_scores = arc_scores_.Arange(tot_arcs,
                                                      tot_arcs + num_arcs_backward);
        step.state_scores = Array1<float>(c_, tot_states + num_states_backward);
        step.forward_state_scores = step.state_scores.Arange(0, tot_states);
        step.backward_state_scores = step.state_scores.Arange(0, num_states_backward);
      } else {
        step.arc_scores = Ragged<float>(bf_shape_prefixes[nb],
                                        arc_scores_.Arange(0, tot_arcs +
                                                           num_arcs_forward));
        step.backward_arc_scores = arc_scores_.Arange(0, tot_arcs);
        step.forward_arc_scores = arc_scores_.Arange(tot_arcs,
                                                     tot_arcs + num_arcs_forward);
        step.state_scores = Array1<float>(c_, tot_states + num_states_forward);
        step.backward_state_scores = step.state_scores.Arange(0, tot_states);
        step.forward_state_scores = step.state_scores.Arange(0, num_states_forward);
      }
    }
  }



  void DoStep0() {
    NVTX_RANGE(__func__);
    // Run step zero of the computation: this initializes the forward probabilities on
    // frame 0, and the backward probabilities on the last frame for each sequence.
    std::vector<float*> backward_state_scores_vec(T_ + 1);
    int32_t tot_states = a_fsas_.TotSize(1);
    for (int32_t t = 0; t <= T_; t++) {
      int32_t bt = steps_[t].backward_t;
      backward_state_scores_vec[bt] = steps_[t].backward_state_scores.Data();
    }
    backward_state_scores_ = Array1<float*>(c_, backward_state_scores_vec);
    float **backward_state_scores_data = backward_state_scores_.Data();
    float *forward_scores_t0 = steps_[0].forward_state_scores.Data();
    int32_t *a_fsas_row_ids1 = a_fsas_.RowIds(1).Data();
    FsaInfo *fsa_info_data = fsa_info_.Data();
    const float minus_inf = -std::numeric_limits<float>::infinity();
    auto lambda_init_state_scores = [=] __host__ __device__ (int32_t state_idx01) -> void {
      int32_t fsa_idx0 = a_fsas_row_ids1[state_idx01];
      FsaInfo this_info = fsa_info_data[fsa_idx0];
      int32_t state_idx0x = this_info.state_offset,
          state_idx1 = state_idx01 - state_idx0x;
      float start_loglike = (state_idx1 == 0 ? 0 : minus_inf),
         end_loglike = (state_idx1 + 1 == this_info.num_states ? 0 :
                      minus_inf);
      float *backward_state_scores_last_frame = backward_state_scores_data[this_info.T];
      forward_scores_t0[state_idx01] = start_loglike;
      backward_state_scores_last_frame[state_idx01] = end_loglike;
    };
    Eval(c_, tot_states, lambda_init_state_scores);
  }

  /* Called for 1 <= t <= T_, does one step of propagation (does forward and
     backward simultaneously, for different time steps) */
  void DoStep(int32_t t) {
    NVTX_RANGE(__func__);
    Step &step = steps_[t], &prev_step = steps_[t-1];

    int32_t forward_num_fsas = step.forward_num_fsas,
           backward_num_fsas = step.backward_num_fsas;
    float *forward_arc_scores_data = step.forward_arc_scores.Data(),
         *backward_arc_scores_data = step.backward_arc_scores.Data(),
     *prev_forward_state_scores_data = prev_step.forward_state_scores.Data(),
  *next_backward_state_scores_data = prev_step.backward_state_scores.Data();

    // the frame from which we need to read scores is actually forward_t - 1,
    // e.g. if we are writing the state-probs on frame t=1 we need to read the
    // scores on t=1.
    int32_t forward_scores_t = step.forward_t - 1,
           backward_state_scores_t = step.backward_t;


    CompressedArc *carcs_data = carcs_.Data();
    FsaInfo *fsa_info_data = fsa_info_.Data();
    float *scores_data = b_fsas_.scores.Data();
    int32_t scores_stride = b_fsas_.scores.ElemStride0();

    auto lambda_set_arc_scores = [=] __host__ __device__ (int32_t arc_idx012) -> void {
      CompressedArc carc = carcs_data[arc_idx012];
      int32_t fsa_idx = carc.fsa_idx;
      FsaInfo fsa_info = fsa_info_data[fsa_idx];
      // First, forward pass.  We read from the src_state of the arc
      if (fsa_idx < forward_num_fsas) {
        int32_t src_state_idx1 = carc.src_state,
               src_state_idx01 = fsa_info.state_offset + src_state_idx1;
        float src_prob = prev_forward_state_scores_data[src_state_idx01];
        float arc_end_prob = src_prob + carc.score +
                             scores_data[carc.label_plus_one +
                                         fsa_info.scores_offset +
                                         scores_stride * forward_scores_t];
        // For the forward pass we write the arc-end probs out-of-order
        // w.r.t. the regular ordering of the arcs, so we can more easily
        // do the reduction at the destination states.
        forward_arc_scores_data[carc.incoming_arc_idx012] = arc_end_prob;
      }
      if (fsa_idx < backward_num_fsas) {
        int32_t dest_state_idx1 = carc.dest_state,
               dest_state_idx01 = fsa_info.state_offset + dest_state_idx1;
        float dest_prob = next_backward_state_scores_data[dest_state_idx01];
        float arc_begin_prob = dest_prob + carc.score +
                             scores_data[carc.label_plus_one +
                                         fsa_info.scores_offset +
                                         scores_stride * backward_state_scores_t];
        backward_arc_scores_data[arc_idx012] = arc_begin_prob;
      }
    };
    Eval(c_, step.arc_scores.NumElements(), lambda_set_arc_scores);
    MaxPerSublist(step.arc_scores, -std::numeric_limits<float>::infinity(),
                  &step.state_scores);
  }

  /*
    Called after DoStep() is done for all time steps, returns the total scores
    minus output_beam_.  (This is what it does in the absence of roundoff error
    making the forward and backward tot_scores differ; when they do, it tries to
    pick a looser beam if there is significant roundoff).
   */
  Array1<float> GetScoreCutoffs() {
    std::vector<float*> forward_state_scores_vec(T_);
    int32_t tot_states = a_fsas_.TotSize(1);
    for (int32_t t = 0; t <= T_; t++) {
      int32_t ft = steps_[t].forward_t;  // actually == t, but it's clearer.
      forward_state_scores_vec[ft] = steps_[t].forward_state_scores.Data();
    }
    forward_state_scores_ = Array1<float*>(c_, forward_state_scores_vec);
    float **forward_state_scores_data = forward_state_scores_.Data();

    float *backward_state_scores_t0 = steps_[0].backward_state_scores.Data();
    FsaInfo *fsa_info_data = fsa_info_.Data();
    Array1<float> score_cutoffs(c_, num_fsas_);
    float *score_cutoffs_data = score_cutoffs.Data();
    float output_beam = output_beam_;
    auto lambda_set_cutoffs = [=] __host__ __device__ (int32_t fsa_idx0) -> void {
      FsaInfo fsa_info = fsa_info_data[fsa_idx0];
      float tot_score_start = backward_state_scores_t0[0],
        tot_score_end = forward_state_scores_data[fsa_info.T]
                 [fsa_info.state_offset + fsa_info.num_states - 1],
        tot_score_avg = 0.5 * (tot_score_start + tot_score_end),
        tot_score_diff = fabs(tot_score_end - tot_score_start);
      // TODO(dan): remove the following after the code is tested.
      K2_CHECK(tot_score_diff < 0.1);
      // subtracting the difference in scores is to help make sure we don't completely prune
      // away all states.
      score_cutoffs_data[fsa_idx0] = tot_score_avg - tot_score_diff - output_beam;
    };
    Eval(c_, num_fsas_, lambda_set_cutoffs);
    K2_LOG(INFO) << "Cutoffs = " << score_cutoffs;
    return score_cutoffs;
  }



  ContextPtr c_;
  FsaVec &a_fsas_;          // a_fsas_: decoding graphs, with same Dim0() as
                            // b_fsas_. Note: a_fsas_ has 3 axes.

  DenseFsaVec &b_fsas_;

  // num_fsas_ equals b_fsas_.shape.Dim0() == a_fsas_.Dim0().
  int32_t num_fsas_;

  // This is just a copy of a_fsas_.arcs, with a couple extra pieces of information.
  struct CompressedArc {
    // src_state of Arc, as uint16 (an idx1)
    uint16_t src_state;
    // dest_state of Arc, as uint16 (an idx1)
    uint16_t dest_state;
    // label of Arc, plus one, as uint16
    uint16_t label_plus_one;
    // FSA index, as uint16.
    uint16_t fsa_idx;
    // The idx012 of the position of this arc in the 'incoming arcs' array as
    // returned by GetIncomingArcs().  This is where we'll write the end-loglike
    // of this arc in the forward propagation, to make the reduction easier.
    int32_t incoming_arc_idx012;
    float score;
  };
  // The arcs in a_fsas_.arcs, converted to int16_t's and with a little more information.
  Array1<CompressedArc> carcs_;

  // incoming_arcs_.shape is a modified version of the shape a_fsas_.arcs.shape,
  // so arcs are arranged by dest_state rather than src_state, as returned by
  // GetIncomingArcs().  It's used to do reductions for the forward-pass
  // computation.
  Ragged<int32_t> incoming_arcs_;


  // forward_then_backward_shape_ is a shape with NumAxes()==3, equal to
  // (incoming_arcs_.shape, a_fsas_.shape) appended together.  The name is
  // because incoming_arcs_.shape is used for reduction of arc-scores to
  // state-scores in the forward pass, and a_fsas_.shape in the backward pass.
  // Prefixes of forward_then_backward_shape_ are used as the shape of the
  // "early" members of fb_state_scores_, those for which t_forward <= t_backward.
  // FSAs active in the forward pass is >= those active in the backward pass.
  RaggedShape forward_then_backward_shape_;

  // backward_then_forward_shape is (a_fsas_.shape, incoming_arcs_.shape)
  // appended together.  Read docs for forward_then_backward_shape_ to
  // understand.
  RaggedShape backward_then_forward_shape_;

  // Temporary array used in combined forward+backward computation, of dimension
  // a_fsas_.TotSize(2) * 2.  Used indirectly through fb_arc_scores_, which
  // share this data.
  Array1<float> arc_scores_;

  struct FsaInfo {
    // T is the number of frames in b_fsas_.scores that we have for this FSA, i.e.
    // `b_fsas_.scores.RowSplits(1)[i+1] -  b_fsas_.scores.RowSplits(1)[i].`
    // The number of copies of the states of a_fsas_ that we have in the total
    // state space equals T+1, i.e. we have copies of those states for times
    // 0 <= t <= T.
    uint16_t T;
    // num_states is the number of states this FSA has.
    uint16_t num_states;
    // scores_offset is the offset of first location in b_fsas_.scores.Data()
    // that is for this FSA, i.e. b_fsas_.scores.Data()[scores_offset] is the
    // score for t=0, symbol=-1 of this FSA.
    int32_t scores_offset;
    // state_offset is the idx0x corresponding to this FSA in a_fsas_.
    int32_t state_offset;
    // arc_offset is the idx0xx corresponding to this FSA in a_fsas_.
    int32_t arc_offset;
  };
  // fsa_info_ is of dimension num_fsas_ + 1 (the last one is not correct in all
  // respects, only certain fields make sense).
  Array1<FsaInfo> fsa_info_;

  struct Step {
    // 0 < forward_t <= T_ is the time whose states we are writing to in the
    // forward pass on this step of the algorithm (we read from those on time
    // forward_t - 1)
    int32_t forward_t;
    // backward_t = T_ - t_forward is the time whose states we are writing
    // to in the backward pass on this step of the algorithm (we read from those
    // on time backward_t + 1)
    int32_t backward_t;

    // true if forward_t <= backward_t.  Affects the order in which we append
    // the states we're processing for the forward and backward passes into a single
    // array (if true, forward goes first).
    bool forward_before_backward;

    // forward_num_fsas == num_fsas_for_t_[forward_t] is the number of FSAs that have states
    // active on t == forward_t.
    int32_t forward_num_fsas;

    // forward_num_fsas_full is num_fsas_ if forward_before_backward == true,
    // else forward_num_fsas.  (This is a padding we use so that we can share the
    // shape information for the arrays, they become prefixes of the same array).
    int32_t forward_num_fsas_full;

    // backward_num_fsas == num_fsas_for_t_[backward_t] is the number of FSAs that have states
    // active on t == backward_t.
    int32_t backward_num_fsas;

    // backward_num_fsas_full is num_fsas_ if backward_before_backward == true,
    // else backward_num_fsas.  (This is a padding we use so that we can share the
    // shape information for the arrays, they become prefixes of the same array).
    int32_t backward_num_fsas_full;

    // Ragged array where we will write the scores of arcs before reduction.
    // The shapes for all of these use shared memory, and so do the floats
    // (they are sub-arrays of arc_scores_).
    // If forward_before_backward it contains the forward scores first;
    // else the backward scores.
    Ragged<float> arc_scores;

    // forward_arc_scores is a sub-array of `arc_scores` containing
    // the arc-end probs we write for the forward pass prior to reduction.
    // Its Dim() is the total num-arcs corresponding to forward_num_fsas.
    Array1<float> forward_arc_scores;

    // backward_arc_scores is a sub-array of `arc_scores` containing
    // the arc-begin probs we write for the backward pass prior to reduction.
    // Its Dim() is the total num-arcs corresponding to backward_num_fsas.
    Array1<float> backward_arc_scores;

    // `state_scores` is where we reduce `arc_scores` to, its Dim() equals
    // arc_scores.TotSize(1).  [arc_scores has 3 axes]. This storage is ACTALLY
    // ALLOCATED HERE, unlike other arrays declared here.  state_scores.Dim() is
    // the num-states corresponding to backward_num_fsas_full plus the
    // num-states corresponding to forward_num_fsas_full (the order depends on
    // whether forward_before_backward == true).
    Array1<float> state_scores;

    // `forward_state_scores` is the sub-array of `state_scores` containing just
    // the forward probs written in this step; its Dim() is the number of states
    // corresponding to forward_num_fsas_full.  It's not needed here directly,
    // but in the next step.
    Array1<float> forward_state_scores;

    // `backward_state_scores` is the sub-array of `state_scores` containing just
    // the backward probs written in this step; its Dim() is the number of
    // states corresponding to backward_num_fsas_full.  It's not needed here
    // directly, but in the next step.
    Array1<float> backward_state_scores;
  };

  // steps_.size() ==  T_ + 1.
  // steps_[0] is "special", on that step we do initialization.
  std::vector<Step> steps_;

  // It happens to be convenient to cache these two things here; they point to
  // data owned in the elements of steps_.forward_state_scores and
  // backward_state_scores.
  Array1<float*> forward_state_scores_;
  Array1<float*> backward_state_scores_;

  float output_beam_;

  int32_t T_;  // == b_fsas_.MaxSize(1)

  // forward_probs_ contains the forward probabilities.
  // Its NumAxes() == 2, it's of dimension [num_fsas][tot_states_per_fsa]
  // where "tot_states_per_fsa" correponds to the elements of tot_states_per_fsa_.
  Ragged<float> forward_probs_;

  // backward_probs_ has the same shape as forward_probs_.  backward_probs_[i] +
  // forward_probs[i] equals the probability of paths including that arc divided
  // by the total-prob of the lattice.
  Ragged<float> backward_probs_;

  // forward_probs_temp_ is a temporary used in computing forward_probs_ on each
  // frames; its dimension is a_fsas_.TotSize(2) * (b_fsas_.Dim0() /
  // a_fsas_.Dim0()), and it's arranged the same way as a_fsas_incoming_.
  Array1<float> forward_probs_temp_;

  // This is as oshape_unpruned_, but after the backward-pass pruning.
  // It is indexed [fsa_id][t][state][arc].
  RaggedShape oshape_pruned_;
};

void IntersectDense(FsaVec &a_fsas, DenseFsaVec &b_fsas, float output_beam,
                    FsaVec *out,
                    Array1<int32_t> *arc_map_a,
                    Array1<int32_t> *arc_map_b) {
  NVTX_RANGE("IntersectDense");
  FsaVec a_vec = FsaToFsaVec(a_fsas);
  MultiGraphDenseIntersect intersector(a_vec, b_fsas,
                                       output_beam);

  intersector.Intersect();
  FsaVec ret = intersector.FormatOutput(arc_map_a, arc_map_b);
  *out = ret;
}
}  // namespace k2
