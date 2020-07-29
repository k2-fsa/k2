#include <compose.h>

// Caution: this is really a .cu file.  It contains mixed host and device code.

/* GLOSSARY

   abs-state-index: index into an FsaVec's RowSplits2(), combines
                    FSA-index and state-index within that FSA.
   abs-arc-index: index into an FsaVec's elems; combines FSA-index, state-index
                    within that FSA, and arc-index within that state.
   frame-state-index: index into FrameInfo::states.values()

 */


// Note: b is FsaVec<Arc>.
void Intersect(const DenseFsa &a, const FsaVec &b, Fsa *c,
               Array<int32_t> *arc_map_a = nullptr,
               Array<int32_t> *arc_map_b = nullptr) {

}



class MultiGraphDenseIntersect {
 public:
  /**
     Pruned intersection (a.k.a. composition) that corresponds to decoding for
     speech recognition-type tasks

       @param [in] a_fsas  The decoding graphs, one per sequence.  E.g. might just be a linear
                      sequence of phones, or might be something nonlinear.
       @param [in] b_fsas  The neural-net output, with each frame containing the log-likes of
                      each phone.  A series of sequences of (in general) different length.
       @param [in] beam    "Default" decoding beam.  The actual beam is dynamic and also depends
                      on max_active and min_active.
       @param [in] max_active  Maximum number of FSA states that are allowed to be active on any given
                     frame for any given intersection/composition task.  This is advisory,
                     in that it will try not to exceed that but may not always succeed.
       @param [in] min_active  Minimum number of FSA states that are allowed to be active on any given
                     frame for any given intersection/composition task.  This is advisory,
                     in that it will try not to have fewer than this number active.
   */
  MultiGraphDenseIntersect(FsaVec &a_fsas,
                           DenseFsaVec &b_fsas,
                           float beam,
                           int32_t max_active,
                           int32_t min_active):
      a_fsas_(a_fsas), b_fsas_(b_fsas), beam_(beam),
      max_active_(max_active), min_active_(min_active),
      dynamic_beams_(a_fsas.Size0(), beam),
      state_map_(a_fsas.TotSize1(), -1) {
    c_ = GetContext(a_fsas, b_fsas);
    assert(beam > 0 && max_active > 0);
    Intersect();
  }


  void Intersect() {
    // Does the main work, but doesn't produce any output.
    /*
      T is the largest number of (frames+1) in any of the FSAs.  Note: each of the FSAs
      has a last-frame that has a NULL pointer to the data, and on this 'fake frame' we
      process final-transitions.
    */
    int32_t T = a_fsas.MaxSize1();

    std::vector<FrameInfo*> frames;
    frames.reserve(T);
    frames.push_back(InitialFrameInfo());

    for (int32_t t = 0; t < T; t++) {
      frames.push_back(PropagateForward(t, frames.back()));
    }

    for (int32_t t = T; t >= 0; t--) {
    }

  }

  /*
    Compute pruning cutoffs for this frame: these are the cutoffs for the arc
    "forward score", one per FSA.  This is a dynamic process involving
    dynamic_beams_ which are updated on each frame (they start off at beam_).

       @param [in] arc_end_probs  The "forward log-probs" (scores) at the
                   end of each arc, i.e. its contribution to the following
                   state.
       @param [out] cutoffs   Outputs the cutoffs, one per FSA (will be -infinity
                    for FSAs that don't have any active states).  These will
                    be of the form: the best score for any arc, minus the
                    dynamic beam.  Note: `cutoffs` does not have to be sized
                    correctly at entry, it will be overwritten.
   */
  void ComputePruningCutoffs(const Ragged3<float> &arc_end_scores,
                             Array1<float> *cutoffs) {
    int32 num_fsas = arc_end_scores.Size0();

    // get the maximum score from each sub-list (i.e. each FSA, on this frame).
    // Note: can probably do this with a cub Reduce operation using an operator
    // that has side effects (that notices when it's operating across a
    // boundary).
    // the max will be -infinity for any FSA-id that doesn't have any active
    // states (e.g. because that stream has finished).
    // Casting to ragged2 just considers the top 2 indexes, ignoring the 3rd.
    // i.e. it's indexed by [fsa_id][state].
    Array1<float> max_per_fsa = MaxPerSubSublist((Ragged2<float>&)end_probs);

    Array1<int> active_states_per_fsa =  end_probs.Sizes1();
    int *active_per_fsa_data = active_states_per_fsa.values.data();
    float *max_per_fsa_data = max_per_fsa.values.data(),
        *dynamic_beams_data = dynamic_beams_.values.data();
    float beam = beam_,
        max_active = max_active_,
        min_active = min_active_;

    auto lambda = __host__ __device__ [=] (int i) -> float {
              float best_loglike = max_per_fsa_data[i],
                  dynamic_beam = dynamic_beams_data[i];
              int num_active = active_per_fsa_data[i];
              float ans;
              if (num_active <= max_active) {
                if (num_active > min_active) {
                  // Neither the max_active nor min_active constraints
                  // apply.  Gradually approach 'beam'
                  ans = 0.8 * dynamic_beam + 0.2 * beam;
                } else {
                  // We violated the min_active constraint -> increase beam
                  if (ans < beam) ans = beam;
                  // gradually make the beam larger as long
                  // as we are below min_active
                  ans *= 1.25;
                }
              } else {
                // We violated the max_active constraint -> decrease beam
                if (ans > beam) ans = beam;
                // Decrease the beam as long as we have more than
                // max_active active states.
                ans *= 0.9;
              }
              return ans;
    };
    Array1<float> new_beam(c_, num_fsas, lambda);
    dynamic_beam_.swap(&new_beam);

    float *dynamic_beam_data = dynamic_beam_.data;
    auto lambda2 = __device__ [dynamic_beam_data,max_per_fsa_data] (int i)->float {
                                return max_per_fsa_data[i] - dynamic_beam_data[i]; };
    *cutoffs = Array1<float>(c_, num_fsas, lambda2);
  }


  /*
    Returns un-pruned list of arcs on this frame, consisting of all arcs
    leaving the states active on 'cur_frame'.

       @param [in] t  The time-index (on which to look up log-likes), t >= 0
       @param [in] cur_frame   The FrameInfo for the current frame; only its
                     'states' member is expected to be set up on entry.
   */
  Array1<Arc> GetUnprunedArcs(int t, FrameInfo *cur_frame) {
    Ragged2<StateInfo> &states = cur_frame->states;
    StateInfo *state_values = states.values.data;
    // in a_fsas_ (the decoding graphs), maps from abs-state-index to abs-arc-index.
    int *fsa_arc_offsets = a_fsas_.RowSplits2().data();

    __host__ __device__ auto num_arcs_lambda = [state_values,fsa_arc_offsets] (int i) -> int {
                     int abs_state_index = state_values[i].abs_state_id;
                     return fsa_arc_offsets[abs_state_index+1]-fsa_arc_offsets[abs_state_index];
                  };
    // `num_arcs` gives the num-arcs for each state in `states`.
    Array<int> num_arcs(c_, states.values.size(), num_arcs_lambda);

    // initialize shape of array that will hold arcs leaving the active states.
    // [fsa_index][state][arc].
    // 'ai' means ArcInfo.
    RaggedShape3 ai_shape = Ragged3FromSizes(states.shape, num_arcs);

    // 'ais' means ArcInfo's
    int *ai_row_ids1 = ai_shape.RowIds1(),   // indexed by frame_state_index == index into state.elems
        *ai_row_ids2 = ai_shape.RowIds2(),   // indexed by an index i into the
                                             // ArcInfo values vector
        *ai_row_splits2 = ai_shape.RowSplits2(),

        *arc_row_splits2 = a_fsas_.RowSplits2();  // indexed by abs-state-index
                                                  // (which combines FSA-id and
                                                  // state-id)

    Arc *arcs = a_fsas_.values.data;
    int *nnet_output_row_ids = b_fsas_.shape.RowIds1();
    float *score_data = b_fsas_.scores.data;
    int score_num_cols = b_fsas_.num_cols;
    const int symbol_shift = 1;
    assert(b_fsas_.symbol_shift == symbol_shift);

    Ragged3<ArcInfo> ai(ai_shape);

    ArcInfo *ai_data = arc_info.values.data();  // uninitialized

    __host__ __device__ auto ai_lambda =
        [state_values, ai_row_ids, ai_row_splits, ai_data, arcs, t,
         fsa_arc_offsets, arcs_row_ids, score_data, score_num_cols, symbol_shift,
         nnet_output_ptrs] (int i) -> void {
          int ai_row_id = ai_row_ids2[i],  // == index into state_values
              fsa_idx = ai_row_ids1[ai_row_id],
              ai_split_start = ai_row_splits2[arc_row_id],
              ai_offset = i - split_start;  // this is the ai_offset'th arc for this state..
          //  note: arc_row_id also corresponds to an index into 'state_values'
          StateInfo sinfo = state_values[ai_row_id];
          int arc_start = arc_row_splits[sinfo.abs_state_id],
              arc_index = arc_start + ai_offset;
          Arc arc = arcs[arc_index];
          int row = nnet_output_row_ids[fsa_idx] + t;
          assert(row < nnet_output_row_ids[fsa_idx+1]);
          assert(static_cast<uint32_t>(arc.symbol + symbol_shift) < static_cast<uint32_t>(score_num_cols))
          // note: symbol_shift is 1, to shift -1 to 0 so we can handle
          // end-of-sequence like a normal symbol.
          float acoustic_loglike = score_data[row * score_num_cols + symbol_shift + arc.symbol];


          ArcInfo ai;
          ai.arc_idx = arc_index;
          float arc_loglike = sinfo.forward_loglike + arc.cost + acoustic_loglike;
          ai.arc_loglike = arc_loglike;
          ai.u.abs_dest_state = sinfo.abs_state_id + arc.dest_state - arc.src_state;
          ai.end_loglike = sinfo.forward_loglike + arc_loglike;
          ai_data[i] = ai;
        };
    Eval(c_, ai.values.size(), ai_lambda);
    return ai;
  }


  /*
    Does the forward-propagation (basically: the decoding step) and
    returns a newly allocated FrameInfo* objecct for the next frame.
   */
  FrameInfo* PropagateForward(int t, FrameInfo *cur_frame) {

    Ragged2<StateInfo> &states = cur_frame->states;
    Array3<ArcInfo> ai = GetUnprunedArcs(t, cur_frame);

    ArcInfo *ai_data = arc_info.elems.data();

    Ragged3<float> ai_loglikes(arc_info.shape,
                               Array1<float>(arc_info.elems.size(),
                                             [ai_data](int i) -> float { return ai_data[i].end_loglike; }));


    Array1<float> cutoffs; // per fsa.
    ComputePruningCutoffs(&ai_loglikes, &cutoffs);

    float *cutoffs_data = cutoffs.data();


    // write certain indexes i (indexes into arc_info.elems) to state_map_.data().
    // Keeps track of the active states and will allow us to assign a numbering to them.
    int *ai_row_ids1 = arc_info.RowIds1().data(),
        *ai_row_ids2 = arc_info.RowIds2().data(),
        *state_map_data = state_map_.data();
    auto lambda3 == __host__ __device__ [ai_data,ai_row_ids1,ai_row_ids2,cutoffs_data](int i) -> void {
                                 int32_t fsa_id = ai_row_ids1[ai_row_ids2[i]];
                                 int32_t abs_dest_state = ai_data[i].abs_dest_state;
                                 float end_loglike = ai_data[i].end_loglike,
                                     cutoff = cutoffs_data[fsa_id];
                                 if (end_loglikes > cutoff) {
                                   // The following is a race condition as multiple threads may write to
                                   // the same location, but it doesn't matter, the point is to assign
                                   // one index i
                                   state_map_data[abs_dest_state] = i;
                                 }
                               };
    Eval(D, arc_info.elems.size(), lambda3);


    // 1 if we keep this arc, else 0.  The unusual way we initialize this is
    // because we need the memory region to cover one extra element, due to how
    // ExclusiveSum() works.
    Array1<char> keep_this_arc = Array1<char>(c_, arc_info.elems.size()+1,
                                              0).Range(0, arc_info.elems.size());

    // This is like `keep_this_arc`, but but only *one* of the arcs leading to
    // any given state has this nonzero (that one becomes the representative
    // that determines the order in which we'll list the destination states).
    Array1<char> keep_this_state = Array1<char>(c_, arc_info.elems.size()+1,
                                                0).Range(0, arc_info.elems.size());

    // Note: we don't just keep arcs that were above the pruning threshold, we
    // keep all arcs whose destination-states survived pruning.  Later we'll
    // prune with the lattice beam, using both forward and backward scores.
    char *keep_this_arc_data = keep_this_arc.data(),
        *keep_this_state_data = keep_this_state.data();
    auto lambda_keep = __host__ __device__ [=](int i) -> void {
                                         int32_t abs_dest_state = ai_data[i].abs_dest_state;
                                         int j = state_map_data[abs_dest_state];
                                         if (j != -1) {
                                           keep_this_arc_data[j] = 1;
                                           if (j == i)
                                             keep_this_state_data[i] = 1;
                                         }
                                       };
    Eval(D, arc_info.elems.size(), lambda_keep);

    // The '+1' is so we can easily get the total num-arcs and num-states.
    Array1<int> arc_reorder(c_, arc_info.elems.size() + 1),
        state_reorder(c_, arc_info.elems.size() + 1);
    ExclusiveSum(keep_this_arc, &arc_reorder);
    ExclusiveSum(keep_this_state, &state_reorder);

    // OK, 'arc_reorder' and 'state_reorder' contain indexes for where we put
    // each kept arc and each kept state.  They map from old index to new index.
    int num_arcs = arc_reorder[arc_reorder.size()-1],
        num_states = state_reorder[state_reorder.size()-1];

    int *arc_reorder_data = arc_reorder.data(),
        *state_reorder_data = state_reorder.data();

    // state_to_fsa_id maps from an index into the next frame's
    // FrameInfo::states.values() vector (i.e. a frame-state-index on the *next*
    // frame) to the FSA-id associated with it.  It should be non-decreasing.
    Array1<int> state_to_fsa_id(c_, num_states);
    { // This block sets 'state_to_fsa_id'.
      int *states_row_ids = states.RowIds1();  // maps from index into `states`,
      // == current frame's
      // frame-state-index to FSA-id.

      int *abs_state_to_fsa_index = a_fsas_.RowIds1().data();
      int *state_to_fsa_id_data = state_to_fsa_id.data();
      auto lambda_stateid = __host__ __device__ [=]->void {
                                                  int this_state_j = state_reorder_data[i],
                                                      next_state_j = state_reorder_data[i+1];
                                                  if (next_state_j > this_state_j) {
                                                    int abs_dest_state = ai_data[i].u.abs_dest_state,
                                                        fsa_id = abs_state_to_fsa_index[abs_dest_state];
                                                    state_to_fsa_id_data[this_state_j] = fsa_id;
                                                  }
                                                };
      Eval(c_, arc_info.elems.size(), lambda_stateid);
      assert(IsMonotonic(state_to_fsa_id));
    }
    // The following creates a structure that contains a subset of the elements
    // of `arc_info`, determined by `keep_this_arc`.  We already computed the
    // exclusive sum so we use that rather than using `keep_this_arc` directly.
    cur_frame->arcs = Ragged3<Arc>(RaggedShape3SubsampledFromNumbering(arc_info.shape,
                                                                       arc_reorder),
                                   Array1<Arc>(c_, num_arcs));

    FrameInfo *ans = new FrameInfo();
    ans->states = Ragged2FromRowIds(num_fsas,
                                    state_to_fsa_id,
                                    Array1<ArcInfo>(D, num_arcs));
    auto lambda_init_loglike = __host__ __device__ [] (int i) -> void {
                           ans->states[i].forward_loglike = FloatToOrderedInt(-std::numeric_limits<BaseFloat>::infinity());
                                                   };
    Eval(c_, num_states, lambda_init_loglike);


    // we'll set up the data of the kept arcs below..
    ArcInfo *kept_ai_data = cur_frame->arcs.values.data();
    StateInfo *kept_states_data = ans->states.data();

    // Modify the elements of `state_map` to refer to the indexes into
    // `ans->states` / `kept_states_data`, rather than the indexes into ai_data.
    // Note: this will decrease some of the values in `state_map`, in general.
    auto lambda_modify_state_map = __host__ __device__ [=](int i) ->void {
               int this_j = state_reorder_data[i],
                   next_j = state_reorder_data[i+1];
               if (next_j > this_j)
                 state_map[ai_data[i].abs_dest_state] = this_j;
            };
    Eval(c_, arc_info.elems.size(), lambda_modify_state_map);

    auto lambda_set_arcs_and_states = __host__ __device__ [=](int i) ->void {
            int this_j = arc_reorder_data[i],
                next_j = arc_reorder_data[i+1];
            // Note: I have a idea to reduce main-memory bandwidth by
            // caching writes in a fixed-size array in shared memory
            // (store: best-prob, index).  We'd go round robin, try e.g.
            // twice.
            // Would have to do this with a functor rather than a lambda,
            // as __shared__ won't work on CPU.

            //
            if (next_j > this_j) {
              // implies this was one of the arcs to keep.
              ArcInfo info = ai_data[i];
              int abs_dest_state = info.u.abs_dest_state;
              // set the dest_frame_state_idx in
              int dest_frame_state_idx = state_map[abs_dest_state];
              info.u.dest_frame_state_idx = dest_frame_state_idx;
              kept_ai_data[this_j] = info;
              // Note: multiple threads may write the same thing, on the next line.
              kept_states_data[dest_frame_state_idx].abs_state_id = abs_dest_state;
              int32_t end_loglike_int = FloatToOrderedInt(info.end_loglike);
              // Set the forward log-like of the dest state to the largest of any of the
              // incoming arcs.
              atomicMax(&(kept_states_data[dest_frame_state_idx].forward_loglike),
                        end_loglike_int);
            }
        };
    Eval(c_, arc_info.elems.size(), lambda_set_arcs_and_states);
    return ans;
  }


  /* Information associated with a state active on a particular frame..  */
  struct StateInfo {
    /* abs_state_id is an abs-state-index as defined in the glossary i.e. an
     * index into a_fsas.row_splits2()/row_ids3(); */
    int32_t abs_state_id;

    /* Caution: this is ACTUALLY A FLOAT that has been bit-twiddled using
       FloatToOrderedInt/OrderedIntToFloat so we can use atomic max.  It
       represents a Viterbi-style 'forward probability'.  (Viterbi, meaning: we
       use max not log-sum).  You can take the pruned lattice and rescore it if
       you want log-sum.  */
    int32_t forward_loglike;

    /* Note: this `backward_loglike` is the best score of any path from here to the
       end, minus the best path in the overall FSA, i.e. it's the backward score
       you get if, at the final-state, you set backward_loglike == forward_loglike.



       has been computed in such a way that if you
       add

    /* Caution: this is ACTUALLY A FLOAT that has been bit-twiddled using
       FloatToOrderedInt/OrderedIntToFloat so we can use atomic max.  It
       represents a Viterbi-style 'forward probability'.  (Viterbi, meaning: we
       use max not log-sum).  You can take the pruned lattice and rescore it if
       you want log-sum.  */
    int32_t backward_loglike;
  };

  struct ArcInfo {    // for an arc that wasn't pruned away...
    int32_t arc_idx;  // arc-index (offset into a_fsas.values).  Note, we can
                      // use this to look up the symbol on the arc, and the
                      // destination state in the decoding graph.
    float arc_loglike;  // loglike on this arc: equals loglike from data (nnet
                        // output, == b_fsas), plus loglike from the arc in a_fsas.

    union {
      int32_t abs_state_id;   // The index into the next Frame Info's state.values
                            // (could call this an abs_state_id).  Note: this is also
                            // used temporarily to store the real abs_dest_state which is the index into
                            // a_fsas_.elems, but in the permanently stored ArcInfo it stores
                            // the index into FrameInfo::state.values.
      int32_t dest_frame_state_idx;  // the frame_state_idx of the destination
                                     // state, i.e. the index into the next
                                     // frame's "states" vector.

      // Note, the reason we don't have to do:
      // int32_t src_frame_state_idx;
      // is that we can get it from the Ragged3 structure's RowIds2().

                         // state_id for fsa_id > 0because it contains an offset
                         // for the FSA).
    } u;
    float end_loglike;  // loglike at the end of the arc just before
                        // (conceptually) it joins the destination state.

  };


  // The information we have for each frame of the pruned-intersection (really: decoding)
  // algorithm.  We keep an array of these, one for each frame, up to the length of the
  // longest sequence we're decoding plus one.
  struct FrameInfo {
    // States that are active at the beginning of this frame.  Indexed
    // [fsa_idx][state_n], where fsa_id is as in `a_fsas` arg to
    // IntersectDensePruned, and state_n is a zero-based index that basically
    // counts the unique states active on this frame, in arbitrary order.
    //
    // We will call the index into states.values a "frame_state_index".
    Ragged2<StateInfo> states;

    // Indexed [fsa_idx][state_n][arc_idx] where arc_idx=0,1,.. just enumerates
    // the arcs that were not pruned away.  Note: there may be indexes [fsa_idx]
    // that have no states, and indexes [fsa_idx][state_idx] that have no arcs,
    // due to pruning.
    Ragged3<ArcInfo> arcs;
  };

  ContextPtr c_;
  FsaVec &a_fsas_;
  DenseFsaVec &b_fsas_;
  float beam_;
  int32_t max_active_;
  int32_t max_active_;
  Array1<float> dynamic_beams_;  // dynamic beams (initially just beam_ but
                                 // change due to max_active/min_active
                                 // constraints).
  Array1<int32_t> state_map_;  // state_map_ is of size (total number of states
                               // in all the FSAS in a_fsas_).  It is used on
                               // each frame to compute and store the mapping
                               // from active states to the position in the
                               // `states` array.  Between frames, all values
                               // have -1 in them.

};

// compose/intersect array of FSAs (multiple streams decoding or training in
// parallel, in a batch)...
void IntersectDensePruned(FsaVec &a_fsas,
                          DenseFsaVec &b_fsas,
                          float beam,
                          int32_t max_active,
                          FsaVec *ofsa,
                          Array1<int32_t> *arc_map_a,
                          Array1<int32_t> *arc_map_b) {

  // We should maybe give these objects some kind of Device() function so we can
  // use a single template to do this kind of thing.
  // We don't need to check device of arc_map_a / arc_map_b since we'll
  // output to them by assignment, overwriting any old data.
  CheckSameDevice(a_fsas, b_fsas);

  int32_t tot_num_active = a_fsas.tot_size1();

  // `state_id_map` is used on each frame to establish a mapping between
  // absolute-state-ids in a_fsas (i.e. indexes into a_fsas.{row_splits2(),row_ids3()})
  // and a linear vector of active states used on each frame.
  Array1<int32_t> state_id_map(tot_num_active, -1);


  /*
     T is the largest number of (frames+1) in any of the FSAs.  Note: each of the FSAs
     has a last-frame that has a NULL pointer to the data, and on this 'fake frame' we
     process final-transitions.
   */
  int32_t T = a_fsas.MaxSize1();


  std::vector


  Array1<int> a_index;
  Array1<int> b_index;

  Array1<int> fsa_id;
  Array1<int> out_state_id;

  Array1<HashKeyType> state_repr_hash;  // hash-value of corresponding elements of a_fsas and b_fsas

  Hash<HashKeyType, int, Hasher> repr_hash_to_id;  // Maps from (fsa_index, hash of state_repr) to






}
