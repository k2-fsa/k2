#include <compose.h>

// Caution: this is really a .cu file.  It contains mixed host and device code.



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
  }


  void ComputePruningCutoffs(const Ragged3<float> &end_probs,
                             Array1<float> *cutoffs) {
    int32 num_fsas = end_probs.Size0();


    // get the maximum score from each sub-list (i.e. each FSA, on this frame).
    // Note: can probably do this with a cub Reduce operation using an operator
    // that has side effects (that notices when it's operating across a
    // boundary).
    // the max will be -infinity for any FSA-id that doesn't have any active
    // states (e.g. because that stream has finished).
    // Casting to ragged2 just considers the top 2 indexes, ignoring the 3rd.
    // i.e. it's indexed by [fsa_id][state].
    Array1<float> max_per_fsa = MaxPerSubSublist(end_probs);

    Array1<int> active_states_per_fsa(c_, end_probs.Sizes1());

    float *max_per_fsa_data = max_per_fsa.values.data(),
        *dynamic_beams_data = dynamic_beams_.values.data();
    float beam = beam_,
        max_active = max_active_, min_active = min_active_;
    int *active_per_fsa_data = active_state_per_fsa.values.data();

    auto lambda = __device__ [max_per_fsa_data, dynamic_beams_data,
                   active_per_fsa_data, beam, max_active, min_active ] (int i) -> float {
                    float best_loglike = max_per_fsa_data[i],
                        dynamic_beam = dynamic_beams_data[i];
                    int num_active = active_per_fsa_data[i];
                    float ans;
                    if (num_active <= max_active) {
                      if (num_active > min_active) {
                        // Gradually approach to 'beam'
                        ans = 0.8 * dynamic_beam + 0.2 * beam;
                      } else {
                        if (ans < beam) ans = beam;
                        // gradually make the beam larger as long
                        // as we are below min_active
                        ans *= 1.5;
                      }
                    } else {
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
    Does the forward-propagation (basically: the decoding step) and
    returns a newly allocated FrameInfo* objecct for the next frame.
   */
  FrameInfo* PropagateForward(int t, FrameInfo *cur_frame) {
    Device D = a_fsas_.Device();  // we'll make sure all objects like this have a Device() function.
    // First create list of arcs that leave states on the current frame.


    Array2<StateInfo> &states = cur_frame->states;


    StateInfo *state_values = states.values.data;
    int *fsa_arc_offsets = a_fsas_.RowSplits2().data;
    auto lambda1 = [state_values,fsa_arc_offsets] (int i) -> int {
                     int abs_state_index = state_values[i].abs_state_id;
                     return fsa_arc_offsets[abs_state_index+1]-fsa_arc_offsets[abs_state_index];
                  };
    // 'sizes' gives the num-arcs for each state in `states`.
    Array<int> sizes(c_, states.values.size(), lambda1);

    // initialize shape of array that will hold arcs leaving the active states.
    // [fsa_index][state][arc].
    RaggedShape3 ai_shape(states.shape, sizes);
    // 'ais' means ArcInfo's
    int *ai_row_ids1 = arcs_shape.RowIds1(),
        *ai_row_ids2 = arcs_shape.RowIds2(),   // indexed by index i into the
                                              // ArcInfo values vector
        *ai_row_splits2 = arcs_shape.RowSplits2(),

        *arc_row_splits2 = a_fsas_.RowSplits2();  // indexed by absolute-state-id
                                                  // (which combines FSA-id and
                                                  // state-id)
    Arc *arcs = a_fsas_.values.data;
    int *nnet_output_row_ids = b_fsas_.shape.RowIds1();
    float *score_data = b_fsas_.scores.data;
    int score_num_cols = b_fsas_.num_cols;
    const int symbol_shift = 1;
    assert(b_fsas_.symbol_shift == symbol_shift);

    __host__ __device__ auto lambda2 =
        [state_values, ai_row_ids, ai_row_splits, arcs, t,
         fsa_arc_offsets, arcs_row_ids, score_data, score_num_cols, symbol_shift,
         nnet_output_ptrs] (int i) -> ArcInfo {
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
          // note: temporarily
          ai.abs_dest_state = sinfo.abs_state_id + arc.dest_state - arc.src_state;
          ai.end_loglike = sinfo.tot_loglike + arc_loglike;
          return ai;
        };

    Ragged3<ArcInfo> arc_info(arcs_shape,
                              Array<ArcInfo>(arcs_shape.TotSize2(),
                                             lambda2));

    ArcInfo *ai_data = arc_info.elems.data;
    Ragged2<float> ai_loglikes(arc_info.shape,
                               Array1<float>(arc_info.elems.size(),
                                             [ai_data](int i) -> float { return ai_data[i].end_loglike; }));


    Array1<float> cutoffs; // per fsa.
    ComputePruningCutoffs(&arc_info, &cutoffs);

    float *cutoffs_data = cutoffs.data;



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
                                   state_map_datax[abs_dest_state] = i;
                                 }
                               };
    Eval(D, arc_info.elems.size(), lambda3);


    // 1 if we keep this arc, else 0.
    Array1<char> keep_this_arc(c_, arc_info.elems.size(), 0);

    // 1 if we keep the destination-state of this arc, else 0; but only one of
    // the arcs leading to any given state has this set.
    Array1<char> keep_this_state(c_, arc_info.elems.size(), 0);

    // Note: we don't just keep arcs that were above the pruning threshold, we
    // keep all arcs whose destination-states survived pruning.  Later we'll
    // prune with the lattice beam, using both forward and backward scores.
    char *keep_this_arc_data = keep_this_arc.data(),
        *keep_this_state_data = keep_this_state.data();
    auto lambda4 = __host__ __device__ [ai_data](int i) -> void {
                                         int32_t abs_dest_state = ai_data[i].abs_dest_state;
                                         int j = state_map_data[abs_dest_state];
                                         if (j != -1) {
                                           keep_this_arc_data[j] = 1;
                                           if (j == i)
                                             keep_this_state_data[i] = 1;
                                         }
                                       };
    Eval(D, arc_info.elems.size(), lambda4);

    // The '+1' is so we can easily get the total num-arcs and num-states.
    Array1<int> arc_reorder(c_, arc_info.elems.size() + 1),
        state_reorder(c_, arc_info.elems.size() + 1);
    ExclusiveSum(keep_this_arc, &arc_reorder);
    ExclusiveSum(keep_this_state, &state_reorder);
    // OK, 'arc_reorder' and 'state_reorder' contain indexes for where we put
    // each kept arc and each kept state.
    int num_arcs = arc_reorder[arc_reorder.size()-1],
        num_states = state_reorder[state_reorder.size()-1];



    int *arc_reorder_data = arc_reorder.data(),
        *state_reorder_data = state_reorder.data();

    cur_frame->arcs = Ragged3FromRowIds(states.shape,
                                        Array1<ArcInfo>(D, num_arcs),


    ArcInfo *kept_ai_data = cur_frame->arcs.values.data();

    FrameInfo *ans = new FrameInfo();
    ans->states = Ragged3(SubsampleFromRowIds());

                          Array1<StateInfo>(D, num_arcs);
    StateInfo *kept_states_data = ans->states.data();

    // Modify the elements of `state_map` to refer to the indexes into
    // `ans->states` / `kept_states_data`, rather than the indexes into ai_data.
    auto lambda5 = __host__ __device__ [=](int i) ->void {
                                         int this_j = state_reorder_data[i],
                                             next_j = state_reorder_data[i+1];
                                         if (next_j > this_j)
                                           state_map[ai_data[i].abs_dest_state] = this_j;
                                       };
    Eval(D, arc_info.elems.size(), lambda5);
    auto lambda6 = __host__ __device__ [=](int i) ->void {
                                         int this_j = arc_reorder_data[i],
                                             next_j = arc_reorder_data[i+1],
                                             this_state_j = state_reorder_data[i],
                                             next_state_j = state_reorder_data[i+1];
                                         // Note: I have a idea to reduce main-memory bandwidth by
                                         // caching writes in a fixed-size array in shared memory
                                         // (store: best-prob, index).  We'd go round robin, try e.g.
                                         // twice.
                                         // would have to do this with a functor rather than a lambda.
                                         if (next_j > this_j) {
                                           // implies this was one of the arcs to keep.
                                           ArcInfo info = ai_data[i];
                                           int abs_dest_state = info.abs_dest_state;
                                           info.abs_dest_state = state_map[info.abs_dest_state];
                                           kept_ai_data[this_j] = info;

                                           if (next_state_j > this_state_j) {
                                             kept_states_data[
                                         }
                                       };
    Eval(D, arc_info.elems.size(), lambda6);






    // IndexableSize2() supports lookup is(i,j) on device.
    IndexableSize2 arcs_per_state = (cur_frame->states.shape.FlatSizes2());

    //IndexableSize2[ FilteredRaggedArray(ans->


  }


  struct StateInfo {       // for a state active on a particular frame..
    int32_t abs_state_id;  // absolute state-id, i.e. index into a_fsas.row_splits2()/row_ids3();
                           // this maps to the origin state-index in the decoding graph.

    float tot_loglike;     // a Viterbi-style 'forward probability'.  (Viterbi,
                           // meaning: we use max not log-sum).  You can take
                           // the pruned lattice and rescore it if you want
                           // log-sum.
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
      int32_t dest_state_idx;

                         // state_id for fsa_id > 0because it contains an offset
                         // for the FSA).
    }
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
