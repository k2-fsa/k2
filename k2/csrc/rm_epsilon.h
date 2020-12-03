
/*
  Epsilon removal algorithm (for tropical semiring):

    Involves first separating and non-epsilon arcs, doing a closure on the
    epsilon part so we have direct epsilon arcs between pairs of states
    that can reach each other by epsilons; then re-combining the
    epsilon and non-epsilon parts of the FSA.  We decide, for each
    epsilon in the closure of the epsilon part, whether to combine it
    with following or preceding non-epsilon arcs based on which would
    produce fewer additional arcs.
*/



/*
  Extract just the epsilon arcs and the states with epsilons entering and
  leaving them (plus the start and final states).  For use inside epsilon-removal
  algorithm.
     @param [in] src   Source FsaVec; must have 3 axes.
     @param [in] dest   Output FsaVec; will contain all the states that had
                       epsilon arcs leaving them or entering them, plus any
                       initial and final final states in `src`.
    @param [out]       Will be set to a new Array1 mapping from the
                       state_idx01's in `dest` to the corresponding state_idx01's in
                       `src`.
    @param [out] arc_map  Will be set to a new Array1, mapping from the
                       arc_idx011's in `dest` to the corresponding arc_idx012's in
                       `src`.
*/
void ComputeEpsilonSubset(FsaVec &src,
       FsaVec *dest,
       Array1<int32_t> *state_map,
       Array1<int32_t> *arc_map);


/*
  Extract just the non-epsilon arcs and the states that have non-epsilons leaving or
  entering them (plus the start and final states).  For use inside epsilon-removal
  algorithm.

     @param [in] src   Source FsaVec; must have 3 axes.
     @param [in] dest   Output FsaVec; will contain all the states that had
                       non-epsilon arcs leaving them or entering them, plus any
                       initial and final final states in `src`.
     @param [out]      Will be set to the renumbering object from the old to new
                       state indexes.
    @param [out] arc_map  Will be set to a new Array1, mapping from the
                       arc_idx011's in `dest` to the corresponding arc_idx012's in
                       `src`.
*/
void ComputeNonEpsilonSubset(FsaVec &src,
       FsaVec *dest,
       Renumbering *state_map,
       Array1<int32_t> *arc_map);


/*
   Map some states of an FsaVec, with the arcs entering and leaving those states
      @param [in] src   Source FsaVec, to be mapped
      @param [in] state_row_splits   The row_splits vector for `dest`, which
                         determines the number of states for each output FSA
      @param [out] state_row_ids  The row_ids vector corresponding to `state_row_splits`
      @param [in] map   Map from state_idx01's in `src` to state_idx01's in
                       `dest`, with -1 for states that are to be removed.  Note:
                        the number of states in `src` may be smaller or larger than
                        state_row_ids.Dim().
      @param [out] dest  Destination FsaVec; at exit, will contain all arcs in
                        `src` whose src_state and dest_state are both kept
                        (i.e. not mapped to -1).
      @param [out] arc_map Will be set to a new Array1 that maps from arc-index
                        in `dest` to original arc-index in `src`.

*/
void MapFsaVecStates(FsaVec &src,
                     const Array1<int32_t> &state_row_splits,
                     const Array1<int32_t> &state_row_ids,
                     const Array1<int32_t> &state_map,
                     FsaVec *dest,
                     Array1<int32_t> *arc_map);


/*
  Compute the closure of an FSA containing just epsilon arcs (as output
  by ComputeEpsilonSubset()).  This means adding epsilon arcs from
  each state s1 to each state s2 which is reachable indirectly by epsilons
  from s1 to s2.  Note: this implicitly assumes the tropical semiring, because
  we are taking only the best epsilon path from any state to any other
  state.

     @param [in] epsilon_fsa   FSA containing only epsilon arcs, as output
                           by ComputeEpsilonSubset()
     @param [out] closure_fsa  FSA containing the closure of the epsilon arcs.
                           Will be arc-sorted, and no state will have more than
                           one arc to any other state.

  Implementation notes from Dan: I suggest to repeatedly call
  ComputeEpsilonClosureOneIter() until there is no further change in the
  FsaVec (this can be by simple comparison on arcs vector, since thanks to sorting
  the order is deterministic).  Obviously the arc_maps from the individual
  iterations must be composed.
*/
void ComputeEpsilonClosure(FsaVec &epsilon_fsa,
                           FsaVec *closure_fsa,
                           Ragged<int32_t> *arc_map);

/*
 One iteration of the algorithm in ComputeEpsilonClosure().
   @param [in] FSA containing only epsilon arcs (possibly already passed through
                   one or more iterations of closure).  Must have 3 axes.
   @param [out] closure_fsa   FSA that is the result of one iteration of closure/
                  Will contain an arc from state s1 to s2 if there was already
                  such an arc in `epsilon_fsa` or if there was a state s3 such
                  that there was an arc from s1 to s2 and one from s2 to s3.
                  Will contain at most one arc from one state to any other state.
    @param [out] arc_map   For each arc in closure_fsa, contains the sequence of
                  arc_idx012's in epsilon_fsa that was the source.

  Implementation notes from Dan: I suggest to over-generate arcs,
  (i.e. for each arc, generate n extra arcs if its dest-state had n arcs leaving
  it), then arc-sort with an operator that sorts on (dest-state then weight),
  then mark arcs to be (kept or not) according to whether the previous arc was
  to the same dest state, then renumber with a Renumbering class.
*/
void ComputeEpsilonClosureOneIter(FsaVec &epsilon_fsa,
                                  FsaVec *closure_fsa,
                                  Ragged<int32_t> *arc_map);

// should be in ragged_ops.h
/*
  Append a single element to each sub-array of a ragged matrix (we consider
  only its last axis).
     @param [in] src     Source ragged tensor
     @param [in] suffix  Array containing elements to append (they will
                         be appended regardless of value, for now).
                         Must have `suffix.Dim() == src.TotSize(src.NumAxes() - 2)`
     @return         Returns ragged tensor with same num-axes as `src`,
                     and NumElements() equal to src.NumElements() +
                     suffix.Dim()
 */
Ragged<int32_t> AddSuffixToRagged(Ragged<int32_t> &src,
                                  Array1<int32_t> &suffix);

// should be in ragged_ops.h
/*
  Prepend a single element to each sub-array of a ragged matrix (we consider
  only its last axis).
     @param [in] src     Source ragged tensor
     @param [in] prefix  Array containing elements to prepend (they will
                         be prepended regardless of value, for now).
                         Must have `prefix.Dim() == src.TotSize(src.NumAxes() - 2)`
     @return         Returns ragged tensor with same num-axes as `src`,
                     and NumElements() equal to src.NumElements() +
                     suffix.Dim()
 */
Ragged<int32_t> AddPrefixToRagged(Ragged<int32_t> &src,
                                  Array1<int32_t> &prefix);


// should be in ragged_ops.h, with documentation.
template <typename T>
Ragged<T> SubsampleRagged(Ragged<T> &src,
                          Renumbering &renumbering) {
  return Ragged<T>(SubsampleRaggedShape(src, renumbering),
                   src.values[renumbering.New2Old()]);
}


/*
  Remove epsilons from FsaVec in `src_fsa`, producing an FsaVec `dest_fsa` which is equivalent
  (in tropical semiring).  Uses an iterative algorithm which tries to minimize the number of
  arcs in the resulting FSA (epsilons are combined with either preceding or following arcs).
*/
void RemoveEpsilonsIterativeTropical(FsaVec &src_fsa,
                                     FsaVec *dest_fsa,
                                     Ragged<int32_t> *arc_map) {
  Array1<int32_t> epsilons_state_map, epsilons_arc_map;
  FsaVec epsilon_fsa;
  ComputeEpsilonSubset(src_fsa, &epsilons, &epsilons_state_map,
                       &epsilons_arc_map);

  FsaVec epsilon_fsa_closure;
  Ragged<int32_t> epsilon_closure_arc_map;
  ComputeEpsilonClosure(epsilon_fsa, &epsilon_fsa_closure,
                        &epsilon_closure_arc_map);
  // make epsilon_closure_arc_map refer back to 'src_fsa'.
  epsilon_closure_arc_map.values = epsilons_arc_map[epsilon_closure_arc_map.values];


  FsaVec non_epsilon_fsa;
  Renumbering non_epsilon_state_renumbering;
  Array1<int32_t> non_epsilon_arc_map;
  ComputeNonEpsilonSubset(src_fsa, &non_epsilon_fsa,
                          &non_epsilon_state_renumbering,
                          &non_epsilon_arc_map);

  // Combine the info in epsilons_state_map
  // and non_epsilon_state_renumbering.Old2New(),
  // to create a state-map from the states in (epsilon_fsa or epsilon_fsa_closure)
  // to those in non_epsilon_fsa (or -1 for those states which
  // are not present in non_epsilon_fsa.
  Array1<int32_t> epsilon_to_noneps_state_map(epsilon_fsa.NumStates());
  // [lambda here to set epsilon_to_noneps_state_map)

  // `epsilon_closure_mapped` will have (a subset of) the arcs of the
  // epsilon-closure FSA in the same numbering as those of non_epsilon_fsa.
  FsaVec epsilon_closure_mapped;
  Array1<int32_t> epsilon_closure_mapped_arc_map1;
  MapFsaVecStates(epsilon_fsa_closure,
                  non_epsilon_fsa.RowSplits(1),
                  non_epsilon_fsa.RowSplits(1),
                  epsilon_to_noneps_state_map,
                  &epsilon_closure_mapped,
                  epsilon_closure_mapped_arc_map1);

  // arc_map from epsilon_closure_mapped back to `src_fsa`.
  Ragged<int32_t> epsilon_closure_mapped_arc_map = Index(
      epsilon_closure_arc_map,
      epsilon_closure_mapped_arc_map1);


  // we will need the row_splits of this to get the number of non-epsilon arcs
  // entering each state in non_epsilon_fsa.
  Ragged<int32_t> non_epsilon_incoming_arcs = GetIncomingArcs(non_epsilon_fsa,
                                                              GetDestStates(non_epsilon_fsa));

  // epsilon_prec_renumbering will remap the arcs in epsilon_closure_mapped to
  // the subset of arcs that we'll combine with the *preceding* arc (i.e. entering
  // their src_state).
  Renumbering epsilon_prec_renumbering(epsilon_closure_mapped.NumElements());
  char *epsilon_prec_renumbering_keep_data = epsilon_prec_renumbering.Keep().Data();

  Array1<int32_t> epsilon_num_foll_arcs(epsilon_closure_mapped.NumElements() + 1);
  int32_t *epsilon_num_foll_arcs_data = ...;


  // Lambda:
  //   For each epsilon arc in epsilon_closure_mapped, we'll decide whether to combine
  //   it with *following* non-epsilon arcs or *preceding* non-epsilon arcs.
  //   We combine it with *following* non-epsilon arcs if it is leaving from
  //   the start-state or if the num-non-epsilon-arcs leaving its dest state
  //   is less than the num-non-epsilon-arcs entering its src state.
  //
  //   If we decided to combine it with following non-epsilon arcs then we set
  //   epsilon_num_foll_arcs_data to the number of non-epsilon-arcs leaving
  //   the dest-state, and set epsilon_prec_renumbering_keep_data to 0.
  //   Else (combining with preceding arcs) we set epsilon_num_foll_arcs_data to 0 and
  //   set epsilon_prec_renumbering_keep_data to 1.


  // `combined_foll` will be set to an FSA, with the same state numbering as
  // `non_epsilon_fsa`, containing the arcs which arose by combining epsilon
  // arcs with non-epsilon arcs following them.
  FsaVec combined_foll;
  Ragged<int32_t> combined_foll_arc_map;
  { // This block will set combined_foll and combined_foll_arc_map
    ExclusiveSum(epsilon_num_foll_arcs, &epsilon_num_foll_arcs);
    Array1<int32_t> &foll_row_splits = epsilon_num_foll_arcs;
    int32_t num_arcs = foll_row_splits.Back();
    Array1<int32_t> foll_row_ids(c, num_arcs);
    RowSplitsToRowIds(...);
    // This shape just says, for each arc in epsilon_closure_mapped
    // that is to be combined with following arcs, how many following
    // arcs it is combined with (else 0).
    RaggedShape foll_shape = RaggedShape2(&foll_row_splits, &foll_row_ids, num_arcs);

    // foll_non_eps_arc_idx will be set in the lambda to the arc-index within
    // non_epsilon_fsa of the following arc which we're combining this epsilon
    // arc with
    Array1<int32_t> foll_non_eps_arc_idx(num_arcs);
    Array<Arc> arcs(num_arcs);
    {
      // lambda that sets foll_non_eps_arc_idx and arcs.
    }

    foll_fsa_shape = RemoveIndex(
        ComposeRaggedShapes(epsilon_closure_mapped.shape, foll_shape), 2);

    combined_foll = FsaVec(foll_fsa_shape, arcs);

    combined_foll_arc_map = AddSuffixToRagged(
        epsilon_closure_mapped_arc_map[foll_row_ids],
        non_epsilon_arc_map[foll_non_eps_arc_idx]);
  }


  FsaVec epsilon_closure_prec =
      SubsampleRagged(epsilon_closure_mapped,
                      epsilon_prec_renumbering);
  Ragged<int32_t> epsilon_closure_prec_arc_map =
      epsilon_closure_mapped_arc_map[epsilon_prec_renumbering.New2Old()];

  // `combined_prec` will be set to an FSA, with the same state numbering as
  // `non_epsilon_fsa`, containing the arcs which arose by combining epsilon
  // arcs with non-epsilon arcs preceding them.
  FsaVec combined_prec;
  Ragged<int32_t> combined_prec_arc_map;

  { //  This block will set combined_prec and combined_prec_arc_map
    //  nonepsilon_num_foll_eps[i] tells us, for each arc in non_epsilon_fsa,
    // how many epsilon arcs leave the state that follows it.
    Array1<int32_t> nonepsilon_num_foll_eps(non_epsilon_fsa.NumArcs() + 1);
    // will set nonepsilon_num_foll_eps using a lambda that uses the row-ids
    // (etc.) of epsilon_closure_prec.


    ExclusiveSum(nonepsilon_num_foll_eps, &nonepsilon_num_foll_eps);
    Array1<int32_t> &prec_row_splits = nonepsilon_num_foll_eps;

    // The basic logic of this block will be similar to the block in
    // which we set combined_foll, except the order of non-epsilon and
    // epsilon are different.  By indexing/numbering things by the *first*
    // of the two arcs (rather than, say, by the epsilon) we ensure
    // that there is no need to re-sort the arcs, which could be slow.
  }



  FsaVec *vecs[3] = { &combined_foll, &combined_prec, &non_epsilon_fsa };
  int32_t axis = 2;

  // Caution: currently Append() does not support axis > 1; but actually this is
  // not a fundamental limitation because it doesn't interact non-trivially with
  // the earlier axes, as long as they are identical among all inputs.  For
  // instance, we could do RemoveAxis() to remove axis 0 of all the inputs, then
  // Append() with axis 1, then do ComposeRaggedShapes() to combine with axis 0
  // of the inputs (we could obtain that by doing RemoveAxis(one_of_the_inputs,
  // 2)).  We just require that the earlier axes all be the same, which they are
  // here.  I'm not saying we need such a recursive implementation, necessarily;
  // only that there is not a fundamental reason why Append() can't work in this
  // case.
  *dest_fsa = Append(axis, 2, vecs);

  // TODO: work out how to combine the arc maps.
  // Can do arc-sorting *after* combining the arc maps, which will make
  // the reordering of the arc-maps.

}
