/**
 * Copyright [2020]  <Xiaomi Corporation> [authors: Daniel Povey]
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * @note
 * CAUTION: this is old code, using older versions of interfaces
 * that no longer exist. Please ignore for now.
 *
 * @todo
 * Right now this just contains some notes on FSA determinization, written in
 * order to clarify my thinking on the right fundamental primitives.
 */

namespace k2 {

//  we'd simultaneously determinize an array of FSAs, in practice.
void DeterminizeFsaArray(Array3<Arc> &FsaVec, Array1<float> &input_scores,
                         Allocator &alloc, ...) {
  // The following mapping stuff would actually be an if...
  // we'll do this mapping only if needed and call
  Array2<Arc> Fsa;
  Array1<int32_t> arc_map;
  Array1<int16_t> fsa_idx;
  Array1<int32_t> start_states;
  ConvertToSingleFsa(FsaVec, &Fsa, &arc_map, fsa_idx, &start_states);

  Determinize(...);

  // Handle reverse mapping
}

// We simultaneously determinize an array of FSAs, in practice (that's why
// it's Ragged3 and not Ragged2 input).
// `begin_states` is the vector of start-states (one for each of the array
// of FSA's that we've merged into one).
void Determinize(const Ragged3<Arc> &input, const Array1<float> &input_scores,
                 Allocator &alloc, ...) {
  Ragged<Arc> output(...);  // indexed [i][num_arcs] where i is just an
                            // arbitrary index.. see output_fsa_idxs and
                            // output_state_idxs for how it maps to the real
                            // output.

  Array1<int32_t> output_fsa_idxs;    // FSA-index in 0..input.dim(0)-1 for each
                                      // state in `output`; indexed by 1st index
                                      // into Output.
  Array1<int32_t> output_state_idxs;  // state-index for each state in `output`,
                                      // i.e. the state within that output FSA.
                                      // It makes the algorithm easier to code,
                                      // and memory management easier, if we put
                                      // it all in one flat array.  Indexed by
                                      // 1st index into `output`.

  // suppose shape3 etc. are just tuples.

  // state_subsets has 2 axes; is indexed first by 'i' == index into axis0 of
  // `output`, and then is a list of state-indexes in the corresponding input
  // FSA (see output_fsa_idxs to see which input FSA).
  Ragged<int32_t> state_subsets(...);

  // score for each element of `state_subsets.elems`..
  Array1<float> state_subset_scores(
      ..., 0.0);  // score for each element of
                  // `state_subsets`... actually the subsets
                  // are weighted subsets.  Note the most
                  // negative score in each subset won't
                  // necessarily be 0; the normalization method
                  // is different in this algorithm.

  // for each element of 'state_subsets/state_subset_scores',
  // 'state_subset_arcs' contains the index of the incoming arc that created
  // it, or -1 if this is
  // the start state.
  Array1<int32_t> state_subset_arcs(..., -1);

  // for each element of 'state_subsets.elems', the index of the previous
  // element in `state_subsets.elems` which 'created' this, or -1 if this was an
  // initial arc.  The arc (in state_subset_arcs) entering that previous state
  // will be a preceding arc on a path vs the arc in state_subset_arcs entering
  // this state.
  Array1<int32_t> state_subset_traceback(..., -1);

  // For each output-state (indexed by 'i' == 1st index into `output` or into
  // `state_subsets`), the input-state which is the starting point for following
  // arcs with a particular sequence of symbols on them to find the set of
  // input-states that comprise that output-state.  Part of the canonical
  // representation of that input-state.
  // See also `output_fsa_idxs` and `symbol_seq`.
  Array1<int32_t> begin_state(...);

  // For each output-state (indexed by 'i' == 1st index into `output` or into
  // `state_subsets`), the sequence of symbols we follow on arcs from
  // `begin_state[i]` to reach the weighted set of input-states that corresponds
  // to this output-state.
  Array2<int32_t> symbol_seq(...);

  // Note: the hash key will be derived from: (output_fsa_idxs[i],
  // begin_state[i], symbol_seq[i]).
  typedef struct {
    int64_t a;
    int64_t b;
  } HashKeyType;
  struct Hasher {
    size_t operator()(const HashKeyType key);
  };

  // Indexed by the 'i' == 1st index into `output`, contains the hash value
  // derived from (output_fsa_idxs[i], begin_state[i],
  // symbol_seq[i]).
  Array1<HashKeyType> state_repr_hash(..);

  // Maps from hash of state-representation to an index 'i' into output, which
  // will correspond to a particular state in a particular output FSA.
  Hash<HashKeyType, int32_t, Hasher> repr_hash_to_idx;

  // TODO: initialize variables defined above, with 1st state.

  // prev_idx is the next index into `output` (i.e. into the leading dim of
  // `output` that we need to process: i.e. that we need to process arcs
  // leaving those states.  (This is also the leading dim of state_subsets).
  int32_t prev_idx = 0;

  // For each output FSA, the index of the next un-allocated output-state.
  Array1<int32_t> next_ostate(num_fsas, 1);

  while (1) {
    int32_t cur_idx = output.size();

    // will swap with prev_ostate.
    HostArray1<int32_t> next_ostate = state_subsets.dim(1).Host();

    // range on axis 1...  this is a shallow op.
    // Let the index into this_state_subsets
    Array2<int32_t> this_state_subsets =
        state_subsets.Range(0, prev_idx, cur_idx);

    struct Filter1 {
      __device__ pair<int32_t, int32_t> operator()(
          typename Array3<int32_t>::iterator iter) const {
        return pair<int32_t, int32_t>(idxs_data[acc.idx0()], acc.elem());
      }
      __device__ Filter1(const Filter1 &other) : data(other.idxs_data) {}
      explicit Filter1(int32_t *idxs_data) : idxs_data(idxs_data) {}
      int32_t *idxs_data;  // device pointer, from output_fsa_idxs.
    };

    // Get an array indexed [i-prev_idx][elem_of_ostate][arc_from_that_elem].
    // containing the flat arc indexes for the transitions we need to process.
    //
    // Note: input.flat_indexes contains the flat indexes into the elements
    // (i.e. the indexes into input.elems) but it is not itself flat; it has
    // the same
    // 3d structure as `input`, so when indexed with a pair<int32_t,int32_t>
    // it gives us a 1-d array of int32_t, hence the extra level of array here.
    //
    // Note: FilteredArray3 is a dynamic creation, it doesn't get physically
    // populated.
    //
    // NOTE: input.flat_indexes or any flat_indexes can be implemented using a
    // special accesor; it just needs access to the shape.  This requires we
    // be able to index a 3-d array with pair<int32_t,int32_t>.
    Array3<int32_t> arc_idxs =
        input.flat_indexes()[FilteredArray3<pair<int32_t, int32_t>>(
            this_state_subsets, Filter1(output_fsa_idxs.elems.data()))];

    // Note on how elements of arc_idxs relate to elements of `state_subsets`:
    // at this point, the combined (1st,2nd) indexes of arc_idxs, i.e.
    // the offset into that 2nd-level array, gives us the flat index into
    // `state_subsets.elems`.  Note, this only works because of how
    // Range() is implemented internally, pointing to the underlying
    // array; if it were copied this wouldn't work.  This is not very ideal.
    struct ToStateSubsetsFilter {
      __device__ int32_t
      operator()(typename Array3<int32_t>::accessor acc) const {
        return acc.offset1();
      }
    };

    // actually the following is an unnecesssary temporary.
    // This is supposed to get the flat indexes into `this_state_subsets` that
    // each element of `arc_idxs` corresponds to.  The + ... is to account
    // for this_state_subsets being a range of state_subsets.
    Array1<int32_t> state_subsets_flat_indexes =
        FilteredArray2<int32_t>(arc_idxs, ToStateSubsetsFilter()).elems() + ...;

    struct ArcToDestStateFilter {
      using ValueType = int32_t;
      Arc *elems;  // device pointer.

      __host__ __device__ ArcToDestStateFilter(const ArcToDestStatFilter &other)
          : elems(other.elems) {}

      __host__ ArcToDestStateFilter(Arc *elems) : elems(elems) {}

      int32_t operator()(typename Array3<int32_t>::accessor acc) const {
        // Note, the arc format we're using is not very efficiently accessed
        // like this, would be better to store src_state, dest_state and so
        // on in
        // separate arrays.
        return elems[acc.elem()].next_state;
      }
    };
    // Use array lookup to get the next-state for each of the src-states..
    // Note: each elem has same 3-dim structure as the Array3's in arc_idxs.
    Array3<int32_t> next_state = FilteredArray3<int32_t>(
        input.elems[arc_idxs], arc_to_dest_state_filter);
    // arc_to_dest_state_filter is a class
    // object with operator () taking Arc and
    // returning int32_t.

    // Discard the last level of array in `next_state`, we don't care about
    // origin state.  Note, we won't resize this, so we can use Array2Base which
    // is not resizable, and which is like a view or iterator (into next_state).
    Array2<int32_t> next_state2 = next_state.concatenate(-1);

    // Obtain a map that reorders `next_state2` within each sub-list, so that
    // the next-states are contiguous.
    // This is a sorting-sublists operation, so it's as if we're sorting the
    // members of each sublist in `next_state2`, and get not the sorted elements
    // but the mapping of indexes.
    // (These are the flat indexes, i.e. into the .elems.)
    bool flat_indexes = true;
    Array2<int32_t> sort_idxs = GetSublistSortOrder(next_state2, flat_indexes);

    // This will be a reordering of `arc_idxs` using `sort_idxs`, so that things
    // with the same next-state are adjacent.
    Array2<int32_t> arc_idxs2 = arc_idxs;
    arc_idxs2 = arc_idxs2.elems[sort_idxs];

    // the flat indexes into `state_subsets` that correspond to each element of
    // arc_idxs2.elems.
    Array1<int32_t> state_subsets_flat_indexes2 =
        state_subset_flat_indexes[sort_idxs];

    // Partition sub-lists of `arc_idxs2`, so that those with the same
    // next-state form individual sub-lists.
    // Same underlying data.
    Array3<int32_t> arc_idxs3 =
        arc_idxs2.partition(next_state2.elems[sort_idxs]);

    // Get the scores with the same structures as arc_idxs3...
    Array3<int32_t> this_scores(arc_idxs3.shape(),
                                input_scores.elems[arc_idxs3.elems]);

    // Get sort_idxs, for reordering `this_scores` within each sub-sub-list
    // so that the one with the best score is first, and if there is a tie
    // on scores, disambiguate with arc-index.
    // (Would give the sort routine a comparator object).
    // TODO: actually we could consider just finding the best in each sublist
    // rather than fully sorting.  Would do that using a different kind of
    // indexing.
    Array3<int32_t> sort_idxs2 = GetSublistSortOrder(this_scores, flat_indexes);

    // this is a reordering of arc_idxs3 with `sort_idxs2`.
    Array3<int32_t> arc_idxs4(arc_idxs3.shape(), arc_idxs3.elems[sort_idxs2]);
    Array1<int32_t> state_subsets_flat_indexes3 =
        state_subset_flat_indexes2[sort_idxs2];

    // this selects the arc indexes with what in Python would be [:,:,0];
    // meaning, take 1st element of each sub-list.  This discards transitions
    // into states that were not the best transition.
    // -1 is the axis (i.e. the last axis).
    Array1<int32_t> elems_to_select =
        arc_idxs4.shape().offsets(-1).range(0, -1);
    Array2<int32_t> arc_idxs5 =
        Array3<int32_t>(arc_idxs4.shape().without_last_level(),
                        arc_idxs4.elems[elems_to_select]);

    // Get the flat indexes into `state_subsets` that correspond to the
    // originating (input-)states for each element of arc_idxs5.elems
    Array1<int32_t> state_subset_idxs =
        state_subsets_flat_indexes2[elems_to_select];

    // max_len is maximum length of any sequence in the states we were
    // extending, plus 1 because we extended with one new arc.
    // We'll have an iteration over this length.
    // Note: in future, to make this more efficient, we could consider
    // reordering the sub-lists of `arc_idxs5` so that the ones which have a
    // larger length in state_repr go first.  that way, the later kernels in the
    // sequence can run fewer threads.  (?)  Or do some other operation that
    // removes unused positions/indexes as we go.
    int32_t max_len = Max(symbol_seq.shape.size(1).range(..., -1) + 1);

    Array2<int32_t> cur_state_subset_idxs = state_subset_idxs;

    /*
      We'd change this once we get away from using a fixed global
      max.
      Anyway, the idea here is that arc_indxs is a list of lists
      that we'll zip to get the list of input arcs that corresponds
      to each of the states that we're creating.

      Each member of arc_indxs *conceptually* represents a list of
      sub-lists of length zero or one, but actually contains
      either the element of the sub-list, or -1 if there is no
      such element.

      We could actually allocate the memory just once upfront,
      as we know the size.
     */
    std::vector<Array1<int32_t>> arc_indxs(max_len, ...);

    // num_new_output_arcs is the num_new_output_states *with duplicates*.  In
    // general the real number of new output-states will be less than this.  but
    // this is the number of new output-arcs (i.e. arcs in the output FSA).
    int32_t num_new_output_arcs = state_subset_idxs.size0();

    /*
      For each potential output state, this will give the number of arcs in the
      input-FSA that we removed from its canonical representation, i.e.  the
      prefix that we removed.  Init with 0; we'll increment in the loop.
    */
    Array1<int32_t> num_arcs_removed(num_new_output_arcs, 0);

    /*
      For each potential output state, this will give the base state from which
      we take that sequence of arcs.  Init with -1.  (I think.  Or might be
      necessary to init with the current base_state, which we can get from
      indexing `state_repr` appropriately, with [some_array][0].
    */
    Array1<int32_t> base_state(...);

    /*
      Before we do the hash lookup, we need to compute the canonical
      representation, which is: base_state, then symbol-sequence.
      The "canonicalization" process consists of finding the longest
      prefix of the symbol-sequence that we can follow and still
      have a unique base_state (i.e there is no branching by then).
    */
    for (i = 0; i < max_len; i++) {
      /*
        TODO: one way to speed this up would be some kind of mask (e.g. one
        element, e.g. byte, per warp) that says whether there is any work to do
        here; we could immediately exit if not, to save resources.
       */

      // init with 1's.  we'll use this to tell whether all the elements we're
      // tracing back are identical.
      Array1<int32_t> is_same(cur_state_subset_idxs.size0(), 1);

      // The syntax below won't be what we'll use, but the idea here is that
      // for each sub-list in `cur_state_subset_indexes` (where each sub-list
      // represents an output state), if we detect that not all elements of the
      // sub-list are the same we set the corresponding element of `is_same` to
      // zero.  This can be done using a simple if-statement in the kernel; it's
      // based on comparing the current element of the sub-list to the 1st
      // element of the sub-list. We could also do it with a simple filter,
      // eval'd, that would run on cur_state_subset_idxs.
      TestIsSame(&cur_state_subset_indexes, &is_same);

      /*  The next code will set arc_indxs[i].  If is_same[j] is true, we'll set
          arc_indxs[i] to an arc-index.  This is an arc on the prefix of the
          path, that we're removing.  Otherwise, we'll set it to -1.
          This is straightforward and fast.
          Note: the reason we need to keep track of the 'removed' arc_indxs is
          so that we can output the derivative information and know how to
          normalize the scores of state subsets.  (Actually this normalization
          only keeps things in a good numerical range, it doesn't affect the
          algorithm).
      */

      /* The next code will increment `num_arcs_removed` if the corresponding
         element of arc_indxs was not -1.  */

      /* The next code will set `base_state` if the corresponding element of
         arc_indxs was not -1 and the current value of base_state is not -1. (Or
         something like that; anyway it's straightforward. */
    }

    /* Now create the new `repr` (representation) for each proposed
       output-state, we'll call this vector-of-vectors `this_repr`.  The
       representation of each proposed state is a sub-list consisting of
       (base_state, list of arcs traversed).  The things that go into this are
       the `base_state` array, the existing list of arcs traversed from the
       `state_repr` (the tail of it, see state_repr.range(...)), the latest arc
       we traversed (arc_idxs5), and the number of arcs removed from the
       existing sequence (num_arcs_removed).

       We'd need to do one pass of exclusive-scan (prefix sum) to work out the
       size of the new `repr` elements; once we compute the `offset vector` and
       `groups vector`, the actual setting of the elements can then be done
       using a fairly simple lookup operation.
    */

    /*
      Next is to look up the vectors in our `this_repr` vector in our map, that
      maps them to state indexes.

      - Stage 1: compute some kind of hash for each sub-list in `this_repr`.
         For now we'll assume there are no collisions here (could be a 128-bit
      hash).
      - Stage 2: lookup of hash in some kind of hash-map object.
        Interface would be something like:
    */
    Array2<int32_t> this_repr(...);

    Array1<int64_t> repr_hash(this_repr.size1);
    ComputeHash(this_repr, &repr_hash);

    Array1<int32_t> repr_idx(repr_hash.size());
    {
      // NOTE: some of the contents of this code block could probably be given
      // some kind of interface as it may be a common pattern.

      /*
        hash_map would be some data structure containing a big array, probably
        of pair<int64_t, int32_t> (key-value pairs); the value is the index into
        state_repr.

        HashLookupOrAdd() does as follows, for each 0 <= i < repr_hash.size():
        - If repr_hash[i] is in the map `hash_map` already, set repr_idx[i] to
        hash_map[repr_hash[i]].
        - Otherwise, set hash_map[repr_hash[i]] to 1000000 + i and set
        repr_idx[i] to that value.
        The behavior is as if the above had been done one by one for all i's,
        but not necessarily in sequential order.  I.e. if more than one element
        of repr_hash is the same and was not already in the map, an arbitrary
        one will `win`.  But they will all get the same retur value, equal to
        1000000 plus the i of the one that 'won'.
      */
      HashLookupOrAdd(repr_hash, &hash_map, &repr_idx, 1000000);

      Array1<char> is_new(repr_idx.size());

      // sets is_new[i] to (repr_idx == i + 1000000 ? 1 : 0).
      // It would be nice if we had some cute templated way to
      // do this.  Actually I'd like to be able to combine this with
      // the next kernel...
      SetEqIPlus(repr_idx, &is_new, 1000000);

      Array1<int32_t> prefix_sum(repr_idx.size());
      // compute exclusive prefix sum of `is_new` -> `prefix_sum`..
      ExclusiveSum(is_new, &prefix_sum);

      int32_t num_new = sort_idxs[repr_idx.size() - 1];
      Array1<int32_t> sort_idxs(num_new + 1);
      // `sort_idxs` will, after the next call, be a vector giving the positions
      // in `repr_hash` of the hash elements that were newly added to the hash
      // (not counting repeats of the same hash element in that vector; only an
      // arbitrarily chosen one).
      GroupsToOffsets(prefix_sum, &sort_idxs);
      sort_idxs.resize(num_new);  // discard last elem.

      Array1<int32_t> new_state_nums =
          range(next_ostate, next_ostate + num_new);
      // The following call will replace the temporary values we put in the hash
      // above (greater than one million) with the "real" values.
      HashSet(repr_hash[sort_idxs], new_state_nums, &hash);

      {  // Now set `state_repr`
        // Append arrays to `state_repr` (NOTE: being able to do this implies we
        // must do the memory management a certain way; would be like
        // std::vector).
        //
        // The .remove_prefix() thing means to remove the first n elements
        // of the `repr`.
        state_repr.Append(
            this_repr[sort_idxs].remove_prefix(num_arcs_removed[sort_idxs]));
      }

      // The following call will ensure that all of the state-ids in `repr_idx`
      // are actually real state-ids (including the `num_new` states we just
      // allocated).
      HashLookup(hash, repr_hash, &repr_idx);

      prev_ostate = next_ostate;
      next_ostate += num_new;
    }

    // Next
  }
}

}  // namespace k2
