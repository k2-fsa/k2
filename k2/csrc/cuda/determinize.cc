

/*
  Right now this just contains some notes on FSA determinization, written in order
  to clarify my thinking on the right fundamental primitives.
 */


// actually we'd simultaneously determinize an array of FSAs, in practice.
void Determinize(Vec<Array2<Arc> > &Fsa,
                 Vec<float*> input_scores,
                 Allocator &alloc,
                 ... ) {

  // OK this may be a bit ugly, but the idea here is that we set the allocator
  // as a thread_local global variable so that it's available whenever
  // we initialize device arrays.  We could pass it to all the constructors,
  // but this would get tedious.
  // TODO: add as opt args to constructors?
  // Note: having explicit control of the allocator may be necessary under
  // some circumstances.
  SetDeviceAllocator(alloc);


  int num_fsas = Fsa.size();


  // outer Vec< > around everything means 'for each of the FSAs we are
  // determinizing'.
  // Array2<Arc> means: list of list of arcs, i.e. list of arcs leaving
  // each state.
  Vec<Array2<Arc> > output(num_fsas);


  Array2<int> one_state_subsets("[ [ 0 ] ]");
  Vec<Array2<int> > state_subsets(num_fsas, one_state_subsets);
                                               // each of these is the set of input-states that
                                               // corresponds to a particular output state.
                                               // The order is arbitrary.
  state_subsets[kI].push_back(Array<int>("0"));

  Vec<Array<float> > state_subset_scores(num_fsas, one_state_subset_scores);  // score for each element of
                                   // `state_subsets`... actually the subsets
                                   // are weighted subsets.  Note the most
                                   // negative score in each subset won't
                                   // necessarily be 0; the normalization method
                                   // is different in this algorithm.
  state_subset_scores[kI].push_back(0.0);

  // for each element of 'state_subsets[i].elems', 'state_subset_arcs[i]'
  // contains the index of the incoming arc that created it, or -1 if this is
  // the start state.
  Vec<Array<int> > state_subset_arcs(num_fsas, one_state_subset_arcs);
  state_subset_arcs[kI].push_back(-1);

  // for each element of 'state_subsets.elems', the flat index of the previous
  // element in `state_subsets` which 'created' this, or -1 if this was an
  // initial arc.  The arc (in state_subset_arcs) entering that previous state
  // will be a preceding arc on a path vs the arc in state_subset_arcs entering
  // this state.
  Array<int> one_state_subset_traceback("0.0");
  Vec<Array<int> > state_subset_traceback(num_fsas, one_state_subset_traceback);


  Vec<Array2<int> > state_repr;      // Canonical representation of each ostate,
                                     // which is a vector: istate, followed by
                                     // symbol sequence that we follow from
                                     // istate.


  typedef struct { int64_t a; int64_t b; } HashKeyType;
  struct Hasher {
    size_t operator () (const HashKeyType key);
  };

  Vec<Array<HashKeyType> > state_repr_hash;  // hash-value of corresponding elements of
                                         // state_repr (note: length of each one
                                         // these is the size1() of the
                                         // corresponding state_repr element.)

  Hash<HashKeyType, int, Hasher> repr_hash_to_id;  // Maps from (fsa_index, hash of state_repr) to
                                                   // corresponding state-id.   Note: it would be the
                                                   // state-id *in that FSA*. (the FSA with index 0 <= fsa_index < num_fsas);
                                                   // we keep everything in one hash.  (TODO: set the num-buckets appropriately).

  // TODO: initialize variables defined above, with 1st state.

  // Note: host vector...
  Vec<int> prev_ostate(num_fsas, 0);

  while (1) {


    // while queue not empty...  (queue is the batch of newly added ostates.)


    // Init as sub-range of state_subsets (subsets from prev_ostate to
    // next_ostate).
    // Note: `range` is a range on the 1st index into the array, the top-level
    // one.
    Vec<Array2Sub<int> > this_state_subsets(
        this_state[kI].range(prev_ostate[kI], this_state[kI].size()));


    // will swap with prev_ostate.
    Vec<int> next_ostate(this_state[kI].size());


    // note, input[i].size() for each i would be a dynamic expression taking the
    // diff between the offsets, i.e. size1() is a scalar but size() is a vector.
    Vec<Array<int> > num_arcs = input[kI].size();


    // Do array lookup to get num-arcs for each ostate, and then form an array3
    // containing the arc indexes for the transitions we need to process.
    // Here, flat_indexes is just replacing elems[i] with i, i.e. the flat indexes
    // into the elems...
    //
    // Indexing with this_state_subsets[kI] means indexing with an Array2;
    // For each scalar k in the array2, input[kI].flat_indexes[k] evaluates
    // to a vector, so the overall thing is an Array3.
    Vec<Array3<int> > arc_idxs(
        input[kI].flat_indexes[this_state_subsets[kI]]);



    // Use array lookup to get the next-state for each of the src-states..
    // Note: each elem has same 3-dim structure as the Array3's in arc_idx's.
    Vec<Array3<int> > next_state(
        input[kI].elems[arc_idxs[kI]],
        arc_to_dest_state_filter);  // arc_to_dest_state_filter is a class object with operator ()
                                    // taking Arc and returning int, and a ValueType that we
                                    // can use in template matching.

    // Discard the last level of array in `next_state`, we don't care about origin state.
    // Not, we won't resize this, so we can use Array2Base which is not resizable,
    // and which is like a view or iterator (into next_state).
    Vec<Array2Base<int> > next_state2 = next_state.collapse_last_level();

    // Obtain a map that reorders `next_state2`, within each sub-list, so that
    // the next-states are contiguous.
    // This is a sorting-sublists operation, so it's as if we're sorting the members
    // of each sublist in `next_state2`, and get not the sorted elements but the
    // mapping of indexes.  (These are the flat indexes, i.e. into the .elems.
    bool flat_indexes = true;
    Vec<Array2<int> > ranks = next_state2[kI].rank_sublists(flat_indexes);

    // This will be a reordering of `arc_idxs` using `ranks`, so that things with the
    // same next-state are adjacent.
    Vec<Array2<int> > arc_idxs2 = next_state2[kI].reorder_sublists(ranks, flat_indexes);

    // This will be sub-lists of `arc_idxs2`, so that those with the same next-state form
    // individual sub-lists.  (Does not have to have reproducible ordering!)
    // Same underlying data.
    // the following syntax may not be the best way.  We could perhaps do this in 2 stages,
    // first getting the structure using a physical array of next-states, then
    // getting the partition and then switching out the elems.
    Vec<Array3<int> > arc_idxs3 = arc_idxs2[kI].partition(inputs[kI].elems[arc_idxs2.elems[kJ]].next_state);

    // Get the scores with the same structures as arc_idxs3...
    Vec<Array3<int> > this_scores(arc_idxs3, input_scores[kI][arc_idxs3[kI]]);

    // Get ranks, reordering `this_scores` within each sub-sub-list so that the
    // one with the best score is first, and if there is a tie on scores,
    // disambiguate with arc-index.  (Would give the sort routine a comparator
    // object).
    Vec<Array3<int> > ranks2 = this_scores[kI].rank_sublists(flat_indexes);


    // this is a reordering of arc_idxs3 with `ranks2`.
    Vec<Array3<int> > arc_idxs4 = arc_idxs3.reorder_sublists(ranks2, flat_indexes);


    // this selects the arc indexes with what in Python would be [:,:,0];
    // meaning, take 1st element of each sub-list.  This discards transitions
    // into states that were not the best transition.
    Vec<Array2<int> > arc_idxs5 = arc_idxs4[kI][kJ][kK][0];


    // Get the flat indexes into `state_subsets` that correspond to the
    // originating (input-)states for each element of arc_idxs5.  Computing it
    // would probably start with some kind of `range` expression and involve
    // ranks and ranks2.
    // MM.  It might be easier to propagate those indexes alongside the
    // arc_idxs, or even use them in place of arc_idxs, and derive arc_idxs from
    // them where necessary.
    Vec<Array<int> > state_subset_idxs = arc_idxs5[kI][kJ].size()


    // max_len is maximum length of any sequence in the states we were
    // extending, plus 1 because we extended with one new arc.
    // We'll have an iteration over this length.
    // Note: in future, to make this more efficient, we'll consider reordering
    // the sub-lists of `arc_idxs5` so that the ones which have a larger
    // length in state_repr go first.  that way, the later kernels in the sequence
    // can run fewer threads.
        int max_len = Max(state_subset  (MaxSublistLen(state_repr.range(...)) - 1) + 1;


    Array2<int> *cur_state_subset_indexes = &state_subset_idxs;


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
     */
    Array<int> *arc_indxs = new Array<int>[max_len]( ...);


    /*
      For each potential output state, this will give the number of
      arcs that we removed from its canonical representation, i.e.
      the prefix that we removed.  Init with 0; we'll increment
      in the loop.
    */
    Array<int> num_arcs_removed(...);

    /*
      For each potential output state, this will give the base state
      from which we take that sequence of arcs.  Init with -1.
      (I think.  Or might be necessary to init with the current
      base_state, which we can get from indexing `state_repr` appropriately,
      with [some_array][0].
    */
    Array<int> base_state(...);

    for (i = 0; i < max_len; i++) {
      /*
        TODO: one way to speed this up would be some kind of mask (e.g. one
        element, e.g. byte, per warp) that says whether there is any work to do
        here; we could immediately exit if not, to save resources.
       */

      // init with 1's.  we'll use this to tell whether all the elements we're
      // tracing back are identical.
      Array<int> is_same(state_subset_idxs.size1, 1);


      // The syntax below won't be what we'll use, but the idea here is that
      // for each sub-list in `cur_state_subset_indexes` (where each sub-list represents
      // an output state), if we detect that not all elements of the sub-list are the
      // same we set the corresponding element of `is_same` to zero.  This
      // can be done using a simple if-statement in the kernel; it's based on
      // comparing the current element of the sub-list to the 1st element of
      // the sub-list.
      TestIsSame(&cur_state_subset_indexes, &is_same);

      /*  The next code will set arc_indxs[i].  If is_same[j] is true, we'll set
          arc_indxs[i] to an arc-index (conceptually an arc on the prefix of the
          path, that we're removing, and otherwise, we'll set it to -1.
          This is straightforward and fast.
      */

      /* The next code will increment `num_arcs_removed` if the corresponding element
         of arc_indxs was not -1.  */

      /* The next code will set `base_state` if the corresponding element of arc_indxs
         was not -1 and the current value of base_state is not -1.  (Or something
         like that; anyway it's straightforward. */
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
         For now we'll assume there are no collisions here (could be a 128-bit hash).
      - Stage 2: lookup of hash in some kind of hash-map object.
        Interface would be something like:
    */
    Array2<int> this_repr(...);


    Array<int64_t> repr_hash(this_repr.size1);
    ComputeHash(this_repr, &repr_hash);


    Array<int> repr_idx(repr_hash.size());
    {
      // NOTE: some of the contents of this code block could probably be given
      // some kind of interface as it may be a common pattern.

      /*
        hash_map would be some data structure containing a big array, probably of
        pair<int64_t, int> (key-value pairs); the value is the index into
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

      Array<char> is_new(repr_idx.size());

      // sets is_new[i] to (repr_idx == i + 1000000 ? 1 : 0).
      // It would be nice if we had some cute templated way to
      // do this.  Actually I'd like to be able to combine this with
      // the next kernel...
      SetEqIPlus(repr_idx, &is_new, 1000000);


      Array<int> prefix_sum(repr_idx.size());
      // compute exclusive prefix sum of `is_new` -> `prefix_sum`..
      ExclusivePrefixSum(is_new, &prefix_sum);

      int num_new = ranks[repr_idx.size() - 1];
      Array<int> ranks(num_new + 1);
      // `ranks` will, after the next call, be a vector giving the positions in
      // `repr_hash` of the hash elements that were newly added to the hash (not
      // counting repeats of the same hash element in that vector; only an
      // arbitrarily chosen one).
      GroupsToOffsets(prefix_sum, &ranks);
      ranks.resize(num_new); // discard last elem.


      Array<int> new_state_nums = range(next_ostate,
                                        next_ostate + num_new);
      // The following call will replace the temporary values we put in the hash above
      // (greater than one million) with the "real" values.
      HashSet(repr_hash[ranks], new_state_nums, &hash);


      {  // Now set `state_repr`

        // Append arrays to `state_repr` (NOTE: being able to do this implies we
        // must do the memory management a certain way; would be like
        // std::vector).
        //
        // The .remove_prefix() thing means to remove the first n elements
        // of the `repr`.
        state_repr.Append(
            this_repr[ranks].remove_prefix(num_arcs_removed[ranks]));

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
