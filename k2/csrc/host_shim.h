/**
 * @brief host_shim  Wrapper functions so we can use our older
 *                 CPU-only code, in host/, with the newer interfaces
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_SHIM_H_
#define K2_CSRC_HOST_SHIM_H_

#include "k2/csrc/fsa.h"
#include "k2/csrc/host/fsa.h"

namespace k2 {

k2host::Fsa FsaToHostFsa(Fsa &fsa);

// get weights of FSA separately
Array1<float> WeightsOfFsa(Fsa &fsa);

struct Arc {
  int32_t src_state;
  int32_t dest_state;
  int32_t symbol;
  float score;  // we have the space to put this here, so...
};

using FsaProperties = uint32_t;

/*
  The FSA properties need to be computed via a reduction over arcs, and we use
  '&' to reduce.  These properties are all things that can be computed locally,
  using an arc and adjacent arcs (and the structural info).
 */
enum FsaBasicProperties {
  kFsaPropertiesValid = 0x01,      // Valid from a formatting perspective *as an
                                   // FsaVec*.  Also require
                                   // kFsaPropertiesSingleFsa == true if
                                   // this is supposed to be a single FSA, not
                                   // an FsaVec.
  kFsaPropertiesNonempty = 0x02,   // Nonempty as in, has at least one arc.
  kFsaPropertiesTopSorted = 0x04,  // FSA is top-sorted, dest_state >= src_state
  kFsaPropertiesTopSortedAndAcyclic =
      0x08,  // Top-sorted and acyclic, dest_state > src_state
  kFsaPropertiesArcSorted =
      0x10,  // Arcs leaving a given state are sorted by symbol
  kFsaPropertiesArcSortedAndDeterministic = 0x20,  // Arcs leaving a given state
                                                   // are *strictly* sorted by
                                                   // symbol, i.e. no duplicates
                                                   // with the same symbol.
  kFsaPropertiesEpsilonFree = 0x40,  // Symbol zero (epsilon) is not present..
  kFsaPropertiesMaybeAccessible = 0x80,  // True if there are no obvious signs
                                         // of states not being accessible or
                                         // co-accessible, i.e. states with no
                                         // arcs entering them
  kFsaPropertiesMaybeCoaccessible =
      0x80,                             // True if there are no obvious signs of
                                        // states not being co-accessible, i.e.
                                        // i.e. states with no arcs leaving them
  kFsaPropertiesSerializable = 0x0100,  // True if there are no FSAs with zero
                                        // states, and if for all fsa-indexes i,
                                        // last-state(i) > first-state(i+1)
                                        // where {last,first}-state is the
                                        // {last,first} state that has an arc
                                        // leaving it.  These properties are
                                        // used in figuring out the boundaries
                                        // between FSAs when we serialize to a
                                        // list of arcs.
  kFsaAllProperties = 0x01FF
};

using Fsa = RaggedShape<Arc>;  // 2 axes: state,arc

using FsaVec = RaggedShape<Arc>;  // 3 axes: fsa,state,arc.  Note, the src_state
                                  // and dest_state in the arc are *within the
                                  // FSA*, i.e. they are idx1 not idx01.

/*
  Vector of FSAs that actually will come from neural net log-softmax outputs (or
  similar).

  Conceptually this is a 3-dimensional tensor of log-probs with the second
  dimension ragged, i.e.  the shape would be [ num_fsas, None, num_symbols+1 ],
  e.g. if this were a TF ragged tensor.  The indexing would be
  [fsa_idx,t,symbol+1], where the "+1" after the symbol is so that we have
  somewhere to put the output for symbol == -1 (remember, -1 is kFinalSymbol,
  used on the last frame).

  Also, if a particular FSA has T frames of neural net output, we actually
  have T+1 potential indexes, 0 through T, so there is space for the terminating
  final-symbol on frame T.  (On the last frame, the final symbol has
  logprob=0, the others have logprob=-inf).
 */
class DenseFsaVec {
  RaggedShape shape;  // has 2 axes; indexed first by FSA-index (this object
                      // represents a list of FSAs!); and then for each FSA,
                      // the state-index (actually the state-index from which
                      // the arcs leave).

  // TODO: construct from a regular matrix containing the nnet output, plus some
  // meta-info saying where the supervisions are.

  // The following variable was removed and can be obtained as scores.Dim1().
  // int32_t num_cols;

  // `scores` is a contiguous matrix of dimension shape.TotSize1()
  // by num_cols (where num_cols == num_symbols+1); the indexes into it are
  // [row_idx, symbol+1], where row_ids is an ind_01 w.r.t. `shape` (see naming
  // convention explained in utils.h).
  //
  //  You can access scores[row_idx,symbol+1] as
  //  scores.Data()[row_ids*scores.Dim1() + symbol+1]
  //
  // `scores` contains -infinity in certain locations: in scores[j,0] where
  // j is not the last row-index for a given FSA-index, and scores[j,k] where
  // j is the last row-index for a given FSA-index and k>0.  The remaining
  // locations contain the neural net output, except scores[j,0] where j
  // is the last row-index for a given FSA-index; this contains zero.
  // (It's the final-transition).
  Array2<float> scores;

  // NOTE: our notion of "arc-index" / arc_idx is an index into scores.Data().
  int32_t NumArcs() { return scores.Size0() * scores.Size1(); }
};

/*
  Create an FSA from a Tensor.  The Tensor is expected to be an N by 4 tensor of
  int32_t, where N is the number of arcs (the format is src_state, dest_state,
  symbol, cost).  The cost is not really an int32_t, it is a float.  This code
  will print an error message and output 'true' to 'error', and return an empty
  FSA (with no states or arcs) if 't' was not interpretable as a valid FSA.
  These requirements for a valid FSA are:

    - src_state values on the arcs must be non-decreasing
    - all arcs with -1 on the label must be to a single state (call this
      final_state) which has no arcs leaving it
    - final_state, if it exists, must be numerically greater than any state
      which has arcs leaving it, and >= any state that has arcs entering it.

  If there are no arcs with -1 on the label, here is how we determine the final
  state:
     - If there were no arcs at all in the FSA we'll return the empty FSA (with
  no states).
     - Otherwise, we'll let `final_state` be the highest-numbered state
       that has any arcs leaving or entering it, plus one.  (This FSA
       has no successful paths but still has states.)

    @param [in] t   Source tensor.  Caution: the returned FSA will share
                    memory with this tensor, so don't modify it afterward!
    @param [out] error   Error flag.  On success this function will write
                        'false' here; on error, it will print an error
                        message to the standard error and write 'true' here.
    @return         The resulting FSA will be returned.

*/
Fsa FsaFromTensor(Tensor t, bool *error);

/*
  Returns a single Tensor that represents the FSA; this is just the vector of
  Arc reinterpreted as  num_arcs by 4 Tensor of int32_t.  It can be converted
  back to an equivalent FSA using `FsaFromTensor`.
 */
Tensor FsaToTensor(const Fsa &fsa);

/*
  Returns a single Tensor that represents the vector of FSAs; this is just the
  vector of Arc reinterpreted as num_arcs by 4 Tensor of int32_t.  It can be
  converted back to an equivalent FsaVec using `FsaVecFromTensor`.
 */
Tensor FsaVecToTensor(const Fsa &fsa);

/*
  Create an FsaVec (vector of FSAs) from a Tensor.  Please see FsaFromTensor for
  how this works for a single FSA.  The reason we can do the same with multiple
  FSAs is that we can use the discontinuities in `src_state` (i.e. where the
  values decrease) to spot where one FSA starts and the next begins.  However
  this only works if all the FSAs were nonempty, i.e. had at least one state.
  This function will die with an assertion failure if any of the provided
  FSAs were empty, so the user should check that beforehand.

  Please see FsaFromTensor() for documentation on what makes the individual
  FSAs valid; however, please note that the FSA with no states (empty FSA)
  cannot appear here, as there is no way to indicate it in a flat
  series of arcs.

    @param [in] t   Source tensor.  Must have dtype == kInt32Dtype and be of
                    shape (N > 0) by 4.  Caution: the returned FSA will share
                    memory with this tensor, so don't modify it afterward!
    @param [out] error   Error flag.  On success this function will write
                        'false' here; on error, it will print an error
                        message to the standard error and write 'true' here.
    @return         The resulting FsaVec (vector of FSAs) will be returned;
                    this is a Ragged<Arc> with 3 axes.

*/
FsaVec FsaVecFromTensor(const Tensor &t, bool *error);

/*
  Return one Fsa in an FsaVec.  Note, this has to make copies of the
  row offsets and strides but can use a sub-array of the arcs array
  directly.

     @param [in] vec   Input FsaVec to make a copy of
     @param [in] i     Index within the FsaVec to select
     @return           Returns the FSA.  Its `values` array will
                       refer to a part of the `values` array of
                       the input `vec`.
 */
Fsa GetFsaVecElement(const FsaVec &vec, int32_t i) { return vec.Index(0, i); }

/*
  Create an FsaVec from a list of Fsas.  Caution: Fsa and FsaVec are really
  the same type, just with different expectations on the number of axes!
 */
FsaVec CreateFsaVec(const FsaVec &vec, int32_t num_fsas, Fsa **fsas) {
  // Implementation goes to this templat:
  //  template <typename T>
  //  Ragged<T> Stack(int32_t axis, int32_t src_size, const Ragged<T> *src);
  K2_CHECK(fsas[0]->NumAxes() == 2);
  return Stack(0, num_fsas, fsas);
}

int32_t GetFsaBasicProperties(const Fsa &fsa);
int32_t GetFsaVecBasicProperties(const FsaVec &fsa_vec);

}  // namespace k2

#endif  // K2_CSRC_HOST_SHIM_H_
