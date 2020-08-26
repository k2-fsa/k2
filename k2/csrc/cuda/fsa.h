// k2/csrc/cuda/fsa.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_FSA_H_
#define K2_CSRC_CUDA_FSA_H_

#include "ragged.h"

namespace k2 {


struct Arc {
  int32_t src_state;
  int32_t dest_state;
  int32_t symbol;
  float score;  // we have the space to put this here, so...
};

using Fsa = RaggedShape<Arc>;  // 2 axes: state,arc

using FsaVec = RaggedShape<Arc>;  // 3 axes: fsa,state,arc.  Note, the src_state
                                  // and dest_state in the arc are *within the
                                  // FSA*, i.e. they are idx1 not idx01.


/*
  Vector of FSAs that actually will come from neural net log-softmax outputs (or similar).

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
  //  You can access scores[row_idx,symbol+1] as scores.Data()[row_ids*scores.Dim1() + symbol+1]
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
     - If there were no arcs we'll return the empty FSA (with no states).
     - Otherwise, we'll let `final_state` be the highest-numbered state
       that has any arcs leaving or entering it, plus one.  (This FSA
       has no successful paths but still has states.)

    @param [in] t   Source tensor.  Caution: the returned FSA will share
                    memory with this tensor, so don't modify it afterward!
    @param [out] error   Error flag.  On success this function will write 'true'
                    here; on error, it will print an error message to
                    the standard error and write 'false' here.
    @return         The resulting FSA will be returned.

*/
Fsa FsaFromTensor(const Tensor &t, bool *error);

/*
  Returns a single Tensor that represents the FSA; this is just the vector of Arc
  reinterpreted as  num_arcs by 4 Tensor of int32_t.  It can be converted back to
  an equivalent FSA using `FsaFromTensor`.
 */
Tensor FsaToTensor(const Fsa &fsa);


/*
  Return one Fsa in an FsaVec.
 */
Fsa GetFsaVecElement(const FsaVec &vec, int32_t index);


/*
  Create an FsaVec from a list of Fsas.
 */
FsaVec CreateFsaVec(const FsaVec &vec, int32_t num_fsas, Fsa *fsas);



}  // namespace k2

#endif  // K2_CSRC_CUDA_FSA_H_
