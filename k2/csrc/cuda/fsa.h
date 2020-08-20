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

using Fsa = RaggedShape2<Arc>;

using FsaVec = RaggedShape3<Arc>;


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
  RaggedShape2 shape;  // Indexed first by FSA-index (this represents a number of
                       // FSAs, and then for each FSA, the state-index (actually
                       // the state-index from which the arcs leave).


  // TODO: construct from a regular matrix containing the nnet output, plus some
  // meta-info saying where the supervisions are.


  // The following variable was removed and can be obtained as scores.Dim1().
  // int32_t num_cols;

  // `scores` is a contiguous matrix of dimension shape.TotSize1()
  // by num_cols (where num_cols == num_symbols+1); the indexes into it are
  // [row_idx, symbol+1], where row_ids is an ind_01 w.r.t. `shape` (see naming
  // convention explained in utils.h).
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


}  // namespace k2

#endif  // K2_CSRC_CUDA_FSA_H_
