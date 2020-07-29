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

  Conceptually this is a 3-dimensional tensor with the second dimension ragged,
  i.e.  the shape would be [ num_fsas, None, num_cols ], e.g. if this were a
  TF ragged tensor.  It's viewed as a list of linear dense FSAs where each one
  has a different num-states/length.

  Example:

   Suppose the row_sizes (the "None" dimension) are 6, 7, 2, the symbol_shift is
   1 and the num_cols is 41.  Consider the first FSA, the one whose row_size
   is 6.  This would be an FSA with 7 states where state 0 is the initial state
   and state 6 is the final state.  For each 0 < i < 6, and for each 0 <= j <
   num_cols, there is conceptually an arc from state i to i+1 with symbol (j -
   symbol_shift) and cost equal to `scores[i,j]` if `scores` were a NumPy matrix;
   actually we'd access this as `scores.data[i*num_cols + j]`.
 */
class DenseFsaVec {
  RaggedShape2 shape;  // shape.Sizes1() gives the number of states in each
                       // dense FSA minus one, i.e. the number of states that
                       // have arcs leaving them (excluding final state).  In an
                       // ASR task, shape.Sizes1()[i] would also equal the
                       // number of frames in that sequence plus one.  (We add
                       // one 'fake' frame to handle arcs to the final state).


  // TODO: construct from a regular matrix containing the nnet output, plus some
  // meta-info saying where the supervisions are.


  int symbol_shift;  // Expected to be 1 (this relates to handling -1 /
                     // final-probs efficiently); see comment above this class.
                     // Symbols are shifted relative to column index in `scores`.

  int num_cols;    // num_cols will be (the number of symbols, including
                     // zero), which is the nnet output dim, plus symbol_shift.

  Array1<float> scores;   // Conceptually a matrix of dimension shape.TotSize2()
                          // by num_cols.  Contains -infinity in certain
                          // locations, as well as the actual neural net output;
                          // see comment above class.  The actual nnet output is
                          // located in scores[:-1,1:], imagining it were a
                          // NumPy matrix; other elements should all be
                          // -infinity, except a 0 in scores[-1,0] which relates
                          // to the transition to the final-state.

};


}  // namespace k2

#endif  // K2_CSRC_CUDA_FSA_H_
