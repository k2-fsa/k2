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

#ifndef K2_CSRC_COMPOSE_H_
#define K2_CSRC_COMPOSE_H_

#include "k2/csrc/array.h"

namespace k2 {


// Note: b is FsaVec<Arc>.
void Intersect(const DenseFsa &a, const FsaVec &b, Fsa *c,
               Array1<int32_t> *arc_map_a = nullptr,
               Array1<int32_t> *arc_map_b = nullptr);



// compose/intersect array of FSAs (multiple streams decoding or training in
// parallel, in a batch)... basically composition with frame-synchronous beam pruning,
// like in speech recognition.
//
// This code is intended to run on GPU (but should also work on CPU).
void IntersectDensePruned(Array3<Arc> &a_fsas,
                          DenseFsaVec &b_fsas,
                          float beam,
                          int32_t max_states,
                          FsaVec *ofsa,
                          Array1<int32_t> *arc_map_a,
                          Array1<int32_t> *arc_map_b);

}  // namespace k2

#endif  // K2_CSRC_COMPOSE_H_
