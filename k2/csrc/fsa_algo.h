// k2/csrc/fsa_algo.h

// Copyright (c)  2020  Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_FSA_ALGO_H_
#define K2_CSRC_FSA_ALGO_H_

#include <utility>
#include <vector>

#include "k2/csrc/fsa.h"

namespace k2 {

void RandomPath(const Fsa &a, const float *a_cost, Fsa *b,
                std::vector<int32_t> *state_map = nullptr);

}  // namespace k2

#endif  // K2_CSRC_FSA_ALGO_H_
