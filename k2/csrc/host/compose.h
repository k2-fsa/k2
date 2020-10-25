/**
 * @brief
 * compose
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_HOST_COMPOSE_H_
#define K2_CSRC_HOST_COMPOSE_H_

#include <vector>

#include "k2/csrc/host/fsa.h"

namespace k2host {

class Compose {
 public:
  Compose(const Fsa &a, const Fsa &b, const int32_t *a_aux_labels,
          const int32_t *b_aux_labels)
      : a_(a),
        b_(b),
        a_aux_labels_(a_aux_labels),
        b_aux_labels_(b_aux_labels) {}

  void GetSizes(Array2Size<int32_t> *fsa_size);

  bool GetOutput(Fsa *c, std::vector<int32_t> *c_aux_labels,
                 int32_t *arc_map_a = nullptr, int32_t *arc_map_b = nullptr);

 private:
  // these are not references due to how we wrap this in fsa_algo.cu, it's
  // convenient to have them be copies.
  Fsa a_;
  Fsa b_;
  const int32_t *a_aux_labels_;
  const int32_t *b_aux_labels_;

  bool status_;
  std::vector<int32_t> arc_indexes_;  // arc_index of fsa_out
  std::vector<Arc> arcs_;             // arcs of fsa_out
  std::vector<int32_t> aux_labels_;   // aux_labels of fsa_out

  std::vector<int32_t> arc_map_a_;
  std::vector<int32_t> arc_map_b_;
};

}  // namespace k2host

#endif  // K2_CSRC_HOST_COMPOSE_H_
