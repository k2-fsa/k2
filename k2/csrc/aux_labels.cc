// k2/csrc/aux_labels.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/aux_labels.h"

#include <numeric>
#include <vector>

#include "glog/logging.h"
#include "k2/csrc/fsa.h"

namespace k2 {

void MapAuxLabels1(const AuxLabels &labels_in,
                   const std::vector<int32_t> &arc_map, AuxLabels *labels_out) {
  CHECK_NOTNULL(labels_out);
  auto &start_pos = labels_out->start_pos;
  auto &labels = labels_out->labels;
  start_pos.clear();
  labels.clear();

  int32_t num_labels = 0;
  for (const auto &arc_index : arc_map) {
    start_pos.push_back(num_labels);
    int32_t pos_start = labels_in.start_pos[arc_index];
    int32_t pos_end = labels_in.start_pos[arc_index + 1];
    for (int32_t pos = pos_start; pos != pos_end; ++pos) {
      int32_t label = labels_in.labels[pos];
      DCHECK_NE(label, kEpsilon);
      labels.push_back(label);
      ++num_labels;
    }
  }
  start_pos.push_back(num_labels);
}

void MapAuxLabels2(const AuxLabels &labels_in,
                   const std::vector<std::vector<int32_t>> &arc_map,
                   AuxLabels *labels_out) {
  CHECK_NOTNULL(labels_out);
  auto &start_pos = labels_out->start_pos;
  auto &labels = labels_out->labels;
  start_pos.clear();
  labels.clear();

  int32_t num_labels = 0;
  for (const auto &arc_indexes : arc_map) {
    start_pos.push_back(num_labels);
    for (const auto &arc_index : arc_indexes) {
      int32_t pos_start = labels_in.start_pos[arc_index];
      int32_t pos_end = labels_in.start_pos[arc_index + 1];
      for (int32_t pos = pos_start; pos != pos_end; ++pos) {
        int32_t label = labels_in.labels[pos];
        DCHECK_NE(label, kEpsilon);
        labels.push_back(label);
        ++num_labels;
      }
    }
  }
  start_pos.push_back(num_labels);
}

}  // namespace k2
