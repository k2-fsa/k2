/**
 * @brief
 * aux_labels
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/aux_labels.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host/properties.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace {

/*
  This function counts how many extra states we need to create for each state in
  the input FSA when we invert an FSA. Generally, if an entering arc of state
  `i` in the input FSA has n olabels, then we need to create n-1 extra states
  for state `i`.

  @param [in] fsa_in      Input FSA
  @param [in] labels_in   Aux-label sequences for the input FSA
  @param [out] num_extra_states For state `i` in `fsa_in`, we need to create
                                extra `num_extra_states[i]` states in the output
                                inverted FSA.
*/
static void CountExtraStates(const k2host::Fsa &fsa_in,
                             const k2host::AuxLabels &labels_in,
                             std::vector<int32_t> *num_extra_states) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(num_extra_states->size(), fsa_in.NumStates());
  auto &states = *num_extra_states;
  const auto arcs = fsa_in.data + fsa_in.indexes[0];
  for (int32_t i = 0; i != fsa_in.size2; ++i) {
    const auto &arc = arcs[i];
    int32_t begin = labels_in.indexes[i];
    int32_t end = labels_in.indexes[i + 1];
    states[arc.dest_state] += std::max(0, end - begin - 1);
  }
}

/*
  Map the state in the input FSA to state in the output inverted FSA.

  @param [in] num_extra_states Output of function `CountExtraStates`
                               which gives how many extra states we need
                               to create for each state in the input FSA.
  @param [out] state_map       Map state `i` in the input FSA to state
                               `state_map[i]` in the output FSA.
                               At exit, it will be
                               state_map[0] = 0,
                               state_map[i] = num_extra_states[0]
                                            + state_map[i-1]
                                            + num_extra_states[i]
                                            + 1, for any i >=1
  @param [out] state_ids       At exit, it will be
                               state_ids[0] = 0,
                               state_ids[1] = num_extra_states[0],
                               state_ids[i] = state_map[i-1], for any i > 1.
*/
static void MapStates(const std::vector<int32_t> &num_extra_states,
                      std::vector<int32_t> *state_map,
                      std::vector<int32_t> *state_ids) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(state_map->size(), num_extra_states.size());
  K2_CHECK_EQ(state_ids->size(), num_extra_states.size());
  auto &s_map = *state_map;
  auto &s_ids = *state_ids;

  auto num_states_in = num_extra_states.size();
  // process from the second state
  s_map[0] = 0;
  s_ids[0] = 0;
  int32_t num_states_out = num_extra_states[0];
  for (auto i = 1; i != num_states_in; ++i) {
    s_ids[i] = num_states_out;
    // `+1` as we did not count state `i` itself in `num_extra_states`
    num_states_out += num_extra_states[i] + 1;
    s_map[i] = num_states_out;
  }
}
}  // namespace

namespace k2host {

void AuxLabels1Mapper::GetSizes(Array2Size<int32_t> *aux_size) const {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(aux_size, nullptr);
  aux_size->size1 = arc_map_.size;
  int32_t num_labels = 0;
  for (auto i = arc_map_.begin; i != arc_map_.end; ++i) {
    const auto arc_index = arc_map_.data[i];
    int32_t begin = labels_in_.indexes[arc_index];
    int32_t end = labels_in_.indexes[arc_index + 1];
    num_labels += end - begin;
  }
  aux_size->size2 = num_labels;
}

void AuxLabels1Mapper::GetOutput(AuxLabels *labels_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(labels_out, nullptr);
  auto &start_pos = labels_out->indexes;
  auto &labels = labels_out->data;
  int32_t num_labels = 0;
  int32_t i = 0;
  for (; i != arc_map_.size; ++i) {
    start_pos[i] = num_labels;
    const auto arc_index = arc_map_.data[i + arc_map_.begin];
    int32_t begin = labels_in_.indexes[arc_index];
    int32_t end = labels_in_.indexes[arc_index + 1];
    for (auto it = begin; it != end; ++it) {
      labels[num_labels++] = labels_in_.data[it];
    }
  }
  start_pos[i] = num_labels;
}

void AuxLabels2Mapper::GetSizes(Array2Size<int32_t> *aux_size) const {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(aux_size, nullptr);
  aux_size->size1 = arc_map_.size1;
  int32_t num_labels = 0;
  for (const auto &arc_index : arc_map_) {
    int32_t begin = labels_in_.indexes[arc_index];
    int32_t end = labels_in_.indexes[arc_index + 1];
    num_labels += end - begin;
  }
  aux_size->size2 = num_labels;
}

void AuxLabels2Mapper::GetOutput(AuxLabels *labels_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(labels_out, nullptr);
  auto &start_pos = labels_out->indexes;
  auto &labels = labels_out->data;
  int32_t num_labels = 0;
  int32_t i = 0;
  for (; i != arc_map_.size1; ++i) {
    start_pos[i] = num_labels;
    for (auto j = arc_map_.indexes[i]; j != arc_map_.indexes[i + 1]; ++j) {
      const auto arc_index = arc_map_.data[j];
      int32_t begin = labels_in_.indexes[arc_index];
      int32_t end = labels_in_.indexes[arc_index + 1];
      for (auto it = begin; it != end; ++it) {
        labels[num_labels++] = labels_in_.data[it];
      }
    }
  }
  start_pos[i] = num_labels;
}

void FstInverter::GetSizes(Array2Size<int32_t> *fsa_size,
                           Array2Size<int32_t> *aux_size) const {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_size, nullptr);
  K2_CHECK_NE(aux_size, nullptr);
  int32_t num_extra_states = 0;
  int32_t num_arcs = 0;
  int32_t num_non_eps_labels = 0;
  for (int32_t i = 0; i != fsa_in_.size2; ++i) {
    const auto &arc = fsa_in_.data[i];
    int32_t begin = labels_in_.indexes[i];
    int32_t end = labels_in_.indexes[i + 1];
    num_extra_states += std::max(0, end - begin - 1);
    num_arcs += std::max(1, end - begin);
    if (arc.label != kEpsilon) ++num_non_eps_labels;
  }
  fsa_size->size1 = num_extra_states + fsa_in_.NumStates();
  fsa_size->size2 = num_arcs;
  aux_size->size1 = num_arcs;
  aux_size->size2 = num_non_eps_labels;
}

void FstInverter::GetOutput(Fsa *fsa_out, AuxLabels *labels_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_out, nullptr);
  K2_CHECK_NE(labels_out, nullptr);

  if (IsEmpty(fsa_in_)) return;

  auto num_states_in = fsa_in_.NumStates();
  // get the number of extra states we need to create for each state
  // in fsa_in when inverting
  std::vector<int32_t> num_extra_states(num_states_in, 0);
  CountExtraStates(fsa_in_, labels_in_, &num_extra_states);

  // map state in fsa_in to state in fsa_out
  std::vector<int32_t> state_map(num_states_in, 0);
  std::vector<int32_t> state_ids(num_states_in, 0);
  MapStates(num_extra_states, &state_map, &state_ids);

  std::vector<Arc> arcs;
  arcs.reserve(fsa_out->size2);
  std::vector<int32_t> start_pos;
  start_pos.reserve(labels_out->size1 + 1);
  std::vector<int32_t> labels;
  labels.reserve(labels_out->size2);
  int32_t final_state_in = fsa_in_.FinalState();

  int32_t num_non_eps_ilabel_processed = 0;
  const auto arcs_in = fsa_in_.data + fsa_in_.indexes[0];
  start_pos.push_back(0);
  for (auto i = 0; i != fsa_in_.size2; ++i) {
    const auto &arc = arcs_in[i];
    int32_t pos_begin = labels_in_.indexes[i];
    int32_t pos_end = labels_in_.indexes[i + 1];
    int32_t src_state = arc.src_state;
    int32_t dest_state = arc.dest_state;
    if (dest_state == final_state_in) {
      // every arc entering the final state must have exactly
      // one olabel == kFinalSymbol (the last one)
      K2_CHECK_LT(pos_begin, pos_end);
      K2_CHECK_EQ(labels_in_.data[pos_end - 1], kFinalSymbol);
    }
    if (pos_end - pos_begin <= 1) {
      int32_t curr_label =
          (pos_end - pos_begin == 0) ? kEpsilon : labels_in_.data[pos_begin];
      if (dest_state != final_state_in) K2_CHECK_NE(curr_label, kFinalSymbol);
      arcs.emplace_back(state_map[src_state], state_map[dest_state], curr_label,
                        arc.weight);
    } else {
      // expand arcs with olabels
      K2_CHECK_NE(labels_in_.data[pos_begin], kFinalSymbol);
      arcs.emplace_back(state_map[src_state], state_ids[dest_state] + 1,
                        labels_in_.data[pos_begin], arc.weight);
      start_pos.push_back(num_non_eps_ilabel_processed);
      for (int32_t pos = pos_begin + 1; pos < pos_end - 1; ++pos) {
        ++state_ids[dest_state];
        K2_CHECK_NE(labels_in_.data[pos], kFinalSymbol);
        arcs.emplace_back(state_ids[dest_state], state_ids[dest_state] + 1,
                          labels_in_.data[pos], 0.0);
        start_pos.push_back(num_non_eps_ilabel_processed);
      }
      ++state_ids[dest_state];
      if (dest_state != final_state_in)
        K2_CHECK_NE(labels_in_.data[pos_end - 1], kFinalSymbol);
      arcs.emplace_back(state_ids[dest_state], state_map[arc.dest_state],
                        labels_in_.data[pos_end - 1], 0.0);
    }
    // push non-epsilon ilabel in fsa_in as olabel of fsa_out
    if (arc.label != kEpsilon) {
      labels.push_back(arc.label);
      ++num_non_eps_ilabel_processed;
    }
    start_pos.push_back(num_non_eps_ilabel_processed);
  }

  // any failure indicates there are some errors
  K2_CHECK_EQ(arcs.size(), fsa_out->size2);
  K2_CHECK_EQ(start_pos.size(), labels_out->size1 + 1);
  K2_CHECK_EQ(labels.size(), labels_out->size2);

  std::vector<int32_t> arc_map;
  ReorderArcs(arcs, fsa_out, &arc_map);
  AuxLabels labels_tmp(labels_out->size1, labels_out->size2, start_pos.data(),
                       labels.data());
  Array1<int32_t *> arc_map_array(0, arc_map.size(), arc_map.data());
  AuxLabels1Mapper aux_mapper(labels_tmp, arc_map_array);
  // don't need to call `GetSizes` here as `labels_out` has been initialized
  aux_mapper.GetOutput(labels_out);
}
}  // namespace k2host
