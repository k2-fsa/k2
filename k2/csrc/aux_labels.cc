// k2/csrc/aux_labels.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/aux_labels.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_util.h"
#include "k2/csrc/properties.h"

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
static void CountExtraStates(const k2::Fsa &fsa_in,
                             const k2::AuxLabels &labels_in,
                             std::vector<int32_t> *num_extra_states) {
  CHECK_EQ(num_extra_states->size(), fsa_in.NumStates());
  auto &states = *num_extra_states;
  for (int32_t i = 0; i != fsa_in.arcs.size(); ++i) {
    const auto &arc = fsa_in.arcs[i];
    int32_t pos_start = labels_in.start_pos[i];
    int32_t pos_end = labels_in.start_pos[i + 1];
    states[arc.dest_state] += std::max(0, pos_end - pos_start - 1);
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
                               state_map[i] = state_map[i-1]
                                            + num_extra_states[i]
                                            + 1, for any i >=1
  @param [out] state_ids       At exit, it will be
                               state_ids[0] = 0,
                               state_ids[i] = state_map[i-1], for any i >= 1.
*/
static void MapStates(const std::vector<int32_t> &num_extra_states,
                      std::vector<int32_t> *state_map,
                      std::vector<int32_t> *state_ids) {
  CHECK_EQ(state_map->size(), num_extra_states.size());
  CHECK_EQ(state_ids->size(), num_extra_states.size());
  auto &s_map = *state_map;
  auto &s_ids = *state_ids;
  // we suppose there's no arcs entering the start state (i.e. state id of the
  // start state in output FSA will be 0), otherwise we may need to create a new
  // state as the real start state.
  CHECK_EQ(num_extra_states[0], 0);
  auto num_states_in = num_extra_states.size();
  // process from the second state
  s_map[0] = 0;
  s_ids[0] = 0;
  int32_t num_states_out = 0;
  for (auto i = 1; i != num_states_in; ++i) {
    s_ids[i] = num_states_out;
    // `+1` as we did not count state `i` itself in `num_extra_states`
    num_states_out += num_extra_states[i] + 1;
    s_map[i] = num_states_out;
  }
}
}  // namespace

namespace k2 {

void Swap(AuxLabels *labels1, AuxLabels *labels2) {
  CHECK_NOTNULL(labels1);
  CHECK_NOTNULL(labels2);
  std::swap(labels1->start_pos, labels2->start_pos);
  std::swap(labels1->labels, labels2->labels);
}

void MapAuxLabels1(const AuxLabels &labels_in,
                   const std::vector<int32_t> &arc_map, AuxLabels *labels_out) {
  CHECK_NOTNULL(labels_out);
  auto &start_pos = labels_out->start_pos;
  auto &labels = labels_out->labels;
  start_pos.clear();
  start_pos.reserve(arc_map.size() + 1);
  labels.clear();

  int32_t num_labels = 0;
  auto labels_in_iter_begin = labels_in.labels.begin();
  for (const auto &arc_index : arc_map) {
    start_pos.push_back(num_labels);
    int32_t pos_start = labels_in.start_pos[arc_index];
    int32_t pos_end = labels_in.start_pos[arc_index + 1];
    labels.insert(labels.end(), labels_in_iter_begin + pos_start,
                  labels_in_iter_begin + pos_end);
    num_labels += pos_end - pos_start;
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
  start_pos.reserve(arc_map.size() + 1);
  labels.clear();

  int32_t num_labels = 0;
  auto labels_in_iter_begin = labels_in.labels.begin();
  for (const auto &arc_indexes : arc_map) {
    start_pos.push_back(num_labels);
    for (const auto &arc_index : arc_indexes) {
      int32_t pos_start = labels_in.start_pos[arc_index];
      int32_t pos_end = labels_in.start_pos[arc_index + 1];
      labels.insert(labels.end(), labels_in_iter_begin + pos_start,
                    labels_in_iter_begin + pos_end);
      num_labels += pos_end - pos_start;
    }
  }
  start_pos.push_back(num_labels);
}

void InvertFst(const Fsa &fsa_in, const AuxLabels &labels_in, Fsa *fsa_out,
               AuxLabels *aux_labels_out) {
  CHECK_NOTNULL(fsa_out);
  CHECK_NOTNULL(aux_labels_out);
  fsa_out->arc_indexes.clear();
  fsa_out->arcs.clear();
  aux_labels_out->start_pos.clear();
  aux_labels_out->labels.clear();

  if (IsEmpty(fsa_in)) {
    aux_labels_out->start_pos.push_back(0);
    return;
  }

  auto num_states_in = fsa_in.NumStates();
  // get the number of extra states we need to create for each state
  // in fsa_in when inverting
  std::vector<int32_t> num_extra_states(num_states_in, 0);
  CountExtraStates(fsa_in, labels_in, &num_extra_states);

  // map state in fsa_in to state in fsa_out
  std::vector<int32_t> state_map(num_states_in, 0);
  std::vector<int32_t> state_ids(num_states_in, 0);
  MapStates(num_extra_states, &state_map, &state_ids);

  // a maximal approximation
  int32_t num_arcs_out = labels_in.labels.size() + fsa_in.arcs.size();
  std::vector<Arc> arcs;
  arcs.reserve(num_arcs_out);
  // `+1` for the end position of the last arc's olabel sequence
  std::vector<int32_t> start_pos;
  start_pos.reserve(num_arcs_out + 1);
  std::vector<int32_t> labels;
  labels.reserve(fsa_in.arcs.size());
  int32_t final_state_in = fsa_in.FinalState();

  int32_t num_non_eps_ilabel_processed = 0;
  start_pos.push_back(0);
  for (auto i = 0; i != fsa_in.arcs.size(); ++i) {
    const auto &arc = fsa_in.arcs[i];
    int32_t pos_start = labels_in.start_pos[i];
    int32_t pos_end = labels_in.start_pos[i + 1];
    int32_t src_state = arc.src_state;
    int32_t dest_state = arc.dest_state;
    if (dest_state == final_state_in) {
      // every arc entering the final state must have exactly
      // one olabel == kFinalSymbol
      CHECK_EQ(pos_start + 1, pos_end);
      CHECK_EQ(labels_in.labels[pos_start], kFinalSymbol);
    }
    if (pos_end - pos_start <= 1) {
      int32_t curr_label =
          (pos_end - pos_start == 0) ? kEpsilon : labels_in.labels[pos_start];
      arcs.emplace_back(state_map[src_state], state_map[dest_state],
                        curr_label);
    } else {
      // expand arcs with olabels
      arcs.emplace_back(state_map[src_state], state_ids[dest_state] + 1,
                        labels_in.labels[pos_start]);
      start_pos.push_back(num_non_eps_ilabel_processed);
      for (int32_t pos = pos_start + 1; pos < pos_end - 1; ++pos) {
        ++state_ids[dest_state];
        arcs.emplace_back(state_ids[dest_state], state_ids[dest_state] + 1,
                          labels_in.labels[pos]);
        start_pos.push_back(num_non_eps_ilabel_processed);
      }
      ++state_ids[dest_state];
      arcs.emplace_back(state_ids[dest_state], state_map[arc.dest_state],
                        labels_in.labels[pos_end - 1]);
    }
    // push non-epsilon ilabel in fsa_in as olabel of fsa_out
    if (arc.label != kEpsilon) {
      labels.push_back(arc.label);
      ++num_non_eps_ilabel_processed;
    }
    start_pos.push_back(num_non_eps_ilabel_processed);
  }

  labels.resize(labels.size());
  arcs.resize(arcs.size());
  start_pos.resize(start_pos.size());

  std::vector<int32_t> arc_map;
  ReorderArcs(arcs, fsa_out, &arc_map);
  AuxLabels labels_tmp;
  labels_tmp.start_pos = std::move(start_pos);
  labels_tmp.labels = std::move(labels);
  MapAuxLabels1(labels_tmp, arc_map, aux_labels_out);
}
}  // namespace k2
