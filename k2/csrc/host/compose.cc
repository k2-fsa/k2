/**
 * @brief
 * compose
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/compose.h"

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/properties.h"
#include "k2/csrc/host/util.h"

namespace {

using StatePair = std::pair<int32_t, int32_t>;

static inline int32_t ComposeState(
    const StatePair &new_state, int32_t *state_index_c,
    std::queue<StatePair> *qstates,
    std::unordered_map<StatePair, int32_t, k2host::PairHash> *state_pair_map) {
  auto result = state_pair_map->insert({new_state, *state_index_c + 1});
  if (result.second) {
    // we have not visited `new_state` before.
    qstates->push(new_state);
    ++(*state_index_c);
  }
  return result.first->second;
}
}  // namespace

namespace k2host {

void Compose::GetSizes(Array2Size<int32_t> *fsa_size) {
  K2_CHECK_NE(fsa_size, nullptr);
  fsa_size->size1 = fsa_size->size2 = 0;
  status_ = true;
  arc_indexes_.clear();
  arcs_.clear();
  aux_labels_.clear();
  arc_map_a_.clear();
  arc_map_b_.clear();

  if (IsEmpty(a_) || IsEmpty(b_)) return;
  // `b` must be epsilon-free and be arc-sorted
  status_ = IsArcSorted(b_) && IsEpsilonFree(b_);
  if (!status_) return;

  int32_t final_state_a = a_.FinalState();
  int32_t final_state_b = b_.FinalState();
  const auto arc_a_begin = a_.data;
  const auto arc_b_begin = b_.data;

  const int32_t final_state_c = -1;  // just as a placeholder
  // no corresponding arc mapping from `c` to `a` or `c` to `b`
  const int32_t arc_map_none = -1;

  const Arc *a_arc_offset = a_.data + a_.indexes[0];
  const Arc *b_arc_offset = b_.data + b_.indexes[0];

  // map state pair to unique id
  std::unordered_map<StatePair, int32_t, PairHash> state_pair_map;
  std::queue<StatePair> qstates;
  qstates.push({0, 0});
  state_pair_map.insert({{0, 0}, 0});
  state_pair_map.insert({{final_state_a, final_state_b}, final_state_c});
  int32_t state_index_c = 0;
  while (!qstates.empty()) {
    arc_indexes_.push_back(static_cast<int32_t>(arcs_.size()));

    auto curr_state_pair = qstates.front();
    qstates.pop();
    // as we have inserted `curr_state_pair` before.
    int32_t curr_state_index = state_pair_map[curr_state_pair];

    auto state_a = curr_state_pair.first;
    auto a_arc_iter_begin = arc_a_begin + a_.indexes[state_a];
    auto a_arc_iter_end = arc_a_begin + a_.indexes[state_a + 1];
    auto state_b = curr_state_pair.second;
    auto b_arc_iter_begin = arc_b_begin + b_.indexes[state_b];
    auto b_arc_iter_end = arc_b_begin + b_.indexes[state_b + 1];

    auto saved_a_arc_iter_begin = a_arc_iter_begin;
    for (; a_arc_iter_begin != a_arc_iter_end; ++a_arc_iter_begin) {
      if (a_aux_labels_[a_arc_iter_begin - a_arc_offset] != kEpsilon) continue;

      StatePair new_state{a_arc_iter_begin->dest_state, state_b};
      int32_t new_state_index =
          ComposeState(new_state, &state_index_c, &qstates, &state_pair_map);
      arcs_.emplace_back(curr_state_index, new_state_index,
                         a_arc_iter_begin->label, a_arc_iter_begin->weight);
      arc_map_a_.push_back(
          static_cast<int32_t>(a_arc_iter_begin - arc_a_begin));
      arc_map_b_.push_back(arc_map_none);
      aux_labels_.push_back(kEpsilon);
    }
    a_arc_iter_begin = saved_a_arc_iter_begin;

    // b_ has no input symbols with kEpsilon
    //
    // The aux labels of arcs in `a_` that enter the final state
    // should be -1, not 0.

    for (; a_arc_iter_begin != a_arc_iter_end; ++a_arc_iter_begin) {
      Arc tmp_arc(0, 0, a_aux_labels_[a_arc_iter_begin - a_arc_offset], 0);
      auto b_arc_range =
          std::equal_range(b_arc_iter_begin, b_arc_iter_end, tmp_arc,
                           [](const Arc &left, const Arc &right) {
                             return left.label < right.label;
                           });
      for (auto it_b = b_arc_range.first; it_b != b_arc_range.second; ++it_b) {
        const Arc &curr_a_arc = *a_arc_iter_begin;
        const Arc &curr_b_arc = *it_b;
        StatePair new_state{curr_a_arc.dest_state, curr_b_arc.dest_state};
        int32_t new_state_index =
            ComposeState(new_state, &state_index_c, &qstates, &state_pair_map);
        arcs_.emplace_back(curr_state_index, new_state_index, curr_a_arc.label,
                           curr_a_arc.weight + curr_b_arc.weight);

        auto curr_arc_index_a =
            static_cast<int32_t>(a_arc_iter_begin - a_arc_offset);
        auto curr_arc_index_b = static_cast<int32_t>(it_b - b_arc_offset);
        arc_map_a_.push_back(curr_arc_index_a);
        arc_map_b_.push_back(curr_arc_index_b);
        aux_labels_.push_back(b_aux_labels_[curr_arc_index_b]);
      }
    }
  }

  // push final state
  arc_indexes_.push_back(static_cast<int32_t>(arcs_.size()));
  ++state_index_c;
  // then replace `final_state_c` with the real index of final state of `c`
  for (auto &arc : arcs_) {
    if (arc.dest_state == final_state_c) arc.dest_state = state_index_c;
  }
  // push a duplicate of final state
  arc_indexes_.emplace_back(arc_indexes_.back());

  K2_CHECK_EQ(state_index_c + 2, arc_indexes_.size());
  fsa_size->size1 = state_index_c + 1;
  fsa_size->size2 = arcs_.size();
}

bool Compose::GetOutput(Fsa *c, std::vector<int32_t> *c_aux_labels,
                        int32_t *arc_map_a /*= nullptr*/,
                        int32_t *arc_map_b /*= nullptr*/) {
  if (IsEmpty(a_) || IsEmpty(b_)) return true;
  if (!status_) return false;

  // output fsa
  K2_CHECK_NE(c, nullptr);
  K2_CHECK_EQ(arc_indexes_.size(), c->size1 + 1);
  std::copy(arc_indexes_.begin(), arc_indexes_.end(), c->indexes);
  K2_CHECK_EQ(arcs_.size(), c->size2);
  std::copy(arcs_.begin(), arcs_.end(), c->data);

  *c_aux_labels = std::move(aux_labels_);

  // output arc map
  if (arc_map_a != nullptr)
    std::copy(arc_map_a_.begin(), arc_map_a_.end(), arc_map_a);
  if (arc_map_b != nullptr)
    std::copy(arc_map_b_.begin(), arc_map_b_.end(), arc_map_b);
  return true;
}

}  // namespace k2host
