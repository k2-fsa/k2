// k2/csrc/intersect.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/intersect.h"

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/properties.h"
#include "k2/csrc/util.h"

namespace {

using StatePair = std::pair<int32_t, int32_t>;

static inline int32_t InsertIntersectionState(
    const StatePair &new_state, int32_t *state_index_c,
    std::queue<StatePair> *qstates,
    std::unordered_map<StatePair, int32_t, k2::PairHash> *state_pair_map) {
  auto result = state_pair_map->insert({new_state, *state_index_c + 1});
  if (result.second) {
    // we have not visited `new_state` before.
    qstates->push(new_state);
    ++(*state_index_c);
  }
  return result.first->second;
}
}  // namespace

namespace k2 {

void Intersection::GetSizes(Array2Size<int32_t> *fsa_size) {
  CHECK_NOTNULL(fsa_size);
  fsa_size->size1 = fsa_size->size2 = 0;
  status_ = true;
  arc_indexes_.clear();
  arcs_.clear();
  arc_map_a_.clear();
  arc_map_b_.clear();

  if (IsEmpty(a_) || IsEmpty(b_)) return;
  // either `a` or `b` must be epsilon-free, both of them should be arc-sorted
  status_ = IsArcSorted(a_) && IsArcSorted(b_) &&
            (IsEpsilonFree(a_) || IsEpsilonFree(b_));
  if (!status_) return;

  int32_t final_state_a = a_.FinalState();
  int32_t final_state_b = b_.FinalState();
  const auto arc_a_begin = a_.data;
  const auto arc_b_begin = b_.data;

  const int32_t final_state_c = -1;  // just as a placeholder
  // no corresponding arc mapping from `c` to `a` or `c` to `b`
  const int32_t arc_map_none = -1;

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

    // As both `a` and `b` are arc-sorted, we first process epsilon arcs.
    // Noted that at most one for-loop below will really run as either `a` or
    // `b` is epsilon-free.
    for (; a_arc_iter_begin != a_arc_iter_end; ++a_arc_iter_begin) {
      if (kEpsilon != a_arc_iter_begin->label) break;

      StatePair new_state{a_arc_iter_begin->dest_state, state_b};
      int32_t new_state_index = InsertIntersectionState(
          new_state, &state_index_c, &qstates, &state_pair_map);
      arcs_.emplace_back(curr_state_index, new_state_index, kEpsilon);
      arc_map_a_.push_back(
          static_cast<int32_t>(a_arc_iter_begin - arc_a_begin));
      arc_map_b_.push_back(arc_map_none);
    }
    for (; b_arc_iter_begin != b_arc_iter_end; ++b_arc_iter_begin) {
      if (kEpsilon != b_arc_iter_begin->label) break;
      StatePair new_state{state_a, b_arc_iter_begin->dest_state};
      int32_t new_state_index = InsertIntersectionState(
          new_state, &state_index_c, &qstates, &state_pair_map);
      arcs_.emplace_back(curr_state_index, new_state_index, kEpsilon);
      arc_map_a_.push_back(arc_map_none);
      arc_map_b_.push_back(
          static_cast<int32_t>(b_arc_iter_begin - arc_b_begin));
    }

    // as both `a` and `b` are arc-sorted, we will iterate over the state with
    // less number of arcs.
    bool swapped = false;
    if ((a_arc_iter_end - a_arc_iter_begin) >
        (b_arc_iter_end - b_arc_iter_begin)) {
      std::swap(a_arc_iter_begin, b_arc_iter_begin);
      std::swap(a_arc_iter_end, b_arc_iter_end);
      swapped = true;
    }

    for (; a_arc_iter_begin != a_arc_iter_end; ++a_arc_iter_begin) {
      auto b_arc_range =
          std::equal_range(b_arc_iter_begin, b_arc_iter_end, *a_arc_iter_begin,
                           [](const Arc &left, const Arc &right) {
                             return left.label < right.label;
                           });
      for (auto it_b = b_arc_range.first; it_b != b_arc_range.second; ++it_b) {
        Arc curr_a_arc = *a_arc_iter_begin;  // copy here as we may swap later
        Arc curr_b_arc = *it_b;
        if (swapped) std::swap(curr_a_arc, curr_b_arc);
        StatePair new_state{curr_a_arc.dest_state, curr_b_arc.dest_state};
        int32_t new_state_index = InsertIntersectionState(
            new_state, &state_index_c, &qstates, &state_pair_map);
        arcs_.emplace_back(curr_state_index, new_state_index, curr_a_arc.label);

        auto curr_arc_index_a = static_cast<int32_t>(
            a_arc_iter_begin - (swapped ? arc_b_begin : arc_a_begin));
        auto curr_arc_index_b =
            static_cast<int32_t>(it_b - (swapped ? arc_a_begin : arc_b_begin));
        if (swapped) std::swap(curr_arc_index_a, curr_arc_index_b);
        arc_map_a_.push_back(curr_arc_index_a);
        arc_map_b_.push_back(curr_arc_index_b);
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

  CHECK_EQ(state_index_c + 2, arc_indexes_.size());
  fsa_size->size1 = state_index_c + 1;
  fsa_size->size2 = arcs_.size();
}

bool Intersection::GetOutput(Fsa *c, int32_t *arc_map_a /*= nullptr*/,
                             int32_t *arc_map_b /*= nullptr*/) {
  if (IsEmpty(a_) || IsEmpty(b_)) return true;
  if (!status_) return false;

  // output fsa
  CHECK_NOTNULL(c);
  CHECK_EQ(arc_indexes_.size(), c->size1 + 1);
  std::copy(arc_indexes_.begin(), arc_indexes_.end(), c->indexes);
  CHECK_EQ(arcs_.size(), c->size2);
  std::copy(arcs_.begin(), arcs_.end(), c->data);

  // output arc map
  if (arc_map_a != nullptr)
    std::copy(arc_map_a_.begin(), arc_map_a_.end(), arc_map_a);
  if (arc_map_b != nullptr)
    std::copy(arc_map_b_.begin(), arc_map_b_.end(), arc_map_b);
  return true;
}

}  // namespace k2
