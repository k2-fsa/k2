// k2/csrc/fsa_algo.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_algo.h"

#include <algorithm>
#include <numeric>
#include <stack>
#include <queue>
#include <unordered_map>
#include <utility>

#include "glog/logging.h"
#include "k2/csrc/properties.h"
#include "k2/csrc/util.h"

namespace {

// depth first search state
struct DfsState {
  int32_t state;      // state number of the visiting node
  int32_t arc_begin;  // arc index of the visiting arc
  int32_t arc_end;    // end of the arc index of the visiting node
};

inline int32_t InsertIntersectionState(
    const k2::StatePair &new_state, int32_t *state_index_c,
    std::queue<k2::StatePair> *qstates,
    std::unordered_map<k2::StatePair, int32_t, k2::PairHash> *state_pair_map) {
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

// The implementation of this function is inspired by
// http://www.openfst.org/doxygen/fst/html/connect_8h_source.html
void ConnectCore(const Fsa &fsa, std::vector<int32_t> *state_map) {
  CHECK_NOTNULL(state_map);

  state_map->clear();
  if (IsEmpty(fsa)) return;

  auto num_states = fsa.NumStates();
  auto final_state = num_states - 1;

  std::vector<bool> accessible(num_states, false);
  std::vector<bool> coaccessible(num_states, false);
  std::vector<bool> visited(num_states, false);

  accessible.front() = true;
  coaccessible.back() = true;

  std::stack<DfsState> stack;
  stack.push({0, fsa.arc_indexes[0], fsa.arc_indexes[1]});
  visited[0] = true;

  while (!stack.empty()) {
    auto &current_state = stack.top();

    if (current_state.arc_begin == current_state.arc_end) {
      // we have finished visiting this state
      auto state = current_state.state;  // get a copy since we will destroy it
      stack.pop();
      if (!stack.empty()) {
        // if it has a parent, set the parent's co-accessible flag
        if (coaccessible[state]) {
          auto &parent = stack.top();
          coaccessible[parent.state] = true;
          ++parent.arc_begin;  // process the next child
        }
      }
      continue;
    }

    const auto &arc = fsa.arcs[current_state.arc_begin];
    auto next_state = arc.dest_state;
    bool is_visited = visited[next_state];
    if (!is_visited) {
      // this is a new discovered state
      visited[next_state] = true;
      auto arc_begin = fsa.arc_indexes[next_state];
      if (next_state != final_state)
        stack.push({next_state, arc_begin, fsa.arc_indexes[next_state + 1]});
      else
        stack.push({next_state, arc_begin, arc_begin});

      if (accessible[current_state.state]) accessible[next_state] = true;
    } else {
      // this is a back arc or forward cross arc;
      // update the co-accessible flag
      auto next_state = arc.dest_state;
      if (coaccessible[next_state]) coaccessible[current_state.state] = true;
      ++current_state.arc_begin;  // go to the next arc
    }
  }

  state_map->reserve(num_states);

  for (auto i = 0; i != num_states; ++i) {
    if (accessible[i] && coaccessible[i]) state_map->push_back(i);
  }
}

void Connect(const Fsa &a, Fsa *b, std::vector<int32_t> *arc_map /*=nullptr*/) {
  CHECK_NOTNULL(b);
  if (arc_map != nullptr) arc_map->clear();

  std::vector<int32_t> state_b_to_a;
  ConnectCore(a, &state_b_to_a);
  if (state_b_to_a.empty()) return;

  b->arc_indexes.resize(state_b_to_a.size());
  b->arcs.clear();
  b->arcs.reserve(a.arcs.size());

  if (arc_map != nullptr) {
    arc_map->clear();
    arc_map->reserve(a.arcs.size());
  }

  std::vector<int32_t> state_a_to_b(a.NumStates(), -1);

  auto num_states_b = b->NumStates();
  for (auto i = 0; i != num_states_b; ++i) {
    auto state_a = state_b_to_a[i];
    state_a_to_b[state_a] = i;
  }

  auto arc_begin = 0;
  auto arc_end = 0;
  auto final_state_a = a.NumStates() - 1;

  for (auto i = 0; i != num_states_b; ++i) {
    auto state_a = state_b_to_a[i];
    arc_begin = a.arc_indexes[state_a];
    if (state_a != final_state_a)
      arc_end = a.arc_indexes[state_a + 1];
    else
      arc_end = arc_begin;

    b->arc_indexes[i] = static_cast<int32_t>(b->arcs.size());
    for (; arc_begin != arc_end; ++arc_begin) {
      auto arc = a.arcs[arc_begin];
      auto dest_state = arc.dest_state;
      auto state_b = state_a_to_b[dest_state];

      if (state_b < 0) continue;  // dest_state is unreachable

      arc.src_state = i;
      arc.dest_state = state_b;
      b->arcs.push_back(arc);
      if (arc_map != nullptr) arc_map->push_back(arc_begin);
    }
  }
}

bool Intersect(const Fsa &a, const Fsa &b, Fsa *c,
               std::vector<int32_t> *arc_map_a /*= nullptr*/,
               std::vector<int32_t> *arc_map_b /*= nullptr*/) {
  CHECK_NOTNULL(c);
  c->arc_indexes.clear();
  c->arcs.clear();
  if (arc_map_a != nullptr) arc_map_a->clear();
  if (arc_map_b != nullptr) arc_map_b->clear();

  if (IsEmpty(a) || IsEmpty(b)) return true;
  if (!IsArcSorted(a) || !IsArcSorted(b)) return false;
  // either `a` or `b` must be epsilon-free
  if (!IsEpsilonFree(a) && !IsEpsilonFree(b)) return false;

  int32_t final_state_a = a.NumStates() - 1;
  int32_t final_state_b = b.NumStates() - 1;
  const auto arc_a_begin = a.arcs.begin();
  const auto arc_a_end = a.arcs.end();
  const auto arc_b_begin = b.arcs.begin();
  const auto arc_b_end = b.arcs.end();
  using ArcIterator = std::vector<Arc>::const_iterator;

  const int32_t final_state_c = -1;  // just as a placeholder
  // no corresponding arc mapping from `c` to `a` or `c` to `b`
  const int32_t arc_map_none = -1;
  auto &arc_indexes_c = c->arc_indexes;
  auto &arcs_c = c->arcs;

  // map state pair to unique id
  std::unordered_map<StatePair, int32_t, PairHash> state_pair_map;
  std::queue<StatePair> qstates;
  qstates.push({0, 0});
  state_pair_map.insert({{0, 0}, 0});
  state_pair_map.insert({{final_state_a, final_state_b}, final_state_c});
  int32_t state_index_c = 0;
  while (!qstates.empty()) {
    arc_indexes_c.push_back(static_cast<int32_t>(arcs_c.size()));

    auto curr_state_pair = qstates.front();
    qstates.pop();
    // as we have inserted `curr_state_pair` before.
    int32_t curr_state_index = state_pair_map[curr_state_pair];

    auto state_a = curr_state_pair.first;
    ArcIterator a_arc_iter_begin = arc_a_begin + a.arc_indexes[state_a];
    ArcIterator a_arc_iter_end =
        (state_a != final_state_a) ? (arc_a_begin + a.arc_indexes[state_a + 1])
                                   : arc_a_end;

    auto state_b = curr_state_pair.second;
    ArcIterator b_arc_iter_begin = arc_b_begin + b.arc_indexes[state_b];
    ArcIterator b_arc_iter_end =
        (state_b != final_state_b) ? (arc_b_begin + b.arc_indexes[state_b + 1])
                                   : arc_b_end;

    // As both `a` and `b` are arc-sorted, we first process epsilon arcs.
    // Noted that at most one for-loop below will really run as either `a` or
    // `b` is epsilon-free.
    for (; a_arc_iter_begin != a_arc_iter_end; ++a_arc_iter_begin) {
      if (kEpsilon != a_arc_iter_begin->label) break;

      StatePair new_state{a_arc_iter_begin->dest_state, state_b};
      int32_t new_state_index = InsertIntersectionState(
          new_state, &state_index_c, &qstates, &state_pair_map);
      arcs_c.push_back({curr_state_index, new_state_index, kEpsilon});
      if (arc_map_a != nullptr)
        arc_map_a->push_back(
            static_cast<int32_t>(a_arc_iter_begin - arc_a_begin));
      if (arc_map_b != nullptr) arc_map_b->push_back(arc_map_none);
    }
    for (; b_arc_iter_begin != b_arc_iter_end; ++b_arc_iter_begin) {
      if (kEpsilon != b_arc_iter_begin->label) break;
      StatePair new_state{state_a, b_arc_iter_begin->dest_state};
      int32_t new_state_index = InsertIntersectionState(
          new_state, &state_index_c, &qstates, &state_pair_map);
      arcs_c.push_back({curr_state_index, new_state_index, kEpsilon});
      if (arc_map_a != nullptr) arc_map_a->push_back(arc_map_none);
      if (arc_map_b != nullptr)
        arc_map_b->push_back(
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
      Arc curr_a_arc = *a_arc_iter_begin;  // copy here as we may swap later
      auto b_arc_range =
          std::equal_range(b_arc_iter_begin, b_arc_iter_end, curr_a_arc,
                           [](const Arc &left, const Arc &right) {
            return left.label < right.label;
          });
      for (ArcIterator it_b = b_arc_range.first; it_b != b_arc_range.second;
           ++it_b) {
        Arc curr_b_arc = *it_b;
        if (swapped) std::swap(curr_a_arc, curr_b_arc);
        StatePair new_state{curr_a_arc.dest_state, curr_b_arc.dest_state};
        int32_t new_state_index = InsertIntersectionState(
            new_state, &state_index_c, &qstates, &state_pair_map);
        arcs_c.push_back({curr_state_index, new_state_index, curr_a_arc.label});

        int32_t curr_arc_index_a = static_cast<int32_t>(
            a_arc_iter_begin - (swapped ? arc_b_begin : arc_a_begin));
        int32_t curr_arc_index_b =
            static_cast<int32_t>(it_b - (swapped ? arc_a_begin : arc_b_begin));
        if (swapped) std::swap(curr_arc_index_a, curr_arc_index_b);
        if (arc_map_a != nullptr) arc_map_a->push_back(curr_arc_index_a);
        if (arc_map_b != nullptr) arc_map_b->push_back(curr_arc_index_b);
      }
    }
  }

  // push final state
  arc_indexes_c.push_back(static_cast<int32_t>(arcs_c.size()));
  ++state_index_c;
  // then replace `final_state_c` with the real index of final state of `c`
  for (auto &arc : arcs_c) {
    if (arc.dest_state == final_state_c) arc.dest_state = state_index_c;
  }
  return true;
}

void ArcSort(const Fsa &a, Fsa *b,
             std::vector<int32_t> *arc_map /*= nullptr*/) {
  CHECK_NOTNULL(b);
  b->arc_indexes = a.arc_indexes;
  b->arcs.clear();
  b->arcs.reserve(a.arcs.size());
  if (arc_map != nullptr) arc_map->clear();

  using ArcWithIndex = std::pair<Arc, int32_t>;
  std::vector<int32_t> indexes(a.arcs.size());  // index mapping
  std::iota(indexes.begin(), indexes.end(), 0);
  const auto arc_begin_iter = a.arcs.begin();
  const auto index_begin_iter = indexes.begin();
  // we will not process the final state as it has no arcs leaving it.
  StateId final_state = a.NumStates() - 1;
  for (StateId state = 0; state < final_state; ++state) {
    int32_t begin = a.arc_indexes[state];
    // as non-empty fsa `a` contains at least two states,
    // we can always access `state + 1` validly.
    int32_t end = a.arc_indexes[state + 1];
    std::vector<ArcWithIndex> arc_range_to_be_sorted;
    arc_range_to_be_sorted.reserve(end - begin);
    std::transform(arc_begin_iter + begin, arc_begin_iter + end,
                   index_begin_iter + begin,
                   std::back_inserter(arc_range_to_be_sorted),
                   [](const Arc & arc, int32_t index)
                       ->ArcWithIndex { return std::make_pair(arc, index); });
    std::sort(arc_range_to_be_sorted.begin(), arc_range_to_be_sorted.end(),
              [](const ArcWithIndex &left, const ArcWithIndex &right) {
      return left.first < right.first;  // sort on arc
    });
    // copy index mappings back to `indexes`
    std::transform(arc_range_to_be_sorted.begin(), arc_range_to_be_sorted.end(),
                   index_begin_iter + begin,
                   [](const ArcWithIndex &v) { return v.second; });
    // move-copy sorted arcs to `b`
    std::transform(arc_range_to_be_sorted.begin(), arc_range_to_be_sorted.end(),
                   std::back_inserter(b->arcs),
                   [](ArcWithIndex &v) { return std::move(v.first); });
  }
  if (arc_map != nullptr) arc_map->swap(indexes);
}

bool TopSort(const Fsa &a, Fsa *b,
             std::vector<int32_t> *state_map /*= nullptr*/) {
  CHECK_NOTNULL(b);
  b->arc_indexes.clear();
  b->arcs.clear();

  if (state_map != nullptr) state_map->clear();

  if (IsEmpty(a)) return true;
  if (!IsConnected(a)) return false;

  static constexpr int8_t kNotVisited = 0;  // a node that has not been visited
  static constexpr int8_t kVisiting = 1;    // a node that is under visiting
  static constexpr int8_t kVisited = 2;     // a node that has been visited

  auto num_states = a.NumStates();
  auto final_state = num_states - 1;
  std::vector<int8_t> state_status(num_states, kNotVisited);

  // map order to state.
  // state 0 has the largest order, i.e., num_states - 1
  // final_state has the least order, i.e., 0
  std::vector<int32_t> order;
  order.reserve(num_states);

  std::stack<DfsState> stack;
  stack.push({0, a.arc_indexes[0], a.arc_indexes[1]});
  state_status[0] = kVisiting;
  bool is_acyclic = true;
  while (is_acyclic && !stack.empty()) {
    auto &current_state = stack.top();
    if (current_state.arc_begin == current_state.arc_end) {
      // we have finished visiting this state
      state_status[current_state.state] = kVisited;
      order.push_back(current_state.state);
      stack.pop();
      continue;
    }
    const auto &arc = a.arcs[current_state.arc_begin];
    auto next_state = arc.dest_state;
    auto status = state_status[next_state];
    switch (status) {
      case kNotVisited: {
        // a new discovered node
        state_status[next_state] = kVisiting;
        auto arc_begin = a.arc_indexes[next_state];
        if (next_state != final_state)
          stack.push({next_state, arc_begin, a.arc_indexes[next_state + 1]});
        else
          stack.push({next_state, arc_begin, arc_begin});
        ++current_state.arc_begin;
        break;
      }
      case kVisiting:
        // this is a back arc indicating a loop in the graph
        is_acyclic = false;
        break;
      case kVisited:
        // this is a forward cross arc, do nothing.
        ++current_state.arc_begin;
        break;
      default:
        LOG(FATAL) << "Unreachable code is executed!";
        break;
    }
  }

  if (!is_acyclic) return false;

  std::vector<int32_t> state_a_to_b(num_states);
  for (auto i = 0; i != num_states; ++i) {
    state_a_to_b[order[num_states - 1 - i]] = i;
  }

  // start state maps to start state
  CHECK_EQ(state_a_to_b.front(), 0);
  // final state maps to final state
  CHECK_EQ(state_a_to_b.back(), final_state);

  b->arcs.reserve(a.arc_indexes.size());
  b->arc_indexes.resize(num_states);

  int32_t arc_begin;
  int32_t arc_end;
  for (auto state_b = 0; state_b != num_states; ++state_b) {
    auto state_a = order[num_states - 1 - state_b];
    arc_begin = a.arc_indexes[state_a];
    if (state_a != final_state)
      arc_end = a.arc_indexes[state_a + 1];
    else
      arc_end = arc_begin;

    b->arc_indexes[state_b] = static_cast<int32_t>(b->arcs.size());
    for (; arc_begin != arc_end; ++arc_begin) {
      auto arc = a.arcs[arc_begin];
      arc.src_state = state_b;
      arc.dest_state = state_a_to_b[arc.dest_state];
      b->arcs.push_back(arc);
    }
  }
  if (state_map != nullptr) {
    std::reverse(order.begin(), order.end());
    state_map->swap(order);
  }
  return true;
}

}  // namespace k2
