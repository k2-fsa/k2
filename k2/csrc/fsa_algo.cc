// k2/csrc/fsa_algo.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_algo.h"

#include <algorithm>
#include <numeric>
#include <stack>
#include <utility>

#include "glog/logging.h"

namespace {

// depth first search state
struct DfsState {
  int32_t state;      // state number of the visiting node
  int32_t arc_begin;  // arc index of the visiting arc
  int32_t arc_end;    // end of the arc index of the visiting node
};

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
        // if it has a parent, set the parent's coaccessible flag
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
  if (arc_map) arc_map->clear();

  std::vector<int32_t> state_b_to_a;
  ConnectCore(a, &state_b_to_a);
  if (state_b_to_a.empty()) return;

  b->arc_indexes.resize(state_b_to_a.size());
  b->arcs.clear();
  b->arcs.reserve(a.arcs.size());

  if (arc_map) {
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
      if (arc_map) {
        arc_map->push_back(arc_begin);
      }
    }
  }
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
                   [](const Arc &arc, int32_t index) -> ArcWithIndex {
                     return std::make_pair(arc, index);
                   });
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
}  // namespace k2
