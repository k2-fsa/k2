// k2/csrc/fsa_algo.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_algo.h"

#include <stack>

namespace {

// depth first search state
struct DfsState {
  int32_t s;      // state number of the visiting node
  int32_t begin;  // arc index of the visiting arc
  int32_t end;    // end of the arc index of the visiting node
};

}  // namespace

namespace k2 {

// The implementation of this function is inspired by
// http://www.openfst.org/doxygen/fst/html/connect_8h_source.html
void ConnectCore(const Fsa &fsa, std::vector<int32_t> *state_map) {
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
    auto &top = stack.top();

    if (top.begin == top.end) {
      // we have finished visiting this state
      auto s = top.s;  // get a copy since we will destroy it
      stack.pop();
      if (!stack.empty()) {
        // if it has a parent, set the parent's coaccessible flag
        if (coaccessible[s]) {
          auto &parent = stack.top();
          coaccessible[parent.s] = true;
          ++parent.begin;  // process the next child
        }
      }
      continue;
    }

    const auto &arc = fsa.arcs[top.begin];
    auto next_state = arc.dest_state;
    bool is_visited = visited[next_state];
    if (!is_visited) {
      // this is a new discovered state
      visited[next_state] = true;
      auto begin = fsa.arc_indexes[next_state];
      if (next_state != final_state)
        stack.push({next_state, begin, fsa.arc_indexes[next_state + 1]});
      else
        stack.push({next_state, begin, begin});

      if (accessible[top.s]) accessible[next_state] = true;
    } else {
      // this is a back arc or forward cross arc;
      // update the coaccesible flag
      auto next_state = arc.dest_state;
      if (coaccessible[next_state]) coaccessible[top.s] = true;
      ++top.begin;  // go to the next arc
    }
  }

  state_map->clear();
  state_map->reserve(num_states);

  for (int32_t i = 0; i != num_states; ++i) {
    if (accessible[i] && coaccessible[i]) {
      state_map->push_back(i);
    }
  }
}

}  // namespace k2
