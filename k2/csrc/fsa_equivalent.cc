// k2/csrc/fsa_equivalent.cc

// Copyright (c)  2020 Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_equivalent.h"

#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/properties.h"

namespace k2 {

bool RandomPath(const Fsa &a, Fsa *b,
                std::vector<int32_t> *state_map /*=nullptr*/) {
  if (IsEmpty(a) || b == nullptr) return false;
  // we cannot do `connect` on `a` here to get a connected fsa
  // as `state_map` will map to states in the connected fsa
  // instead of in `a` if we do that.
  if (!IsConnected(a)) return false;

  int32_t num_states = a.NumStates();
  std::vector<int32_t> state_map_b2a;
  std::vector<int32_t> state_map_a2b(num_states, -1);
  // `visited_arcs[i]` stores `arcs` leaving from state `i` in `b`
  std::vector<std::unordered_set<Arc, ArcHash>> visited_arcs;

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int32_t> distribution(0);

  int32_t num_visited_arcs = 0;
  int32_t num_visited_state = 0;
  int32_t state = 0;
  int32_t final_state = num_states - 1;
  while (true) {
    if (state_map_a2b[state] == -1) {
      state_map_a2b[state] = num_visited_state;
      state_map_b2a.push_back(state);
      visited_arcs.push_back(std::unordered_set<Arc, ArcHash>());
      ++num_visited_state;
    }
    if (state == final_state) break;
    int32_t begin = a.arc_indexes[state];
    int32_t end = a.arc_indexes[state + 1];
    // since `a` is valid, so every states contains at least one arc.
    int32_t arc_index = begin + (distribution(generator) % (end - begin));
    int32_t state_id_in_b = state_map_a2b[state];
    const auto &curr_arc = a.arcs[arc_index];
    if (visited_arcs[state_id_in_b].insert(curr_arc).second) ++num_visited_arcs;
    state = curr_arc.dest_state;
  }

  // create `b`
  b->arc_indexes.resize(num_visited_state);
  b->arcs.resize(num_visited_arcs);
  int32_t n = 0;
  for (int32_t i = 0; i < num_visited_state; ++i) {
    b->arc_indexes[i] = n;
    for (const auto &arc : visited_arcs[i]) {
      auto &b_arc = b->arcs[n];
      b_arc.src_state = i;
      b_arc.dest_state = state_map_a2b[arc.dest_state];
      b_arc.label = arc.label;
      ++n;
    }
  }
  if (state_map != nullptr) {
    state_map->swap(state_map_b2a);
  }
  b->arc_indexes.emplace_back(b->arc_indexes.back());
  return true;
}

}  // namespace k2
