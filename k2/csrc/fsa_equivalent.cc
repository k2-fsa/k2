// k2/csrc/fsa_equivalent.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_equivalent.h"

#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/properties.h"
#include "k2/csrc/fsa_algo.h"

namespace k2 {

bool IsRandEquivalent(const Fsa &a, const Fsa &b, std::size_t npath /*=100*/) {
  // We will do `intersect` later which requires either `a` or `b` is
  // epsilon-free, considering they should hold same set of arc labels, so both
  // of them should be epsilon-free.
  // TODO(haowen): call `RmEpsilon` here instead of checking.
  if (!IsEpsilonFree(a) || !IsEpsilonFree(b)) return false;

  Fsa connected_a, connected_b, valid_a, valid_b;
  Connect(a, &connected_a);
  Connect(b, &connected_b);
  ArcSort(connected_a, &valid_a);  // required by `intersect`
  ArcSort(connected_b, &valid_b);
  if (IsEmpty(valid_a) && IsEmpty(valid_b)) return true;
  if (IsEmpty(valid_a) || IsEmpty(valid_b)) return false;

  // Check that arc labels are compatible.
  std::unordered_set<int32_t> labels_a, labels_b;
  for (const auto &arc : valid_a.arcs) labels_a.insert(arc.label);
  for (const auto &arc : valid_b.arcs) labels_b.insert(arc.label);
  if (labels_a != labels_b) return false;

  Fsa c, connected_c, valid_c;
  if (!Intersect(valid_a, valid_b, &c)) return false;
  Connect(c, &connected_c);
  ArcSort(connected_c, &valid_c);
  if (IsEmpty(valid_c)) return false;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution coin(0.5);
  for (auto i = 0; i != npath; ++i) {
    const auto &fsa = coin(gen) ? valid_a : valid_b;
    Fsa path, valid_path;
    RandomPath(fsa, &path);  // path is already connected
    ArcSort(path, &valid_path);
    Fsa cpath, valid_cpath;
    Intersect(valid_path, valid_c, &cpath);
    Connect(cpath, &valid_cpath);
    if (IsEmpty(valid_cpath)) return false;
  }

  return true;
}

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
      visited_arcs.emplace_back(std::unordered_set<Arc, ArcHash>());
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
