// k2/csrc/fsa_algo.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_algo.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <stack>
#include <unordered_map>
#include <utility>

#include "glog/logging.h"
#include "k2/csrc/properties.h"
#include "k2/csrc/util.h"

namespace {

static constexpr int8_t kNotVisited = 0;  // a node that has not been visited
static constexpr int8_t kVisiting = 1;    // a node that is under visiting
static constexpr int8_t kVisited = 2;     // a node that has been visited
// depth first search state
struct DfsState {
  int32_t state;      // state number of the visiting node
  int32_t arc_begin;  // arc index of the visiting arc
  int32_t arc_end;    // end of the arc index of the visiting node
};

using StatePair = std::pair<int32_t, int32_t>;

inline int32_t InsertIntersectionState(
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

// This function uses "Tarjan's strongly connected components algorithm"
// (see
// https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)
// to find co-accessible states in a single pass of the graph.
//
// The notations "lowlink", "dfnumber", and "onstack" are from the book
// "The design and analysis of computer algorithms", on page 192,
// Fig. 5.15 "Procedure to compute LOWLINK", written by John E. Hopcroft.
//
// http://www.openfst.org/doxygen/fst/html/connect_8h_source.html
// is used as a reference while implementing this function.
bool ConnectCore(const Fsa &fsa, std::vector<int32_t> *state_map) {
  CHECK_NOTNULL(state_map);

  state_map->clear();
  if (IsEmpty(fsa)) return true;

  auto num_states = fsa.NumStates();
  auto final_state = num_states - 1;

  std::vector<bool> accessible(num_states, false);
  std::vector<bool> coaccessible(num_states, false);
  std::vector<int8_t> state_status(num_states, kNotVisited);

  // ssc is short for "strongly connected component"
  // the following block of variables are for
  // "Tarjan's strongly connected components algorithm"
  //
  // Refer to the comment above the function for the meaning of them
  std::vector<int32_t> ssc_stack;
  ssc_stack.reserve(num_states);
  std::vector<bool> onstack(num_states, false);
  std::vector<int32_t> dfnumber(num_states,
                                std::numeric_limits<int32_t>::max());
  auto df_count = 0;
  std::vector<int32_t> lowlink(num_states, std::numeric_limits<int32_t>::max());

  accessible.front() = true;
  coaccessible.back() = true;

  std::stack<DfsState> stack;
  stack.push({0, fsa.arc_indexes[0], fsa.arc_indexes[1]});
  state_status[0] = kVisiting;

  dfnumber[0] = df_count;
  lowlink[0] = df_count;
  ++df_count;
  ssc_stack.push_back(0);
  onstack[0] = true;

  // map order to state.
  // state 0 has the largest order, i.e., num_states - 1
  // final_state has the least order, i.e., 0
  std::vector<int32_t> order;
  order.reserve(num_states);
  bool is_acyclic = true;  // order and is_acyclic are for topological sort

  while (!stack.empty()) {
    auto &current_state = stack.top();

    if (current_state.arc_begin == current_state.arc_end) {
      // we have finished visiting this state
      auto state = current_state.state;  // get a copy since we will destroy it
      stack.pop();
      state_status[state] = kVisited;

      order.push_back(state);

      if (dfnumber[state] == lowlink[state]) {
        // this is the root of the strongly connected component
        bool scc_coaccessible = false;  // if any node in scc is co-accessible,
                                        // it will be set to true
        auto k = ssc_stack.size() - 1;
        auto num_nodes = 0;  // number of nodes in the scc

        auto tmp = 0;
        do {
          tmp = ssc_stack[k--];
          if (coaccessible[tmp]) scc_coaccessible = true;
          ++num_nodes;
        } while (tmp != state);

        // if this cycle is not removed in the output fsa
        // set is_acyclic to false
        if (num_nodes > 1 && scc_coaccessible) is_acyclic = false;

        // now pop ssc_stack and set co-accessible of each node
        do {
          tmp = ssc_stack.back();
          if (scc_coaccessible) coaccessible[tmp] = true;
          ssc_stack.pop_back();
          onstack[tmp] = false;
        } while (tmp != state);
      }

      if (!stack.empty()) {
        // if it has a parent, set the parent's co-accessible flag
        auto &parent = stack.top();
        if (coaccessible[state]) coaccessible[parent.state] = true;

        ++parent.arc_begin;  // process the next child

        lowlink[parent.state] = std::min(lowlink[parent.state], lowlink[state]);
      }
      continue;
    }

    const auto &arc = fsa.arcs[current_state.arc_begin];
    auto next_state = arc.dest_state;
    auto status = state_status[next_state];
    switch (status) {
      case kNotVisited: {
        // a new discovered node
        state_status[next_state] = kVisiting;
        auto arc_begin = fsa.arc_indexes[next_state];
        stack.push({next_state, arc_begin, fsa.arc_indexes[next_state + 1]});

        dfnumber[next_state] = df_count;
        lowlink[next_state] = df_count;
        ++df_count;
        ssc_stack.push_back(next_state);
        onstack[next_state] = true;

        if (accessible[current_state.state]) accessible[next_state] = true;
        break;
      }
      case kVisiting:
        // this is a back arc, which means there is a loop in the fsa;
        // but this loop may be removed in the output fsa
        //
        // Refer to the above book for what the meaning of back arc is
        lowlink[current_state.state] =
            std::min(lowlink[current_state.state], dfnumber[next_state]);

        if (coaccessible[next_state]) coaccessible[current_state.state] = true;
        ++current_state.arc_begin;  // go to the next arc
        break;
      case kVisited:
        // this is a forward or cross arc;
        if (dfnumber[next_state] < dfnumber[current_state.state] &&
            onstack[next_state])
          lowlink[current_state.state] =
              std::min(lowlink[current_state.state], dfnumber[next_state]);

        // update the co-accessible flag
        if (coaccessible[next_state]) coaccessible[current_state.state] = true;
        ++current_state.arc_begin;  // go to the next arc
        break;
      default:
        LOG(FATAL) << "Unreachable code is executed!";
        break;
    }
  }

  state_map->reserve(num_states);
  if (!is_acyclic) {
    for (auto i = 0; i != num_states; ++i) {
      if (accessible[i] && coaccessible[i]) state_map->push_back(i);
    }
    return false;
  }

  // now for the acyclic case,
  // we return a state_map of a topologically sorted fsa
  const auto rend = order.rend();
  for (auto rbegin = order.rbegin(); rbegin != rend; ++rbegin) {
    auto s = *rbegin;
    if (accessible[s] && coaccessible[s]) state_map->push_back(s);
  }
  return true;
}

bool Connect(const Fsa &a, Fsa *b, std::vector<int32_t> *arc_map /*=nullptr*/) {
  CHECK_NOTNULL(b);
  if (arc_map != nullptr) arc_map->clear();

  std::vector<int32_t> state_b_to_a;
  bool is_acyclic = ConnectCore(a, &state_b_to_a);
  if (state_b_to_a.empty()) return true;

  b->arc_indexes.resize(state_b_to_a.size() + 1);
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
    arc_end = a.arc_indexes[state_a + 1];

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
  b->arc_indexes[num_states_b] = b->arc_indexes[num_states_b - 1];
  return is_acyclic;
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
  const auto arc_b_begin = b.arcs.begin();
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
    ArcIterator a_arc_iter_end = arc_a_begin + a.arc_indexes[state_a + 1];
    auto state_b = curr_state_pair.second;
    ArcIterator b_arc_iter_begin = arc_b_begin + b.arc_indexes[state_b];
    ArcIterator b_arc_iter_end = arc_b_begin + b.arc_indexes[state_b + 1];

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
  // push a duplicate of final state, see the constructor of `Fsa` in
  // `k2/csrc/fsa.h`
  arc_indexes_c.emplace_back(arc_indexes_c.back());
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
  int32_t final_state = a.NumStates() - 1;
  for (int32_t state = 0; state < final_state; ++state) {
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

bool TopSort(const Fsa &a, Fsa *b,
             std::vector<int32_t> *state_map /*= nullptr*/) {
  CHECK_NOTNULL(b);
  b->arc_indexes.clear();
  b->arcs.clear();

  if (state_map != nullptr) state_map->clear();

  if (IsEmpty(a)) return true;
  if (!IsConnected(a)) return false;

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
        stack.push({next_state, arc_begin, a.arc_indexes[next_state + 1]});
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
    arc_end = a.arc_indexes[state_a + 1];

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
  b->arc_indexes.emplace_back(b->arc_indexes.back());
  return true;
}

std::unique_ptr<Fsa> CreateFsa(const std::vector<Arc> &arcs) {
  if (arcs.empty()) return nullptr;

  std::vector<std::vector<Arc>> vec;
  for (const auto &arc : arcs) {
    auto src_state = arc.src_state;
    auto dest_state = arc.dest_state;
    auto new_size = std::max(src_state, dest_state);
    if (new_size >= vec.size()) vec.resize(new_size + 1);
    vec[src_state].push_back(arc);
  }

  std::stack<DfsState> stack;
  std::vector<char> state_status(vec.size(), kNotVisited);
  std::vector<int32_t> order;

  auto num_states = static_cast<int32_t>(vec.size());
  for (auto i = 0; i != num_states; ++i) {
    if (state_status[i] == kVisited) continue;
    stack.push({i, 0, static_cast<int32_t>(vec[i].size())});
    state_status[i] = kVisiting;
    while (!stack.empty()) {
      auto &current_state = stack.top();
      auto state = current_state.state;

      if (current_state.arc_begin == current_state.arc_end) {
        state_status[state] = kVisited;
        order.push_back(state);
        stack.pop();
        continue;
      }

      const auto &arc = vec[state][current_state.arc_begin];
      auto next_state = arc.dest_state;
      auto status = state_status[next_state];
      switch (status) {
        case kNotVisited:
          state_status[next_state] = kVisiting;
          stack.push(
              {next_state, 0, static_cast<int32_t>(vec[next_state].size())});
          ++current_state.arc_begin;
          break;
        case kVisiting:
          LOG(FATAL) << "there is a cycle: " << state << " -> " << next_state;
          break;
        case kVisited:
          ++current_state.arc_begin;
          break;
        default:
          LOG(FATAL) << "Unreachable code is executed!";
          break;
      }
    }
  }

  CHECK_EQ(num_states, static_cast<int32_t>(order.size()));

  std::reverse(order.begin(), order.end());

  std::unique_ptr<Fsa> fsa(new Fsa);
  fsa->arc_indexes.resize(num_states + 1);
  fsa->arcs.reserve(arcs.size());

  std::vector<int32_t> old_to_new(num_states);
  for (auto i = 0; i != num_states; ++i) old_to_new[order[i]] = i;

  for (auto i = 0; i != num_states; ++i) {
    auto old_state = order[i];
    fsa->arc_indexes[i] = static_cast<int32_t>(fsa->arcs.size());
    for (auto arc : vec[old_state]) {
      arc.src_state = i;
      arc.dest_state = old_to_new[arc.dest_state];
      fsa->arcs.push_back(arc);
    }
  }

  fsa->arc_indexes.back() = static_cast<int32_t>(fsa->arcs.size());

  return fsa;
}

}  // namespace k2
