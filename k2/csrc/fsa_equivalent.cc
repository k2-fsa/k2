// k2/csrc/fsa_equivalent.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_equivalent.h"

#include <algorithm>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "k2/csrc/fsa.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/properties.h"
#include "k2/csrc/util.h"
#include "k2/csrc/weights.h"

namespace {
// out_weights[i] = weights[arc_map1[arc_map2[i]]]
static void GetArcWeights(const float *weights,
                          const std::vector<int32_t> &arc_map1,
                          const std::vector<int32_t> &arc_map2,
                          std::vector<float> *out_weights) {
  CHECK_NOTNULL(out_weights);
  auto &arc_weights = *out_weights;
  for (auto i = 0; i != arc_weights.size(); ++i) {
    arc_weights[i] = weights[arc_map1[arc_map2[i]]];
  }
}

// c = (a - b) + (b-a)
static void SetDifference(const std::unordered_set<int32_t> &a,
                          const std::unordered_set<int32_t> &b,
                          std::unordered_set<int32_t> *c) {
  CHECK_NOTNULL(c);
  c->clear();
  for (const auto &v : a) {
    if (b.find(v) == b.end()) c->insert(v);
  }
  for (const auto &v : b) {
    if (a.find(v) == a.end()) c->insert(v);
  }
}

static bool RandomPathHelper(const k2::Fsa &a, k2::Fsa *b, bool no_epsilon_arc,
                             std::vector<int32_t> *state_map = nullptr) {
  using k2::Arc;
  using k2::ArcHash;
  using k2::kEpsilon;
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
    const Arc *curr_arc = nullptr;
    int32_t curr_state = state;
    do {
      int32_t begin = a.arc_indexes[curr_state];
      int32_t end = a.arc_indexes[curr_state + 1];
      // since `a` is valid, so every states contains at least one arc.
      int32_t arc_index = begin + (distribution(generator) % (end - begin));
      curr_arc = &a.arcs[arc_index];
      curr_state = curr_arc->dest_state;
    } while (no_epsilon_arc && curr_arc->label == kEpsilon);
    int32_t state_id_in_b = state_map_a2b[state];
    if (visited_arcs[state_id_in_b]
            .insert({state, curr_arc->dest_state, curr_arc->label})
            .second)
      ++num_visited_arcs;
    state = curr_arc->dest_state;
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

}  // namespace

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

template <FbWeightType Type>
bool IsRandEquivalent(const Fsa &a, const float *a_weights, const Fsa &b,
                      const float *b_weights, float beam /*=kFloatInfinity*/,
                      float delta /*=1e-6*/, bool top_sorted /*=true*/,
                      std::size_t npath /*= 100*/) {
  CHECK_GT(beam, 0);
  CHECK_NOTNULL(a_weights);
  CHECK_NOTNULL(b_weights);
  Fsa connected_a, connected_b, valid_a, valid_b;
  std::vector<int32_t> connected_a_arc_map, connected_b_arc_map,
      valid_a_arc_map, valid_b_arc_map;
  Connect(a, &connected_a, &connected_a_arc_map);
  Connect(b, &connected_b, &connected_b_arc_map);
  ArcSort(connected_a, &valid_a, &valid_a_arc_map);  // required by `intersect`
  ArcSort(connected_b, &valid_b, &valid_b_arc_map);
  if (IsEmpty(valid_a) && IsEmpty(valid_b)) return true;
  if (IsEmpty(valid_a) || IsEmpty(valid_b)) return false;

  // Get arc weights
  std::vector<float> valid_a_weights(valid_a.arcs.size());
  std::vector<float> valid_b_weights(valid_b.arcs.size());
  ::GetArcWeights(a_weights, connected_a_arc_map, valid_a_arc_map,
                  &valid_a_weights);
  ::GetArcWeights(b_weights, connected_b_arc_map, valid_b_arc_map,
                  &valid_b_weights);

  // Check that arc labels are compatible.
  std::unordered_set<int32_t> labels_a, labels_b, labels_difference;
  for (const auto &arc : valid_a.arcs) labels_a.insert(arc.label);
  for (const auto &arc : valid_b.arcs) labels_b.insert(arc.label);
  SetDifference(labels_a, labels_b, &labels_difference);
  if (labels_difference.size() >= 2 ||
      (labels_difference.size() == 1 &&
       (*(labels_difference.begin())) != kEpsilon))
    return false;

  double loglike_cutoff_a, loglike_cutoff_b;
  if (beam != kFloatInfinity) {
    // TODO(haowen): remove fsa_creator here after replacing FSA with Array2
    loglike_cutoff_a =
        ShortestDistance<Type>(
            FsaCreator(valid_a.arcs, valid_a.FinalState()).GetFsa(),
            valid_a_weights.data()) -
        beam;
    loglike_cutoff_b =
        ShortestDistance<Type>(
            FsaCreator(valid_b.arcs, valid_b.FinalState()).GetFsa(),
            valid_b_weights.data()) -
        beam;
    if (Type == kMaxWeight &&
        !DoubleApproxEqual(loglike_cutoff_a, loglike_cutoff_b))
      return false;
  } else {
    loglike_cutoff_a = kDoubleNegativeInfinity;
    loglike_cutoff_b = kDoubleNegativeInfinity;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution coin(0.5);
  std::size_t n = 0;
  while (n < npath) {
    const auto &fsa = coin(gen) ? valid_a : valid_b;
    Fsa path, valid_path;
    RandomPathWithoutEpsilonArc(fsa, &path);  // path is already connected
    ArcSort(path, &valid_path);

    Fsa a_compose_path, b_compose_path;
    std::vector<float> a_compose_weights, b_compose_weights;
    Intersect(valid_a, valid_a_weights.data(), path, &a_compose_path,
              &a_compose_weights);
    Intersect(valid_b, valid_b_weights.data(), path, &b_compose_path,
              &b_compose_weights);
    // TODO(haowen): we may need to implement a version of `ShortestDistance`
    // for non-top-sorted FSAs, but we prefer to decide this later as there's no
    // such scenarios (input FSAs are not top-sorted) currently. If we finally
    // find out that we don't need that version, we will remove flag
    // `top_sorted` and add requirements as comments in the header file.
    CHECK(top_sorted);
    double cost_a = ShortestDistance<Type>(
        FsaCreator(a_compose_path.arcs, a_compose_path.FinalState()).GetFsa(),
        a_compose_weights.data());
    double cost_b = ShortestDistance<Type>(
        FsaCreator(b_compose_path.arcs, b_compose_path.FinalState()).GetFsa(),
        b_compose_weights.data());

    if (cost_a < loglike_cutoff_a && cost_b < loglike_cutoff_b) continue;

    if (!DoubleApproxEqual(cost_a, cost_b, delta)) return false;

    ++n;
  }
  return true;
}

// explicit instantiation here
template bool IsRandEquivalent<kMaxWeight>(const Fsa &a, const float *a_weights,
                                           const Fsa &b, const float *b_weights,
                                           float beam, float delta,
                                           bool top_sorted, std::size_t npath);
template bool IsRandEquivalent<kLogSumWeight>(
    const Fsa &a, const float *a_weights, const Fsa &b, const float *b_weights,
    float beam, float delta, bool top_sorted, std::size_t npath);

bool IsRandEquivalentAfterRmEpsPrunedLogSum(
    const Fsa &a, const float *a_weights, const Fsa &b, const float *b_weights,
    float beam, bool top_sorted /*= true*/, std::size_t npath /*= 100*/) {
  CHECK_GT(beam, 0);
  CHECK_NOTNULL(a_weights);
  CHECK_NOTNULL(b_weights);
  Fsa connected_a, connected_b, valid_a, valid_b;
  std::vector<int32_t> connected_a_arc_map, connected_b_arc_map,
      valid_a_arc_map, valid_b_arc_map;
  Connect(a, &connected_a, &connected_a_arc_map);
  Connect(b, &connected_b, &connected_b_arc_map);
  ArcSort(connected_a, &valid_a, &valid_a_arc_map);  // required by `intersect`
  ArcSort(connected_b, &valid_b, &valid_b_arc_map);
  if (IsEmpty(valid_a) && IsEmpty(valid_b)) return true;
  if (IsEmpty(valid_a) || IsEmpty(valid_b)) return false;

  // Get arc weights
  std::vector<float> valid_a_weights(valid_a.arcs.size());
  std::vector<float> valid_b_weights(valid_b.arcs.size());
  ::GetArcWeights(a_weights, connected_a_arc_map, valid_a_arc_map,
                  &valid_a_weights);
  ::GetArcWeights(b_weights, connected_b_arc_map, valid_b_arc_map,
                  &valid_b_weights);

  // Check that arc labels are compatible.
  std::unordered_set<int32_t> labels_a, labels_b, labels_difference;
  for (const auto &arc : valid_a.arcs) labels_a.insert(arc.label);
  for (const auto &arc : valid_b.arcs) labels_b.insert(arc.label);
  SetDifference(labels_a, labels_b, &labels_difference);
  if (labels_difference.size() >= 2 ||
      (labels_difference.size() == 1 &&
       (*(labels_difference.begin())) != kEpsilon))
    return false;
  // `b` is the FSA after epsilon-removal, so it should be epsilon-free
  if (labels_b.find(kEpsilon) != labels_b.end()) return false;

  double loglike_cutoff_a =
      ShortestDistance<kLogSumWeight>(
          FsaCreator(valid_a.arcs, valid_a.FinalState()).GetFsa(),
          valid_a_weights.data()) -
      beam;
  double loglike_cutoff_b =
      ShortestDistance<kLogSumWeight>(
          FsaCreator(valid_b.arcs, valid_b.FinalState()).GetFsa(),
          valid_b_weights.data()) -
      beam;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution coin(0.5);
  std::size_t n = 0;
  while (n < npath) {
    bool random_path_from_a = coin(gen);
    const auto &fsa = random_path_from_a ? valid_a : valid_b;
    Fsa path, valid_path;
    RandomPathWithoutEpsilonArc(fsa, &path);  // path is already connected
    ArcSort(path, &valid_path);

    Fsa a_compose_path, b_compose_path;
    std::vector<float> a_compose_weights, b_compose_weights;
    Intersect(valid_a, valid_a_weights.data(), path, &a_compose_path,
              &a_compose_weights);
    Intersect(valid_b, valid_b_weights.data(), path, &b_compose_path,
              &b_compose_weights);
    // TODO(haowen): we may need to implement a version of `ShortestDistance`
    // for non-top-sorted FSAs, but we prefer to decide this later as there's no
    // such scenarios (input FSAs are not top-sorted) currently.
    CHECK(top_sorted);
    double cost_a = ShortestDistance<kLogSumWeight>(
        FsaCreator(a_compose_path.arcs, a_compose_path.FinalState()).GetFsa(),
        a_compose_weights.data());
    double cost_b = ShortestDistance<kLogSumWeight>(
        FsaCreator(b_compose_path.arcs, b_compose_path.FinalState()).GetFsa(),
        b_compose_weights.data());
    if (random_path_from_a) {
      if (cost_a < loglike_cutoff_a) continue;
      // there is no corresponding path in `b`
      // (cost_b == kDoubleNegativeInfinity < cost_a)
      // or it has weight less than its weight in `a`.
      if (cost_a > cost_b) return false;
    } else {
      if (cost_b < loglike_cutoff_b) continue;
      // there's no corresponding path in `a` or it has weight greater than its
      // weights in `b`.
      if (cost_a == kDoubleNegativeInfinity || cost_a > cost_b) return false;
    }
    ++n;
  }
  return true;
}

bool RandomPath(const Fsa &a, Fsa *b,
                std::vector<int32_t> *state_map /*=nullptr*/) {
  return RandomPathHelper(a, b, false, state_map);
}

bool RandomPathWithoutEpsilonArc(
    const Fsa &a, Fsa *b, std::vector<int32_t> *state_map /*= nullptr*/) {
  return RandomPathHelper(a, b, true, state_map);
}

void Intersect(const Fsa &a, const float *a_weights, const Fsa &b, Fsa *c,
               std::vector<float> *c_weights,
               std::vector<int32_t> *arc_map_a /*= nullptr*/,
               std::vector<int32_t> *arc_map_b /*= nullptr*/) {
  CHECK_NOTNULL(c);
  CHECK_NOTNULL(c_weights);
  c->arc_indexes.clear();
  c->arcs.clear();
  c_weights->clear();
  if (arc_map_a != nullptr) arc_map_a->clear();
  if (arc_map_b != nullptr) arc_map_b->clear();

  if (IsEmpty(a) || IsEmpty(b)) return;
  CHECK(IsArcSorted(a));
  CHECK(IsArcSorted(b));
  CHECK(IsEpsilonFree(b));

  int32_t final_state_a = a.NumStates() - 1;
  int32_t final_state_b = b.NumStates() - 1;
  const auto arc_a_begin = a.arcs.begin();
  const auto arc_b_begin = b.arcs.begin();
  using ArcIterator = std::vector<Arc>::const_iterator;

  constexpr int32_t kFinalStateC = -1;  // just as a placeholder
  // no corresponding arc mapping from `c` to `a` or `c` to `b`
  constexpr int32_t kArcMapNone = -1;
  auto &arc_indexes_c = c->arc_indexes;
  auto &arcs_c = c->arcs;

  using StatePair = std::pair<int32_t, int32_t>;
  // map state pair to unique id
  std::unordered_map<StatePair, int32_t, PairHash> state_pair_map;
  std::queue<StatePair> qstates;
  qstates.push({0, 0});
  state_pair_map.insert({{0, 0}, 0});
  state_pair_map.insert({{final_state_a, final_state_b}, kFinalStateC});
  int32_t state_index_c = 0;
  while (!qstates.empty()) {
    arc_indexes_c.push_back(static_cast<int32_t>(arcs_c.size()));

    auto curr_state_pair = qstates.front();
    qstates.pop();
    // as we have inserted `curr_state_pair` before.
    int32_t curr_state_index = state_pair_map[curr_state_pair];

    auto state_a = curr_state_pair.first;
    auto a_arc_iter_begin = arc_a_begin + a.arc_indexes[state_a];
    auto a_arc_iter_end = arc_a_begin + a.arc_indexes[state_a + 1];
    auto state_b = curr_state_pair.second;
    auto b_arc_iter_begin = arc_b_begin + b.arc_indexes[state_b];
    auto b_arc_iter_end = arc_b_begin + b.arc_indexes[state_b + 1];

    // As both `a` and `b` are arc-sorted, we first process epsilon arcs in `a`.
    for (; a_arc_iter_begin != a_arc_iter_end; ++a_arc_iter_begin) {
      if (kEpsilon != a_arc_iter_begin->label) break;

      StatePair new_state{a_arc_iter_begin->dest_state, state_b};
      auto result = state_pair_map.insert({new_state, state_index_c + 1});
      if (result.second) {
        // we have not visited `new_state` before.
        qstates.push(new_state);
        ++state_index_c;
      }
      int32_t new_state_index = result.first->second;
      arcs_c.emplace_back(curr_state_index, new_state_index, kEpsilon);
      c_weights->push_back(a_weights[a_arc_iter_begin - arc_a_begin]);
      if (arc_map_a != nullptr)
        arc_map_a->push_back(
            static_cast<int32_t>(a_arc_iter_begin - arc_a_begin));
      if (arc_map_b != nullptr) arc_map_b->push_back(kArcMapNone);
    }

    // `b` is usually a path generated from `RandNonEpsilonPath`, it may hold
    // less number of arcs in each state, so we iterate over `b` here to save
    // time.
    for (; b_arc_iter_begin != b_arc_iter_end; ++b_arc_iter_begin) {
      const Arc &curr_b_arc = *b_arc_iter_begin;
      auto a_arc_range =
          std::equal_range(a_arc_iter_begin, a_arc_iter_end, curr_b_arc,
                           [](const Arc &left, const Arc &right) {
                             return left.label < right.label;
                           });
      for (auto it_a = a_arc_range.first; it_a != a_arc_range.second; ++it_a) {
        const Arc &curr_a_arc = *it_a;
        StatePair new_state{curr_a_arc.dest_state, curr_b_arc.dest_state};
        auto result = state_pair_map.insert({new_state, state_index_c + 1});
        if (result.second) {
          qstates.push(new_state);
          ++state_index_c;
        }
        int32_t new_state_index = result.first->second;
        arcs_c.emplace_back(curr_state_index, new_state_index,
                            curr_a_arc.label);
        c_weights->push_back(a_weights[it_a - arc_a_begin]);
        if (arc_map_a != nullptr)
          arc_map_a->push_back(static_cast<int32_t>(it_a - arc_a_begin));
        if (arc_map_b != nullptr)
          arc_map_b->push_back(
              static_cast<int32_t>(b_arc_iter_begin - arc_b_begin));
      }
    }
  }

  // push final state
  arc_indexes_c.push_back(static_cast<int32_t>(arcs_c.size()));
  ++state_index_c;
  // then replace `kFinalStateC` with the real index of final state of `c`
  for (auto &arc : arcs_c) {
    if (arc.dest_state == kFinalStateC) arc.dest_state = state_index_c;
  }
  // push a duplicate of final state, see the constructor of `Fsa` in
  // `k2/csrc/fsa.h`
  arc_indexes_c.emplace_back(arc_indexes_c.back());
}

}  // namespace k2
