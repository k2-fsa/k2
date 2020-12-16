/**
 * @brief
 * fsa_equivalent
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/fsa_equivalent.h"

#include <algorithm>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "k2/csrc/host/arcsort.h"
#include "k2/csrc/host/connect.h"
#include "k2/csrc/host/fsa.h"
#include "k2/csrc/host/fsa_util.h"
#include "k2/csrc/host/intersect.h"
#include "k2/csrc/host/properties.h"
#include "k2/csrc/host/topsort.h"
#include "k2/csrc/host/util.h"
#include "k2/csrc/host/weights.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace {
/*
  Version of Connection that writes the output FSA to an FsaCreator `fsa_out`;
  see its documentation.
  Usually user will call `fsa_out.GetFsa()` to get the output FSA after the
  function call, the memory of the output FSA is managed by `fsa_out` and
  will be released automatically if `fsa_out`is out of scope.
 */
static bool Connect(const k2host::Fsa &fsa_in, k2host::FsaCreator *fsa_out,
                    std::vector<int32_t> *arc_map = nullptr) {
  NVTX_RANGE(K2_FUNC);
  k2host::Connection connection(fsa_in);
  k2host::Array2Size<int32_t> fsa_size;

  connection.GetSizes(&fsa_size);

  fsa_out->Init(fsa_size);
  auto &connected_fsa = fsa_out->GetFsa();
  if (arc_map != nullptr) arc_map->resize(fsa_size.size2);
  bool status = connection.GetOutput(
      &connected_fsa, arc_map == nullptr ? nullptr : arc_map->data());
  return status;
}

/*
  Version of ArcSorter that writes the output FSA to an FsaCreator `fsa_out`;
  see its documentation.
  Usually user will call `fsa_out.GetFsa()` to get the output FSA after the
  function call, the memory of the output FSA is managed by `fsa_out` and
  will be released automatically if `fsa_out`is out of scope.
 */
static void ArcSort(const k2host::Fsa &fsa_in, k2host::FsaCreator *fsa_out,
                    std::vector<int32_t> *arc_map = nullptr) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_out, nullptr);
  k2host::ArcSorter sorter(fsa_in);
  k2host::Array2Size<int32_t> fsa_size;
  sorter.GetSizes(&fsa_size);

  fsa_out->Init(fsa_size);
  auto &sorted_fsa = fsa_out->GetFsa();
  if (arc_map != nullptr) arc_map->resize(fsa_size.size2);
  sorter.GetOutput(&sorted_fsa, arc_map == nullptr ? nullptr : arc_map->data());
}

/*
  Version of TopSorter that writes the output FSA to an FsaCreator `fsa_out`;
  see its documentation.
  Usually user will call `fsa_out.GetFsa()` to get the output FSA after the
  function call, the memory of the output FSA is managed by `fsa_out` and
  will be released automatically if `fsa_out`is out of scope.
 */
static void TopSort(const k2host::Fsa &fsa_in, k2host::FsaCreator *fsa_out,
                    std::vector<int32_t> *arc_map = nullptr) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_out, nullptr);
  k2host::TopSorter sorter(fsa_in);
  k2host::Array2Size<int32_t> fsa_size;
  sorter.GetSizes(&fsa_size);

  fsa_out->Init(fsa_size);
  auto &sorted_fsa = fsa_out->GetFsa();
  if (arc_map != nullptr) arc_map->resize(fsa_size.size2);
  sorter.GetOutput(&sorted_fsa, arc_map == nullptr ? nullptr : arc_map->data());
}

/*
  Version of Intersection that writes the output FSA to an FsaCreator `c`;
  see its documentation.
  Usually user will call `c.GetFsa()` to get the output FSA after the
  function call, the memory of the output FSA is managed by `c` and will
  be released automatically if `c`is out of scope.
 */
static bool Intersect(const k2host::Fsa &a, const k2host::Fsa &b,
                      bool treat_epsilons_specially, k2host::FsaCreator *c,
                      std::vector<int32_t> *arc_map_a = nullptr,
                      std::vector<int32_t> *arc_map_b = nullptr) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(c, nullptr);
  k2host::Intersection intersection(a, b, treat_epsilons_specially);
  k2host::Array2Size<int32_t> fsa_size;
  intersection.GetSizes(&fsa_size);

  c->Init(fsa_size);
  auto &composed_fsa = c->GetFsa();
  if (arc_map_a != nullptr) arc_map_a->resize(fsa_size.size2);
  if (arc_map_b != nullptr) arc_map_b->resize(fsa_size.size2);
  bool status = intersection.GetOutput(
      &composed_fsa, arc_map_a == nullptr ? nullptr : arc_map_a->data(),
      arc_map_b == nullptr ? nullptr : arc_map_b->data());
  return status;
}

/*
  Version of RandPath that writes the output path to an FsaCreator `path`;
  see its documentation.
  Usually user will call `path.GetFsa()` to get the output path after
  the function call, the memory of the output path is managed by `path`
  and will be released automatically if `path`is out of scope.
 */
static bool RandomPath(const k2host::Fsa &fsa_in, bool no_eps_arc,
                       k2host::FsaCreator *path,
                       std::vector<int32_t> *arc_map = nullptr) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(path, nullptr);
  k2host::RandPath rand_path(fsa_in, no_eps_arc);
  k2host::Array2Size<int32_t> fsa_size;
  rand_path.GetSizes(&fsa_size);

  path->Init(fsa_size);
  auto &path_fsa = path->GetFsa();
  if (arc_map != nullptr) arc_map->resize(fsa_size.size2);
  bool status = rand_path.GetOutput(
      &path_fsa, arc_map == nullptr ? nullptr : arc_map->data());
  return status;
}

// c = (a - b) + (b-a)
static void SetDifference(const std::unordered_set<int32_t> &a,
                          const std::unordered_set<int32_t> &b,
                          std::unordered_set<int32_t> *c) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(c, nullptr);
  c->clear();
  for (const auto &v : a) {
    if (b.find(v) == b.end()) c->insert(v);
  }
  for (const auto &v : b) {
    if (a.find(v) == a.end()) c->insert(v);
  }
}

}  // namespace

namespace k2host {

bool IsRandEquivalent(const Fsa &a, const Fsa &b,
                      bool treat_epsilons_specially /*=true*/,
                      std::size_t npath /*=100*/) {
  NVTX_RANGE(K2_FUNC);
  if (!IsValid(a) || !IsValid(b)) {
    K2_LOG(WARNING) << "One or more of the inputs is not valid.";
    return false;
  }
  FsaCreator valid_a_storage, valid_b_storage;
  ::Connect(a, &valid_a_storage);
  ::Connect(b, &valid_b_storage);
  ArcSort(&valid_a_storage.GetFsa());  // required by `intersect`
  ArcSort(&valid_b_storage.GetFsa());
  const auto &valid_a = valid_a_storage.GetFsa();
  const auto &valid_b = valid_b_storage.GetFsa();

  if (IsEmpty(valid_a) && IsEmpty(valid_b)) return true;
  if (IsEmpty(valid_a) || IsEmpty(valid_b)) return false;

  // Check that arc labels are compatible: the two input Fsas must have the same
  // set of labels after connection, it's fine that one has epsilon arcs and the
  // other doesn't have.
  std::unordered_set<int32_t> labels_a, labels_b, labels_difference;
  for (const auto &arc : valid_a) labels_a.insert(arc.label);
  for (const auto &arc : valid_b) labels_b.insert(arc.label);
  SetDifference(labels_a, labels_b, &labels_difference);
  if (labels_difference.size() >= 2 ||
      (labels_difference.size() == 1 &&
       (*(labels_difference.begin())) != kEpsilon))
    return false;

  FsaCreator c_storage, valid_c_storage;
  if (!::Intersect(valid_a, valid_b, treat_epsilons_specially, &c_storage))
    return false;
  ::Connect(c_storage.GetFsa(), &valid_c_storage);
  ArcSort(&valid_c_storage.GetFsa());
  const auto &valid_c = valid_c_storage.GetFsa();
  if (IsEmpty(valid_c)) return false;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution coin(0.5);
  for (auto i = 0; i != npath; ++i) {
    const auto &fsa = coin(gen) ? valid_a : valid_b;
    FsaCreator valid_path_storage;
    if (!::RandomPath(fsa, false, &valid_path_storage)) continue;
    // path is already connected
    ArcSort(&valid_path_storage.GetFsa());
    FsaCreator cpath_storage, valid_cpath_storage;
    ::Intersect(valid_path_storage.GetFsa(), valid_c, treat_epsilons_specially,
                &cpath_storage);
    ::Connect(cpath_storage.GetFsa(), &valid_cpath_storage);
    if (IsEmpty(valid_cpath_storage.GetFsa())) return false;
  }

  return true;
}

template <FbWeightType Type>
bool IsRandEquivalent(const Fsa &a, const Fsa &b,
                      float beam /*=kFloatInfinity*/,
                      bool treat_epsilons_specially /*=true*/,
                      float delta /*=1e-6*/, bool top_sorted /*=true*/,
                      std::size_t npath /*= 100*/) {
  NVTX_RANGE(K2_FUNC);
  if (!IsValid(a) || !IsValid(b)) {
    K2_LOG(WARNING) << "One or more of the inputs is not valid.";
    return false;
  }
  K2_CHECK_GT(beam, 0);
  // TODO(haowen): for now we only support top-sorted input Fsas
  K2_CHECK(top_sorted);
  FsaCreator connected_a_storage, connected_b_storage, valid_a_storage,
      valid_b_storage;
  std::vector<int32_t> connected_a_arc_map, connected_b_arc_map,
      valid_a_arc_map, valid_b_arc_map;
  ::Connect(a, &connected_a_storage, &connected_a_arc_map);
  ::Connect(b, &connected_b_storage, &connected_b_arc_map);
  ::ArcSort(connected_a_storage.GetFsa(), &valid_a_storage,
            &valid_a_arc_map);  // required by `intersect`
  ::ArcSort(connected_b_storage.GetFsa(), &valid_b_storage, &valid_b_arc_map);
  const auto &valid_a = valid_a_storage.GetFsa();
  const auto &valid_b = valid_b_storage.GetFsa();
  if (IsEmpty(valid_a) && IsEmpty(valid_b)) return true;
  if (IsEmpty(valid_a) || IsEmpty(valid_b)) return false;

  // Check that arc labels are compatible.
  std::unordered_set<int32_t> labels_a, labels_b, labels_difference;
  for (const auto &arc : valid_a) labels_a.insert(arc.label);
  for (const auto &arc : valid_b) labels_b.insert(arc.label);
  SetDifference(labels_a, labels_b, &labels_difference);
  if (labels_difference.size() >= 2 ||
      (labels_difference.size() == 1 &&
       (*(labels_difference.begin())) != kEpsilon))
    return false;

  double dist_a = ShortestDistance<Type>(valid_a),
         dist_b = ShortestDistance<Type>(valid_b),
         loglike_cutoff_a = dist_a - beam, loglike_cutoff_b = dist_b - beam;
  if (abs(dist_a - dist_b) > delta) return false;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution coin(0.5);

  for (std::size_t n = 0; n < npath; ++n) {
    // Note: for now we `continue` when paths are not above the beam, which may
    // mean that for some FSAs we don't test any paths at all.
    const auto &fsa = coin(gen) ? valid_a : valid_b;
    FsaCreator valid_path_storage;
    if (!::RandomPath(fsa, false, &valid_path_storage)) continue;
    // path is already connected so we will not call `::Connect` here
    ArcSort(&valid_path_storage.GetFsa());
    const auto &valid_path = valid_path_storage.GetFsa();

    FsaCreator a_compose_path_storage, b_compose_path_storage;
    std::vector<int32_t> arc_map_a_path, arc_map_b_path;
    ::Intersect(valid_a, valid_path, treat_epsilons_specially,
                &a_compose_path_storage, &arc_map_a_path);
    ::Intersect(valid_b, valid_path, treat_epsilons_specially,
                &b_compose_path_storage, &arc_map_b_path);

    // TODO(haowen): we may need to implement a version of `ShortestDistance`
    // for non-top-sorted FSAs.
    // As the random path and a (or b) could be both non-epsilon-free, even if
    // a (or b) is top-sorted, we may still get non-top-sorted composed path.
    FsaCreator a_top_sorted_path_storage, b_top_sorted_path_storage;
    ::TopSort(a_compose_path_storage.GetFsa(), &a_top_sorted_path_storage);
    ::TopSort(b_compose_path_storage.GetFsa(), &b_top_sorted_path_storage);
    double cost_a = ShortestDistance<Type>(a_top_sorted_path_storage.GetFsa());
    double cost_b = ShortestDistance<Type>(b_top_sorted_path_storage.GetFsa());

    if (cost_a < loglike_cutoff_a && cost_b < loglike_cutoff_b) continue;

    if (!DoubleApproxEqual(cost_a, cost_b, delta)) return false;
  }
  return true;
}

// explicit instantiation here
template bool IsRandEquivalent<kMaxWeight>(const Fsa &a, const Fsa &b,
                                           float beam,
                                           bool treat_epsilons_specially,
                                           float delta, bool top_sorted,
                                           std::size_t npath);
template bool IsRandEquivalent<kLogSumWeight>(const Fsa &a, const Fsa &b,
                                              float beam,
                                              bool treat_epsilons_specially,
                                              float delta, bool top_sorted,
                                              std::size_t npath);

bool IsRandEquivalentAfterRmEpsPrunedLogSum(const Fsa &a, const Fsa &b,
                                            float beam,
                                            bool top_sorted /*= true*/,
                                            std::size_t npath /*= 100*/) {
  NVTX_RANGE(K2_FUNC);
  // TODO(haowen): the implementation may be not correct, we need to check this
  K2_LOG(FATAL)
      << "Don't call this function for now, the implementation may be "
         "incorrect";
  K2_CHECK_GT(beam, 0);
  FsaCreator connected_a_storage, connected_b_storage, valid_a_storage,
      valid_b_storage;
  std::vector<int32_t> connected_a_arc_map, connected_b_arc_map,
      valid_a_arc_map, valid_b_arc_map;
  ::Connect(a, &connected_a_storage, &connected_a_arc_map);
  ::Connect(b, &connected_b_storage, &connected_b_arc_map);
  ::ArcSort(connected_a_storage.GetFsa(), &valid_a_storage,
            &valid_a_arc_map);  // required by `intersect`
  ::ArcSort(connected_b_storage.GetFsa(), &valid_b_storage, &valid_b_arc_map);
  const auto &valid_a = valid_a_storage.GetFsa();
  const auto &valid_b = valid_b_storage.GetFsa();
  if (IsEmpty(valid_a) && IsEmpty(valid_b)) return true;
  if (IsEmpty(valid_a) || IsEmpty(valid_b)) return false;

  // Check that arc labels are compatible.
  std::unordered_set<int32_t> labels_a, labels_b, labels_difference;
  for (const auto &arc : valid_a) labels_a.insert(arc.label);
  for (const auto &arc : valid_b) labels_b.insert(arc.label);
  SetDifference(labels_a, labels_b, &labels_difference);
  if (labels_difference.size() >= 2 ||
      (labels_difference.size() == 1 &&
       (*(labels_difference.begin())) != kEpsilon))
    return false;
  // `b` is the FSA after epsilon-removal, so it should be epsilon-free
  if (labels_b.find(kEpsilon) != labels_b.end()) return false;

  double loglike_cutoff_a = ShortestDistance<kLogSumWeight>(valid_a) - beam;
  double loglike_cutoff_b = ShortestDistance<kLogSumWeight>(valid_b) - beam;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution coin(0.5);
  std::size_t n = 0;
  while (n < npath) {
    bool random_path_from_a = coin(gen);
    const auto &fsa = random_path_from_a ? valid_a : valid_b;
    FsaCreator valid_path_storage;
    // fail to generate a epsilon-free path (note that
    // our current implementation of `Intersection` requires
    // one of the two input FSAs must be epsilon-free).
    if (!::RandomPath(fsa, true, &valid_path_storage)) continue;
    // path is already connected so we will not call `::Connect` here
    ArcSort(&valid_path_storage.GetFsa());
    const auto &valid_path = valid_path_storage.GetFsa();

    FsaCreator a_compose_path_storage, b_compose_path_storage;
    std::vector<int32_t> arc_map_a_path, arc_map_b_path;
    // note that `valid_path` is epsilon-free
    ::Intersect(valid_a, valid_path, true, &a_compose_path_storage,
                &arc_map_a_path);
    ::Intersect(valid_b, valid_path, true, &b_compose_path_storage,
                &arc_map_b_path);

    // TODO(haowen): we may need to implement a version of `ShortestDistance`
    // for non-top-sorted FSAs, but we prefer to decide this later as there's no
    // such scenarios (input FSAs are not top-sorted) currently.
    K2_CHECK(top_sorted);
    double cost_a =
        ShortestDistance<kLogSumWeight>(a_compose_path_storage.GetFsa());
    double cost_b =
        ShortestDistance<kLogSumWeight>(b_compose_path_storage.GetFsa());
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

void RandPath::GetSizes(Array2Size<int32_t> *fsa_size) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_size, nullptr);
  fsa_size->size1 = fsa_size->size2 = 0;

  arc_indexes_.clear();
  arcs_.clear();
  arc_map_.clear();

  status_ = !IsEmpty(fsa_in_) && IsConnected(fsa_in_);
  if (!status_) return;

  int32_t num_states = fsa_in_.NumStates();
  std::vector<int32_t> state_map_in_to_out(num_states, -1);
  // `visited_arcs[i]` maps `arcs` leaving from state `i` in the output `path`
  // to arc-index in the input FSA.
  std::vector<std::unordered_map<Arc, int32_t, ArcHash>> visited_arcs;

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int32_t> distribution(0);

  int32_t num_visited_arcs = 0;
  int32_t num_visited_state = 0;
  int32_t state = 0;
  int32_t final_state = fsa_in_.FinalState();
  while (true) {
    if (state_map_in_to_out[state] == -1) {
      state_map_in_to_out[state] = num_visited_state;
      visited_arcs.emplace_back(std::unordered_map<Arc, int32_t, ArcHash>());
      ++num_visited_state;
    }
    if (state == final_state) break;
    const Arc *curr_arc = nullptr;
    int32_t arc_index_in = -1;
    int32_t tries = 0;
    do {
      int32_t begin = fsa_in_.indexes[state];
      int32_t end = fsa_in_.indexes[state + 1];
      // since `fsa_in_` is valid, so every state contains at least one arc.
      arc_index_in = begin + (distribution(generator) % (end - begin));
      curr_arc = &fsa_in_.data[arc_index_in];
      ++tries;
    } while (no_epsilon_arc_ && curr_arc->label == kEpsilon &&
             tries < eps_arc_tries_);
    if (no_epsilon_arc_ && curr_arc->label == kEpsilon &&
        tries >= eps_arc_tries_) {
      status_ = false;
      return;
    }
    int32_t state_id_out = state_map_in_to_out[state];
    if (visited_arcs[state_id_out]
            .insert({{state, curr_arc->dest_state, curr_arc->label,
                      curr_arc->weight},
                     arc_index_in - fsa_in_.indexes[0]})
            .second)
      ++num_visited_arcs;
    state = curr_arc->dest_state;
  }

  arc_indexes_.resize(num_visited_state);
  arcs_.resize(num_visited_arcs);
  arc_map_.resize(num_visited_arcs);
  int32_t n = 0;
  for (int32_t i = 0; i != num_visited_state; ++i) {
    arc_indexes_[i] = n;
    for (const auto &arc_with_index : visited_arcs[i]) {
      const auto &arc = arc_with_index.first;
      auto &output_arc = arcs_[n];
      output_arc.src_state = i;
      output_arc.dest_state = state_map_in_to_out[arc.dest_state];
      output_arc.label = arc.label;
      arc_map_[n] = arc_with_index.second;
      ++n;
    }
  }
  arc_indexes_.emplace_back(arc_indexes_.back());

  fsa_size->size1 = num_visited_state;
  fsa_size->size2 = num_visited_arcs;
}

bool RandPath::GetOutput(Fsa *fsa_out, int32_t *arc_map /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_out, nullptr);
  if (!status_) return false;

  // output fsa
  K2_CHECK_NE(fsa_out, nullptr);
  K2_CHECK_EQ(arc_indexes_.size(), fsa_out->size1 + 1);
  std::copy(arc_indexes_.begin(), arc_indexes_.end(), fsa_out->indexes);
  K2_CHECK_EQ(arcs_.size(), fsa_out->size2);
  std::copy(arcs_.begin(), arcs_.end(), fsa_out->data);

  // output arc map
  if (arc_map != nullptr) std::copy(arc_map_.begin(), arc_map_.end(), arc_map);

  return true;
}

}  // namespace k2host
