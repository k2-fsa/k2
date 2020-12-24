/**
 * @brief
 * fsa_util
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/fsa_util.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <random>
#include <sstream>
#include <stack>
#include <unordered_set>
#include <utility>
#include <vector>

#include "k2/csrc/host/connect.h"
#include "k2/csrc/host/properties.h"
#include "k2/csrc/host/util.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace {

// Generate a uniformly distributed random variable of type int32_t.
class RandInt {
 public:
  // Set `seed` to non-zero for reproducibility.
  explicit RandInt(int32_t seed = 0) : gen_(rd_()) {
    if (seed != 0) gen_.seed(seed);
  }

  // Get the next random number on the **closed** interval [low, high]
  int32_t operator()(int32_t low = std::numeric_limits<int32_t>::min(),
                     int32_t high = std::numeric_limits<int32_t>::max()) {
    std::uniform_int_distribution<int32_t> dis(low, high);
    return dis(gen_);
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
};

/** Convert a string to an integer.

  @param [in]   s     The input string.
  @param [out]  is    If non-null, it is set to true when the conversion
                      is successful; false otherwise
  @return The converted integer.
 */
int32_t StringToInt(const std::string &s, bool *is_ok = nullptr) {
  K2_CHECK_EQ(s.empty(), false);

  bool ok = false;
  char *p = nullptr;
  // std::strtol requires a `long` type
  long n = std::strtol(s.c_str(), &p, 10);  // NOLINT
  if (*p == '\0') ok = true;

  auto res = static_cast<int32_t>(n);
  if (n != res) ok = false;  // out of range
  if (is_ok != nullptr) *is_ok = ok;

  return res;
}

/** Convert `std::vector<std::string>` to `std::vector<int32_t>`.
 */
std::vector<int32_t> StringVectorToIntVector(
    const std::vector<std::string> &in) {
  std::vector<int32_t> res;
  res.reserve(in.size());
  for (const auto &s : in) {
    bool ok = false;
    auto n = StringToInt(s, &ok);
    K2_CHECK(ok);
    res.push_back(n);
  }
  return res;
}

// trim leading and trailing spaces of a string
void TrimString(std::string *s) {
  K2_CHECK_NE(s, nullptr);
  auto not_space = [](int32_t c) { return std::isspace(c) == 0; };

  s->erase(s->begin(), std::find_if(s->begin(), s->end(), not_space));
  s->erase(std::find_if(s->rbegin(), s->rend(), not_space).base(), s->end());
}

/** split a string to a vector of strings using a set of delimiters.

   Example usage:

   @code
    std::string in = "1 2 3";
    const char *delim = " \t";
    std::vector<std::string> out;
    SplitStringToVector(in, delim, &out);
   @endcode

   @param [in]  in    The input string to be split.
   @param [in]  delim A string of delimiters.
   @param [out] out   It saves the split result.
*/
void SplitStringToVector(const std::string &in, const char *delim,
                         std::vector<std::string> *out) {
  K2_CHECK_NE(delim, nullptr);
  K2_CHECK_NE(out, nullptr);
  out->clear();
  std::size_t start = 0;
  while (true) {
    auto pos = in.find_first_of(delim, start);
    if (pos == std::string::npos) break;

    auto sub = in.substr(start, pos - start);
    start = pos + 1;

    TrimString(&sub);
    if (!sub.empty()) out->emplace_back(std::move(sub));
  }

  if (start < in.size()) {
    auto sub = in.substr(start);
    TrimString(&sub);
    if (!sub.empty()) out->emplace_back(std::move(sub));
  }
}

}  // namespace

namespace k2host {

void GetEnteringArcs(const Fsa &fsa, Array2<int32_t *, int32_t> *arc_indexes) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(arc_indexes, nullptr);
  K2_CHECK_EQ(arc_indexes->size1, fsa.size1);
  K2_CHECK_EQ(arc_indexes->size2, fsa.size2);

  auto num_states = fsa.NumStates();
  std::vector<std::vector<int32_t>> vec(num_states);
  int32_t k = 0;
  for (const auto &arc : fsa) {
    auto dest_state = arc.dest_state;
    vec[dest_state].push_back(k);
    ++k;
  }

  auto indexes = arc_indexes->indexes;
  auto data = arc_indexes->data;
  int32_t curr_state = 0;
  int32_t num_arcs = 0;
  for (const auto &indices : vec) {
    indexes[curr_state++] = num_arcs;
    std::copy(indices.begin(), indices.end(), data + num_arcs);
    num_arcs += indices.size();
  }
  K2_CHECK_EQ(curr_state, num_states);
  K2_CHECK_EQ(num_arcs, fsa.size2);
  indexes[curr_state] = num_arcs;
}

void GetArcWeights(const float *arc_weights_in,
                   const Array2<int32_t *, int32_t> &arc_map,
                   float *arc_weights_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(arc_weights_in, nullptr);
  K2_CHECK_NE(arc_weights_out, nullptr);
  for (int32_t i = 0; i != arc_map.size1; ++i) {
    float sum_weights = 0.0f;
    for (int32_t j = arc_map.indexes[i]; j != arc_map.indexes[i + 1]; ++j) {
      int32_t arc_index_in = arc_map.data[j];
      sum_weights += arc_weights_in[arc_index_in];
    }
    *arc_weights_out++ = sum_weights;
  }
}

void GetArcWeights(const float *arc_weights_in, const int32_t *arc_map,
                   int32_t num_arcs, float *arc_weights_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(arc_weights_in, nullptr);
  K2_CHECK_NE(arc_weights_out, nullptr);
  for (int32_t i = 0; i != num_arcs; ++i) {
    *arc_weights_out++ = arc_weights_in[arc_map[i]];
  }
}

void ReorderArcs(const std::vector<Arc> &arcs, Fsa *fsa,
                 std::vector<int32_t> *arc_map /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa, nullptr);
  if (arc_map != nullptr) arc_map->clear();

  // as fsa has been initialized (fsa.size1 = 0 && fsa.size2 == 0),
  // we don't need to do anything here.
  if (arcs.empty()) return;

  using ArcWithIndex = std::pair<Arc, int32_t>;
  int32_t arc_id = 0;
  std::size_t num_states = fsa->size1;
  std::vector<std::vector<ArcWithIndex>> vec(num_states);
  for (const auto &arc : arcs) {
    auto src_state = arc.src_state;
    auto dest_state = arc.dest_state;
    K2_CHECK_LT(src_state, num_states);
    K2_CHECK_LT(dest_state, num_states);
    vec[src_state].push_back({arc, arc_id++});
  }

  std::vector<int32_t> arc_map_out;
  arc_map_out.reserve(arcs.size());

  int32_t num_arcs = 0;
  for (auto i = 0; i != num_states; ++i) {
    fsa->indexes[i] = num_arcs;
    for (auto arc_with_index : vec[i]) {
      fsa->data[num_arcs++] = arc_with_index.first;
      arc_map_out.push_back(arc_with_index.second);
    }
  }
  fsa->indexes[num_states] = num_arcs;
  if (arc_map != nullptr) arc_map->swap(arc_map_out);
}

void ConvertIndexes1(const int32_t *arc_map, int32_t num_arcs,
                     int64_t *indexes_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(arc_map, nullptr);
  K2_CHECK_GE(num_arcs, 0);
  K2_CHECK_NE(indexes_out, nullptr);
  std::copy(arc_map, arc_map + num_arcs, indexes_out);
}

void GetArcIndexes2(const Array2<int32_t *, int32_t> &arc_map,
                    int64_t *indexes1, int64_t *indexes2) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(indexes1, nullptr);
  K2_CHECK_NE(indexes2, nullptr);
  std::copy(arc_map.data + arc_map.indexes[0],
            arc_map.data + arc_map.indexes[arc_map.size1], indexes1);
  int32_t num_arcs = 0;
  for (int32_t i = 0; i != arc_map.size1; ++i) {
    int32_t curr_arc_mappings = arc_map.indexes[i + 1] - arc_map.indexes[i];
    std::fill_n(indexes2 + num_arcs, curr_arc_mappings, i);
    num_arcs += curr_arc_mappings;
  }
}

void StringToFsa::GetSizes(Array2Size<int32_t> *fsa_size) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_size, nullptr);
  fsa_size->size1 = fsa_size->size2 = 0;

  static constexpr const char *kDelim = " \t";
  std::istringstream is(s_);
  std::string line;
  bool finished = false;  // when the final state is read, set it to true.
  int32_t num_arcs = 0;
  while (std::getline(is, line)) {
    std::vector<std::string> splits;
    SplitStringToVector(line, kDelim, &splits);
    if (splits.empty()) continue;

    K2_CHECK_EQ(finished, false);

    auto fields = StringVectorToIntVector(splits);
    auto num_fields = fields.size();
    if (num_fields == 3u) {
      Arc arc{};
      arc.src_state = fields[0];
      arc.dest_state = fields[1];
      arc.label = fields[2];

      auto new_size = std::max(arc.src_state, arc.dest_state);
      if (new_size >= arcs_.size()) arcs_.resize(new_size + 1);

      arcs_[arc.src_state].push_back(arc);
      ++num_arcs;
    } else if (num_fields == 1u) {
      finished = true;
      K2_CHECK_EQ(fields[0] + 1, static_cast<int32_t>(arcs_.size()));
    } else {
      K2_LOG(FATAL) << "invalid line: " << line;
    }
  }

  K2_CHECK_EQ(finished, true) << "The last line should be the final state";
  K2_CHECK_GT(num_arcs, 0) << "An empty fsa is detected!";

  fsa_size->size1 = static_cast<int32_t>(arcs_.size());
  fsa_size->size2 = num_arcs;
}

void StringToFsa::GetOutput(Fsa *fsa_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_out, nullptr);
  K2_CHECK_EQ(fsa_out->size1, arcs_.size());

  int32_t num_arcs = 0;
  for (auto i = 0; i != fsa_out->size1; ++i) {
    fsa_out->indexes[i] = num_arcs;
    std::copy(arcs_[i].begin(), arcs_[i].end(), fsa_out->data + num_arcs);
    num_arcs += arcs_[i].size();
  }
  fsa_out->indexes[fsa_out->size1] = num_arcs;
}

std::string FsaToString(const Fsa &fsa) {
  NVTX_RANGE(K2_FUNC);
  if (IsEmpty(fsa)) return "";

  static constexpr const char *kSep = " ";
  std::ostringstream os;

  for (const auto &arc : fsa) {
    os << arc.src_state << kSep << arc.dest_state << kSep << arc.label << kSep
       << arc.weight << "\n";
  }
  os << fsa.FinalState() << "\n";
  return os.str();
}

RandFsaOptions::RandFsaOptions() {
  RandInt rand;
  num_syms = 2 + rand(1) % 5;
  num_states = 3 + rand(1) % 10;
  num_arcs = 5 + rand(1) % 30;
  allow_empty = true;
  acyclic = false;
  seed = 0;
  nonzero_weights = false;
}

void RandFsaGenerator::GetSizes(Array2Size<int32_t> *fsa_size) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_size, nullptr);
  fsa_size->size1 = fsa_size->size2 = 0;

  K2_CHECK_GT(opts_.num_syms, 1);
  K2_CHECK_GT(opts_.num_states, 1);
  K2_CHECK_GT(opts_.num_arcs, 1);

  RandInt rand(opts_.seed);

  // index is state_id
  std::vector<std::vector<Arc>> state_to_arcs(opts_.num_states);
  int32_t src_state;
  int32_t dest_state;
  int32_t label;
  auto num_states = static_cast<int32_t>(opts_.num_states);

  int32_t num_fails = -1;
  int32_t max_loops = 100 * opts_.num_arcs;
  do {
    ++num_fails;
    if (num_fails > 100)
      K2_LOG(FATAL) << "Cannot generate a rand fsa. Please increase num_states "
                       "and num_arcs";

    std::unordered_set<std::pair<int32_t, int32_t>, PairHash> seen;
    int32_t tried = 0;
    for (auto i = 0;
         i != static_cast<int32_t>(opts_.num_arcs) && tried < max_loops;
         ++tried) {
      src_state = rand(0, num_states - 2);
      if (!opts_.acyclic)
        dest_state = rand(0, num_states - 1);
      else
        dest_state = rand(src_state + 1, num_states - 1);

      if (seen.count(std::make_pair(src_state, dest_state)) != 0) continue;

      seen.insert(std::make_pair(src_state, dest_state));

      if (dest_state == num_states - 1)
        label = kFinalSymbol;
      else
        label = rand(0, static_cast<int32_t>(opts_.num_syms - 1));

      float weight =
          static_cast<float>(opts_.nonzero_weights ? rand(0, 10) * 0.5 : 0.0);

      state_to_arcs[src_state].emplace_back(src_state, dest_state, label,
                                            weight);
      ++i;
    }

    std::vector<Arc> tmp_arcs;
    for (const auto &arcs : state_to_arcs) {
      tmp_arcs.insert(tmp_arcs.end(), arcs.begin(), arcs.end());
    }
    FsaCreator tmp_creator(tmp_arcs, num_states - 1);
    auto &tmp_fsa = tmp_creator.GetFsa();

    Connection connection(tmp_fsa);
    Array2Size<int32_t> out_fsa_size;
    connection.GetSizes(&out_fsa_size);

    fsa_creator_.Init(out_fsa_size);
    connection.GetOutput(&fsa_creator_.GetFsa());
  } while (!opts_.allow_empty && IsEmpty(fsa_creator_.GetFsa()));

  if (opts_.acyclic) K2_CHECK(IsAcyclic(fsa_creator_.GetFsa()));

  const auto &generated_fsa = fsa_creator_.GetFsa();
  fsa_size->size1 = generated_fsa.size1;
  fsa_size->size2 = generated_fsa.size2;
}

void RandFsaGenerator::GetOutput(Fsa *fsa_out) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_out, nullptr);

  const auto &fsa = fsa_creator_.GetFsa();
  K2_CHECK_EQ(fsa_out->size1, fsa.size1);
  K2_CHECK_EQ(fsa_out->size2, fsa.size2);
  std::copy(fsa.indexes, fsa.indexes + fsa.size1 + 1, fsa_out->indexes);
  std::copy(fsa.data, fsa.data + fsa.size2, fsa_out->data);
}

void CreateTopSortedFsa(const std::vector<Arc> &arcs, Fsa *fsa,
                        std::vector<int32_t> *arc_map /*=null_ptr*/) {
  NVTX_RANGE(K2_FUNC);
  using dfs::DfsState;
  using dfs::kNotVisited;
  using dfs::kVisited;
  using dfs::kVisiting;
  K2_CHECK_NE(fsa, nullptr);
  if (arcs.empty()) return;

  using ArcWithIndex = std::pair<Arc, int32_t>;
  int32_t arc_id = 0;
  std::vector<std::vector<ArcWithIndex>> vec;
  for (const auto &arc : arcs) {
    auto src_state = arc.src_state;
    auto dest_state = arc.dest_state;
    auto new_size = std::max(src_state, dest_state);
    if (new_size >= vec.size()) vec.resize(new_size + 1);
    vec[src_state].push_back({arc, arc_id++});
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

      const auto &arc = vec[state][current_state.arc_begin].first;
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
          K2_LOG(FATAL) << "there is a cycle: " << state << " -> "
                        << next_state;
          break;
        case kVisited:
          ++current_state.arc_begin;
          break;
        default:
          K2_LOG(FATAL) << "Unreachable code is executed!";
          break;
      }
    }
  }

  K2_CHECK_EQ(num_states, static_cast<int32_t>(order.size()));
  std::reverse(order.begin(), order.end());

  K2_CHECK_EQ(fsa->size1, num_states);
  K2_CHECK_EQ(fsa->size2, arcs.size());
  std::vector<int32_t> arc_map_out;
  arc_map_out.reserve(arcs.size());

  std::vector<int32_t> old_to_new(num_states);
  for (auto i = 0; i != num_states; ++i) old_to_new[order[i]] = i;

  int32_t num_arcs = 0;
  for (auto i = 0; i != num_states; ++i) {
    auto old_state = order[i];
    fsa->indexes[i] = num_arcs;
    for (auto arc_with_index : vec[old_state]) {
      auto &arc = arc_with_index.first;
      arc.src_state = i;
      arc.dest_state = old_to_new[arc.dest_state];
      fsa->data[num_arcs++] = arc;
      arc_map_out.push_back(arc_with_index.second);
    }
  }
  fsa->indexes[num_states] = num_arcs;
  if (arc_map != nullptr) arc_map->swap(arc_map_out);
}

void CreateFsa(const std::vector<Arc> &arcs, Fsa *fsa,
               std::vector<int32_t> *arc_map /*=null_ptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa, nullptr);
  if (arcs.empty()) return;

  int32_t old_final_state = -1;
  using ArcWithIndex = std::pair<Arc, int32_t>;
  int32_t arc_id = 0;
  std::vector<std::vector<ArcWithIndex>> state_to_arcs;  // indexed by states
  for (const auto &arc : arcs) {
    if (arc.label == -1) {
      // if old_final_state != -1, there must be arc.dest_state ==
      // old_final_state, as we suppose there is only one final state.
      K2_CHECK(old_final_state == -1 || arc.dest_state == old_final_state);
      old_final_state = arc.dest_state;
    }
    int32_t src_state = arc.src_state;
    int32_t dest_state = arc.dest_state;
    int32_t new_size = std::max(src_state, dest_state);
    if (new_size >= state_to_arcs.size()) state_to_arcs.resize(new_size + 1);
    state_to_arcs[src_state].push_back({arc, arc_id++});
  }

  int32_t num_states = static_cast<int32_t>(state_to_arcs.size());
  K2_CHECK_EQ(fsa->size1, num_states);
  K2_CHECK_EQ(fsa->size2, arcs.size());
  // there's no leaving arc from final state
  K2_CHECK_EQ(state_to_arcs[old_final_state].size(), 0);

  int32_t final_state = num_states - 1;
  std::vector<int32_t> new_to_old(num_states);
  // we move old_final_state to the end so that final state will be
  // the largest state.
  for (int32_t i = 0; i < old_final_state; ++i) new_to_old[i] = i;
  for (int32_t i = old_final_state; i < final_state; ++i) new_to_old[i] = i + 1;
  new_to_old[final_state] = old_final_state;
  std::vector<int32_t> old_to_new(num_states);
  for (int32_t i = 0; i != num_states; ++i) old_to_new[new_to_old[i]] = i;

  std::vector<int32_t> arc_map_out;
  arc_map_out.reserve(arcs.size());

  int32_t num_arcs = 0;
  for (int32_t i = 0; i != num_states; ++i) {
    int32_t old_state = new_to_old[i];
    fsa->indexes[i] = num_arcs;
    for (auto arc_with_index : state_to_arcs[old_state]) {
      auto &arc = arc_with_index.first;
      arc.src_state = i;
      arc.dest_state = old_to_new[arc.dest_state];
      fsa->data[num_arcs++] = arc;
      arc_map_out.push_back(arc_with_index.second);
    }
  }

  fsa->indexes[num_states] = num_arcs;
  if (arc_map != nullptr) arc_map->swap(arc_map_out);
}

}  // namespace k2host
