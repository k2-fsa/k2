// k2/csrc/fsa_util.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Haowen Qiu

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_util.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/properties.h"
#include "k2/csrc/util.h"

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
  CHECK_EQ(s.empty(), false);

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
    CHECK(ok);
    res.push_back(n);
  }
  return res;
}

// trim leading and trailing spaces of a string
void TrimString(std::string *s) {
  CHECK_NOTNULL(s);
  auto not_space = [](int c) { return std::isspace(c) == 0; };

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
  CHECK_NOTNULL(delim);
  CHECK_NOTNULL(out);
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

namespace k2 {

void GetEnteringArcs(const Fsa &fsa, std::vector<int32_t> *arc_index,
                     std::vector<int32_t> *end_index) {
  auto num_states = fsa.NumStates();
  std::vector<std::vector<int32_t>> vec(num_states);
  int32_t k = 0;
  for (const auto &arc : fsa.arcs) {
    auto dest_state = arc.dest_state;
    vec[dest_state].push_back(k);
    ++k;
  }
  arc_index->clear();
  end_index->clear();

  arc_index->reserve(fsa.arcs.size());
  end_index->reserve(num_states);

  for (const auto &indices : vec) {
    arc_index->insert(arc_index->end(), indices.begin(), indices.end());
    auto end = static_cast<int32_t>(arc_index->size());
    end_index->push_back(end);
  }
}

void GetArcWeights(const float *arc_weights_in,
                   const std::vector<std::vector<int32_t>> &arc_map,
                   float *arc_weights_out) {
  CHECK_NOTNULL(arc_weights_in);
  CHECK_NOTNULL(arc_weights_out);
  for (const auto &arcs : arc_map) {
    float sum_weights = 0.0f;
    for (auto arc : arcs) sum_weights += arc_weights_in[arc];
    *arc_weights_out++ = sum_weights;
  }
}

void GetArcWeights(const float *arc_weights_in,
                   const std::vector<int32_t> &arc_map,
                   float *arc_weights_out) {
  CHECK_NOTNULL(arc_weights_in);
  CHECK_NOTNULL(arc_weights_out);
  for (const auto &arc : arc_map) {
    *arc_weights_out++ = arc_weights_in[arc];
  }
}

void ReorderArcs(const std::vector<Arc> &arcs, Fsa *fsa,
                 std::vector<int32_t> *arc_map /*= nullptr*/) {
  CHECK_NOTNULL(fsa);
  fsa->arc_indexes.clear();
  fsa->arcs.clear();
  if (arc_map != nullptr) arc_map->clear();

  if (arcs.empty()) return;

  using ArcWithIndex = std::pair<Arc, int32_t>;
  int arc_id = 0;
  std::vector<std::vector<ArcWithIndex>> vec;
  for (const auto &arc : arcs) {
    auto src_state = arc.src_state;
    auto dest_state = arc.dest_state;
    auto new_size = std::max(src_state, dest_state);
    if (new_size >= vec.size()) vec.resize(new_size + 1);
    vec[src_state].push_back({arc, arc_id++});
  }

  std::size_t num_states = vec.size();
  fsa->arc_indexes.resize(num_states + 1);
  fsa->arcs.reserve(arcs.size());
  std::vector<int32_t> arc_map_out;
  arc_map_out.reserve(arcs.size());

  for (auto i = 0; i != num_states; ++i) {
    fsa->arc_indexes[i] = static_cast<int32_t>(fsa->arcs.size());
    for (auto arc_with_index : vec[i]) {
      fsa->arcs.emplace_back(arc_with_index.first);
      arc_map_out.push_back(arc_with_index.second);
    }
  }
  fsa->arc_indexes.back() = static_cast<int32_t>(fsa->arcs.size());
  if (arc_map != nullptr) arc_map->swap(arc_map_out);
}

void Swap(Fsa *a, Fsa *b) {
  CHECK_NOTNULL(a);
  CHECK_NOTNULL(b);
  std::swap(a->arc_indexes, b->arc_indexes);
  std::swap(a->arcs, b->arcs);
}

std::unique_ptr<Fsa> StringToFsa(const std::string &s) {
  static constexpr const char *kDelim = " \t";

  std::istringstream is(s);
  std::string line;
  std::vector<std::vector<Arc>> vec;  // index is state number
  bool finished = false;  // when the final state is read, set it to true.
  int32_t num_arcs = 0;
  while (std::getline(is, line)) {
    std::vector<std::string> splits;
    SplitStringToVector(line, kDelim, &splits);
    if (splits.empty()) continue;

    CHECK_EQ(finished, false);

    auto fields = StringVectorToIntVector(splits);
    auto num_fields = fields.size();
    if (num_fields == 3u) {
      Arc arc{};
      arc.src_state = fields[0];
      arc.dest_state = fields[1];
      arc.label = fields[2];

      auto new_size = std::max(arc.src_state, arc.dest_state);
      if (new_size >= vec.size()) vec.resize(new_size + 1);

      vec[arc.src_state].push_back(arc);
      ++num_arcs;
    } else if (num_fields == 1u) {
      finished = true;
      CHECK_EQ(fields[0] + 1, static_cast<int32_t>(vec.size()));
    } else {
      LOG(FATAL) << "invalid line: " << line;
    }
  }

  CHECK_EQ(finished, true) << "The last line should be the final state";
  CHECK_GT(num_arcs, 0) << "An empty fsa is detected!";

  std::unique_ptr<Fsa> fsa(new Fsa);
  fsa->arc_indexes.resize(vec.size());
  fsa->arcs.reserve(num_arcs);
  int32_t i = 0;
  for (const auto &v : vec) {
    fsa->arc_indexes[i] = (static_cast<int32_t>(fsa->arcs.size()));
    fsa->arcs.insert(fsa->arcs.end(), v.begin(), v.end());
    ++i;
  }
  fsa->arc_indexes.emplace_back(fsa->arc_indexes.back());
  return fsa;
}

std::string FsaToString(const Fsa &fsa) {
  if (IsEmpty(fsa)) return "";

  static constexpr const char *kSep = " ";
  std::ostringstream os;

  for (const auto &arc : fsa.arcs) {
    os << arc.src_state << kSep << arc.dest_state << kSep << arc.label << "\n";
  }
  os << fsa.NumStates() - 1 << "\n";
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
}

void GenerateRandFsa(const RandFsaOptions &opts, Fsa *fsa) {
  CHECK_NOTNULL(fsa);
  CHECK_GT(opts.num_syms, 1);
  CHECK_GT(opts.num_states, 1);
  CHECK_GT(opts.num_arcs, 1);

  RandInt rand(opts.seed);

  // index is state_id
  std::vector<std::vector<Arc>> state_to_arcs(opts.num_states);
  int32_t src_state;
  int32_t dest_state;
  int32_t label;
  auto num_states = static_cast<int32_t>(opts.num_states);

  int32_t num_fails = -1;
  int32_t max_loops = 100 * opts.num_arcs;
  do {
    ++num_fails;
    if (num_fails > 100)
      LOG(FATAL) << "Cannot generate a rand fsa. Please increase num_states "
                    "and num_arcs";

    std::unordered_set<std::pair<int32_t, int32_t>, PairHash> seen;
    int32_t tried = 0;
    for (auto i = 0;
         i != static_cast<int32_t>(opts.num_arcs) && tried < max_loops;
         ++tried) {
      src_state = rand(0, num_states - 2);
      if (!opts.acyclic)
        dest_state = rand(0, num_states - 1);
      else
        dest_state = rand(src_state + 1, num_states - 1);

      if (seen.count(std::make_pair(src_state, dest_state)) != 0) continue;

      seen.insert(std::make_pair(src_state, dest_state));

      if (dest_state == num_states - 1)
        label = kFinalSymbol;
      else
        label = rand(0, static_cast<int32_t>(opts.num_syms - 1));

      state_to_arcs[src_state].emplace_back(src_state, dest_state, label);
      ++i;
    }

    Fsa tmp;
    tmp.arc_indexes.reserve(opts.num_states + 1);
    tmp.arcs.reserve(opts.num_arcs);

    for (const auto &arcs : state_to_arcs) {
      tmp.arc_indexes.push_back(static_cast<int32_t>(tmp.arcs.size()));
      tmp.arcs.insert(tmp.arcs.end(), arcs.begin(), arcs.end());
    }

    tmp.arc_indexes.push_back(tmp.arc_indexes.back());

    Connect(tmp, fsa);
  } while (!opts.allow_empty && IsEmpty(*fsa));

  if (opts.acyclic) CHECK(IsAcyclic(*fsa));
}

}  // namespace k2
