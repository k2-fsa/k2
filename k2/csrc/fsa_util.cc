// k2/csrc/fsa_util.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_util.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <utility>
#include <vector>

#include "glog/logging.h"

namespace {
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

  int32_t res = static_cast<int32_t>(n);
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
  auto not_space = [](int c) { return std::isspace(c) == false; };

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
  size_t start = 0;
  size_t end = in.size();
  while (true) {
    auto pos = in.find_first_of(delim, start);
    if (pos == in.npos) break;

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
      Arc arc;
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

}  // namespace k2
