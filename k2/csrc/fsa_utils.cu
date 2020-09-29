/**
 * @brief Utilities for reading, writing and creating FSAs.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <sstream>
#include <utility>
#include <vector>

#include "k2/csrc/context.h"
#include "k2/csrc/fsa_utils.h"

namespace k2 {

// field separator within a line for a text form FSA
static constexpr const char *kDelim = " \t";

// Convert a string to an integer. Abort the program on failure.
static int32_t StringToInt(const std::string &s) {
  K2_CHECK(!s.empty());

  bool ok = false;
  char *p = nullptr;
  // std::strtol requires a `long` type
  long n = std::strtol(s.c_str(), &p, 10);  // NOLINT
  if (*p == '\0') ok = true;

  auto res = static_cast<int32_t>(n);
  if (n != res) ok = false;  // out of range

  K2_CHECK(ok) << "Failed to convert " << s << " to an integer";

  return res;
}

// Convert a string to a float. Abort the program on failure.
static float StringToFloat(const std::string &s) {
  K2_CHECK(!s.empty());
  char *p = nullptr;
  float f = std::strtof(s.c_str(), &p);
  if (*p != '\0') K2_LOG(FATAL) << "Failed to convert " << s << " to a float";
  return f;
}

// Trim leading and trailing spaces of a string.
static void TrimString(std::string *s) {
  K2_CHECK_NE(s, nullptr);
  auto not_space = [](int32_t c) -> bool { return std::isspace(c) == 0; };

  s->erase(s->begin(), std::find_if(s->begin(), s->end(), not_space));
  s->erase(std::find_if(s->rbegin(), s->rend(), not_space).base(), s->end());
}

/* Split a string to a vector of strings using a set of delimiters.

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
static void SplitStringToVector(const std::string &in, const char *delim,
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

/* Check that arcs are valid.

   It will abort the program if there is an invalid arc.
   An arc is invalid if one of the following happens:

    - its symbol is -1, but its dest state is not final state.
    - its dest state is larger than final state

  @param [in] state_arcs   Indexed by state numbers.
  @param [in] final_state  The final state.
 */
static void CheckStateArcs(const std::vector<std::vector<Arc>> &state_arcs,
                           int32_t final_state) {
  int32_t num_states = static_cast<int32_t>(state_arcs.size());
  K2_CHECK_EQ(num_states, final_state + 1);
  K2_CHECK(!state_arcs[final_state].empty());

  for (int32_t i = 0; i != num_states; ++i) {
    const auto &arcs = state_arcs[i];
    for (const auto &arc : arcs) {
      K2_CHECK_EQ(arc.src_state, i);
      K2_CHECK_LE(arc.dest_state, final_state);
      if (arc.symbol == -1) K2_CHECK_EQ(arc.dest_state, final_state);
    }
  }
}

/* Build an Fsa from a list of lists of arcs.

  @param [in] state_arcs Indexed by states.
  @param [in] num arcs   It is the number of arcs in total.
                         Provided for optimization.
 */
static Fsa AcceptorFromStateArcs(
    const std::vector<std::vector<Arc>> &state_arcs, int32_t num_arcs) {
  int32_t num_states = static_cast<int32_t>(state_arcs.size());
  std::vector<int32_t> row_splits(num_states + 1);
  std::vector<Arc> arcs;
  arcs.reserve(num_arcs);

  row_splits[0] = 0;
  for (int32_t i = 1; i != num_states; ++i) {
    row_splits[i] =
        row_splits[i - 1] + static_cast<int32_t>(state_arcs[i - 1].size());
    arcs.insert(arcs.end(), state_arcs[i - 1].begin(), state_arcs[i - 1].end());
  }
  // the final_state has no leaving arcs, so there is nothing
  // to add to `arcs` and row_splits[num_states]
  row_splits[num_states] = row_splits[num_states - 1];

  auto cpu_context = GetCpuContext();
  Array1<int32_t> _row_splits(cpu_context, row_splits);
  std::vector<RaggedShapeDim> axes(1);
  axes[0].row_splits = _row_splits;
  axes[0].cached_tot_size = -1;

  RaggedShape shape(axes);
  Array1<Arc> values(cpu_context, arcs);

  return Fsa(shape, values);
}

// Create an acceptor from a stream.
static Fsa AcceptorFromStream(std::string first_line, std::istringstream &is,
                              bool negate_scores) {
  std::vector<std::vector<Arc>> state_arcs;  // indexed by states
  std::string line = std::move(first_line);
  std::vector<std::string> splits;

  float scale = 1;
  if (negate_scores) scale = -1;

  int32_t final_state = -1;  // invalid state
  int32_t num_arcs = 0;
  do {
    SplitStringToVector(line, kDelim,
                        &splits);  // splits is cleared in the function
    if (splits.empty()) continue;  // this is an empty line

    auto num_fields = splits.size();

    Arc arc{};  // it has a trivial destructor
    if (num_fields == 4u) {
      //   0            1          2      3
      // src_state  dest_state  symbol  score
      ++num_arcs;
      arc.src_state = StringToInt(splits[0]);
      arc.dest_state = StringToInt(splits[1]);
      arc.symbol = StringToInt(splits[2]);
      arc.score = scale * StringToFloat(splits[3]);
    } else if (num_fields == 1u) {
      final_state = StringToInt(splits[0]);
      if (static_cast<int32_t>(state_arcs.size()) <= final_state)
        state_arcs.resize(final_state + 1);
      break;  // finish reading
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nIt expects a line with 4 fields";
    }

    if (static_cast<int32_t>(state_arcs.size()) <= arc.src_state)
      state_arcs.resize(arc.src_state + 1);

    state_arcs[arc.src_state].emplace_back(arc);
  } while (std::getline(is, line));

  K2_CHECK(is) << "Failed to read";
  K2_CHECK_NE(final_state, -1) << "Found no final_state!";

  CheckStateArcs(state_arcs, final_state);
  return AcceptorFromStateArcs(state_arcs, num_arcs);
}

static Fsa TransducerFromStream(std::string first_line, std::istringstream &is,
                                bool negate_scores,
                                Array1<int32_t> *aux_labels) {
  K2_CHECK(aux_labels != nullptr);

  std::vector<std::vector<int32_t>> state_aux_labels;  // indexed by states
  std::vector<std::vector<Arc>> state_arcs;            // indexed by states
  std::string line = std::move(first_line);
  std::vector<std::string> splits;

  float scale = 1;
  if (negate_scores) scale = -1;

  int32_t final_state = -1;  // invalid state
  int32_t num_arcs = 0;
  do {
    SplitStringToVector(line, kDelim,
                        &splits);  // splits is cleared in the function
    if (splits.empty()) continue;  // this is an empty line
    auto num_fields = splits.size();
    Arc arc{};
    if (num_fields == 5u) {
      //   0           1         2         3        4
      // src_state  dest_state  symbol  aux_label score
      ++num_arcs;
      arc.src_state = StringToInt(splits[0]);
      arc.dest_state = StringToInt(splits[1]);
      arc.symbol = StringToInt(splits[2]);
      arc.score = scale * StringToFloat(splits[4]);
    } else if (num_fields == 1u) {
      final_state = StringToInt(splits[0]);
      if (static_cast<int32_t>(state_arcs.size()) <= final_state)
        state_arcs.resize(final_state + 1);
      state_aux_labels.resize(final_state + 1);
      break;  // finish reading
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nIt expects a line with 5 fields";
    }
    if (static_cast<int32_t>(state_arcs.size()) <= arc.src_state) {
      state_arcs.resize(arc.src_state + 1);
      state_aux_labels.resize(arc.src_state + 1);
    }

    state_arcs[arc.src_state].emplace_back(arc);
    state_aux_labels[arc.src_state].push_back(StringToInt(splits[3]));
  } while (std::getline(is, line));

  K2_CHECK(is) << "Failed to read";
  K2_CHECK_NE(final_state, -1) << "Found no final_state!";

  std::vector<int32_t> labels;
  labels.reserve(num_arcs);
  for (const auto &v : state_aux_labels) {
    labels.insert(labels.end(), v.begin(), v.end());
  }

  *aux_labels = Array1<int32_t>(GetCpuContext(), labels);

  CheckStateArcs(state_arcs, final_state);
  return AcceptorFromStateArcs(state_arcs, num_arcs);
}

Fsa FsaFromString(const std::string &s, bool negate_scores /*= false*/,
                  Array1<int32_t> *aux_labels /*= nullptr*/) {
  std::istringstream is(s);
  std::string line;
  std::getline(is, line);
  K2_CHECK(is);

  std::vector<std::string> splits;
  SplitStringToVector(line, kDelim, &splits);
  auto num_fields = splits.size();
  if (num_fields == 4u)
    return AcceptorFromStream(std::move(line), is, negate_scores);
  else if (num_fields == 5u)
    return TransducerFromStream(std::move(line), is, negate_scores, aux_labels);

  K2_LOG(FATAL) << "Expected number of fields: 4 or 5."
                << "Actual: " << num_fields << "\n"
                << "First line is: " << line;

  return Fsa();  // unreachable code
}

std::string FsaToString(const Fsa &fsa, bool negate_scores /*= false*/,
                        const Array1<int32_t> *aux_labels /*= nullptr*/) {
  K2_CHECK_EQ(fsa.NumAxes(), 2);
  K2_CHECK_EQ(fsa.Context()->GetDeviceType(), kCpu);
  const Array1<int32_t> &row_splits = fsa.shape.RowSplits(1);
  const Array1<Arc> &arcs = fsa.values;

  const int32_t *p = nullptr;
  if (aux_labels != nullptr) {
    K2_CHECK(IsCompatible(fsa, *aux_labels));
    K2_CHECK_EQ(aux_labels->Dim(), arcs.Dim());
    p = aux_labels->Data();
  }
  float scale = 1;
  if (negate_scores) scale = -1;

  std::ostringstream os;

  int32_t n = arcs.Dim();
  char sep = ' ';
  char line_sep = '\n';
  for (int32_t i = 0; i != n; ++i) {
    const auto &arc = arcs[i];
    os << arc.src_state << sep << arc.dest_state << sep << arc.symbol << sep;
    if (p != nullptr) os << p[i] << sep;
    os << (scale * arc.score) << line_sep;
  }
  os << (fsa.shape.Dim0() - 1) << line_sep;
  return os.str();
}

}  // namespace k2
