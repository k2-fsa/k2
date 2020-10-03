/**
 * @brief Utilities for reading, writing and creating FSAs.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *                      Guoguo Chen
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <algorithm>
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

// Create an acceptor from a stream.
static Fsa AcceptorFromStream(std::string first_line, std::istringstream &is,
                              bool openfst) {
  std::vector<Arc> arcs;
  std::string line = std::move(first_line);
  std::vector<std::string> splits;

  float scale = 1;
  if (openfst) scale = -1;

  int32_t max_state = -1;
  std::vector<int32_t> original_final_states;
  std::vector<float> original_final_weights;
  do {
    SplitStringToVector(line, kDelim,
                        &splits);  // splits is cleared in the function
    if (splits.empty()) continue;  // this is an empty line

    auto num_fields = splits.size();

    if (num_fields == 4u) {
      //   0            1          2      3
      // src_state  dest_state  symbol  score
      int32_t src_state = StringToInt(splits[0]);
      int32_t dest_state = StringToInt(splits[1]);
      int32_t symbol = StringToInt(splits[2]);
      float score = scale * StringToFloat(splits[3]);
      arcs.emplace_back(src_state, dest_state, symbol, score);
      max_state = std::max(max_state, std::max(src_state, dest_state));
    } else if (num_fields == 2u) {
      //   0            1
      // final_state  score
      // In this case, openfst is true. There could be multiple final states, so
      // we first have to collect all the final states, and then work out the
      // super final state.
      K2_CHECK(openfst) << "Invalid line: " << line
                        << "\nFinal state with weight detected in K2 format";
      original_final_states.push_back(StringToInt(splits[0]));
      original_final_weights.push_back(StringToFloat(splits[1]));
      max_state = std::max(max_state, original_final_states.back());
    } else if (num_fields == 1u) {
      //   0
      // final_state
      (void)StringToInt(splits[0]);  // this is a final state
      break;                         // finish reading
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nIt expects a line with 1, 2 or 4 fields";
    }
  } while (std::getline(is, line));

  // Post processing on final states. When openfst is true, we may have multiple
  // final states with weights associated with them. We will have to add a super
  // final state, and convert that into the K2 format (final state with no
  // weight).
  if (original_final_states.size() > 0) {
    K2_CHECK_EQ(openfst, true);
    K2_CHECK_EQ(original_final_states.size(), original_final_weights.size());
    int32_t super_final_state = max_state + 1;
    for (auto i = 0; i != original_final_states.size(); ++i) {
      arcs.emplace_back(original_final_states[i],
                        super_final_state,
                        -1,     // kFinalSymbol
                        scale * original_final_weights[i]);
    }
  }

  // Sort arcs so that source states are in non-decreasing order.
  std::sort(arcs.begin(), arcs.end());

  bool error = true;
  Array1<Arc> array(GetCpuContext(), arcs);
  auto fsa = FsaFromArray1(array, &error);
  K2_CHECK_EQ(error, false);

  return fsa;
}

static Fsa TransducerFromStream(std::string first_line, std::istringstream &is,
                                bool openfst,
                                Array1<int32_t> *aux_labels) {
  K2_CHECK(aux_labels != nullptr);

  std::vector<int32_t> state_aux_labels;
  std::vector<Arc> arcs;
  std::string line = std::move(first_line);
  std::vector<std::string> splits;

  float scale = 1;
  if (openfst) scale = -1;

  int32_t max_state = -1;
  std::vector<int32_t> original_final_states;
  std::vector<float> original_final_weights;
  do {
    SplitStringToVector(line, kDelim,
                        &splits);  // splits is cleared in the function
    if (splits.empty()) continue;  // this is an empty line

    auto num_fields = splits.size();
    if (num_fields == 5u) {
      //   0           1         2         3        4
      // src_state  dest_state  symbol  aux_label score
      int32_t src_state = StringToInt(splits[0]);
      int32_t dest_state = StringToInt(splits[1]);
      int32_t symbol = StringToInt(splits[2]);
      int32_t aux_label = StringToInt(splits[3]);
      float score = scale * StringToFloat(splits[4]);
      arcs.emplace_back(src_state, dest_state, symbol, score);
      state_aux_labels.push_back(aux_label);
      max_state = std::max(max_state, std::max(src_state, dest_state));
    } else if (num_fields == 2u) {
      //   0            1
      // final_state  score
      // In this case, openfst is true. There could be multiple final states, so
      // we first have to collect all the final states, and then work out the
      // super final state.
      K2_CHECK(openfst) << "Invalid line: " << line
                        << "\nFinal state with weight detected in K2 format";
      original_final_states.push_back(StringToInt(splits[0]));
      original_final_weights.push_back(StringToFloat(splits[1]));
      max_state = std::max(max_state, original_final_states.back());
    } else if (num_fields == 1u) {
      //   0
      // final_state
      (void)StringToInt(splits[0]);
      break;  // finish reading
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nIt expects a line with 1, 2 or 5 fields";
    }
  } while (std::getline(is, line));

  // Post processing on final states. When openfst is true, we may have multiple
  // final states with weights associated with them. We will have to add a super
  // final state, and convert that into the K2 format (final state with no
  // weight).
  if (original_final_states.size() > 0) {
    K2_CHECK_EQ(openfst, true);
    K2_CHECK_EQ(original_final_states.size(), original_final_weights.size());
    int32_t super_final_state = max_state + 1;
    for (auto i = 0; i != original_final_states.size(); ++i) {
      arcs.emplace_back(original_final_states[i],
                        super_final_state,
                        -1,             // kFinalSymbol
                        scale * original_final_weights[i]);
      // TODO(guoguo) We are not sure yet what to put as the auxiliary label for
      //              arcs entering the super final state. The only real choices
      //              are kEpsilon or kFinalSymbol. We are using kEpsilon for
      //              now.
      state_aux_labels.push_back(0);    // kEpsilon
    }
  }

  // Sort arcs so that source states are in non-decreasing order. We have to do
  // this simultaneously for both arcs and auxiliary labels. The following
  // implementation makes a pair of (Arc, AuxLabel) for sorting.
  // TODO(guoguo) Optimize this when necessary.
  std::vector<std::pair<Arc, int32_t>> arcs_and_aux_labels;
  K2_CHECK_EQ(state_aux_labels.size(), arcs.size());
  arcs_and_aux_labels.resize(arcs.size());
  for (auto i = 0; i < arcs.size(); ++i) {
    arcs_and_aux_labels[i] = std::make_pair(arcs[i], state_aux_labels[i]);
  }
  // Default pair comparison should work for us.
  std::sort(arcs_and_aux_labels.begin(), arcs_and_aux_labels.end());
  for (auto i = 0; i < arcs.size(); ++i) {
    arcs[i] = arcs_and_aux_labels[i].first;
    state_aux_labels[i] = arcs_and_aux_labels[i].second;
  }

  auto cpu_context = GetCpuContext();
  *aux_labels = Array1<int32_t>(cpu_context, state_aux_labels);
  Array1<Arc> array(cpu_context, arcs);

  bool error = true;
  auto fsa = FsaFromArray1(array, &error);
  K2_CHECK_EQ(error, false);

  return fsa;
}

Fsa FsaFromString(const std::string &s, bool openfst /*= false*/,
                  Array1<int32_t> *aux_labels /*= nullptr*/) {
  std::istringstream is(s);
  std::string line;
  std::getline(is, line);
  K2_CHECK(is);

  std::vector<std::string> splits;
  SplitStringToVector(line, kDelim, &splits);
  auto num_fields = splits.size();
  if (num_fields == 4u)
    return AcceptorFromStream(std::move(line), is, openfst);
  else if (num_fields == 5u)
    return TransducerFromStream(std::move(line), is, openfst, aux_labels);

  K2_LOG(FATAL) << "Expected number of fields: 4 or 5."
                << "Actual: " << num_fields << "\n"
                << "First line is: " << line;

  return Fsa();  // unreachable code
}

std::string FsaToString(const Fsa &fsa, bool openfst /*= false*/,
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
  if (openfst) scale = -1;

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
