/**
 * @brief Utilities for creating FSAs.
 *
 * Note that serializations are done in Python.
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

// Create an acceptor from a stream.
static Fsa AcceptorFromStream(std::string first_line, std::istringstream &is,
                              bool negate_scores) {
  std::vector<Arc> arcs;
  std::string line = std::move(first_line);
  std::vector<std::string> splits;

  float scale = 1;
  if (negate_scores) scale = -1;

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
    } else if (num_fields == 1u) {
      (void)StringToInt(splits[0]);  // this is a final state
      break;                         // finish reading
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nIt expects a line with 4 fields";
    }
  } while (std::getline(is, line));

  K2_CHECK(is) << "Failed to read fsa from string";

  bool error = true;
  Array1<Arc> array(GetCpuContext(), arcs);
  auto fsa = FsaFromArray1(array, &error);
  K2_CHECK_EQ(error, false);

  return fsa;
}

static Fsa TransducerFromStream(std::string first_line, std::istringstream &is,
                                bool negate_scores,
                                Array1<int32_t> *aux_labels) {
  K2_CHECK(aux_labels != nullptr);

  std::vector<int32_t> state_aux_labels;
  std::vector<Arc> arcs;
  std::string line = std::move(first_line);
  std::vector<std::string> splits;

  float scale = 1;
  if (negate_scores) scale = -1;

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
    } else if (num_fields == 1u) {
      (void)StringToInt(splits[0]);
      break;  // finish reading
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nIt expects a line with 5 fields";
    }
  } while (std::getline(is, line));

  K2_CHECK(is) << "Failed to read fsa from string";

  auto cpu_context = GetCpuContext();
  *aux_labels = Array1<int32_t>(cpu_context, state_aux_labels);
  Array1<Arc> array(cpu_context, arcs);

  bool error = true;
  auto fsa = FsaFromArray1(array, &error);
  K2_CHECK_EQ(error, false);

  return fsa;
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
