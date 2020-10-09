/**
 * @brief Utilities for creating FSAs.
 *
 * Note that serializations are done in Python.
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
// TODO(guoguo): We may run into locale problems, with comma vs. period for
//               decimals. We have to test if the C code will behave the same
//               w.r.t. locale as Python does.
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

/* Create an acceptor from a stream, assuming the acceptor is in the k2 format:

   src_state1 dest_state1 label1 score1
   src_state2 dest_state2 label2 score2
   ... ...
   final_state

   The source states will be in non-descending order, and the final state does
   not bear a cost/score -- we put the cost/score on the arc that connects to
   the final state and set its label to -1.

   @param [in]  is    The input stream that contains the acceptor.

   @return It returns an Fsa on CPU.
*/
static Fsa K2AcceptorFromStream(std::istringstream &is) {
  std::vector<Arc> arcs;
  std::vector<std::string> splits;
  std::string line;

  bool finished = false;  // when the final state is read, set it to true.
  while (std::getline(is, line)) {
    SplitStringToVector(line, kDelim,
                        &splits);  // splits is cleared in the function
    if (splits.empty()) continue;  // this is an empty line

    K2_CHECK_EQ(finished, false);

    auto num_fields = splits.size();
    if (num_fields == 4u) {
      //   0            1          2      3
      // src_state  dest_state   label  score
      int32_t src_state = StringToInt(splits[0]);
      int32_t dest_state = StringToInt(splits[1]);
      int32_t symbol = StringToInt(splits[2]);
      float score = StringToFloat(splits[3]);
      arcs.emplace_back(src_state, dest_state, symbol, score);
    } else if (num_fields == 1u) {
      //   0
      // final_state
      (void)StringToInt(splits[0]);  // this is a final state
      finished = true;               // set finish
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nk2 acceptor expects a line with 1 (final_state) or "
                       "4 (src_state dest_state label score) fields";
    }
  }

  K2_CHECK_EQ(finished, true) << "The last line should be the final state";

  bool error = true;
  Array1<Arc> array(GetCpuContext(), arcs);
  auto fsa = FsaFromArray1(array, &error);
  K2_CHECK_EQ(error, false);

  return fsa;
}

/* Create a transducer from a stream, assuming the transducer is in the K2
   format:

   src_state1 dest_state1 label1 aux_label1 score1
   src_state2 dest_state2 label2 aux_label2 score2
   ... ...
   final_state

   The source states will be in non-descending order, and the final state does
   not bear a cost/score -- we put the cost/score on the arc that connects to
   the final state and set its label to -1.

   @param [in]  is    The input stream that contains the transducer.

   @return It returns an Fsa on CPU.
*/
static Fsa K2TransducerFromStream(std::istringstream &is,
                                  Array1<int32_t> *aux_labels) {
  K2_CHECK(aux_labels != nullptr);

  std::vector<int32_t> aux_labels_internal;
  std::vector<Arc> arcs;
  std::vector<std::string> splits;
  std::string line;

  bool finished = false;  // when the final state is read, set it to true.
  while (std::getline(is, line)) {
    SplitStringToVector(line, kDelim,
                        &splits);  // splits is cleared in the function
    if (splits.empty()) continue;  // this is an empty line

    K2_CHECK_EQ(finished, false);

    auto num_fields = splits.size();
    if (num_fields == 5u) {
      //   0           1         2         3        4
      // src_state  dest_state label   aux_label  score
      int32_t src_state = StringToInt(splits[0]);
      int32_t dest_state = StringToInt(splits[1]);
      int32_t symbol = StringToInt(splits[2]);
      int32_t aux_label = StringToInt(splits[3]);
      float score = StringToFloat(splits[4]);
      arcs.emplace_back(src_state, dest_state, symbol, score);
      aux_labels_internal.push_back(aux_label);
    } else if (num_fields == 1u) {
      //   0
      // final_state
      (void)StringToInt(splits[0]);
      finished = true;               // set finish
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nk2 transducer expects a line with 1 (final_state) or "
                       "5 (src_state dest_state label aux_label score) fields";
    }
  }

  K2_CHECK_EQ(finished, true) << "The last line should be the final state";

  auto cpu_context = GetCpuContext();
  *aux_labels = Array1<int32_t>(cpu_context, aux_labels_internal);
  Array1<Arc> array(cpu_context, arcs);

  bool error = true;
  auto fsa = FsaFromArray1(array, &error);
  K2_CHECK_EQ(error, false);

  return fsa;
}

/* Create an acceptor from a stream, assuming the acceptor is in the OpenFST
   format:

   src_state1 dest_state1 label1 score1
   src_state2 dest_state2 label2 score2
   ... ...
   final_state final_score

   We will negate the cost/score when we read them in. Also note, OpenFST may
   omit the cost/score if it is 0.0.

   We always create the super final state. If there are final state(s) in the
   original FSA, then we add arc(s) from the original final state(s) to the
   super final state, with the (negated) old final state cost/score as its
   cost/score, and -1 as its label.

   @param [in]  is    The input stream that contains the acceptor.

   @return It returns an Fsa on CPU.
*/
static Fsa OpenFstAcceptorFromStream(std::istringstream &is) {
  std::vector<Arc> arcs;
  std::vector<std::vector<Arc>> state_to_arcs;            // indexed by states
  std::vector<std::string> splits;
  std::string line;

  int32_t max_state = -1;
  int32_t num_arcs = 0;
  std::vector<int32_t> original_final_states;
  std::vector<float> original_final_weights;
  while (std::getline(is, line)) {
    SplitStringToVector(line, kDelim,
                        &splits);  // splits is cleared in the function
    if (splits.empty()) continue;  // this is an empty line

    auto num_fields = splits.size();
    if (num_fields == 3u || num_fields == 4u) {
      //   0            1          2
      // src_state  dest_state   label
      //
      // or
      //
      //   0            1          2      3
      // src_state  dest_state   label  score
      int32_t src_state = StringToInt(splits[0]);
      int32_t dest_state = StringToInt(splits[1]);
      int32_t symbol = StringToInt(splits[2]);
      float score = 0.0f;
      if (num_fields == 4u)
        score = -1.0f * StringToFloat(splits[3]);

      // Add the arc to "state_to_arcs".
      ++num_arcs;
      max_state = std::max(max_state, std::max(src_state, dest_state));
      if (static_cast<int32_t>(state_to_arcs.size()) <= src_state)
        state_to_arcs.resize(src_state + 1);
      state_to_arcs[src_state].emplace_back(src_state, dest_state, symbol,
                                            score);
    } else if (num_fields == 1u || num_fields == 2u) {
      //   0            1
      // final_state  score
      float score = 0.0f;
      if (num_fields == 2u)
        score = -1.0f * StringToFloat(splits[1]);
      original_final_states.push_back(StringToInt(splits[0]));
      original_final_weights.push_back(score);
      max_state = std::max(max_state, original_final_states.back());
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nOpenFST acceptor expects a line with 1 (final_state),"
                       " 2 (final_state score), 3 (src_state dest_state label) "
                       "or 4 (src_state dest_state label score) fields.";
    }
  }

  K2_CHECK(is.eof());

  // Post processing on final states. If there are final state(s) in the
  // original FSA, we add the super final state as well as arc(s) from original
  // final state(s) to the super final state. Otherwise, the super final state
  // will be added by FsaFromArray1 (since there's no arc with label
  // kFinalSymbol).
  if (original_final_states.size() > 0) {
    K2_CHECK_EQ(original_final_states.size(), original_final_weights.size());
    int32_t super_final_state = max_state + 1;
    state_to_arcs.resize(super_final_state);
    for (std::size_t i = 0; i != original_final_states.size(); ++i) {
      state_to_arcs[original_final_states[i]].emplace_back(
          original_final_states[i], super_final_state,
          -1,  // kFinalSymbol
          original_final_weights[i]);
      ++num_arcs;
    }
  }

  // Move arcs from "state_to_arcs" to "arcs".
  int32_t arc_index = 0;
  arcs.resize(num_arcs);
  for (std::size_t s = 0; s < state_to_arcs.size(); ++s) {
    for (std::size_t a = 0; a < state_to_arcs[s].size(); ++a) {
      K2_CHECK_GT(num_arcs, arc_index);
      arcs[arc_index] = state_to_arcs[s][a];
      ++arc_index;
    }
  }
  K2_CHECK_EQ(num_arcs, arc_index);

  bool error = true;
  Array1<Arc> array(GetCpuContext(), arcs);
  // FsaFromArray1 will add a super final state if the original FSA doesn't have
  // a final state.
  auto fsa = FsaFromArray1(array, &error);
  K2_CHECK_EQ(error, false);

  return fsa;
}

/* Create a transducer from a stream, assuming the transducer is in the OpenFST
   format:

   src_state1 dest_state1 label1 aux_label1 score1
   src_state2 dest_state2 label2 aux_label2 score2
   ... ...
   final_state final_score

   We will negate the cost/score when we read them in. Also note, OpenFST may
   omit the cost/score if it is 0.0.

   We always create the super final state. If there are final state(s) in the
   original FST, then we add arc(s) from the original final state(s) to the
   super final state, with the (negated) old final state cost/score as its
   cost/score, -1 as its label and 0 as its aux_label.

   @param [in]  is    The input stream that contains the transducer.

   @return It returns an Fsa on CPU.
*/
static Fsa OpenFstTransducerFromStream(std::istringstream &is,
                                       Array1<int32_t> *aux_labels) {
  K2_CHECK(aux_labels != nullptr);

  std::vector<std::vector<int32_t>> state_to_aux_labels;  // indexed by states
  std::vector<std::vector<Arc>> state_to_arcs;            // indexed by states
  std::vector<int32_t> aux_labels_internal;
  std::vector<Arc> arcs;
  std::vector<std::string> splits;
  std::string line;

  int32_t max_state = -1;
  int32_t num_arcs = 0;
  std::vector<int32_t> original_final_states;
  std::vector<float> original_final_weights;
  while (std::getline(is, line)) {
    SplitStringToVector(line, kDelim,
                        &splits);  // splits is cleared in the function
    if (splits.empty()) continue;  // this is an empty line

    auto num_fields = splits.size();
    if (num_fields == 4u || num_fields == 5u) {
      //   0           1         2         3
      // src_state  dest_state label   aux_label
      //
      // or
      //
      //   0           1         2         3        4
      // src_state  dest_state label   aux_label  score
      int32_t src_state = StringToInt(splits[0]);
      int32_t dest_state = StringToInt(splits[1]);
      int32_t symbol = StringToInt(splits[2]);
      int32_t aux_label = StringToInt(splits[3]);
      float score = 0.0f;
      if (num_fields == 5u)
        score = -1.0f * StringToFloat(splits[4]);

      // Add the arc to "state_to_arcs", and aux_label to "state_to_aux_labels"
      ++num_arcs;
      max_state = std::max(max_state, std::max(src_state, dest_state));
      if (static_cast<int32_t>(state_to_arcs.size()) <= src_state) {
        state_to_arcs.resize(src_state + 1);
        state_to_aux_labels.resize(src_state + 1);
      }
      state_to_arcs[src_state].emplace_back(src_state, dest_state, symbol,
                                            score);
      state_to_aux_labels[src_state].push_back(aux_label);
    } else if (num_fields == 1u || num_fields == 2u) {
      //   0
      // final_state
      //
      // or
      //
      //   0            1
      // final_state  score
      // There could be multiple final states, so we first have to collect all
      // the final states, and then work out the super final state.
      float score = 0.0f;
      if (num_fields == 2u)
        score = -1.0f * StringToFloat(splits[1]);
      original_final_states.push_back(StringToInt(splits[0]));
      original_final_weights.push_back(score);
      max_state = std::max(max_state, original_final_states.back());
    } else {
      K2_LOG(FATAL) << "Invalid line: " << line
                    << "\nOpenFST transducer expects a line with "
                       "1 (final_state), 2 (final_state score), "
                       "4 (src_state dest_state label aux_label) or "
                       "5 (src_state dest_state label aux_label score) fields.";
    }
  }

  K2_CHECK(is.eof());

  // Post processing on final states. If there are final state(s) in the
  // original FST, we add the super final state as well as arc(s) from original
  // final state(s) to the super final state. Otherwise, the super final state
  // will be added by FsaFromArray1 (since there's no arc with label
  // kFinalSymbol).
  if (original_final_states.size() > 0) {
    K2_CHECK_EQ(original_final_states.size(), original_final_weights.size());
    int32_t super_final_state = max_state + 1;
    state_to_arcs.resize(super_final_state);
    state_to_aux_labels.resize(super_final_state);
    for (std::size_t i = 0; i != original_final_states.size(); ++i) {
      state_to_arcs[original_final_states[i]].emplace_back(
          original_final_states[i], super_final_state,
          -1,  // kFinalSymbol
          original_final_weights[i]);
      // TODO(guoguo) We are not sure yet what to put as the auxiliary label for
      //              arcs entering the super final state. The only real choices
      //              are kEpsilon or kFinalSymbol. We are using kEpsilon for
      //              now.
      state_to_aux_labels[original_final_states[i]].push_back(0);  // kEpsilon
      ++num_arcs;
    }
  }

  // Move arcs from "state_to_arcs" to "arcs", and aux_labels from
  // "state_to_aux_labels" to "aux_labels_internal"
  int32_t arc_index = 0;
  arcs.resize(num_arcs);
  aux_labels_internal.resize(num_arcs);
  K2_CHECK_EQ(state_to_arcs.size(), state_to_aux_labels.size());
  for (std::size_t s = 0; s < state_to_arcs.size(); ++s) {
    K2_CHECK_EQ(state_to_arcs[s].size(), state_to_aux_labels[s].size());
    for (std::size_t a = 0; a < state_to_arcs[s].size(); ++a) {
      K2_CHECK_GT(num_arcs, arc_index);
      arcs[arc_index] = state_to_arcs[s][a];
      aux_labels_internal[arc_index] = state_to_aux_labels[s][a];
      ++arc_index;
    }
  }
  K2_CHECK_EQ(num_arcs, arc_index);

  auto cpu_context = GetCpuContext();
  *aux_labels = Array1<int32_t>(cpu_context, aux_labels_internal);
  Array1<Arc> array(cpu_context, arcs);

  bool error = true;
  // FsaFromArray1 will add a super final state if the original FSA doesn't have
  // a final state.
  auto fsa = FsaFromArray1(array, &error);
  K2_CHECK_EQ(error, false);

  return fsa;
}

Fsa FsaFromString(const std::string &s, bool openfst /*= false*/,
                  Array1<int32_t> *aux_labels /*= nullptr*/) {
  std::istringstream is(s);
  K2_CHECK(is);

  if (openfst == false && aux_labels == nullptr)
    return K2AcceptorFromStream(is);
  else if (openfst == false && aux_labels != nullptr)
    return K2TransducerFromStream(is, aux_labels);
  else if (openfst == true && aux_labels == nullptr)
    return OpenFstAcceptorFromStream(is);
  else if (openfst == true && aux_labels != nullptr)
    return OpenFstTransducerFromStream(is, aux_labels);

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
