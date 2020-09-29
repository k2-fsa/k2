/**
 * @brief Utilities for reading, writing and creating FSAs.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_FSA_UTILS_H_
#define K2_CSRC_FSA_UTILS_H_

#include <string>

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"

namespace k2 {

/*
  Create an Fsa from string.

  The string `s` consists of lines. Every line, except the line for
  the final state, has one of the following two formats:

  (1)
      src_state dest_state label cost
  which means the string represents an acceptor.

  (2)
      src_state dest_state label aux_label cost
  which indicates the string is a transducer.

  The line for the final state has the following format:

      final_state

  CAUTION: The line for the final state contains NO cost.
  When an arc's dest state is the final state, we put the cost
  on the arc and set its label to -1.


  @param [in]   s   The input string. See the above description for its format.
  @param [in]   negate_scores
                    If true, the string form has the weights as costs,
                    not scores, so we negate as we read.
  @param [out]  aux_labels
                    Used only when it is a transducer. It contains the
                    aux_label of each arc.

  @return It returns an Fsa.
 */
Fsa FsaFromString(const std::string &s, bool negate_scores = false,
                  Array1<int32_t> *aux_labels = nullptr);

/*
  Write an Fsa to file.

  @param [in]   fsa       The fsa to be written.
  @param [in]   filename
  @param [in]   binary    True to save in binary format; false for text format.
  @param [in]   aux_labels
                    Not NULL when the fsa is a transducer. Leave it NULL
                    when the fsa is an acceptor.
*/
void WriteFsa(const Fsa &fsa, const std::string &filename, bool binary = true,
              const Array1<int32_t> *aux_labels = nullptr);

/*
  Read an Fsa from file.

  @param [in]   filename
  @param [in]   binary    true to read in binary format;
                          false to read in text format.
  @param [out]  aux_labels
                          If the file contains a transducer,
                          it will contain the aux_labels if not NULL.
 */
Fsa ReadFsa(const std::string &filename, bool binary = true,
            Array1<int32_t> *aux_labels = nullptr);

}  // namespace k2

#endif  //  K2_CSRC_FSA_UTILS_H_
