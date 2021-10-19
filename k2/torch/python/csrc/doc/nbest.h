/**
 * @copyright
 * Copyright      2021  Xiaomi Corp.  (authors: Wei Kang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef K2_TORCH_PYTHON_CSRC_DOC_NBEST_H_
#define K2_TORCH_PYTHON_CSRC_DOC_NBEST_H_

namespace k2 {

static constexpr const char *kNbestGetBestMatchingStatsDoc = R"doc(
For "query" sentences, this function gets the mean and variance of
scores from the best matching words-in-context in a set of provided "key"
sentences. This matching process matches the word and the words preceding
it, looking for the highest-order match it can find (it's intended for
approximating the scores of models that see only left-context,
like language models). The intended application is in estimating the scores
of hypothesized transcripts, when we have actually computed the scores for
only a subset of the hypotheses.

CAUTION:
  This function only runs on CPU for now.

Args:
  tokens:
    A ragged tensor of int32_t with 2 or 3 axes. If 2 axes, this represents
    a collection of key and query sequences. If 3 axes, this represents a
    set of such collections.

      2-axis example:
        [ [ the, cat, said, eos ], [ the, cat, fed, eos ] ]
      3-axis example:
        [ [ [ the, cat, said, eos ], [ the, cat, fed, eos ] ],
          [ [ hi, my, name, is, eos ], [ bye, my, name, is, eos ] ], ... ]

    where the words would actually be represented as integers,
    The eos symbol is required if this code is to work as intended
    (otherwise this code will not be able to recognize when we have reached
    the beginnings of sentences when comparing histories).
    bos symbols are allowed but not required.

  scores:
    A one dim torch.tensor with scores.size() == tokens.NumElements(),
    this is the item for which we are requesting best-matching values
    (as means and variances in case there are multiple best matches).
    In our anticipated use, these would represent scores of words in the
    sentences, but they could represent anything.
  counts:
    An one dim torch.tensor with counts.size() == tokens.NumElements(),
    containing 1 for words that are considered "keys" and 0 for
    words that are considered "queries".  Typically some entire
    sentences will be keys and others will be queries.
  eos:
    The value of the eos (end of sentence) symbol; internally, this
    is used as an extra padding value before the first sentence in each
    collection, so that it can act like a "bos" symbol.
  min_token:
    The lowest possible token value, including the bos
    symbol (e.g., might be -1).
  max_token:
    The maximum possible token value.  Be careful not to
    set this too large the implementation contains a part which
    takes time and space O(max_token - min_token).
  max_order:
    The maximum n-gram order to ever return in the
    `ngram_order` output; the output will be the minimum of max_order
    and the actual order matched; or max_order if we matched all the
    way to the beginning of both sentences. The main reason this is
    needed is that we need a finite number to return at the
    beginning of sentences.

Returns:
  Returns a tuple of four torch.tensor (mean, var, counts_out, ngram_order)
    mean:
      For query positions, will contain the mean of the scores at the
      best matching key positions, or zero if that is undefined because
      there are no key positions at all.  For key positions,
      you can treat the output as being undefined (actually they
      are treated the same as queries, but won't match with only
      themselves because we don't match at singleton intervals).
    var:
      Like `mean`, but contains the (centered) variance
      of the best matching positions.
    counts_out:
      The number of key positions that contributed to the `mean`
      and `var` statistics.  This should only be zero if `counts`
      was all zero.
    ngram_order:
      The n-gram order corresponding to the best matching
      positions found at each query position, up to a maximum of
      `max_order`; will be `max_order` if we matched all
      the way to the beginning of a sentence.
)doc";

}  // namespace k2

#endif  // K2_TORCH_PYTHON_CSRC_DOC_NBEST_H_
