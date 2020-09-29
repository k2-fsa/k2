/**
 * @brief
 * fsa_renderer
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <string>

#include "k2/csrc/host/fsa.h"

#ifndef K2_CSRC_HOST_FSA_RENDERER_H_
#define K2_CSRC_HOST_FSA_RENDERER_H_

namespace k2host {

// Get a GraphViz representation of an fsa.
class FsaRenderer {
 public:
  explicit FsaRenderer(const Fsa &fsa) : fsa_(fsa) {}

  // Return a GraphViz representation of the fsa
  std::string Render() const;

 private:
  const Fsa &fsa_;
};

}  // namespace k2host

#endif  // K2_CSRC_HOST_FSA_RENDERER_H_
