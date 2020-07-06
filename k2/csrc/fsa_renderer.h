// k2/csrc/fsa_renderer.h

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include <string>

#include "k2/csrc/fsa.h"

#ifndef K2_CSRC_FSA_RENDERER_H_
#define K2_CSRC_FSA_RENDERER_H_

namespace k2 {

// Get a GraphViz representation of an fsa.
class FsaRenderer {
 public:
  explicit FsaRenderer(const Fsa &fsa, const float *arc_weights = nullptr)
      : fsa_(fsa), arc_weights_(arc_weights) {}

  // Return a GraphViz representation of the fsa
  std::string Render() const;

 private:
  const Fsa &fsa_;
  const float *arc_weights_;
};

}  // namespace k2

#endif  // K2_CSRC_FSA_RENDERER_H_
