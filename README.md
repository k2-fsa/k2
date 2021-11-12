<div align="center">
<a href="https://k2-fsa.github.io/k2/">
  <img src="https://raw.githubusercontent.com/k2-fsa/k2/master/docs/source/_static/logo.png" width=88>
</a>

<br/>

[![Documentation Status](https://github.com/k2-fsa/k2/actions/workflows/build-doc.yml/badge.svg)](https://k2-fsa.github.io/k2/)

</div>

# k2

The vision of k2 is to be able to seamlessly integrate Finite State Automaton
(FSA) and Finite State Transducer (FST) algorithms into autograd-based machine
learning toolkits like PyTorch and TensorFlow.  For speech recognition
applications, this should make it easy to interpolate and combine various
training objectives such as cross-entropy, CTC and MMI and to jointly optimize a
speech recognition system with multiple decoding passes including lattice
rescoring and confidence estimation.  We hope k2 will have many other
applications as well.

One of the key algorithms that we have implemented is
pruned composition of a generic FSA with a "dense" FSA (i.e. one that
corresponds to log-probs of symbols at the output of a neural network).  This
can be used as a fast implementation of decoding for ASR, and for CTC and
LF-MMI training.  This won't give a direct advantage in terms of Word Error Rate when
compared with existing technology; but the point is to do this in a much more
general and extensible framework to allow further development of ASR technology.

## Implementation

 A few key points on our implementation strategy.

 Most of the code is in C++ and CUDA.  We implement a templated class `Ragged`,
 which is quite like TensorFlow's `RaggedTensor` (actually we came up with the
 design independently, and were later told that TensorFlow was using the same
 ideas).  Despite a close similarity at the level of data structures, the
 design is quite different from TensorFlow and PyTorch.  Most of the time we
 don't use composition of simple operations, but rely on C++11 lambdas defined
 directly in the C++ implementations of algorithms.  The code in these lambdas operate
 directly on data pointers and, if the backend is CUDA, they can run in parallel
 for each element of a tensor.  (The C++ and CUDA code is mixed together and the
 CUDA kernels get instantiated via templates).

 It is difficult to adequately describe what we are doing with these `Ragged`
 objects without going in detail through the code.  The algorithms look very
 different from the way you would code them on CPU because of the need to avoid
 sequential processing.  We are using coding patterns that make the most
 expensive parts of the computations "embarrassingly parallelizable"; the only
 somewhat nontrivial CUDA operations are generally reduction-type operations
 such as exclusive-prefix-sum, for which we use NVidia's `cub` library.  Our
 design is not too specific to the NVidia hardware and the bulk of the code we
 write is fairly normal-looking C++; the nontrivial CUDA programming is mostly
 done via the cub library, parts of which we wrap with our own convenient
 interface.

 The Finite State Automaton object is then implemented as a Ragged tensor templated
 on a specific data type (a struct representing an arc in the automaton).

## Autograd

 If you look at the code as it exists now, you won't find any references to
 autograd.  The design is quite different to TensorFlow and PyTorch (which is
 why we didn't simply extend one of those toolkits).  Instead of making autograd
 come from the bottom up (by making individual operations differentiable) we are
 implementing it from the top down, which is much more efficient in this case
 (and will tend to have better roundoff properties).

 An example: suppose we are finding the best path of an FSA, and we need
 derivatives.  We implement this by keeping track of, for each arc in the output
 best-path, which input arc it corresponds to.  (For more complex algorithms an arc
 in the output might correspond to a sum of probabilities of a list of input arcs).
 We can make this compatible with PyTorch/TensorFlow autograd at the Python level,
 by, for example, defining a Function class in PyTorch that remembers this relationship
 between the arcs and does the appropriate (sparse) operations to propagate back the
 derivatives w.r.t. the weights.

## Current state of the code

 We have wrapped all the C++ code to Python with [pybind11](https://github.com/pybind/pybind11)
 and have finished the integration with [PyTorch](https://github.com/pytorch/pytorch).

 We are currently writing speech recognition recipes using k2, which are hosted in a
 separate repository. Please see <https://github.com/k2-fsa/icefall>.

## Plans after initial release

 We are currently trying to make k2 ready for production use (see the branch
 [v2.0-pre](https://github.com/k2-fsa/k2/tree/v2.0-pre)).

## Quick start

Want to try it out without installing anything? We have setup a [Google Colab][1].
You can find more Colab notebooks using k2 in speech recognition at
<https://icefall.readthedocs.io/en/latest/recipes/librispeech/conformer_ctc.html>.

[1]: https://colab.research.google.com/drive/1qbHUhNZUX7AYEpqnZyf29Lrz2IPHBGlX?usp=sharing
