<div align="center">
<img src="https://raw.githubusercontent.com/k2-fsa/k2/master/docs/source/_static/logo.png" width=376>

[![Documentation Status](https://readthedocs.org/projects/k2/badge/?version=latest)](https://k2.readthedocs.io/en/latest/?badge=latest)

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

One of the key algorithms that we want to make efficient in the short term is
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

 A lot of the code is still unfinished (Sep 11, 2020).
 We finished the CPU versions of many algorithms and this code is in `k2/csrc/host/`;
 however, after that we figured out how to implement things on the GPU and decided
 to change the interfaces so the CPU and GPU code had a more unified interface.
 Currently in `k2/csrc/` we have more GPU-oriented implementations (although
 these algorithms will also work on CPU).  We had almost finished the Python
 wrapping for the older code, in the `k2/python/` subdirectory, but we decided not to
 release code with that wrapping because it would have had to be reworked to be compatible
 with our GPU algorithms.  Instead we will use the interfaces drafted in `k2/csrc/`
 e.g. the Context object (which encapsulates things like memory managers from external
 toolkits) and the Tensor object which can be used to wrap tensors from external toolkits;
 and wrap those in Python (using pybind11).  The code in host/ will eventually
 be either deprecated, rewritten or wrapped with newer-style interfaces.

## Plans for initial release

 We hope to get the first version working in early October.  The current
 short-term aim is to finish the GPU implementation of pruned composition of a
 normal FSA with a dense FSA, which is the same as decoder search in speech
 recognition and can be used to implement CTC training and lattice-free MMI (LF-MMI) training.  The
 proof-of-concept that we will release initially is something that's like CTC
 but allowing more general supervisions (general FSAs rather than linear
 sequences).  This will work on GPU.  The same underlying code will support
 LF-MMI so that would be easy to implement soon after.  We plan to put
 example code in a separate repository.

## Plans after initial release

 We will then gradually implement more algorithms in a way that's compatible
 with the interfaces in `k2/csrc/`.  Some of them will be CPU-only to start
 with.  The idea is to eventually have very rich capabilities for operating on
 collections of sequences, including methods to convert from a lattice to a
 collection of linear sequences and back again (for purposes of neural language
 model rescoring, neural confidence estimation and the like).

## Quick start

Want to try it out without installing anything? We have setup a [Google Colab][1].

Caution: k2 is not nearly ready for actual use!  We are still coding the core
algorithms, and hope to have an early version working by early October.

[1]: https://colab.research.google.com/drive/1qbHUhNZUX7AYEpqnZyf29Lrz2IPHBGlX?usp=sharing
