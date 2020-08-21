

 So far this directory just contains some notes on implementation; all the code
 is just a VERY EARLY DRAFT.  The goal here is to show *in principle* how we parallelize
 things, building up from low-level primitives, but without actually creating any
 CUDA code.

 Actually we probably shouldn't separate this into a separate directory from the CPU code,
 since most of it is general purpose.

 Notes on build, and types of file:

 Currently the plan is for *all* of these files to be put through the CUDA compiler
 (nvcc).  Most of it is host code, but some of it leads to CUDA dependencies
 (e.g. one of the constructors of Array1 is a template which can instantiate
 CUDA code).

 Eventually I'd like to make compilation conditional, so we can create a version of this
 that runs on CPU with no CUDA dependency.  That can be done later though.
 (Would involve a bunch of #ifdefs, plus defining things like __host__ and __device__ to
 be the empty string).

 For CUDA streams, I intend to always use cudaStreamPerThread as the stream.  This will
 keep usage of the library relatively simple (no need to pass streams around).
