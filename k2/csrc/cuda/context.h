// k2/csrc/cuda/context.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_CONTEXT_H_
#define K2_CSRC_CUDA_CONTEXT_H_

#include <cassert>
#include <memory>

namespace k2 {

enum class DeviceType {
  kUnk,
  kCuda,
  kCpu,
};

constexpr DeviceType kUnk = DeviceType::kUnk;
constexpr DeviceType kCuda = DeviceType::kCuda;
constexpr DeviceType kCpu = DeviceType::kCpu;

class Context;
using ContextPtr = std::shared_ptr<Context>;



/**
   class Context is the main surface of interaction with external tensor
   libraries like PyTorch; it allows us to use their notions of device and of
   memory allocation.  (in the case of PyTorch it would probably contain a
   Device and an Allocator).

   We will sub-class this in several ways: with versions that wrap external
   toolkits like PyTorch, and also a "native" version that's mostly for
   testing purposes.

   This object should be allocated with std::shared_ptr, as that's how we store
   pointers to it.
*/
class Context: public std::enable_shared_from_this<Context> {
 public:
  virtual ~Context() = default;

  // note: shared_from_this(), which returns a std::shared_ptr<Context>, is
  // public, inherited from std::enable_shared_from_this<Context>.

  // Return CPU version of this context.  May or may not return the
  // same value as ::k2::GetCpuContext()... this is so, for instance,
  // if you have a GPU PyTorch context you can get a CPU PyTorch context.
  virtual ContextPtr GetCpuContext();

  // Returns a (CPU) context that will allocate pinned memory.  (This is CPU
  // memory that's pinned for faster GPU memory transfers).  May or may not
  // return the same value as ::k2::GetCpuContext()... this is so, for instance,
  // if you have a GPU PyTorch context you can get a CPU PyTorch context.
  // NOTE: for now this won't do anything, we can do without pinned memory
  // for the time being.
  virtual ContextPtr GetPinnedContext();

  // Returns kCuda if this device is a CUDA device, or kCpu if it's the CPU.
  virtual DeviceType GetDeviceType() const = 0;

  // Return the cuda stream associated with this context.  Once we support
  // multiple GPUs, this is expected to also set the GPU to be the correct one.
  virtual cudaStream_t GetCudaStream() const {
    LOG(FATAL) << "Code error: not a CUDA context\n";
  }

  /*
    Allocate memory on this device (raise an exception on failure, which we likely
    won't attempt to catch).

        @param [in] bytes   Number of bytes to allocate; may be zero, in which
                            case NULL will be returned.   Let's assume the alignment
                            of the returned memory is at least as strict as
                            for malloc() with the same values.
        @param [out] decoder_context   If more information than the returned pointer
                            is required in order to deallocat this memory, then
                            that information will be supplied as 'deleter_context'.
                            In some cases this will be zero.
        @return    Returns the allocated block, or NULL if bytes == 0.
  */
  virtual void *Allocate(size_t bytes, void **deleter_context) = 0;

  /*
    Free memory that was allocated by Context::Allocate() from this Context object
    (or, in general, memory obtained from an external toolkit that this Context object
    knows how to delete).
           @param [in] data       The memory to delete (may be NULL)
           @param [in] deleter_context    Some Context objects may require additional
                            context information to delete the memory, like
                            the concept of 'context' in PyTorch's DataPtr.  Or may be
                            NULL in some instances.  In general, whatever was output
                            by Allocate() to deleter_context should be supplied to
                            Deallocate().
  */
  virtual void Deallocate(void *data, void *deleter_context) = 0;

  /*
    Return true if this is the same device as 'other' (essentially: that it
    lives in the same physical memory space).  Must always return true if this
    == &other.  In most cases it will be sufficient to test whether the pointers
    are the same.
  */
  virtual bool IsCompatible(const Context &other) const = 0;


  /*
    For CPU contexts, does nothing.  For CUDA contexts, synchronizes the CUDA
    stream associated with the context.  This will ensure, for instance, that
    any GPU-to-CPU transfers have completed.  (Note: we may end up using
    something more fine-grained.)
   */
  virtual void Sync() { }

};


/*
  NOTE: let's leave this for later, this won't be needed initially.

  Used to run a task "in the background" (with a thread pool), for parallelism.
  This should generally be used together with the GetChild() of the context
  object so that in case we're using a GPU the GPU stream doesn't cause the
  tasks to be serialized.

  General usage would be:
     ContextPtr c;  // passed in
     BackgroundRunner br;
     for (int i = 0; i < N; i++) {
        std::function<void()> lambda = [=] () {
        ContextPtr c_child = c.Child();
           // do something here, possibly with multiple steps...
        }
        br.Background(lambda);
     }
     br.Wait();

  This is necessariy because if you do something that isn't just a simple
  Eval() but requires, for instance, copying a number back to the CPU,
  just parallelizing the GPU streams using c.Child() isn't enough because
  it will synchronize in the loop.
 */
class BackgroundRunner {
 public:
  // TODO: may at some point add in a "cost estimate" that can help the code
  // decide whether the overhead of creating a thread is worth it.
  // Also, there will be a global semaphore that limits the number of threads
  // that can exist at any one time.
  void Background(std::function<void()> &f);

  //  Waits for all (CPU) threads launched by Background() on this object since
  // the last call to Wait(), to terminate.
  void Wait();
 private:
  // TODO.  My current thinking on this is for Background() to create threads
  // that Wait() can wait on, and to have the number of threads limited by a
  // global semaphore.
};



template <typename T>
ContextPtr GetContext(const T &t) {
  // suppose T has member method `Context`
  return t.Context();
}

template <typename First, typename... Rest>
ContextPtr GetContext(const First &first, const Rest &... rest) {
  ContextPtr ans1 = GetContext(first), ans2 = GetContext(rest...);
  assert(ans1->IsCompatible(*ans2) && "Contexts are not compatible");
  return ans1;
}

/*
  Note: Region will always be allocated with std::make_shared<Region>(...), and
  holders of a Region will hold it via shared_ptr.

  To enable resizable (extendable) arrays to work when multiple objects may
  point to the same memory: there will be a convention that if an Array covers
  all the bytes used in a memory region it gets to use the remaining bytes
  allocated (i.e. it gets to increased bytes_used up until num_bytes).
  That means only one of the Arrays pointing to that memory region will 'take'
  that memory. Once it gets too large and can't fit in the Region, it allocates
  a new Region.
*/
struct Region: public std::enable_shared_from_this<Region> {
  // note: the inheritance from std::enable_shared_from_this<Region>
  // means that this object has a function
  //  std::shared_ptr<Region> shared_from_this();


  ContextPtr context;  // Context from which this memory region was allocated

  void *data;         // Pointer to the start of the allocated memory region
  void *deleter_context;  // if non-NULL, this is provided to the context in the
                          // destructor instead of 'data'.  It will be NULL for
                          // some Contexts, non-NULL for others.
  size_t num_bytes;   // number of bytes allocated.
  size_t bytes_used;  // largest number of bytes used/covered by any Array that
                      // points to this Region (this is relevant for things that
                      // behave like resizable vectors).

  // You need template arg to invoke this, e.g. region->GetData<int>();
  // You can also choose to template additionally on the device-type, like
  // region->GetData<int,kGpu>(), to activate a check that it's on the expected
  // device.
  template <typename T = void, DeviceType d = kUnk>
  T *GetData() {
    if (d != kUnk) assert(d == context->GetDeviceType());
    return reinterpret_cast<T*>(data);
  }

  ~Region() {
    context->Deallocate(deleter_context != nullptr ? deleter_context : data);
  }
};

using RegionPtr = std::shared_ptr<Region>;

// Return a k2-native Context object suitable for work on the CPU.  Note: for use with
// external toolkits you will probably want to use
ContextPtr GetCpuContext();


// Return a basic Context object suitable for work with CUDA, with specified
// GPU-id (or the first one we grab, if gpu_id == -1).  This will be a *native*
// context, one that uses k2's own memory manager, and which will mostly be used
// for testing purposes without an external neural-network toolkit.  If you want
// to use (say) PyTorch's memory manager, you should use a Context passed in
// from PyTorch
ContextPtr GetCudaContext(int gpu_id = -1);

// Returns a (CPU) context that will allocate pinned memory.
ContextPtr GetPinnedContext();


/**
   Allocate a new Region.

     @param [in] context   Context from which to allocate the region
                          (specifies the device and allocator)
     @param [in] num_bytes  Number of bytes to allocate.  Note: zero bytes
                          is OK and will be handled in the same way as
                          nonzero allocations.
   Returns a new region.   Raises exception (TBD, may be dependent on the
                          context) on error such as allocation failure.
                          The returned
                          region will have bytes_used == num_bytes; if the user
                          wants to change this they can do it afterward.
*/
RegionPtr NewRegion(Context &context, size_t num_bytes) {
  // .. fairly straightforward.  Sets bytes_used to num_bytes, caller can
  // overwrite if needed.
  std::shared_ptr<Region> ans = std::make_shared<Region>();
  ans->context = context.shared_from_this();
  ans->data = context->Allocate(num_bytes, &ans->deleter_context);
  ans->num_bytes = num_bytes;
  ans->bytes_used = num_bytes;
  return ans;
}

/*
  Convenience wrapper for NewRegion() that takes the context from a provided
  region.
 */
std::shared_ptr<Region> NewRegion(Region &region, size_t num_bytes) {
  return NewRegion(region.context, num_bytes);
}

// Objects from k2 generally have a Context() method, so this template
// will work to get the device-type for pretty arbitrary objects.
template <typename T>
inline DeviceType DeviceOf(const T &t) {
  return t.Context().GetDeviceType();
}


template <typename LambdaT>
__global__ eval_lambda(int32_t n, LambdaT lambda) {
  int i =  blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    lambda(i);
  }
}


inline int32 NumBlocks(int32_t size, int32_t block_size) {
  return (size + block_size - 1) / block_size;
}

template <typename ContextPtrType,   // Context*  or ContextPtr == std::shared_ptr<Context>
          typename LambdaT>
void Eval(ContextPtrType c, int32_t n, LambdaT &lambda) {
  if (n <= 0)
    return;  // actually it would be an error if n < 0.
  DeviceType t = c->GetDeviceType();
  if (t == kCpu) {
    // TODO: if n is very large, we'll eventually support running this with
    // multiple threads.
    for (int32_t i = 0; i < n; i++) {
      lambda(i);
    }
  } else {
    assert(c == kCuda);
    int dim_block = 256;
    int dim_grid = NumBlocks(n, dim_block);
    if (dim_grid == 1)
      dim_block = n;
    eval_lambda<LambdaT><<<dim_block, dim_grid, 0, c->GetCudaStream()>>> (n, lambda);


    cudaError_t err = cudaGetLastError();
    assert(err == 0);
  }
}

// OK, want to do:
// ContextPtr c = ...;  ///
// auto d = Dependency({out_region1, out_region2},
//                      in_region1, in_region2, in_region3...);
// Eval(d, n_elems, lambda...);
//
//
// struct DepType {
//   std::vector<out_region> out_regs;
//   std::vector<in_region> in_regs;
//   Context *c;  // out_regs[0]->context.
// }
//
// Note: these dependencies are WITHIN CONTEXT for now...
//
// void* ContextPtr::ProcessDep(std::vector<Region> &out_deps,
//                              std::vector<Region> &in_deps);
//
//  WITHIN-CONTEXT OPS
//
// For GPU, when executing:
//
//  (i) Decide on output stream, e.g. create new stream for this op.
//  (ii) Find list of input dependencies' events that are not already
//       terminated (mark them if so!) and set the output stream to
//       wait on them.
//  (iii) Run kernel
//  (iv)  For each out_dep:
//         Write the event (to wait on) in the Region.
//
// For *simple* CPU, when executing:
//
//   Just execute, ignoring deps.
//
// For multi-threaded CPU, when executing.
//
//  (i) get list of Tasks that we depend on that have not terminated yet,
//      using try_wait() on their mutexes.
//
//    - If that list was empty:
//          create a new Task that's not finished;
//          queue a job that will run the lambda and then
//          mark the Task as finished.
//
//    - Mark all output regions as depending on that new Task as well as
//      any preceding Tasks running in those regions that have not yet terminated
//      (assuming this Task didn't depend on those...)
//
//    -
//
//
//
//  (ii)
//
//  Let the job be a lambda that will:
//    (ii) increment the wait_count on the destination memory regions
//
//
//    (ii) if that list is empty:
//        Run
//
// c_.Eval()...
//


}  // namespace k2

#endif  // K2_CSRC_CUDA_CONTEXT_H_
