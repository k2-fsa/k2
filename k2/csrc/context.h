/**
 * @brief
 * context
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
 *                                                   Haowen Qiu
 *                                                   Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_CONTEXT_H_
#define K2_CSRC_CONTEXT_H_

#include <algorithm>
#include <cassert>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <type_traits>
#include <vector>

#include "k2/csrc/log.h"
#include "k2/csrc/nvtx.h"
#include "k2/csrc/semaphore.h"

namespace k2 {

enum class DeviceType {
  kUnk,
  kCuda,
  kCpu,
};

constexpr DeviceType kUnk = DeviceType::kUnk;
constexpr DeviceType kCuda = DeviceType::kCuda;
constexpr DeviceType kCpu = DeviceType::kCpu;

// Intended for use in debugging
inline std::ostream &operator<<(std::ostream &stream, const DeviceType type) {
  switch (type) {
    case kUnk:
      stream << "kUnk";
      break;
    case kCuda:
      stream << "kCuda";
      break;
    case kCpu:
      stream << "kCpu";
      break;
    default:
      K2_LOG(FATAL) << "Unreachable code!";
  }
  return stream;
}

class Context;
using ContextPtr = std::shared_ptr<Context>;

#define kCudaStreamInvalid ((cudaStream_t)(~((size_t)0)))

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
class Context : public std::enable_shared_from_this<Context> {
 public:
  virtual ~Context() = default;

  // note: shared_from_this(), which returns a std::shared_ptr<Context>, is
  // public, inherited from std::enable_shared_from_this<Context>.

  // Returns kCuda if this device is a CUDA device, or kCpu if it's the CPU.
  virtual DeviceType GetDeviceType() const = 0;

  // Returns the device id that the context is bound to, note we always return
  // -1 for CPU context. For GPU context, the sub-class will override this.
  // Note: we may not actually need this for a while.  For now it is not used.
  virtual int32_t GetDeviceId() const { return -1; }

  // Return the cuda stream associated with this context, or
  // kCudaStreamInvalid if this is not a CUDA context.
  virtual cudaStream_t GetCudaStream() const { return kCudaStreamInvalid; }

  /*
    Allocate memory on this device (raise an exception on failure, which we
    likely won't attempt to catch).

        @param [in] bytes   Number of bytes to allocate; may be zero, in which
                            case NULL will be returned.   Let's assume the
                            alignment of the returned memory is at least as
                            strict as for malloc() with the same values.
        @param [out] deleter_context   If more information than the returned
                            pointer is required in order to deallocate this
                            memory, then that information will be supplied
                            as 'deleter_context'. In some cases this will
                            be zero.
        @return    Returns the allocated block, or NULL if bytes == 0.
  */
  virtual void *Allocate(size_t bytes, void **deleter_context) = 0;

  /*
    Free memory that was allocated by Context::Allocate() from this Context
    object (or, in general, memory obtained from an external toolkit that this
    Context object knows how to delete).
           @param [in] data       The memory to delete (may be NULL)
           @param [in] deleter_context    Some Context objects may require
                              additional context information to delete the
                              memory, like the concept of 'context'
                              in PyTorch's DataPtr.  Or may be NULL in some
                              instances.  In general, whatever was output by
                              Allocate() to deleter_context should be
                              supplied to Deallocate().
  */
  virtual void Deallocate(void *data, void *deleter_context) = 0;

  /*
    Return true if this is the same device as 'other' (essentially: that it
    lives in the same physical memory space).  Must always return true if this
    == &other.  */
  virtual bool IsCompatible(const Context &other) const = 0;

  /*
    For CPU contexts, does nothing.  For CUDA contexts, synchronizes the CUDA
    stream associated with the context.  This will ensure, for instance, that
    any GPU-to-CPU transfers have completed.  (Note: we may end up using
    something more fine-grained.)
   */
  virtual void Sync() const {}

  /* Copy data between contexts.

     - For copying from host to host, it uses memcpy. We assume that src and dst
       do not overlap.

     - For copying from host to device, it allocates a block of pinned memory as
       an intermediate buffer. The data is first copied to the buffer
       using memcpy and then it is copied from the buffer to the device
       using cudaMemcpyAsync.

     - For copying from device to device, it uses cudaMemcpyAsync.

     - For copying from device to host, it uses cudaMemcpy.

     @param [in]   num_bytes  Number of bytes to be copied.
     @param [in]   src   The src pointer. It has to point to a memory block
                         allocated by `this` context.
     @param [in]   dst_context  The context of `dst` from which its memory
                                 gets allocated.
     @param [in]   dst   The dst pointer. It has to point to a memory block
                         allocated by `dst_context`.
   */
  virtual void CopyDataTo(size_t num_bytes, const void *src,
                          ContextPtr dst_context, void *dst) = 0;
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
     for (int32_t i = 0; i < N; ++i) {
        std::function<void()> lambda = [=] () {
        ContextPtr c_child = c.Child();
           // do something here, possibly with multiple steps...
        }
        br.Background(lambda);
     }
     br.Wait();

  This is necessary because if you do something that isn't just a simple
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
  // TODO:  My current thinking on this is for Background() to create threads
  // that Wait() can wait on, and to have the number of threads limited by a
  // global semaphore.
};

template <typename T1, typename T2>
bool IsCompatible(const T1 &t1, const T2 &t2) {
  // suppose both T1 and T2 have member method `Context`
  return t1.Context()->IsCompatible(*t2.Context());
}

template <typename T>
ContextPtr GetContext(const T &t) {
  // suppose T has member method `Context`
  return t.Context();
}

template <typename First, typename... Rest>
ContextPtr GetContext(const First &first, const Rest &... rest) {
  ContextPtr ans1 = GetContext(first), ans2 = GetContext(rest...);
  K2_CHECK(ans1->IsCompatible(*ans2)) << "Contexts are not compatible";
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
struct Region : public std::enable_shared_from_this<Region> {
  // note: the inheritance from std::enable_shared_from_this<Region>
  // means that this object has a function
  //  std::shared_ptr<Region> shared_from_this();

  ContextPtr context;  // Context from which this memory region was allocated

  void *data;             // Pointer to the start of the allocated memory region
  void *deleter_context;  // if non-NULL, this is provided to the context in the
                          // destructor instead of 'data'.  It will be NULL for
                          // some Contexts, non-NULL for others.
  size_t num_bytes;       // number of bytes allocated.
  size_t bytes_used;  // largest number of bytes used/covered by any Array that
                      // points to this Region (this is relevant for things that
                      // behave like resizable vectors).

  // You need template arg to invoke this, e.g. region->GetData<int32_t>();
  // You can also choose to template additionally on the device-type, like
  // region->GetData<int32_t,kCuda>(), to activate a check that it's on the
  // expected device.
  template <typename T = void, DeviceType d = kUnk>
  T *GetData() {
    if (d != kUnk) K2_CHECK_EQ(d, context->GetDeviceType());
    return reinterpret_cast<T *>(data);
  }

  /* Extends the region (this is like realloc; and in fact, in future, we might
     decide to use realloc-type things inside the implementation).
        @param [in] new_bytes_used   New size of this region; if this is
                         <= bytes_used nothing is done.  At exit, the
                         bytes_used of this region will equal new_bytes_used.
                         if num_bytes < new_bytes_used this region will be
                         reallocated according to some heuristic (e.g. the
                         larger of double the current size, or
                         the next power of 2 greater than `new_bytes_used`). */
  void Extend(size_t new_bytes_used) {
    NVTX_RANGE(K2_FUNC);
    if (new_bytes_used <= bytes_used) return;
    if (num_bytes < new_bytes_used) {  // reallocate and copy
      size_t new_size = std::max<size_t>(num_bytes * 2, new_bytes_used);
      size_t i = 4;
      while (i < new_size / 8) i <<= 3;
      while (i < new_size) i <<= 1;
      new_size = i;  // Round up `new_size` to a power of 2.
      void *new_deleter_context;
      void *new_data = context->Allocate(new_size, &new_deleter_context);
      context->CopyDataTo(bytes_used, data, context, new_data);
      context->Deallocate(data, deleter_context);
      data = new_data;
      deleter_context = new_deleter_context;
      num_bytes = new_size;
    }
    bytes_used = new_bytes_used;
  }

  ~Region() { context->Deallocate(data, deleter_context); }
};

using RegionPtr = std::shared_ptr<Region>;

// Return a k2-native Context object suitable for work on the CPU.  Note: for
// use with external toolkits you will probably want to use
ContextPtr GetCpuContext();

// Return a basic Context object suitable for work with CUDA, with specified
// GPU-id (or the first one we grab, if gpu_id == -1).  This will be a *native*
// context, one that uses k2's own memory manager, and which will mostly be used
// for testing purposes without an external neural-network toolkit.  If you want
// to use (say) PyTorch's memory manager, you should use a Context passed in
// from PyTorch
//
// CAUTION: If there are no CUDA capable GPUs, it returns a CPU context!
ContextPtr GetCudaContext(int32_t gpu_id = -1);

/* Returns a (CPU) context that will allocate pinned memory.  (This is CPU
   memory that's pinned for faster GPU memory transfers).  May or may not
   return the same value as ::k2::GetCpuContext()... this is so, for instance,
   if you have a GPU PyTorch context you can get a CPU PyTorch context.

   CAUTION: If there are no CUDA capable GPUs, it returns a CPU context!
 */
ContextPtr GetPinnedContext();

/* Return a (CPU) context that will allocate pinned memory if device_type
   is kCuda. It is equivalent to GetCpuContext() if device_type is kCpu.

   @param [in] device_type  If device_type is kCpu, it is equivalent
                            to `GetCpuContext()`. If device_type is kCuda,
                            it is equivalent to `GetPinnedContext()`.
*/
ContextPtr GetContextForTransfer(DeviceType device_type);

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
RegionPtr NewRegion(ContextPtr context, std::size_t num_bytes);

/*
  Convenience wrapper for NewRegion() that takes the context from a provided
  region.
 */
inline RegionPtr NewRegion(Region &region, std::size_t num_bytes) {
  return NewRegion(region.context, num_bytes);
}

// Objects from k2 generally have a Context() method, so this template
// will work to get the device-type for pretty arbitrary objects.
template <typename T>
inline DeviceType DeviceOf(const T &t) {
  return t.Context()->GetDeviceType();
}

// This is for use by ParallelRunner and Context.  Users probably should not
// interact with this directly.  The idea is that the Context object will call
// this to possibly override its default thread.  The user would
// create a new stream by calling ParallelRunner's NewStream() method, and
// do `With w(stream);` which calls Push(stream), and later Pop(stream) when it
// goes out of scope.
class CudaStreamOverride {
 public:
  inline cudaStream_t OverrideStream(cudaStream_t stream) {
    if (stream_override_ != 0 && stream != kCudaStreamInvalid)
      return stream_override_;
    else
      return stream;
  }

  void Push(cudaStream_t stream);

  void Pop(cudaStream_t stream);

  CudaStreamOverride() : stream_override_(0x0) {}

 private:
  cudaStream_t stream_override_;
  std::vector<cudaStream_t> stack_;
};

static thread_local CudaStreamOverride g_stream_override;

class With {
 public:
  explicit With(cudaStream_t stream) : stream_(stream) {
    g_stream_override.Push(stream_);
  }
  ~With() { g_stream_override.Pop(stream_); }

 private:
  cudaStream_t stream_;
};


/*
  Our class Semaphore is a slight extension of std::counting_semaphore that also
  takes care of stream synchronization.  The projected use-case is when two
  threads (possibly with different CUDA streams, if we are using CUDA) have a
  producer-consumer relationship, such that one is waiting for the other.
  The projected use is:
    - Construct semaphore
    - Producing thread (maybe repeatedly) calls semaphore.Signal(ctx);
    - Consuming thread (maybe repeatedly) calls semaphore.Wait(ctx);
 */
class Semaphore {
 public:
  Semaphore(): device_type_(kUnk), semaphore_(0) { }

  void Signal(ContextPtr c);

  void Wait(ContextPtr c);

 private:
  DeviceType device_type_;  // makes sure it's always used with the same device
                            // type.
  k2std::counting_semaphore semaphore_;
  std::mutex events_mutex_;
  std::deque<cudaEvent_t> events_;
};



/*
  Class ParallelRunner allows you to invoke CUDA kernels in parallel.
  It works for CUDA and CPU, but for CPU it currently just executes things
  sequentially.  It works by creating a separate stream each time you
  call NewStream(),, and using CUDA events to ensure correct ordering of kernels
  with respect to the CUDA stream in the supplied context.

  Note: it's important to destroy this at the right time.  The usage pattern
  should be:

   (a) Do whatever you were doing before (i.e. previous tasks in the
       stream of ContextPtr c).
   (b) Create this object
   (c) For each task to be done in parallel:
        - Call NewStream() on this object, and then either pass the stream to
          Eval() directly or do `With w(pr.Stream());` and call any function
          while that's in scope (it automagically swaps the stream that
          the Context returns for the newly created one).
          [Note: if you give the tasks the same stream they'll execute
          sequentially so there is no point in calling NewStream() just once].
   (d) Destroy this object by letting it go out of scope
   (e) Do whatever you need to do after the parallel jobs (i.e. following
       tasks in the stream of ContextPtr c)

  Note the order of (a) and (b), and (d) and (e).  If you get this wrong,
*/
class ParallelRunnerActive {
 public:
  explicit ParallelRunnerActive(ContextPtr c);

  // create a new stream, that first syncs with stream of c_ via an event.  The
  // destructor will cause the stream of c_ to wait on this stream in the
  // destructor of `this` You can pass this into the Eval() and Eval2()
  // functions, or invoke kernels directly with it; but if you want it
  // to be used in called functions you should do something like
  //  With w(pr.NewStream());
  // with that object alive in the scope where you want the stream to be
  // used.
  //
  // NOTE: this will also push the stream onto the stack g_stream_override
  // (after popping that of any previous stream that this class generated)
  // so that you won't need to directly pass this into Eval(); the context
  // will call CudaStreamOverride::OverrideStream() and replace it
  // with this stream automatically.
  //
  //    @param [in] num_work_items   Provided by the caller, saying how many
  //               elements approximately may be processed with this stream.
  //               If it's less than some internal threshold (e.g. 10k, we'd
  //               tune it), we would not create a new stream; instead, we
  //               just return stream 0 (the default stream).
  //    @return Returns the created new stream, or stream 0 if num_work_items
  //            is less than the internal threshold (i.e. 10k for now).
  cudaStream_t NewStream(std::size_t num_work_items = 0);

  // Calling Finish() is equivalent to calling the destructor early.
  // But user should never call this directly if they use
  // `With w(pr.NewStream())` and `w` is not destructed; instead, they should
  // wait the destructor of `pr` to call this.
  void Finish();

  ~ParallelRunnerActive() { Finish(); }

 private:
  ContextPtr c_;
  std::vector<cudaStream_t> streams_;
  cudaEvent_t event_;
};

// use the one that does nothing.  Turns out cudaStreamCreate and
// cudaEventRecord() are too slow to make this worthwhile.
class ParallelRunnerDummy {
 public:
  explicit ParallelRunnerDummy(ContextPtr c) : stream_(c->GetCudaStream()) {}
  cudaStream_t NewStream() { return stream_; }
  void Finish() {}

 private:
  cudaStream_t stream_;
};

using ParallelRunner = ParallelRunnerActive;
}  // namespace k2

#endif  // K2_CSRC_CONTEXT_H_
