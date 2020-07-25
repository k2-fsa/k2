
enum DeviceType {
  kUnk,
  kCuda,
  kCpu
};


/**
   class Context is the main surface of interaction with external tensor libraries like
   PyTorch; it allows us to use their notions of device and of memory allocation.
   (in the case of PyTorch it would probably contain a Device and an Allocator).

   We will sub-class this in several ways: with versions that wrap external toolkits
   like PyTorch, and also a "native" version that's mostly for testing purposes.

   This object should be allocated with std::shared_ptr, as that's how we store
   pointers to it.
*/
class Context {

  /*
    Return either a copy of this object that can be used for (e.g.) creating new
    arrays, or nullptr.  If nullptr is returned, then the user should use the
    std::shared_ptr to this object instead.

    The reason this is necessary is that originally when we get this Context
    we'll get it from some kind of tensor from an external toolkit, and this
    might be part of an Array that points to memory held by a Tensor that came
    from that toolkit.  In that case the right way to free the memory inside
    Deallocate() would be to destroy our locally held copy of that Tensor, and
    the Context will know this.  When we call Duplicate() on this type of
    Context, it will give us a "fresh" Context that will allocate and deallocate
    in the normal way.
   */
  virtual std::shared_ptr<Context> Duplicate();

  // Returns kCuda if this device is a CUDA device, or kCpu if it's the CPU.
  virtual DeviceType GetDeviceType();

  // Allocate memory on this device (raise an exception on failure, which we won't
  // attempt to catch because it will be specific to the external toolkit).
  // Note: will return NULL if bytes == 0.
  virtual void* Allocate(size_t bytes);

  // Return true if this is the same device as 'other' (essentially: that it
  // lives in the same physical memory space).  Must always return true if this
  // == &other.
  virtual bool IsSame(const Context &other) const;

  bool operator == (const Context &other) const {
    return this == &other || this->IsSame(other);
  }

  // Free memory that was allocated by Context::Allocate() on the same device.
  // In general it is an error if `data` was not allocated this way; however,
  // this will still do the right thing (generally: nothing) if the data was
  // allocated by an external toolkit because the Context object will remember
  // that it's held by a Region whose data belongs to an external toolkit.
  // See also Duplicate() for more information on this special case.
  virtual void Deallocate(void *data);

  virtual ~Context();
};

typedef std::shared_ptr<Context> ContextPtr;

/*
  Note: Region will always be allocated with std::make_shared<Region>(...), and
  holders of a Region will hold it via shared_ptr.

  To enable resizable (extendable) arrays to work when multiple objects may
  point to the same memory: there will be a convention that if an Array covers
  all the bytes used in a memory region it gets to use the remaining bytes allocated
  (i.e. it gets to increased bytes_used up until num_bytes).  That means
  only one of the Arrays pointing to that memory region will 'take' that memory.
  Once it gets too large and can't fit in the Region, it allocates a new Region.
*/
struct Region {
  // The 'context' is an object that
  ContextPtr context;
  void *data;        // Pointer to the start of the allocated memory region
  size_t num_bytes;  // number of bytes allocated.
  size_t bytes_used;  // largest number of bytes used/covered by any Array that
                      // points to this Region (this is relevant for things that
                      // behave like resizable vectors).
  // You need template arg to invoke this, e.g. region->GetData<int>();
  // You can also choose to template additionally on the device-type, like
  // region->GetData<int,kGpu>(), to activate a check that it's on the expected
  // device.
  template <typename T=void, DeviceType d = kUnk> T *GetData() {
    if (d != kUnk) { assert(d == context->GetDeviceType()); }
    return reinterpret_cast<T*>(data);
  }

  ~Region() { deallocator(data); }
};


// Return a basic Context object suitable for work on the CPU.
std::shared_ptr<Context> CpuContext();


// Return a basic Context object suitable for work with CUDA,
// with specified GPU-id (or the first one we grab, if gpu_id == -1).
// This will be a *native* context, one that used k2's own memory
// manager.  If you want to use (say) PyTorch's memory manager,  you
// should use a Context passed in from PyTorch
std::shared_ptr<Context> CudaContext(int gpu_id=-1);


/**
   Allocate a new Region.

     @param [in] context   Context from which to allocate the region
                          (specifies the device and allocator)
     @param [in] num_bytes  Number of bytes to allocate.  Note: zero bytes
                          is OK and will be handled in the same was as
                          nonzero allocations.
   Returns a new region.   Raises exception (TBD, may be dependent on the context)
                         on error such as allocation failure.  The returned
                         region will have bytes_used == num_bytes; if the user
                         wants to change this they can do it afterward.
*/
std::shared_ptr<Region> NewRegion(const std::shared_ptr<Context> &context,
                                  size_t num_bytes) {
  // .. fairly straightforward.  Sets bytes_used to num_bytes, caller can overwrite if needed.
  std::shared_ptr<Region> ans = std::make_shared<Region>();
  if (!(ans->context = context->Duplicate())) {
    // we'll almost always go inside this if-statement.  The only time when
    // Duplicate() returns a non-NULL pointer is when we have a Context that is
    // attached to
    ans->context = context;
  }
  ans->data = context->Allocate(num_bytes);
  ans->num_bytes = num_bytes;
  ans->bytes_used = num_bytes;
}


/*
  Convenience wrapper for NewRegion() that takes the context from a provided region.
 */
std::shared_ptr<Region> NewRegion(const Region &region,
                                  size_t num_bytes) {
  return NewRegion(region->context, num_bytes);
}


// Objects from k2 generally have a Context() method, so this template
// will work to get the device-type for pretty arbitrary objects.
template <typename T>
inline DeviceType DeviceOf(T t) {
  return t.Context().GetDeviceType();
}
