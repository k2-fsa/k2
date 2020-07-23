#include "k2/csrc/cuda/context.h"


/*
  Array1* is a 1-dimensional contiguous array (that doesn't support a stride).
*/
template <typename T> class Array1 {
 public:
  int size() const { return size_; }  // dimension of only (1st) axis

  T *data();  // Returns pointer to 1st elem



  // generally L will be some kind of lambda or function object; it should be
  // possible to evaluate it on the CUDA device (if we're compiling with CUDA)
  // and also on the CPU.  We'll do src(i) to evaluate element i.
  // NOTE: we assume this thread is already set to use the device associated with the
  // context in 'ctx', if it's a CUDA context.
  template <typename L>
  Array1(ContextPtr ctx, int size, L lambda) {
    Init(ctx, size);

    Eval(ctx->DeviceType(), data, size, lambda);
  }

  DeviceType Device() { return region->device; }

  // Resizes, copying old contents if we could not re-use the same memory location.
  // It will always at least double the allocated size if it has to reallocate.
  // See Region::num_bytes vs. Region::bytes_used.
  void Resize(int new_size);

 private:
  int size_;
  int byte_offset_;
  std::shared_ptr<Region> region_; // Region that `data` is a part of.  Device
                                   // type is stored here.  For an Array1 with
                                   // zero size (e.g. created using empty
                                   // constructor), will point to an empty
                                   // Region.


  void Init(DeviceType d, int size) {
    // .. takes care of allocation etc.
  }

};
