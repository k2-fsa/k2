// k2/csrc/cuda/types.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_CUDA_TYPES_H_
#define K2_CSRC_CUDA_TYPES_H_

namespace k2 {

/*
  Mostly some notes.

  Target scenario:
    For now we'll assume there is just one GPU and mostly we'll assume there's
    one thread using it

  First, a note on the type hierarchy.


  BaseType:
     we use this name for the template arg when we mean a type like short, int,
       int64_t, long, or structs... any POD type.
  I, J:
     As BaseType, but when we definitely expect the type to be an integer.
*/


// all will have a `using CategoryType = ... ` which will evaluate to
// one of the following:
//
//  Ptr1d
//  Vec1d
//  Vec<Ptr1d>
//  Vec<Vec1d>
// ... this will guide template selection.


template <typename BaseType> class DPtrBase { };  // for things that behave like DPtr.. used in template args
template <typename BaseType> class DVecBase { };  // for things that behave like DVec.. used in template args.


template <typename BaseType>
class DPtr: public DPtrBase<BaseType> {
 public:
  using ValueType = BaseType;
  using PtrType = DPtr<BaseType>;  // see DVec and Plus for why this is useful,
                                   // it's for when we want to remove the size
                                   // element from something.
  BaseType *data;
  __device__ int operator [] (int i) const {
    return data[i];
  }
  __device__ int &operator [] (int i) {
    return data[i];
  }
  // constructor on host and device.. device constructor is needed so we can pass this
  // into CUDA functions.
  __host__ __device__ DPtr(const DPtr<BaseType> &src):
      data(src.data) { }

};


template <typename BaseType>
class DVec: public DPtr<BaseType> {
 public:
  using ValueType = BaseType;
  using PtrType = DPtr<BaseType>;

  __host__ __device__ size_t size() const { return size_; }
  size_t size_;  // extra member `size`.  Note, if the hierarchy isn't convenient
                 // or clear we may just define DVec separately from DPtr.

  __host__ __device__ Dvec(const Dvec<BaseType> &src):
      DPtr<BaseType>(src), size_(src.size_) { }

};


// template to do type promotion.
// later we can use tricks like the following:
// https://stackoverflow.com/questions/17892770/c-template-automatic-type-promotion
template <typename A, typename B> class TypePromotion;

template
class TypePromotion<int, long int> {
 public:
  using T = long int;
}


// Note: A and B will be types that behave like DVec or DPtr.  If you want the
// result to be DVec-like (i.e. have a `size()`, you should make sure the LEFT
// HAND arg has a size.  If in future we need to handle cases where the size
// is only there on the right, we may need to specialize the template or discover
// some fancy template tricks.
template <typename A, typename B>
class Plus {
 public:
  using AType = typename A::ValueType;
  using BType = typename B::ValueType;
  using ValueType = TypePromotion<AType,BType>::T;
  using PtrType = Plus<typename A::PtrType, B>;
  A a;
  typename B::PtrType b;  // B, but ignore the `size` member, if any.

  __device__ ValueType operator [] (int i) const {
    return a(i) + b(i);
  }
  __host__ __device__ size_t size() const { return a.size(); }

 Plus(const A &a, const B &b): a(a), b(b) { }
};


template <typename A, typename B>
class Bracket {  // bracket/indexing [] operator
 public:
  using AType = typename A::ValueType;
  using BType = typename B::ValueType;
  using ValueType = AType;
  using PtrType = Plus<A, typename B::PtrType>;
  typename A::PtrType a;
  B b;

  __device__ ValueType operator [] (int i) const {
    return a(b(i));
  }
  // Note: in case B is a PtrType (i.e. doesn't have a size()), size() wouldn't
  // compile, and this would break compilation on some compilers like MSVC even
  // if it wasn't used...  It might be necessary either to figure out a way to
  // specialize this template for that case somehow, or to make size() a
  // non-member that can be handled with a function template.
  //
  __host__ __device__ size_t size() const { return b.size(); }

  Bracket(const A &a, const B &b): a(a), b(b) { }
};


template<typename A, typename B>
Plus<A,B> operator + (const A &a, const B &b) {
  return Plus(a, b);
}



template <typename A> __host__ void eval_kernel(A a) {
  // get thread id.. it's something like this, anyway..
  int tid = blockIdx.x + blockDim.x * gridIdx.x;
  if (tid < a.size()) {
    a(tid);  // just eval the expression.
  }
}


// Evaluate the expression A on the GPU; note, this will only be useful
// if the expression A has side effects, e.g. it contains assignment,
// +=, and so on.
template <typename A> __host__ void Eval(const A &a) {
  dim3 threads, blocks;
  eval_kernel<A> <<<threads, blocks>>>(A);
}



/*
  DRVec is a device vector that's resizable.  Of course we'll enable things like
  reserve().  The difference from a DVec is that the DRVec owns its data and is
  responsible for resizing.  The corresponding DRVec behaves in some ways like
  an iterator.
 */
template <typename BaseType>
class DRVec {
 public:
  // Templated constructor, this is important... there will be various versions
  // of this.
  template <typename T> DRVec(T):
      size_(T.size()) {
    // set `allocated` and `data` (we'll probably acquire the allocator object
    // from a thread_local global variable that we set earlier, to keep the
    // syntax clear).  set `allocator` to the allocator object that we get from
    // that thread_local variable...

    // The following is a slightly lazy way of implementing this constructor,
    // making use of the Eval() template.
    // We rely on the `=` expression being overloaded in the same way we overloaded
    // `+`->Plus above.
    Eval( static_cast<Dvec<T> >(*this) = T );
  }

  void Resize(size_t size);  // will allocate + memcpy if needed; won't
                             // apply any constructor or zeroing on new elements.


  DVec<T> Range(size_t begin, size_t end);

  template <typename T> void Append(const T &t) {
    size_t cur_size = size_;
    Resize(cur_size + t.size());
    // Below, relying on the Eval template to do the actual implementation.
    Eval(this->Range(cur_size, this->size_) = t);
  }



  DVec operator DVec() { .. } // casting operator

  Allocator allocator_;  // to be used when we delete this object or need to
                        // reallocate.
  BaseType *data_;
  size_t size_;  //  Number of elements in `data`, <= allocated
  size_t allocated_;  // Number of elements allocated in the memory region.
};



/*
   Class Array has a different purpose than DVec etc.  Here the base-type T
   is expected to usually be something like DVec or an expression,
   and Array actually contains an array of C++ objects, it doesn't contain
   device pointers directly.

   Partly, Array functions as a syntactic sugar when you want to do the same
   operation in parallel for a bunch of objects (all of the same type).
   But it also has its own version of the Eval() template which will
   invoke a special form of the kernel that can deal with multiple
   separate linear arrays of different sizes.

   The simplest implementation would be use 2 dimensions where one is
   the num of elements in the Array and the other is the largest size() of
   any of the elements.  We can work on smarter implementations later.

   In the actual kernel, we'd have the array of T (which of course contains
   metadata)
 */
template <typename T> class Array {




  size_t size_;
  std::shared_ptr<T> *elems_;  // owned here.
};


// Here, Vec is a CPU-based vector, something that has a function size()
// and an operator [] that can be evaluated on CPU.
// This would have to be implemented specially, so w
template <typename Vec, typename T>
void Resize(Vec sizes, Array<<DRVec<T> > *thing_to_be_resized) {
  for (..) { // resize manually, if needed (hopefully this won't be necessary to often..)

  }
}

template <typename Vec1, typename Vec2, typename A>
Array Range(const Vec1 &begin, const Vec2 &end, const Array<A> &thing_to_get_range_of) {


}


}  // namespace k2

#endif  // K2_CSRC_CUDA_TYPES_H_
