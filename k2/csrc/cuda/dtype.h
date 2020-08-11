#include <types.h>



namespace k2 {

class DtypeTraits {
 public:
  int NumBytes() { return num_bytes; }
  BaseType BaseType() { return static_cast<BaseType>(base_type); }

  DtypeTraits(BaseType base_type, int num_bytes, int num_scalars = 1, int misc = 0):
      base_type_(static_cast<char>(base_type)),
      num_scalars_(num_scalars),
      misc_(misc),
      num_bytes_(num_bytes) { }

 private:
  // We may add more
  char base_type_;     // BaseType converted to char
  char num_scalars_;  // currently always 1; may be greater for vector types in future.  Must
                     // divide num_bytes exactly.
  char misc_;        // field that is normally 0, but we may use for extensions in future.
  char num_bytes_;    // sizeof() this type in bytes, gives stride.  The size per
                     // scalar element is given by bytes_per_elem / num_scalars;
                     // we do it this way so that the stride in bytes is easily
                     // extractable.


};

// We initialize this in dtype.cc
extern DtypeTraits g_dtype_traits_array[];

inline DtypeTraits TraitsOf(Dtype dtype) {
  return g_dtype_traits_array[(int)dtype];
}


enum BaseType {  // BaseType is the *general type*
  kUnknownBase = 0,   // e.g. can use this for structs
  kFloatBase = 1,
  kIntBase = 2,  // signed int
  kUintBase = 3,  // unsigned int
};

// It's just an enum, we can use TraitsOf(dtype).NumBytes() and so on..
class enum Dtype {
  kFloatDtype,
  kDoubleDtype,
  kInt8Dtype,
  kInt32Dtype,
  kInt64Dtype,
  kUint32Dtype,
  kUint64Dtype
};


template<typename T> struct DtypeOf;
template<> struct DtypeOf<int32_t> {  constexpr Dtype dtype = kInt32Dtype; };
template<> struct DtypeOf<int64_t> {  constexpr Dtype dtype = kInt64Dtype; };
// and so on.
