/**
 * @brief
 * dtype
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_DTYPE_H_
#define K2_CSRC_DTYPE_H_

#include <cstdint>

#include "k2/csrc/log.h"

namespace k2 {

enum BaseType : int8_t {  // BaseType is the *general type*
  kUnknownBase = 0,       // e.g. can use this for structs
  kFloatBase = 1,         // real numbers, e.g., float or double
  kIntBase = 2,           // signed int, e.g., int8_t, int32_t
  kUintBase = 3,          // unsigned int, e.g, uint32_t, uint64_t
};

class DtypeTraits {
 public:
  int32_t NumBytes() const { return num_bytes_; }
  BaseType GetBaseType() const { return static_cast<BaseType>(base_type_); }
  const char *Name() const { return name_; }

  DtypeTraits(BaseType base_type, int32_t num_bytes, const char *name,
              int32_t num_scalars = 1, int32_t misc = 0)
      : base_type_(static_cast<char>(base_type)),
        num_scalars_(num_scalars),
        misc_(misc),
        num_bytes_(num_bytes),
        name_(name) {
    K2_CHECK_EQ(num_bytes_ % num_scalars_, 0);
  }

 private:
  // We may add more
  char base_type_;    // BaseType converted to char
  char num_scalars_;  // currently always 1; may be greater for vector types in
                      // future.  Must divide num_bytes exactly.
  char misc_;  // field that is normally 0, but we may use for extensions in
               // future.
  char num_bytes_;  // sizeof() this type in bytes, gives stride.  The size per
                    // scalar element is given by bytes_per_elem / num_scalars;
                    // we do it this way so that the stride in bytes is easily
                    // extractable.
  const char *name_;  // name, e.g. "float", "int8", "int32"
};

// We initialize this in dtype.cu
extern const DtypeTraits g_dtype_traits_array[];

// It's just an enum, we can use TraitsOf(dtype).NumBytes() and so on..
enum Dtype {
  kFloatDtype,
  kDoubleDtype,
  kInt8Dtype,
  kInt16Dtype,
  kInt32Dtype,
  kInt64Dtype,
  kUint32Dtype,
  kUint64Dtype,
};

inline DtypeTraits TraitsOf(Dtype dtype) {
  return g_dtype_traits_array[static_cast<int32_t>(dtype)];
}

template <typename T>
struct DtypeOf;

template <>
struct DtypeOf<float> {
  static const Dtype dtype = kFloatDtype;
};

template <>
struct DtypeOf<double> {
  static const Dtype dtype = kDoubleDtype;
};

template <>
struct DtypeOf<int8_t> {
  static const Dtype dtype = kInt8Dtype;
};

template <>
struct DtypeOf<int16_t> {
  static const Dtype dtype = kInt16Dtype;
};

template <>
struct DtypeOf<int32_t> {
  static const Dtype dtype = kInt32Dtype;
};

template <>
struct DtypeOf<int64_t> {
  static const Dtype dtype = kInt64Dtype;
};

template <>
struct DtypeOf<uint32_t> {
  static const Dtype dtype = kUint32Dtype;
};

template <>
struct DtypeOf<uint64_t> {
  static const Dtype dtype = kUint64Dtype;
};

/*
  Evaluates Expr for TypeName being all dtypes.  E.g.
     FOR_ALL_DTYPES(t.GetDtype(), T, SomeFuncCall<T>(a,b,c..));
 */
#define FOR_ALL_DTYPES(DtypeValue, TypeName, ...)                        \
  do {                                                                   \
    switch (DtypeValue) {                                                \
      case kFloatDtype: {                                                \
        using TypeName = float;                                          \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      case kDoubleDtype: {                                               \
        using TypeName = double;                                         \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      case kInt8Dtype: {                                                 \
        using TypeName = int8_t;                                         \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      case kInt16Dtype: {                                                \
        using TypeName = int16_t;                                        \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      case kInt32Dtype: {                                                \
        using TypeName = int32_t;                                        \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      case kInt64Dtype: {                                                \
        using TypeName = int64_t;                                        \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      case kUint32Dtype: {                                               \
        using TypeName = uint32_t;                                       \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      case kUint64Dtype: {                                               \
        using TypeName = uint64_t;                                       \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      default:                                                           \
        K2_LOG(FATAL)                                                    \
            << "Dtype " << TraitsOf(DtypeValue).Name()                   \
            << " not covered in switch statement.  p not supported for " \
               "this type?";                                             \
        break;                                                           \
    }                                                                    \
  } while (0)

#define FOR_REAL_AND_INT32_TYPES(DtypeValue, TypeName, ...)              \
  do {                                                                   \
    switch (DtypeValue) {                                                \
      case kFloatDtype: {                                                \
        using TypeName = float;                                          \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      case kDoubleDtype: {                                               \
        using TypeName = double;                                         \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      case kInt32Dtype: {                                                \
        using TypeName = int32_t;                                        \
        __VA_ARGS__;                                                     \
        break;                                                           \
      }                                                                  \
      default:                                                           \
        K2_LOG(FATAL)                                                    \
            << "Dtype " << TraitsOf(DtypeValue).Name()                   \
            << " not covered in switch statement.  p not supported for " \
               "this type?";                                             \
        break;                                                           \
    }                                                                    \
  } while (0)

}  // namespace k2

#endif  // K2_CSRC_DTYPE_H_
