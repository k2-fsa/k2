/**
 * @brief
 * array_inl
 *
 * @note
 * Don't include this file directly; it is included by array.h.
 * It contains implementation code.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_ARRAY_INL_H_
#define K2_CSRC_ARRAY_INL_H_

#ifndef IS_IN_K2_CSRC_ARRAY_H_
#error "this file is supposed to be included only by array.h"
#endif

#include <algorithm>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cub/cub.cuh"
#include "k2/csrc/macros.h"
#include "k2/csrc/utils.h"

namespace k2 {

template <typename T>
Array1<T> Array1<T>::Clone() const {
  NVTX_RANGE(K2_FUNC);
  Array1<T> ans(Context(), Dim());
  ans.CopyFrom(*this);
  return ans;
}

template <typename T>
void Array1<T>::CopyFrom(const Array1<T> &src) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(dim_, src.dim_);
  if (dim_ == 0) return;

  const T *src_data = src.Data();
  T *dst_data = this->Data();

  src.Context()->CopyDataTo(Dim() * ElementSize(), src_data, Context(),
                            dst_data);
}
template <typename T>
std::ostream &operator<<(std::ostream &stream, const Array1<T> &array) {
  if (array.GetRegion() == nullptr) return stream << "<invalid Array1>";
  stream << "[ ";
  Array1<T> to_print = array.To(GetCpuContext());
  const T *to_print_data = to_print.Data();
  int32_t dim = to_print.Dim();
  for (int32_t i = 0; i < dim; ++i)
    stream << ToPrintable(to_print_data[i]) << ' ';
  return stream << ']';
}

template <typename T>
std::ostream &operator<<(std::ostream &stream, const Array2<T> &array) {
  if (array.GetRegion() == nullptr) return stream << "<invalid Array2>";
  stream << "\n[";
  Array2<T> array_cpu = array.To(GetCpuContext());
  int32_t num_rows = array_cpu.Dim0();
  for (int32_t i = 0; i < num_rows; ++i) {
    stream << ToPrintable(array_cpu.Row(i));
    if (i + 1 < num_rows) stream << '\n';
  }
  return stream << "\n]";
}

template <typename T>
std::istream &operator>>(std::istream &is, Array2<T> &array) {
  std::vector<T> vec;
  int32_t row_length = 0, num_rows = 0;
  is >> std::ws;  // eat whitespace
  int c = is.peek();
  if (c != '[') {
    is.setstate(std::ios::failbit);
    return is;
  } else {
    is.get();
  }
  while (1) {
    char c;
    is >> std::ws >> c;  // eat whitespace, read c
    if (!is.good() || (c != ']' && c != '[')) {
      is.setstate(std::ios::failbit);
      return is;
    }
    if (c == '[') {  // read next row.
      while (1) {
        is >> std::ws;
        if (is.peek() == ']') {
          is.get();
          num_rows++;
          if (num_rows == 1) {
            row_length = vec.size();
          } else if (static_cast<int32_t>(vec.size()) !=
                     row_length * num_rows) {
            is.setstate(std::ios::failbit);
            return is;
          }
          break;
        } else {
          InputFixer<T> t;
          is >> t;
          if (!is.good()) {
            is.setstate(std::ios::failbit);
            return is;
          }
          vec.push_back(t);
        }
      }
    } else {  // c == ']'
      Array1<T> a(GetCpuContext(), vec);
      array = Array2<T>(a, num_rows, row_length);
      return is;
    }
  }
}

template <typename T>
std::istream &operator>>(std::istream &is, Array1<T> &array) {
  std::vector<T> vec;
  is >> std::ws;  // eat whitespace
  int c = is.peek();
  if (c != '[') {
    is.setstate(std::ios::failbit);
    return is;
  } else {
    is.get();
  }
  while (1) {
    is >> std::ws;  // eat whitespace
    if (is.peek() == ']') {
      is.get();
      array = Array1<T>(GetCpuContext(), vec);
      return is;
    }
    InputFixer<T> t;
    is >> t;
    if (!is.good()) {
      is.setstate(std::ios::failbit);
      return is;
    }
    vec.push_back(t);
  }
}

template <typename T>
Array1<T>::Array1(const std::string &str) : Array1() {
  std::istringstream is(str);
  is >> *this;
  if (!is.good()) {
    K2_LOG(FATAL) << "Failed to initialize Array1 from string: " << str;
  }
}

template <typename T>
Array2<T>::Array2(const std::string &str) : Array2() {
  std::istringstream is(str);
  is >> *this;
  if (!is.good())
    K2_LOG(FATAL) << "Failed to initialize Array2 from string: " << str;
}

}  // namespace k2

#endif  // K2_CSRC_ARRAY_INL_H_
