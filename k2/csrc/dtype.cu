/**
 * Copyright      2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "k2/csrc/dtype.h"

namespace k2 {

const DtypeTraits g_dtype_traits_array[] = {
  {kUnknownBase, 0, "Any", 0},
  {kFloatBase, 4, "half"}, {kFloatBase, 4, "float"}, {kFloatBase, 8, "double"},
  {kIntBase, 1, "int8"}, {kIntBase, 2, "int16"},
  {kIntBase, 4, "int32"},  {kIntBase, 8, "int64"},
  {kUintBase, 1, "uint8"}, {kUintBase, 2, "uint16"},
  {kUintBase, 4, "uint32"}, {kUintBase, 8, "uint64"},
  {kUnknownBase, 16, "Arc", 4}, {kUnknownBase, 0, "Other", 0}
};

const Dtype DtypeOf<Any>::dtype;
const Dtype DtypeOf<float>::dtype;
const Dtype DtypeOf<double>::dtype;
const Dtype DtypeOf<int8_t>::dtype;
const Dtype DtypeOf<int16_t>::dtype;
const Dtype DtypeOf<int32_t>::dtype;
const Dtype DtypeOf<int64_t>::dtype;
const Dtype DtypeOf<uint8_t>::dtype;
const Dtype DtypeOf<uint16_t>::dtype;
const Dtype DtypeOf<uint32_t>::dtype;
const Dtype DtypeOf<uint64_t>::dtype;


std::ostream &operator<<(std::ostream &os, Dtype dtype) {
  os << TraitsOf(dtype).Name();
  return os;
}

}  // namespace k2
