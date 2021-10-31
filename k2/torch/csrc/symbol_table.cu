/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include <fstream>
#include <sstream>

#include "k2/csrc/log.h"
#include "k2/torch/csrc/symbol_table.h"

namespace k2 {

SymbolTable::SymbolTable(const std::string &filename) {
  std::ifstream is(filename);
  std::string sym;
  int32_t id;
  while (is >> sym >> id) {
    K2_CHECK(!sym.empty());
    K2_CHECK_EQ(sym2id_.count(sym), 0) << "Duplicated symbol: " << sym;
    K2_CHECK_EQ(id2sym_.count(id), 0) << "Duplicated ID: " << id;
    sym2id_.insert({sym, id});
    id2sym_.insert({id, sym});
  }
  K2_CHECK(is.eof());
}

std::string SymbolTable::ToString() const {
  std::ostringstream os;
  char sep = ' ';
  for (const auto &p : sym2id_) {
    os << p.first << sep << p.second << "\n";
  }
  return os.str();
}

const std::string &SymbolTable::operator[](int32_t id) const {
  return id2sym_.at(id);
}

int32_t SymbolTable::operator[](const std::string &sym) const {
  return sym2id_.at(sym);
}

bool SymbolTable::contains(int32_t id) const { return id2sym_.count(id) != 0; }

bool SymbolTable::contains(const std::string &sym) const {
  return sym2id_.count(sym) != 0;
}

std::ostream &operator<<(std::ostream &os, const SymbolTable &symbol_table) {
  return os << symbol_table.ToString();
}

}  // namespace k2
