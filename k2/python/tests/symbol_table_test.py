#!/usr/bin/env python3
#
# Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To run this single test, use
#
#  ctest --verbose -R symbol_table_test_py

import unittest

import k2


class TestSymbolTable(unittest.TestCase):

    def test(self):
        s = '''
        a 1
        b 2
        '''
        symbol_table = k2.SymbolTable.from_str(s)
        assert symbol_table.get('a') == 1
        assert symbol_table.get(1) == 'a'
        assert symbol_table.get('b') == 2
        assert symbol_table.get(2) == 'b'

        assert symbol_table.get(0) == '<eps>'
        assert symbol_table.get('<eps>') == 0

        assert symbol_table['a'] == 1
        assert symbol_table[1] == 'a'
        assert symbol_table['b'] == 2
        assert symbol_table[2] == 'b'

        assert symbol_table[0] == '<eps>'
        assert symbol_table['<eps>'] == 0

        assert 1 in symbol_table
        assert 'a' in symbol_table
        assert 2 in symbol_table
        assert 'b' in symbol_table

        assert symbol_table.ids == [0, 1, 2]
        assert symbol_table.symbols == ['<eps>', 'a', 'b']

        symbol_table.add('c')
        assert symbol_table['c'] == 3

        symbol_table.add('d', 10)
        assert symbol_table['d'] == 10

        symbol_table.add('e')
        assert symbol_table['e'] == 11

        assert symbol_table.ids == [0, 1, 2, 3, 10, 11]
        assert symbol_table.symbols == ['<eps>', 'a', 'b', 'c', 'd', 'e']

        s = '''
        a 1
        b 2
        h 12
        '''
        sym = k2.SymbolTable.from_str(s)

        merged = symbol_table.merge(sym)
        assert merged.ids == [0, 1, 2, 3, 10, 11, 12]
        assert merged.symbols == ['<eps>', 'a', 'b', 'c', 'd', 'e', 'h']
        assert merged[12] == 'h'
        assert merged['h'] == 12

        assert merged['e'] == 11
        assert merged[11] == 'e'

        copied = k2.SymbolTable.from_str(merged.to_str())
        assert merged == copied


if __name__ == '__main__':
    unittest.main()
