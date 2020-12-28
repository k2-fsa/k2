#!/usr/bin/env python3
#
# Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

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


if __name__ == '__main__':
    unittest.main()
