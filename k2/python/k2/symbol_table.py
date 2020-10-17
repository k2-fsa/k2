# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import Dict
from typing import Union
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolTable(object):
    _id2sym: Dict[int, str]
    '''Map an integer to a symbol.
    '''

    _sym2id: Dict[str, int]
    '''Map a symbol to an integer.
    '''

    def __post_init__(self):
        for idx, sym in self._id2sym.items():
            assert self._sym2id[sym] == idx
            assert idx >= 0

        for sym, idx in self._sym2id.items():
            assert idx >= 0
            assert self._id2sym[idx] == sym

        eps_sym = '<eps>'
        if 0 not in self._id2sym:
            self._id2sym[0] = eps_sym
            self._sym2id[eps_sym] = 0
        else:
            assert self._id2sym[0] == eps_sym
            assert self._sym2id[eps_sym] == 0

    @staticmethod
    def from_str(s: str) -> 'SymbolTable':
        '''Build a symbol table from a string.

        The string consists of lines. Every line has two fields separated
        by space(s), tab(s) or both. The first field is the symbol and the
        second the integer id of the symbol.

        Args:
          s:
            The input string with the format described above.
        Returns:
          An instance of :class:`SymbolTable`.
        '''
        id2sym: Dict[int, str] = dict()
        sym2id: Dict[str, int] = dict()

        for line in s.split('\n'):
            fields = line.split()
            if len(fields) == 0:
                continue  # skip empty lines
            assert len(fields) == 2, \
                    f'Expect a line with 2 fields. Given: {len(fields)}'
            sym, idx = fields[0], int(fields[1])
            assert sym not in sym2id, f'Duplicated symbol {sym}'
            assert idx not in id2sym, f'Duplicated id {idx}'
            id2sym[idx] = sym
            sym2id[sym] = idx

        return SymbolTable(_id2sym=id2sym, _sym2id=sym2id)

    @staticmethod
    def from_file(filename: str) -> 'SymbolTable':
        '''Build a symbol table from file.

        Every line in the symbol table file has two fields separated by
        space(s), tab(s) or both. The following is an example file:

        .. code-block::

            <eps> 0
            a 1
            b 2
            c 3

        Args:
          filename:
            Name of the symbol table file. Its format is documented above.

        Returns:
          An instance of :class:`SymbolTable`.

        '''
        with open(filename, 'r') as f:
            return SymbolTable.from_str(f.read().strip())

    def get(self, k: Union[int, str]) -> Union[str, int]:
        '''Get a symbol for an id or get an id for a symbol

        Args:
          k:
            If it is an id, it tries to find the symbol corresponding
            to the id; if it is a symbol, it tries to find the id
            corresponding to the symbol.

        Returns:
          An id or a symbol depending on the given ``k``.
        '''
        if isinstance(k, int):
            return self._id2sym[k]
        elif isinstance(k, str):
            return self._sym2id[k]
        else:
            raise ValueError(f'Unsupported type {type(k)}.')
