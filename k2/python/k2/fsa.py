# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Guoguo Chen
#
# See ../../../LICENSE for clarification regarding multiple authors

from collections import OrderedDict
from typing import Any
from typing import Iterator
from typing import Optional
from typing import Tuple

import torch

from _k2 import RaggedArc
from _k2 import _as_int
from _k2 import _as_float
from _k2 import _fsa_from_str
from _k2 import _fsa_from_tensor


class Fsa(object):
    '''This class represents a single fsa or a vector of fsas.

    When it denotes a single FSA, its attribute :attr:`shape` is a tuple
    containing two elements ``(num_states, None)``; when it represents
    a vector of FSAs it is a tuple with three
    elements ``(num_fsas, None, None)``.  (Caution: it's possible
    for a vector of FSAs to have zero or one elements).

    An instance of FSA has the following attributes:

    - ``arcs``: You will NOT use it directly in Python. It is an instance
                of ``_k2.RaggedArc`` with only one method ``values()`` which
                returns a 2-D `torch.Tensor`` of dtype ``torch.int32`` with 4
                columns. Its number of rows indicates the number of arcs in the
                FSA. The first column represents the source states, second
                column the destination states, third column the labels and the
                fourth column is the score. Note that the score is actually
                a float number but it is **reinterpreted** as an integer.

    - ``score``: A 1-D ``torch.Tensor`` of dtype ``torch.float32``. It has
                 as many entries as the number of arcs representing the score
                 of every arc.

    - ``labels``: A 1-D ``torch.Tensor`` of dtype ``torch.int32``. It has as
                  many entries as the number of arcs representing the label of
                  every arc.

    - ``arc_labels_sorted``: A boolean. True if arcs of every state are sorted
                             by labels in increasing order. False means arcs
                             are not sorted by ``labels``.

    It MAY have the following attributes:

    - ``symbols``: An instance of ``k2.SymbolTable``. It maps an entry in
                   ``labels`` to an integer and vice versa. It is used for
                   visualization only.

    - ``aux_labels`: A 1-D ``torch.Tensor`` of dtype ``torch.int32``. It has the
                     same shape as ``labels``.

    - ``aux_symbols``: An instance of ``k2.SymbolTable. It maps an entry in
                       ``aux_labels`` to an integer and vice versa.

    - ``arc_aux_labels_sorted``: A boolean. True if arcs of every state are
                                 sorted by ``aux_labels`` in increasing order.
                                 False means arcs are not sorted by
                                 ``aux_labels``.

    It MAY have other attributes that set by users.

    CAUTION:
      When an attribute is an instance of ``torch.Tensor``, its ``shape[0]``
      has to be equal to the number arcs. Otherwise, an assertion error
      will be thrown.

    NOTE:
      ``symbols`` and ``aux_symbols`` are symbol tables, while ``labels``
      and ``aux_labels`` are instances of ``torch.Tensor``.
    '''

    def __init__(self,
                 tensor: torch.Tensor,
                 aux_labels: Optional[torch.Tensor] = None) -> 'Fsa':
        '''Build an Fsa from a tensor with optional aux_labels.

        It is useful when loading an Fsa from file.

        Args:
          tensor:
            A torch tensor of dtype `torch.int32` with 4 columns.
            Each row represents an arc. Column 0 is the src_state,
            column 1 the dest_state, column 2 the label, and column
            3 the score.

            Caution:
              Scores are floats and their binary pattern is
              **reinterpreted** as integers and saved in a tensor
              of dtype `torch.int32`.

          aux_labels:
            Optional. If not None, it associates an aux_label with every arc,
            so it has as many rows as `tensor`. It is a 1-D tensor of dtype
            `torch.int32`.

        Returns:
          An instance of Fsa.
        '''
        self._init_internal()
        self.arcs: RaggedArc = _fsa_from_tensor(tensor)
        self.arc_labels_sorted = False
        self._tensor_attr['score'] = _as_float(self.arcs.values()[:, -1])
        if aux_labels is not None:
            self.aux_labels = aux_labels.to(torch.int32)
            self.arc_aux_labels_sorted = False

    def _init_internal(self) -> None:
        self._tensor_attr = OrderedDict()
        self._non_tensor_attr = OrderedDict()

    def __setattr__(self, name: str, value: Any) -> None:
        if value is None:
            return
        if name in ('_tensor_attr', '_non_tensor_attr', 'arcs'):
            object.__setattr__(self, name, value)
        elif isinstance(value, torch.Tensor):
            assert value.shape[0] == self.arcs.values().shape[0]
            if name == 'labels':
                assert value.dtype == torch.int32
                self.arcs.values()[:, 2] = value
                return

            self._tensor_attr[name] = value.clone()

            if name == 'score':
                assert value.dtype == torch.float32
                # NOTE: we **reinterpret** the float patterns
                # to integer patterns here.
                self.arcs.values()[:, -1] = _as_int(value.detach())
        else:
            self._non_tensor_attr[name] = value

    def __getattr__(self, name: str) -> Any:
        if name == 'labels':
            return self.arcs.values()[:, 2]
        elif name in self._tensor_attr:
            return self._tensor_attr[name]
        elif name in self._non_tensor_attr:
            return self._non_tensor_attr[name]

        raise AttributeError(f'Unknown attribute {name}')

    def __delattr__(self, name: str) -> None:
        assert name not in ('arcs', 'score', 'labels')

        if name in self._tensor_attr:
            del self._tensor_attr[name]
        elif name in self._non_tensor_attr:
            del self._non_tensor_attr[name]
        else:
            super().__delattr__(name)

    def invert_(self) -> 'Fsa':
        '''Swap the ``labels`` and ``aux_labels``.

        CAUTION:
          The function name ends with an underscore which means this
          is an **in-place** operation.
        '''
        if hasattr(self, 'aux_labels'):
            aux_labels = self.aux_labels
            self.aux_labels = self.labels
            self.labels = aux_labels

            arc_aux_labels_sorted = getattr(self, 'arc_aux_labels_sorted',
                                            False)
            self.arc_aux_labels_sorted = self.arc_labels_sorted
            self.arc_labels_sorted = arc_aux_labels_sorted

        symbols = getattr(self, 'symbols', None)
        aux_symbols = getattr(self, 'aux_symbols', None)
        self.symbols, self.aux_symbols = aux_symbols, symbols

        return self

    def named_tensor_attr(self) -> Iterator[Tuple[str, torch.Tensor]]:
        '''Return an iterator over tensor attributes containing both
        the name of the attribute as well as the tensor value.

        Returns:
          A tuple containing the name and the value.
        '''
        for name, value in self._tensor_attr.items():
            yield name, value

    def named_non_tensor_attr(self) -> Iterator[Tuple[str, Any]]:
        '''Return an iterator over non-tensor attributes containing both
        the name of the attribute as well as the value.

        Returns:
          A tuple containing the name and the value.
        '''
        for name, value in self._non_tensor_attr.items():
            yield name, value

    @property
    def shape(self) -> Tuple[int, ...]:
        '''
        Returns:
          ``(num_states, None)` if this is an Fsa;
          ``(num_fsas, None, None)` if this is an FsaVec.
        '''
        if self.arcs.num_axes() == 2:
            return (self.arcs.dim0(), None)
        elif self.arcs.num_axes() == 3:
            return (self.arcs.dim0(), None, None)
        else:
            raise ValueError(f'Unsupported num_axes: {self.arcs.num_axes()}')

    @classmethod
    def from_ragged_arc(cls, ragged_arc: RaggedArc) -> 'Fsa':
        '''Create an Fsa from a RaggedArc directly.

        Note:
          Fsa algorithms will always produce some RaggedArc output. You can
          use this function to construct a Python FSA from RaggedArc.

        Args:
          ragged_arc:
            The input ragged arc. It is usually generated by some FSA
            algorithms. You do not need to know how to construct it in Python.
        Returns:
          An Fsa.
        '''
        ans = cls.__new__(cls)
        super(Fsa, ans).__init__()
        ans._init_internal()
        ans.arcs = ragged_arc
        ans.arc_labels_sorted = False
        ans._tensor_attr['score'] = _as_float(ans.arcs.values()[:, -1])
        return ans

    @classmethod
    def from_str(cls, s: str) -> 'Fsa':
        '''Create an Fsa from a string.

        The given string `s` consists of lines with the following format:

        (1) When it represents an acceptor:

                src_state dest_state label score

        (2) When it represents a transducer:

                src_state dest_state label aux_label score

        The line for the final state consists of only one field:

                final_state

        Note:
          Fields are separated by space(s), tab(s) or both. The `score`
          field is a float, while other fields are integers.

        Caution:
          The first column has to be non-decreasing.

        Caution:
          The final state has the largest state number. There is only
          one final state. All arcs that are connected to the final state
          have label -1.

        Args:
          s:
            The input string. Refer to the above comment for its format.
        '''
        # Figure out acceptor/transducer for k2 fsa.
        acceptor = True
        line = s.strip().split('\n', 1)[0]
        fields = line.strip().split()
        assert len(fields) == 4 or len(fields) == 5
        if len(fields) == 5:
            acceptor = False

        ans = cls.__new__(cls)
        super(Fsa, ans).__init__()
        ans._init_internal()
        arcs, aux_labels = _fsa_from_str(s, acceptor, False)
        ans.arcs = arcs
        ans.arc_labels_sorted = False
        ans._tensor_attr['score'] = _as_float(ans.arcs.values()[:, -1])
        if aux_labels is not None:
            ans.aux_labels = aux_labels.to(torch.int32)
            ans.arc_aux_labels_sorted = False
        return ans

    @classmethod
    def from_openfst(cls, s: str, acceptor: bool = True) -> 'Fsa':
        '''Create an Fsa from a string in OpenFST format.

        The given string `s` consists of lines with the following format:

        (1) When it represents an acceptor:

                src_state dest_state label score

        (2) When it represents a transducer:

                src_state dest_state label aux_label score

        The line for the final state consists of two fields:

                final_state score

        Note:
          Fields are separated by space(s), tab(s) or both. The `score`
          field is a float, while other fields are integers.

          There might be multiple final states. Also, OpenFst may omit the score
          if it is 0.0.

        Args:
          s:
            The input string. Refer to the above comment for its format.
          acceptor:
            Optional. If true, interpret the input string as an acceptor;
            otherwise, interpret it as a transducer.
        '''
        ans = cls.__new__(cls)
        super(Fsa, ans).__init__()
        ans._init_internal()
        arcs, aux_labels = _fsa_from_str(s, acceptor, True)
        ans.arcs = arcs
        ans.arc_labels_sorted = False
        ans._tensor_attr['score'] = _as_float(ans.arcs.values()[:, -1])
        if aux_labels is not None:
            ans.aux_labels = aux_labels.to(torch.int32)
            ans.arc_aux_labels_sorted = False
        return ans
