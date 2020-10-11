# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Guoguo Chen
#
# See ../../../LICENSE for clarification regarding multiple authors

from collections import OrderedDict
from typing import Any
from typing import Optional
from typing import Tuple

import torch

from _k2 import RaggedArc
from _k2 import _as_float
from _k2 import _fsa_from_str
from _k2 import _fsa_from_tensor


class Fsa(object):
    '''This class represents a single fsa or a vector of fsas.

    When it denotes a single fsa, its attribute :attr:`shape` is a tuple
    containing two elements ``(num_states, None)``; it is a tuple with three
    elements ``(num_fsas, None, None)`` for a vector of fsas.
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
        if aux_labels is not None:
            self.aux_labels = aux_labels

    def _init_internal(self) -> None:
        self._attr_dict = OrderedDict()

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'score':
            score = self.arcs.values()[:, -1]
            score = _as_float(score)
            score.copy_(value)
        elif isinstance(value, torch.Tensor):
            assert value.shape[0] == self.arcs.values().shape[0]
            self._attr_dict[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        if name == 'score':
            score = self.arcs.values()[:, -1]
            return _as_float(score)
        if name in self._attr_dict:
            return self._attr_dict[name]

        raise AttributeError(f'Unknown attribute {name}')

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
        if aux_labels is not None:
            ans.aux_labels = aux_labels
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
        if aux_labels is not None:
            ans.aux_labels = aux_labels
        return ans
