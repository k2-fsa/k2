# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import Optional
from typing import Union

import torch

from _k2 import _Fsa
from _k2 import _as_float
from _k2 import _fsa_from_str
from _k2 import _fsa_from_tensor
from _k2 import _fsa_to_str


class Fsa(object):

    def __init__(self, s: str, negate_scores: bool = False):
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
          negate_scores:
            Optional. If true, the string form has the weights as costs,
            not scores, so we negate as we read.
        '''
        fsa: _Fsa
        aux_labels: Optional[torch.Tensor]

        fsa, aux_labels = _fsa_from_str(s, negate_scores)

        self._fsa = fsa
        self._aux_labels = aux_labels

    @classmethod
    def _create(cls, fsa: _Fsa,
                aux_labels: Optional[torch.Tensor] = None) -> 'Fsa':
        '''For internal use only.
        '''
        ans = cls.__new__(cls)
        super(Fsa, ans).__init__()
        ans._fsa = fsa
        ans._aux_labels = aux_labels
        return ans

    @classmethod
    def from_tensor(cls,
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
        ans = cls.__new__(cls)
        super(Fsa, ans).__init__()
        ans._fsa = _fsa_from_tensor(tensor)
        ans._aux_labels = aux_labels
        return ans

    def to_str(self, negate_scores: bool = False) -> str:
        '''Convert an Fsa to a string.

        Note:
          The returned string can be used to construct an Fsa.

        Args:
          negate_scores:
            Optional. If true, we negate the score during the conversion,

        Returns:
          A string representation of the Fsa.
        '''
        return _fsa_to_str(self._fsa, negate_scores, self._aux_labels)

    @property
    def arcs(self) -> torch.Tensor:
        '''Return the arcs of the Fsa.

        Caution:
          The scores are not contained in the returned tensor.
          Please use the `weights` property if you need it.

        Returns:
          A tensor of dtype `torch.int32` with 3 columns. Each row
          represents an arc. Column 0 is the src_state, column 1
          the dest_state, and column 2 the label.
        '''
        return self._fsa.values.tensor()[:, :-1]

    @property
    def weights(self) -> torch.Tensor:
        '''Returns the weights of arcs in the Fsa.

        Returns:
          A 1-D tensor of dtype `torch.float32`. It has
          as many rows as `arcs`.
        '''
        return _as_float(self._fsa.values.tensor()[:, -1])

    @property
    def aux_labels(self) -> Union[torch.Tensor, None]:
        '''Return the aux_labels associated with `arcs`, if any.

        Returns:
          None or a 1-D tensor of dtype `torch.int32` if the Fsa
          is a transducer. It has as many rows as `arcs`.
        '''
        return self._aux_labels

    def to(self, device: Union[str, torch.device]) -> 'Fsa':
        '''Convert an Fsa to a new Fsa on a given device.

        Caution:
          It is NOT an in-place operation. It returns a NEW instance.

        Args:
          device:
            A torch device. Currently it supports only CUDA and CPU devices.

        Returns:
          A new Fsa on the given device.
        '''
        if isinstance(device, str):
            device = torch.device(device)
        assert device.type in ['cpu', 'cuda']

        if device.type == 'cuda':
            fsa = self._fsa.cuda(device.index if device.index else -1)
        else:
            fsa = self._fsa.cpu()

        if self._aux_labels is not None:
            aux_labels = self._aux_labels.to(device)
        else:
            aux_labels = None
        return Fsa._create(fsa, aux_labels)
