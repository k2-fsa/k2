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

# TODO(fangjun): add documentation to all functions


class Fsa(object):

    def __init__(self, s: str, negate_scores: bool = False):
        fsa: _Fsa
        aux_labels: Optional[torch.Tensor]

        fsa, aux_labels = _fsa_from_str(s, negate_scores)

        self._fsa = fsa
        self._aux_labels = aux_labels

    @classmethod
    def _create(cls, fsa: _Fsa,
                aux_labels: Optional[torch.Tensor] = None) -> 'Fsa':
        ans = cls.__new__(cls)
        super(Fsa, ans).__init__()
        ans._fsa = fsa
        ans._aux_labels = aux_labels
        return ans

    @classmethod
    def from_tensor(cls,
                    tensor: torch.Tensor,
                    aux_labels: Optional[torch.Tensor] = None) -> 'Fsa':
        '''This is useful when building an FSA from file.
        '''
        ans = cls.__new__(cls)
        super(Fsa, ans).__init__()
        ans._fsa = _fsa_from_tensor(tensor)
        ans._aux_labels = aux_labels
        return ans

    def to_str(self, negate_scores: bool = False) -> str:
        return _fsa_to_str(self._fsa, negate_scores, self._aux_labels)

    @property
    def arcs(self) -> torch.Tensor:
        return self._fsa.values.tensor()[:, :-1]

    @property
    def weights(self) -> torch.Tensor:
        return _as_float(self._fsa.values.tensor()[:, -1])

    @property
    def aux_labels(self) -> Union[torch.Tensor, None]:
        return self._aux_labels

    def to(self, device: Union[str, torch.device]) -> 'Fsa':
        if isinstance(device, str):
            device = torch.device(device)
        assert device.type in ['cpu', 'cuda']

        if device.type == 'cpu':
            fsa = self._fsa.cuda(device.index)
        else:
            fsa = self._fsa.cpu()

        if self._aux_labels is not None:
            aux_labels = self._aux_labels.to(device)
        else:
            aux_labels = None
        return Fsa._create(fsa, aux_labels)
