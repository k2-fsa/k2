# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Guoguo Chen
#
# See ../../../LICENSE for clarification regarding multiple authors

from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple

import torch
import _k2

from _k2 import RaggedArc
from _k2 import _as_float
from _k2 import _as_int
from _k2 import _fsa_from_str
from _k2 import _fsa_from_tensor
from _k2 import _fsa_to_str
from _k2 import _fsa_to_tensor


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

    - ``scores``: A 1-D ``torch.Tensor`` of dtype ``torch.float32``. It has
                  as many entries as the number of arcs representing the score
                  of every arc.

    - ``labels``: A 1-D ``torch.Tensor`` of dtype ``torch.int32``. It has as
                  many entries as the number of arcs representing the label of
                  every arc.


    It MAY have the following attributes:

    - ``symbols``: An instance of ``k2.SymbolTable``. It maps an entry in
                   ``labels`` to an integer and vice versa. It is used for
                   visualization only.

    - ``aux_labels`: A 1-D ``torch.Tensor`` of dtype ``torch.int32``. It has the
                     same shape as ``labels``. NOTE: We will change it to a
                     ragged tensor in the future.

    - ``aux_symbols``: An instance of ``k2.SymbolTable. It maps an entry in
                       ``aux_labels`` to an integer and vice versa.

    - ``properties``: An integer that encodes the properties of the FSA. It is
                      returned by :func:`get_properties`.

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
                 aux_labels: Optional[torch.Tensor] = None) -> None:
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
        self._init_properties()
        self._tensor_attr['scores'] = _as_float(self.arcs.values()[:, -1])
        if aux_labels is not None:
            self.aux_labels = aux_labels.to(torch.int32)

    def __str__(self) -> str:
        '''Return a string representation of this object (note: does not
           contain all the information in it for now)'''
        if hasattr(self, 'aux_labels'):
            aux_labels = self.aux_labels.to(torch.int32)
        else:
            aux_labels = None
        if self.arcs.num_axes() == 2:
            ans = "k2.Fsa: " + _fsa_to_str(self.arcs, False, aux_labels)
        else:
            ans = "k2.FsaVec: \n"
            for i in range(self.shape[0]):
                # get the i-th Fsa
                ragged_arc, start = self.arcs.index(0, i)
                end = start + ragged_arc.values().shape[0]
                ans += "FsaVec[" + str(i) + "]: " + _fsa_to_str(
                    ragged_arc, False,
                    None if aux_labels is None else aux_labels[start:end])
        ans += "properties_str = " + _k2.fsa_properties_as_str(
            self._properties) + "."
        return ans

    def _init_properties(self) -> None:
        if self.arcs.num_axes() == 2:
            properties = _k2.get_fsa_basic_properties(self.arcs)
        else:
            properties = _k2.get_fsa_vec_basic_properties(self.arcs)
        self._properties = properties
        if properties & 1 != 1:
            raise ValueError(
                "Fsa is not valid, properties are: {} = {}, arcs are: {}".
                format(properties, _k2.fsa_properties_as_str(properties),
                       str(self.arcs)))

    def _init_internal(self) -> None:
        # There are three kinds of attribute dictionaries:
        #
        # - `_tensor_attr`
        #     It saves attribute values of type torch.Tensor. `shape[0]` of
        #     attribute values have to be equal to the number of arcs
        #     in the FSA.
        #
        # - `_non_tensor_attr`
        #     It saves non-tensor attributes, e.g., :class:`SymbolTable`.
        #
        # - `_grad_cache`
        #     It contains tensors for autograd. Users should NOT manipulate it.
        #     The dict is filled in automagically.
        self._tensor_attr = OrderedDict()
        self._non_tensor_attr = OrderedDict()

        self._grad_cache = OrderedDict()
        # The `_grad_cache` dict contains the following attributes:
        #
        #  - `state_batches`:
        #           returned by :func:`_k2._get_state_batches`
        #  - `dest_states`:
        #           returned by :func:`_k2._get_dest_states`
        #  - `incoming_arcs`:
        #           returned by :func:`_k2._get_incoming_arcs`
        #  - `entering_arc_batches`:
        #           returned by :func:`_k2._get_entering_arc_index_batches`
        #  - `leaving_arc_batches`:
        #           returned by :func:`_k2._get_leaving_arc_index_batches`
        #  - `forward_scores_tropical`:
        #           returned by :func:`_k2._get_forward_scores_float`
        #           with `log_semiring=False`
        #  - `forward_scores_log`:
        #           returned by :func:`_k2._get_forward_scores_float` or
        #           :func:`_get_forward_scores_double` with `log_semiring=True`
        #  - `tot_scores_tropical`:
        #           returned by :func:`_k2._get_tot_scores_float` or
        #           :func:`_k2._get_tot_scores_double` with
        #           `forward_scores_tropical`.
        #  - `tot_scores_log`:
        #           returned by :func:`_k2._get_tot_scores_float` or
        #           :func:`_k2._get_tot_scores_double` with
        #           `forward_scores_log`.
        #  - `backward_scores_tropical`:
        #           returned by :func:`_k2._get_backward_scores_float` or
        #           :func:`_k2._get_backward_scores_double` with
        #           `log_semiring=False`
        #  - `backward_scores_log_semiring`:
        #           returned by :func:`_k2._get_backward_scores_float` or
        #           :func:`_k2._get_backward_scores_double` with
        #           `log_semiring=True`
        #  - `entering_arcs`:
        #           returned by :func:`_k2._get_forward_scores_float` or
        #           :func:`_get_forward_scores_double` with `log_semiring=False`

    def __setattr__(self, name: str, value: Any) -> None:
        '''
        Caution:
          We save a reference to ``value``. If you need to change ``value``
          afterwards, please consider passing a copy of it.
        '''
        if name in ('_tensor_attr', '_non_tensor_attr', 'arcs', '_properties',
                    '_grad_cache'):
            object.__setattr__(self, name, value)
        elif isinstance(value, torch.Tensor):
            assert value.shape[0] == self.arcs.values().shape[0]
            if name == 'labels':
                assert value.dtype == torch.int32
                self.arcs.values()[:, 2] = value
                return

            self._tensor_attr[name] = value

            if name == 'scores':
                assert value.dtype == torch.float32
                # NOTE: we **reinterpret** the float patterns
                # to integer patterns here.
                self.arcs.values()[:, -1] = _as_int(value.detach())
        else:
            assert name != 'properties'
            self._non_tensor_attr[name] = value

    def __getattr__(self, name: str) -> Any:
        if name == 'labels':
            return self.arcs.values()[:, 2]
        elif name in self._tensor_attr:
            return self._tensor_attr[name]
        elif name in self._non_tensor_attr:
            return self._non_tensor_attr[name]
        elif name in self._grad_cache:
            return self._grad_cache[name]
        elif name == 'properties':
            return self._properties
        elif name == 'properties_str':
            return _k2.fsa_properties_as_str(self._properties)
        elif name == 'requires_grad':
            return self.scores.requires_grad
        elif name == 'grad':
            return self.scores.grad

        raise AttributeError(f'Unknown attribute {name}')

    def __delattr__(self, name: str) -> None:
        assert name not in ('arcs', 'scores', 'labels', 'properties',
                            '_properties', '_grad_cache')

        if name in self._tensor_attr:
            del self._tensor_attr[name]
        elif name in self._non_tensor_attr:
            del self._non_tensor_attr[name]
        elif name in self._grad_cache:
            del self._grad_cache[name]
        else:
            super().__delattr__(name)

    def _update_cache(self, name: str, value: Any) -> None:
        self._grad_cache[name] = value

    def get_state_batches(self) -> _k2.RaggedInt:
        '''Get (and compute if necessary) cached property self.state_batches.
           For use by internal k2 code.  Used in many algorithms.'''
        if hasattr(self, 'state_batches') is False:
            state_batches = _k2._get_state_batches(self.arcs, transpose=True)
            self._update_cache('state_batches', state_batches)
        return self.state_batches

    def get_dest_states(self) -> torch.Tensor:
        '''Get (and compute if necessary) cached property self.dest_states.
           For use by internal k2 code, relates to best-path.'''
        if hasattr(self, 'dest_states') is False:
            dest_states = _k2._get_dest_states(self.arcs, as_idx01=True)
            self._update_cache('dest_states', dest_states)
        return self.dest_states

    def get_incoming_arcs(self) -> _k2.RaggedInt:
        '''Get (and compute if necessary) cached property self.incoming_arcs
           For use by internal k2 code, relates to best-path'''
        if hasattr(self, 'incoming_arcs') is False:
            dest_states = self.get_dest_states()
            incoming_arcs = _k2._get_incoming_arcs(self.arcs, dest_states)
            self._update_cache('incoming_arcs', incoming_arcs)
        return self.incoming_arcs

    def get_entering_arc_batches(self) -> _k2.RaggedInt:
        '''Get (and compute if necessary) cached property self.entering_arc_batches
           For use by internal k2 code, used in many algorithms.'''
        if hasattr(self, 'entering_arc_batches') is False:
            incoming_arcs = self.get_incoming_arcs()
            state_batches = self.get_state_batches()
            entering_arc_batches = _k2._get_entering_arc_index_batches(
                self.arcs,
                incoming_arcs=incoming_arcs,
                state_batches=state_batches)
            self._update_cache('entering_arc_batches', entering_arc_batches)
        return self.entering_arc_batches

    def get_leaving_arc_batches(self) -> _k2.RaggedInt:
        '''Get (and compute if necessary) cached property self.leaving_arc_batches
           For use by internal k2 code, used in many algorithms.'''
        if hasattr(self, 'leaving_arc_batches') is False:
            state_batches = self.get_state_batches()
            leaving_arc_batches = _k2._get_leaving_arc_index_batches(
                self.arcs, state_batches)
            self._update_cache('leaving_arc_batches', leaving_arc_batches)
        return self.leaving_arc_batches

    def get_forward_scores_tropical(self, use_float_scores) -> torch.Tensor:
        '''Get (and compute if necessary) cached property
        self.forward_scores_tropical.

        For use by internal k2 code, used in getting best-path or (tropical)
        total-scores.  These are raw forward-scores and not differentiable.'''
        name = 'forward_scores_tropical' + ('float'
                                            if use_float_scores else 'double')
        if hasattr(self, name) is False:
            if use_float_scores:
                func = _k2._get_forward_scores_float
            else:
                func = _k2._get_forward_scores_double

            state_batches = self.get_state_batches()
            entering_arc_batches = self.get_entering_arc_batches()

            forward_scores_tropical, entering_arcs = func(
                self.arcs,
                state_batches=state_batches,
                entering_arc_batches=entering_arc_batches,
                log_semiring=False)
            self._update_cache(name, forward_scores_tropical)
            self._update_cache('entering_arcs', entering_arcs)
        return getattr(self, name)

    def get_forward_scores_log(self, use_float_scores) -> torch.Tensor:
        '''Get (and compute if necessary) cached property
        self.forward_scores_log.

        For use by internal k2 code, used in getting total-score for
        log semiring
        '''
        if hasattr(self, 'forward_scores_log') is False \
                or (use_float_scores is True and self.forward_scores_log.dtype == torch.float64) \
                or (use_float_scores is False and self.forward_scores_log.dtype == torch.float32): # noqa
            if use_float_scores:
                func = _k2._get_forward_scores_float
            else:
                func = _k2._get_forward_scores_double

            state_batches = self.get_state_batches()
            entering_arc_batches = self.get_entering_arc_batches()

            forward_scores_log, _ = func(
                self.arcs,
                state_batches=state_batches,
                entering_arc_batches=entering_arc_batches,
                log_semiring=True)
            self._update_cache('forward_scores_log', forward_scores_log)
        return self.forward_scores_log

    def get_tot_scores_tropical(self, use_float_scores) -> torch.Tensor:
        '''Compute total-scores in tropical semiring (one per FSA), which is the same
           as the best-path score.
           CAUTION: these are just the raw total-scores and are
           not differentiable.   Use k2.get_tot_scores(self) to
           get differentiable total-scores.
        '''
        name = 'tot_scores_tropical_' + ('float'
                                         if use_float_scores else 'double')
        if hasattr(self, name) is False:
            if use_float_scores is True:
                func = _k2._get_tot_scores_float
            else:
                func = _k2._get_tot_scores_double
            forward_scores_tropical = self.get_forward_scores_tropical(
                use_float_scores)
            tot_scores_tropical = func(self.arcs, forward_scores_tropical)
            self._update_cache(name, tot_scores_tropical)
        return getattr(self, name)

    def get_tot_scores_log(self, use_float_scores) -> torch.Tensor:
        '''Compute total-scores in log semiring (one per FSA).
           as the best-path score.
           CAUTION: these are just the raw total-scores and are not
           differentiable.  Use k2.get_tot_scores(self) to get differentiable
           total-scores.
        '''
        name = 'tot_scores_log_' + ('float' if use_float_scores else 'double')
        if hasattr(self, name) is False:
            if use_float_scores is True:
                func = _k2._get_tot_scores_float
            else:
                func = _k2._get_tot_scores_double
            forward_scores_log = self.get_forward_scores_log(use_float_scores)
            tot_scores_log = func(self.arcs, forward_scores_log)
            self._update_cache(name, tot_scores_log)
        return getattr(self, name)

    def get_backward_scores_tropical(self, use_float_scores) -> torch.Tensor:
        '''Compute backward-scores in tropical semiring, i.e. best-path-to-end
           costs.  For internal k2 use.  Not differentiable.
        '''
        name = 'backward_scores_tropical_' + ('float' if use_float_scores else
                                              'double')
        if hasattr(self, name) is False:
            if use_float_scores:
                func = _k2._get_backward_scores_float
            else:
                func = _k2._get_backward_scores_double

            state_batches = self.get_state_batches()
            leaving_arc_batches = self.get_leaving_arc_batches()
            tot_scores_tropical = self.get_tot_scores_tropical(
                use_float_scores)
            backward_scores_tropical = func(
                self.arcs,
                state_batches=state_batches,
                leaving_arc_batches=leaving_arc_batches,
                tot_scores=tot_scores_tropical,
                log_semiring=False)
            self._update_cache(name, backward_scores_tropical)
        return getattr(self, name)

    def get_backward_scores_log(self, use_float_scores) -> torch.Tensor:
        '''Compute backward-scores in tropical semiring, i.e. total-score-to-end.
           for each state.  For internal k2 use.  Not differentiable.
        '''
        name = 'backward_scores_log_' + ('float'
                                         if use_float_scores else 'double')
        if hasattr(self, name) is False:
            if use_float_scores:
                func = _k2._get_backward_scores_float
            else:
                func = _k2._get_backward_scores_double

            state_batches = self.get_state_batches()
            leaving_arc_batches = self.get_leaving_arc_batches()
            tot_scores_log = self.get_tot_scores_log(use_float_scores)
            backward_scores_log = func(self.arcs,
                                       state_batches=state_batches,
                                       leaving_arc_batches=leaving_arc_batches,
                                       tot_scores=tot_scores_log,
                                       log_semiring=True)
            self._update_cache(name, backward_scores_log)
        return getattr(self, name)

    def get_entering_arcs(self, use_float_scores) -> torch.Tensor:
        '''Compute, for each state, the index of the best arc entering it.
           For internal k2 use.
        '''
        if hasattr(self, 'entering_arcs') is False:
            self.get_forward_scores_tropical(use_float_scores)
        # If the following fails, there may be some kind of code error.
        # It should be computed when we call get_forward_scores_tropical,
        # unless someone
        return self.entering_arcs

    def requires_grad_(self, requires_grad: bool) -> 'Fsa':
        '''Change if autograd should record operations on this FSA:

        Sets the `scores`'s requires_grad attribute in-place.
        Returns this FSA.
        You can test whether this object has the requires_grad property
        true or false by accessing self.requires_grad (handled in
        __getattr__).

        Caution:
          This is an **in-place** operation as you can see that the function
          name ends with `_`.

        Args:
          requires_grad:
            If autograd should record operations on this FSA or not.

        Returns:
          This FSA itself.
        '''
        self.scores.requires_grad_(requires_grad)
        return self

    def invert_(self) -> 'Fsa':
        '''Swap the ``labels`` and ``aux_labels``.

        If there are symbol tables associated with ``labels`` and
        ``aux_labels``, they are also swapped.

        It is a no-op if the FSA contains no ``aux_labels``.

        CAUTION:
          The function name ends with an underscore which means this
          is an **in-place** operation.

        Returns:
          Return ``self``.
        '''
        if hasattr(self, 'aux_labels'):
            aux_labels = self.aux_labels
            self.aux_labels = self.labels.clone()
            self.labels = aux_labels

        symbols = getattr(self, 'symbols', None)
        aux_symbols = getattr(self, 'aux_symbols', None)

        if symbols is not None:
            del self.symbols

        if aux_symbols is not None:
            del self.aux_symbols

        if symbols is not None:
            self.aux_symbols = symbols

        if aux_symbols is not None:
            self.symbols = aux_symbols

        self._init_properties()
        return self

    def is_cpu(self) -> bool:
        '''Return true if this FSA is on CPU.

        Returns:
          True if the FSA is on CPU; False otherwise.
        '''
        return self.arcs.is_cpu()

    def is_cuda(self) -> bool:
        '''Return true if this FSA is on GPU.

        Returns:
          True if the FSA is on GPU; False otherwise.
        '''
        return self.arcs.is_cuda()

    @property
    def device(self) -> torch.device:
        return self.scores.device

    def __getitem__(self, i: int) -> 'Fsa':
        '''Get the i-th FSA.

        Caution:
          `self` has to be an FsaVec.

        Args:
          i:
            The i-th FSA to select. 0 <= i < self.arcs.dim0().

        Returns:
          The i-th FSA. Note it is a single FSA.
        '''
        assert len(self.shape) == 3
        assert 0 <= i < self.shape[0]
        ragged_arc, start = self.arcs.index(0, i)
        end = start + ragged_arc.values().shape[0]

        out_fsa = Fsa.from_ragged_arc(ragged_arc)
        for name, value in self.named_tensor_attr():
            setattr(out_fsa, name, value[start:end])

        for name, value in self.named_non_tensor_attr():
            setattr(out_fsa, name, value)

        out_fsa._init_properties()
        return out_fsa

    def arcs_as_tensor(self) -> torch.Tensor:
        '''Return the core part of the Fsa (the arcs) serialized to a Tensor.
           This can be passed to the constructor, along with the aux_labels if
           present, to reconstruct this object.
           A more convenient way to serialize a Tensor is to use `as_dict`
           and `from_dict`
        '''
        return _fsa_to_tensor(self.arcs)

    def as_dict(self) -> Dict[str, Any]:
        '''Convert this Fsa to a dict (probably for purposes of serialization
          with, e.g., torch.save).
        '''
        ans = dict()
        ans['arcs'] = _fsa_to_tensor(self.arcs)
        if hasattr(self, 'aux_labels'):
            ans['aux_labels'] = self.aux_labels
        # TODO(dan): add other properties, e.g. from _tensor_attr and
        # _non_tensor_attr.
        return ans

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> 'Fsa':
        # TODO(dan): deal with other properties, e.g. that will go to from
        # _tensor_attr and _non_tensor_attr.
        return Fsa(dict_in['arcs'], aux_labels=dict_in.get('aux_labels', None))

    def to_(self, device: torch.device) -> 'Fsa':
        '''Move the FSA onto a given device.

        Caution:
          This is an in-place operation.

        Args:
          device:
            An instance of `torch.device`. It supports only cpu and cuda.

        Returns:
          Return `self`.
        '''
        assert device.type in ('cpu', 'cuda')
        if device.type == 'cpu' and self.is_cpu():
            return self
        elif device.type == 'cuda' and self.is_cuda():
            return self

        if device.type == 'cpu':
            self.arcs = self.arcs.to_cpu()
        else:
            self.arcs = self.arcs.to_cuda(device.index)

        for name, value in self.named_tensor_attr():
            setattr(self, name, value.to(device))

        self._grad_cache = OrderedDict()

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
          ``(num_states, None)`` if this is an Fsa;
          ``(num_fsas, None, None)`` if this is an FsaVec.
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
        ans._init_properties()
        ans._tensor_attr['scores'] = _as_float(ans.arcs.values()[:, -1])
        return ans

    @classmethod
    def from_str(cls, s: str) -> 'Fsa':
        '''Create an Fsa from a string in the k2 format.
        (See also from_openfst).

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
        ans._init_properties()
        ans._tensor_attr['scores'] = _as_float(ans.arcs.values()[:, -1])
        if aux_labels is not None:
            ans.aux_labels = aux_labels.to(torch.int32)
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
        ans._init_properties()
        ans._tensor_attr['scores'] = _as_float(ans.arcs.values()[:, -1])
        if aux_labels is not None:
            ans.aux_labels = aux_labels.to(torch.int32)
        return ans
