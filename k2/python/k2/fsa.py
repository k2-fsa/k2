# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corp.       (author: Daniel Povey, Haowen Qiu)
#                      Guoguo Chen
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import Union

import os
import shutil
import torch

import k2
import k2.ragged
import _k2

from _k2 import RaggedArc
from k2 import fsa_properties


class Fsa(object):
    '''This class represents a single fsa or a vector of fsas.

    When it denotes a single FSA, its attribute :attr:`shape` is a tuple
    containing two elements `(num_states, None)`; when it represents
    a vector of FSAs it is a tuple with three
    elements `(num_fsas, None, None)`.

    CAUTION:
      It's possible for a vector of FSAs to have zero or one elements.

    An instance of FSA has the following attributes:

    arcs
      You will NOT use it directly in Python. It is an instance of
      `_k2.RaggedArc` with only one method `values()` which
      returns a 2-D `torch.Tensor` of dtype `torch.int32` with 4
      columns. Its number of rows indicates the number of arcs in the
      FSA. The first column represents the source states, second
      column the destination states, third column the labels and the
      fourth column is the score. Note that the score is actually
      a float number but it is **reinterpreted** as an integer.

    scores
      A 1-D `torch.Tensor` of dtype `torch.float32`. It has
      as many entries as the number of arcs representing the score
      of every arc.

    labels
      1-D `torch.Tensor` of dtype `torch.int32`. It has as
      many entries as the number of arcs representing the label of
      every arc.


    It MAY have the following attributes:

    symbols
      An instance of `k2.SymbolTable`. It maps an entry in
      `labels` to an integer and vice versa. It is used for
      visualization only.

    aux_labels
     A 1-D `torch.Tensor` of dtype `torch.int32` or a ragged tensor with type
     `_k2.RaggedInt`. It contains auxiliary labels per arc.  If it's a tensor,
     `aux_labels.numel()` equals to the number of arcs.  if it's
     `_k2.RaggedInt`, then `aux_labels.dim0()` equals to the number of arcs.

    aux_symbols
      An instance of `k2.SymbolTable`. It maps an entry in
      `aux_labels` to an integer and vice versa.

    properties
      An integer that encodes the properties of the FSA. It is
      accessed as fsa.properties (read-only!)

    It MAY have other attributes that set by users.  Tensor attributes should
    have the same 1st dimension as the number of arcs in the FSA, Ragged
    attributes should have the same `dim0` as the number of arcs in the FSA.

    CAUTION:
      When an attribute is an instance of `torch.Tensor`, its `shape[0]`
      has to be equal to the number arcs. Otherwise, an assertion error
      will be thrown.
      When an attribute is an instance of `_k2.RaggedInt`, its `dim0()`
      has to be equal to the number arcs. Otherwise, an assertion error
      will be thrown.

    NOTE:
      `symbols` and `aux_symbols` are symbol tables, while `labels`
      is instances of `torch.Tensor` and `aux_labels` is instances of
      `torch.Tensor` or `_k2.RaggedInt`.

      Implementation note: most of this class's attributes are not
      real attributes in the object's dict; the real attributes are
      `arcs`, `_non_tensor_attr`, `_tensor_attr`, `_properties`,
      `_cache`.

    '''

    def __init__(
            self,
            arcs: Union[torch.Tensor, RaggedArc],
            aux_labels: Optional[Union[torch.Tensor, _k2.RaggedInt]] = None,
            properties=None) -> None:
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
            `torch.int32` or `_k2.RaggedInt` whose `dim0()` equals to the
            number of arcs.

          properties:
            Tensor properties if known (should only be provided by
            internal code, as they are not checked; intended for use
            by `Fsa.clone()`)

        Returns:
          An instance of Fsa.
        '''
        if isinstance(arcs, torch.Tensor):
            arcs: RaggedArc = _k2.fsa_from_tensor(arcs)
        assert isinstance(arcs, RaggedArc)

        # Accessing self.__dict__ bypasses __setattr__.
        self.__dict__['arcs'] = arcs
        self.__dict__['_properties'] = properties

        # - `_tensor_attr`
        #     It saves attribute values of type torch.Tensor. `shape[0]` of
        #     attribute values have to be equal to the number of arcs
        #     in the FSA.  There are a couple of standard ones, 'aux_labels'
        #     (present for transducers), and 'scores'. It also saves
        #     attribute values of type _k2.RaggedInt, e.g. `aux_labels` if
        #     it has type of _k2.RaggedInt instead of torch.Tensor.
        #
        # - `_non_tensor_attr`
        #     It saves non-tensor attributes, e.g., :class:`SymbolTable`.
        #
        # - `_cache`
        #     It contains tensors for autograd. Users should NOT manipulate it.
        #     The dict is filled in automagically.
        #
        # The `_cache` dict contains the following attributes:
        #
        #  - `state_batches`:
        #           returned by :func:`_k2.get_state_batches`
        #  - `dest_states`:
        #           returned by :func:`_k2.get_dest_states`
        #  - `incoming_arcs`:
        #           returned by :func:`_k2.get_incoming_arcs`
        #  - `entering_arc_batches`:
        #           returned by :func:`_k2.get_entering_arc_index_batches`
        #  - `leaving_arc_batches`:
        #           returned by :func:`_k2.get_leaving_arc_index_batches`
        #  - `forward_scores_tropical`:
        #           returned by :func:`_k2.get_forward_scores_float`
        #           with `log_semiring=False`
        #  - `forward_scores_log`:
        #           returned by :func:`_k2.get_forward_scores_float` or
        #           :func:`_get_forward_scores_double` with `log_semiring=True`
        #  - `tot_scores_tropical`:
        #           returned by :func:`_k2.get_tot_scores_float` or
        #           :func:`_k2.get_tot_scores_double` with
        #           `forward_scores_tropical`.
        #  - `tot_scores_log`:
        #           returned by :func:`_k2.get_tot_scores_float` or
        #           :func:`_k2.get_tot_scores_double` with
        #           `forward_scores_log`.
        #  - `backward_scores_tropical`:
        #           returned by :func:`_k2.get_backward_scores_float` or
        #           :func:`_k2.get_backward_scores_double` with
        #           `log_semiring=False`
        #  - `backward_scores_log_semiring`:
        #           returned by :func:`_k2.get_backward_scores_float` or
        #           :func:`_k2.get_backward_scores_double` with
        #           `log_semiring=True`
        #  - `entering_arcs`:
        #           returned by :func:`_k2.get_forward_scores_float` or
        #           :func:`_get_forward_scores_double` with `log_semiring=False`

        for name in ['_tensor_attr', '_non_tensor_attr', '_cache']:
            self.__dict__[name] = dict()

        self._tensor_attr['scores'] = _k2.as_float(self.arcs.values()[:, -1])
        if aux_labels is not None:
            if isinstance(aux_labels, torch.Tensor):
                self.aux_labels = aux_labels.to(torch.int32)
            else:
                # ragged tensor
                self.aux_labels = aux_labels
        # Access the properties field (it's a @property, i.e. it has a
        # getter) which sets up the properties and also checks that
        # the FSA is valid.
        _ = self.properties

    def __str__(self) -> str:
        '''Return a string representation of this object

        For visualization and debug only.
        '''
        if hasattr(self, 'aux_labels') and isinstance(self.aux_labels,
                                                      torch.Tensor):
            aux_labels = self.aux_labels.to(torch.int32)
        else:
            aux_labels = None
        if self.arcs.num_axes() == 2:
            ans = 'k2.Fsa: ' + _k2.fsa_to_str(self.arcs, False, aux_labels)
        else:
            ans = 'k2.FsaVec: \n'
            for i in range(self.shape[0]):
                # get the i-th Fsa
                ragged_arc, start = self.arcs.index(0, i)
                end = start + ragged_arc.values().shape[0]
                ans += 'FsaVec[' + str(i) + ']: ' + _k2.fsa_to_str(
                    ragged_arc, False,
                    None if aux_labels is None else aux_labels[start:end])
        ans += 'properties_str = ' + _k2.fsa_properties_as_str(
            self._properties) + '.'
        for name, value in self.named_tensor_attr(include_scores=False):
            sep = '\n'
            ans += f'{sep}{name}: {value}'
        for name, value in self.named_non_tensor_attr():
            if name == 'symbols':
                continue
            sep = '\n'
            ans += f'{sep}{name}: {value}'

        return ans

    def draw(self, filename: Optional[str],
             title: Optional[str] = None) -> 'Digraph':  # noqa
        '''
        Render FSA as an image via graphviz, and return the Digraph object;
        and optionally save to file `filename`.
        `filename` must have a suffix that graphviz understands, such as
        `pdf`, `svg` or `png`.

        Args:
           filename:
              Filename to (optionally) save to, e.g. 'foo.png', 'foo.svg',
              'foo.png'  (must have a suffix that graphviz understands).
           title:
              Title to be displayed in image, e.g. 'A simple FSA example'
        '''

        digraph = k2.utils.to_dot(self, title=title)

        _, extension = os.path.splitext(filename)
        if extension == '' or extension[0] != '.':
            raise ValueError(
                "Filename needs to have a suffix like .png, .pdf, .svg: {}".
                format(filename))

        if filename:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_fn = digraph.render(filename='temp',
                                         directory=tmp_dir,
                                         format=extension[1:],
                                         cleanup=True)

                shutil.move(temp_fn, filename)
        return digraph

    def __setattr__(self, name: str, value: Any) -> None:
        '''
        Caution:
          We save a reference to `value`. If you need to change `value`
          afterwards, please consider passing a copy of it.

        Args:
          name:
            Name of the attribute.
          value:
            Value of the attribute.
        '''

        assert name not in ('_tensor_attr', '_non_tensor_attr', 'arcs',
                            '_cache', '_properties', 'properties')

        if isinstance(value, torch.Tensor):
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
                self.arcs.values()[:, -1] = _k2.as_int(value.detach())
        elif isinstance(value, _k2.RaggedInt):
            assert value.dim0() == self.arcs.values().shape[0]
            self._tensor_attr[name] = value
        else:
            self._non_tensor_attr[name] = value

    @property
    def labels(self) -> torch.Tensor:
        '''Return the labels.

        Returns:
          Return a 1-D `torch.Tensor` with dtype `torch.int32`.
        '''
        try:
            return self.arcs.values()[:, 2]
        except Exception as e:
            # print the exception because it will probably be lost, since
            # python's getting code will back off to __getattr__.
            import traceback
            traceback.print_exc()
            raise e

    @labels.setter
    def labels(self, values) -> None:
        '''Set labels.

        Args:
          values:
            A 1-D `torch.tensor` with dtype `torch.int32`.
        '''
        assert values.dtype == torch.int32
        self.arcs.values()[:, 2] = values
        # Invalidate the properties since we changed the labels.
        self.__dict__['_properties'] = None

    @property
    def properties(self) -> int:
        # instead of accessing self._properties, we use
        # self.__dict__.{get,set}('_properties') in order to
        # avoid calling __getattr__ and any complexity involved in that.
        properties = self.__dict__.get('_properties', None)
        if properties is not None:
            return properties  # Return cached properties.

        if self.arcs.num_axes() == 2:
            properties = _k2.get_fsa_basic_properties(self.arcs)
        else:
            properties = _k2.get_fsa_vec_basic_properties(self.arcs)
        self.__dict__['_properties'] = properties
        if properties & fsa_properties.VALID != 1:
            raise ValueError(
                'Fsa is not valid, properties are: {} = {}, arcs are: {}'.
                format(properties, fsa_properties.to_str(properties),
                       str(self.arcs)))
        return properties

    @property
    def properties_str(self) -> str:
        return _k2.fsa_properties_as_str(self.properties)

    @property
    def requires_grad(self) -> bool:
        return self.scores.requires_grad

    @property
    def grad(self) -> torch.Tensor:
        return self.scores.grad

    def __getattr__(self, name: str) -> Any:
        '''
        Note: for attributes that exist as properties, e.g.
        self.labels, self.properties, self.requires_grad, we won't
        reach this code because Python checks the class dict before
        calling getattr.  The same is true for instance attributes
        such as self.{_tensor_attr,_non_tensor_attr,_cache,_properties}

        The 'virtual' members of this class are those in self._tensor_attr
        and self._non_tensor_attr.
        '''
        if name in self._tensor_attr:
            return self._tensor_attr[name]
        elif name in self._non_tensor_attr:
            return self._non_tensor_attr[name]
        elif name in self._cache:
            return self._cache[name]

        raise AttributeError(f'Unknown attribute {name}')

    def __delattr__(self, name: str) -> None:
        # We won't allow deletion of class attributes such as @property
        # getters
        assert name not in Fsa.__dict__
        # ... or instance attributes such as self._tensor_attr or
        # self._properties
        assert name not in self.__dict__

        if name in self._tensor_attr:
            del self._tensor_attr[name]
        elif name in self._non_tensor_attr:
            del self._non_tensor_attr[name]
        elif name in self._cache:
            del self._cache[name]
        else:
            raise AttributeError('No such attribute in Fsa: ' + name)

    def get_state_batches(self) -> _k2.RaggedInt:
        '''Get (and compute if necessary) cached property self.state_batches.
           For use by internal k2 code.  Used in many algorithms.'''
        name, cache = 'state_batches', self._cache
        if name not in cache:
            cache[name] = _k2.get_state_batches(self.arcs, transpose=True)
        return cache[name]

    def get_dest_states(self) -> torch.Tensor:
        '''Get (and compute if necessary) cached property self.dest_states.
           For use by internal k2 code, relates to best-path.'''
        name, cache = 'dest_states', self._cache
        if name not in cache:
            cache[name] = _k2.get_dest_states(self.arcs, as_idx01=True)
        return cache[name]

    def get_incoming_arcs(self) -> _k2.RaggedInt:
        '''Get (and compute if necessary) cached property self.incoming_arcs
           For use by internal k2 code, relates to best-path'''
        name, cache = 'incoming_arcs', self._cache
        if name not in cache:
            cache[name] = _k2.get_incoming_arcs(self.arcs,
                                                self.get_dest_states())
        return cache[name]

    def get_entering_arc_batches(self) -> _k2.RaggedInt:
        '''Get (and compute if necessary) cached property self.entering_arc_batches
           For use by internal k2 code, used in many algorithms.'''
        name, cache = 'entering_arc_batches', self._cache
        if name not in cache:
            cache[name] = _k2.get_entering_arc_index_batches(
                self.arcs,
                incoming_arcs=self.get_incoming_arcs(),
                state_batches=self.get_state_batches())
        return cache[name]

    def get_leaving_arc_batches(self) -> _k2.RaggedInt:
        '''Get (and compute if necessary) cached property self.leaving_arc_batches
           For use by internal k2 code, used in many algorithms.'''
        name, cache = 'leaving_arc_batches', self._cache
        if name not in cache:
            cache[name] = _k2.get_leaving_arc_index_batches(
                self.arcs, self.get_state_batches())
        return cache[name]

    def get_forward_scores_tropical(self,
                                    use_double_scores: bool) -> torch.Tensor:
        '''Get (and compute if necessary) cached property
        self.forward_scores_tropical.

        For use by internal k2 code, used in getting best-path or (tropical)
        total-scores.  These are raw forward-scores and not differentiable.

        Args:
          use_double_scores:
            True to use `double precision` floating point.
            False to use `single precision`.
        '''
        name = 'forward_scores_tropical' + ('double'
                                            if use_double_scores else 'float')
        cache = self._cache
        if name not in cache:
            if use_double_scores:
                func = _k2.get_forward_scores_double
            else:
                func = _k2.get_forward_scores_float
            forward_scores_tropical, entering_arcs = func(
                self.arcs,
                state_batches=self.get_state_batches(),
                entering_arc_batches=self.get_entering_arc_batches(),
                log_semiring=False)
            cache[name] = forward_scores_tropical
            cache['entering_arcs'] = entering_arcs
        return cache[name]

    def get_forward_scores_log(self, use_double_scores: bool) -> torch.Tensor:
        '''Get (and compute if necessary) cached property
        self.forward_scores_log.

        For use by internal k2 code, used in getting total-score for
        log semiring.

        Args:
          use_double_scores:
            True to use `double precision` floating point.
            False to use `single precision`.
        '''
        name = 'forward_scores_log' + ('double'
                                       if use_double_scores else 'float')
        cache = self._cache
        if name not in cache:
            if use_double_scores:
                func = _k2.get_forward_scores_double
            else:
                func = _k2.get_forward_scores_float
            cache[name], _ = func(
                self.arcs,
                state_batches=self.get_state_batches(),
                entering_arc_batches=self.get_entering_arc_batches(),
                log_semiring=True)
        return cache[name]

    def get_tot_scores_tropical(self, use_double_scores: bool) -> torch.Tensor:
        '''Compute total-scores in tropical semiring (one per FSA), which is the same
        as the best-path score.

        CAUTION:
          These are just the raw total-scores and are not differentiable.
          Use `k2.get_tot_scores(self)` to get differentiable total-scores.

        Args:
          use_double_scores:
            True to use `double precision` floating point.
            False to use `single precision`.
        '''
        name = 'tot_scores_tropical_' + ('double'
                                         if use_double_scores else 'false')
        cache = self._cache
        if name not in cache:
            if use_double_scores is True:
                func = _k2.get_tot_scores_double
            else:
                func = _k2.get_tot_scores_float
            forward_scores_tropical = self.get_forward_scores_tropical(
                use_double_scores)
            cache[name] = func(self.arcs, forward_scores_tropical)
        return cache[name]

    def get_tot_scores_log(self, use_double_scores: bool) -> torch.Tensor:
        '''Compute total-scores in log semiring (one per FSA) as the
        best-path score.

        CAUTION:
          These are just the raw total-scores and are not differentiable.
          Use `k2.get_tot_scores(self)` to get differentiable total-scores.

        Args:
          use_double_scores:
            True to use `double precision` floating point.
            False to use `single precision`.
        '''
        name = 'tot_scores_log_' + ('double' if use_double_scores else 'float')
        cache = self._cache
        if name not in cache:
            if use_double_scores is True:
                func = _k2.get_tot_scores_double
            else:
                func = _k2.get_tot_scores_float
            forward_scores_log = self.get_forward_scores_log(use_double_scores)
            cache[name] = func(self.arcs, forward_scores_log)
        return cache[name]

    def get_backward_scores_tropical(self,
                                     use_double_scores: bool) -> torch.Tensor:
        '''Compute backward-scores in tropical semiring, i.e. best-path-to-end
        costs.  For internal k2 use.  Not differentiable.

        Args:
          use_double_scores:
            True to use `double precision` floating point.
            False to use `single precision`.
        '''
        name = 'backward_scores_tropical_' + ('double' if use_double_scores
                                              else 'float')
        cache = self._cache
        if name not in cache:
            if use_double_scores:
                func = _k2.get_backward_scores_double
            else:
                func = _k2.get_backward_scores_float

            state_batches = self.get_state_batches()
            leaving_arc_batches = self.get_leaving_arc_batches()
            tot_scores_tropical = self.get_tot_scores_tropical(
                use_double_scores)
            backward_scores_tropical = func(
                self.arcs,
                state_batches=state_batches,
                leaving_arc_batches=leaving_arc_batches,
                tot_scores=tot_scores_tropical,
                log_semiring=False)
            cache[name] = backward_scores_tropical
        return cache[name]

    def get_backward_scores_log(self, use_double_scores: bool) -> torch.Tensor:
        '''Compute backward-scores in tropical semiring, i.e. total-score-to-end.
        for each state.  For internal k2 use.  Not differentiable.

        Args:
          use_double_scores:
            True to use `double precision` floating point.
            False to use `single precision`.
        '''
        name = 'backward_scores_log_' + ('double'
                                         if use_double_scores else 'float')
        cache = self._cache
        if name not in cache:
            if use_double_scores:
                func = _k2.get_backward_scores_double
            else:
                func = _k2.get_backward_scores_float

            state_batches = self.get_state_batches()
            leaving_arc_batches = self.get_leaving_arc_batches()
            tot_scores_log = self.get_tot_scores_log(use_double_scores)
            cache[name] = func(self.arcs,
                               state_batches=state_batches,
                               leaving_arc_batches=leaving_arc_batches,
                               tot_scores=tot_scores_log,
                               log_semiring=True)
        return cache[name]

    def get_entering_arcs(self, use_double_scores: bool) -> torch.Tensor:
        '''Compute, for each state, the index of the best arc entering it.
        For internal k2 use.

        Args:
          use_double_scores:
            True to use `double precision` floating point.
            False to use `single precision`.
        '''
        name, cache = 'entering_arcs', self._cache
        if name not in cache:
            # the following will set self._cache['entering_arcs']
            self.get_forward_scores_tropical(use_double_scores)
        return cache[name]

    def requires_grad_(self, requires_grad: bool) -> 'Fsa':
        '''Change if autograd should record operations on this FSA:

        Sets the `scores`'s requires_grad attribute in-place.
        Returns this FSA.
        You can test whether this object has the requires_grad property
        true or false by accessing self.requires_grad (handled in
        `__getattr__`).

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
        '''Swap the `labels` and `aux_labels`.

        If there are symbol tables associated with `labels` and
        `aux_labels`, they are also swapped.

        It is an error if the FSA contains no `aux_labels`.

        CAUTION:
          The function name ends with an underscore which means this
          is an **in-place** operation.

        Returns:
          Return `self`.
        '''
        if not hasattr(self, 'aux_labels'):
            raise RuntimeError(
                'invert_ cannot be called on acceptors (no aux_labels)')
        if not isinstance(self.aux_labels, torch.Tensor):
            raise RuntimeError('current invert_ method only supports case '
                               'where aux_labels is a tensor')

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
        self.__dict__['_properties'] = None
        # access self.properties which will do a validity check on the modified
        # FSA after getting the properties
        self.properties
        return self

    def invert(self) -> 'Fsa':
        '''Swap the `labels` and `aux_labels`.

        If there are symbol tables associated with `labels` and
        `aux_labels`, they are also swapped.

        It is an error if the FSA contains no `aux_labels`.

        Returns:
          Return a new Fsa.
        '''
        return self.clone().invert_()

    def is_cpu(self) -> bool:
        '''Return true if this FSA is on CPU.

        Returns:
          True if the FSA is on CPU; False otherwise.
        '''
        return self.device.type == 'cpu'

    def is_cuda(self) -> bool:
        '''Return true if this FSA is on GPU.

        Returns:
          True if the FSA is on GPU; False otherwise.
        '''
        return self.device.type == 'cuda'

    @property
    def device(self) -> torch.device:
        return self.scores.device

    def __getitem__(self, i: int) -> 'Fsa':
        '''Get the i-th FSA.

        Caution:
          `self` has to be an FsaVec, i.e. len(self.shape) == 3
        Args:
          i: The i-th FSA to select. 0 <= i < self.arcs.dim0().
        Returns:
          The i-th FSA. Note it is a single FSA.
        '''
        assert len(self.shape) == 3
        assert 0 <= i < self.shape[0]
        ragged_arc, start = self.arcs.index(0, i)
        end = start + ragged_arc.values().shape[0]

        out_fsa = Fsa(ragged_arc)
        for name, value in self.named_tensor_attr(include_scores=False):
            if isinstance(value, torch.Tensor):
                setattr(out_fsa, name, value[start:end])
            else:
                assert isinstance(value, _k2.RaggedInt)
                setattr(out_fsa, name,
                        _k2.ragged_int_arange(value, 0, start, end))

        for name, value in self.named_non_tensor_attr():
            setattr(out_fsa, name, value)

        # The following is a magic invocation to make sure
        # the backprop on the scores happens.
        k2.autograd_utils.phantom_set_scores_to(out_fsa,
                                                self.scores[start:end])

        return out_fsa

    def arcs_as_tensor(self) -> torch.Tensor:
        '''Return the core part of the Fsa (the arcs) serialized to a Tensor.
           This can be passed to the constructor, along with the aux_labels if
           present, to reconstruct this object.
           A more convenient way to serialize a Tensor is to use `as_dict`
           and `from_dict`
        '''
        return _k2.fsa_to_tensor(self.arcs)

    def as_dict(self) -> Dict[str, Any]:
        '''Convert this Fsa to a dict (probably for purposes of serialization
        , e.g., torch.save).

        Caution:
          `self.requires_grad` attribute is not saved.
        Returns:
          A `dict` that can be used to reconstruct this FSA by using
          `Fsa.from_dict`.
        '''
        ans = dict()
        ans['arcs'] = _k2.fsa_to_tensor(self.arcs)

        for name, value in self.named_tensor_attr(include_scores=False):
            ans[name] = value

        for name, value in self.named_non_tensor_attr():
            ans[name] = value

        return ans

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> 'Fsa':
        fsa = Fsa(dict_in['arcs'], aux_labels=dict_in.get('aux_labels', None))
        for key, value in dict_in.items():
            if key in ['arcs', 'aux_labels']:
                continue
            setattr(fsa, key, value)
        return fsa

    def to(self, device: Union[str, torch.device]) -> 'Fsa':
        '''Move the FSA onto a given device.

        Args:
          device:
            An instance of `torch.device` or a string that can be used to
            construct a `torch.device`, e.g., 'cpu', 'cuda:0'.
            It supports only cpu and cuda devices.

        Returns:
          Returns a new Fsa which is this object copied to the given device
          (or this object itself, if the device was the same)
        '''
        # Keep this code in sync with that in clone()
        if isinstance(device, str):
            device = torch.device(device)

        assert device.type in ('cpu', 'cuda')
        if device == self.scores.device:
            return self

        ans = Fsa(self.arcs.to(device), properties=self.properties)

        for name, value in self.named_tensor_attr(include_scores=False):
            setattr(ans, name, value.to(device))

        for name, value in self.named_non_tensor_attr():
            setattr(ans, name, value)

        # Don't copy members of self._cache.  They don't all have convenient
        # .to() methods.

        # The following is a magic invocation to make sure
        # the backprop happens.
        k2.autograd_utils.phantom_set_scores_to(ans, self.scores.to(device))

        return ans

    def clone(self) -> 'Fsa':
        '''
        Return an Fsa that is a clone of this one, i.e. a close approximation
        to what you'd get if you did .clone() on all its tensor members.
        Any non-tensor attributes are copied over.
        '''
        # Keep this code in sync with that in to()
        ans = Fsa(self.arcs.clone(), properties=self.properties)

        for name, value in self.named_tensor_attr(include_scores=False):
            setattr(ans, name, value.clone())

        for name, value in self.named_non_tensor_attr():
            setattr(ans, name, value)

        # Just copy elements of the _cache that we might already have..
        # These don't directly participate in autograd, and are not supposed to
        # be modified by the user, so this should be safe (i.e. it should
        # be safe to do this without clone(); these are mostly not tensors
        # anyway.
        for name, value in self._cache:
            ans._cache[name] = value

        # The following is a magic invocation to make sure
        # the backprop happens.
        k2.autograd_utils.phantom_set_scores_to(ans, self.scores)

        return ans

    def detach(self) -> 'Fsa':
        '''
        Return an Fsa that shares the underlying data with this one,
        except gradients are not tracked.
        Any non-tensor attributes are copied over.
        '''
        ans = Fsa(self.arcs, properties=self.properties)

        for name, value in self.named_tensor_attr(include_scores=False):
            setattr(ans, name, value.detach())

        for name, value in self.named_non_tensor_attr():
            setattr(ans, name, value)

        # Just copy elements of the _cache that we might already have..
        # These don't directly participate in autograd, and are not supposed to
        # be modified by the user, so this should be safe (i.e. it should
        # be safe to do this without clone(); these are mostly not tensors
        # anyway.
        for name, value in self._cache:
            ans._cache[name] = value
        return ans

    def named_tensor_attr(self, include_scores: bool = True
                         ) -> Iterator[Tuple[str, torch.Tensor]]:  # noqa
        '''Return an iterator over tensor attributes containing both
        the name of the attribute as well as the tensor value.

        Returns:
          A tuple containing the name and the value.
        '''
        if include_scores:
            for name, value in self._tensor_attr.items():
                yield name, value
        else:
            for name, value in self._tensor_attr.items():
                if name != 'scores':
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
          `(num_states, None)` if this is an Fsa;
          `(num_fsas, None, None)` if this is an FsaVec.
        '''
        if self.arcs.num_axes() == 2:
            return (self.arcs.dim0(), None)
        elif self.arcs.num_axes() == 3:
            return (self.arcs.dim0(), None, None)
        else:
            raise ValueError(f'Unsupported num_axes: {self.arcs.num_axes()}')

    @classmethod
    def from_str(cls, s: str) -> 'Fsa':
        '''Create an Fsa from a string in the k2 format.
        (See also from_openfst).

        The given string `s` consists of lines with the following format:

        (1) When it represents an acceptor::

                src_state dest_state label score

        (2) When it represents a transducer::

                src_state dest_state label aux_label score

        The line for the final state consists of only one field::

                final_state

        Note:
          Fields are separated by space(s), tab(s) or both. The `score`
          field is a float, while other fields are integers.

        Caution:
          The first column has to be non-decreasing.

        Caution:
          The final state has the largest state number. There is **ONLY**
          ONE final state. All arcs that are connected to the final state
          have label -1. If there are aux_labels, they are also -1 for
          arcs entering the final state.

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
        arcs, aux_labels = _k2.fsa_from_str(s, acceptor, False)
        ans = Fsa(arcs, aux_labels=aux_labels)
        return ans

    @classmethod
    def from_openfst(cls, s: str, acceptor: bool = True) -> 'Fsa':
        '''Create an Fsa from a string in OpenFST format.

        The given string `s` consists of lines with the following format:

        (1) When it represents an acceptor::

                src_state dest_state label score

        (2) When it represents a transducer::

                src_state dest_state label aux_label score

        The line for the final state consists of two fields::

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
        arcs, aux_labels = _k2.fsa_from_str(s, acceptor, True)
        return Fsa(arcs, aux_labels=aux_labels)

    def set_scores_stochastic_(self, scores) -> None:
        '''Normalize the given `scores` and assign it to `self.scores`.

        Args:
          scores:
            Tensor of scores of dtype torch.float32, and shape equal to
            `self.scores.shape` (one axis). Will be normalized so the
            sum, after exponentiating, of the scores leaving each state
            that has at least one arc leaving it is 1.

        Caution:
          The function name ends with an underline indicating this function
          will modify `self` **in-place**.
        '''
        assert scores.ndim == 1
        assert scores.dtype == torch.float32
        assert scores.numel() == self.scores.numel()

        ragged_scores = k2.ragged.RaggedFloat(
            self.arcs.shape().to(scores.device), scores)
        ragged_scores = k2.ragged.normalize_scores(ragged_scores)

        # Note we use `to` here since `scores` and `self.scores` may not
        # be on the same device.
        self.scores = ragged_scores.values.to(self.scores.device)
