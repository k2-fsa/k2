# Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corp.       (author: Daniel Povey, Haowen Qiu)
#                      Guoguo Chen
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

from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import os
import re
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
            by :func:`clone`)

        Returns:
          An instance of Fsa.
        '''
        if isinstance(arcs, torch.Tensor):
            arcs: RaggedArc = _k2.fsa_from_tensor(arcs)
        assert isinstance(arcs, RaggedArc)

        # Accessing self.__dict__ bypasses __setattr__.
        # Here we are setting self.arcs and self._properties.
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

    def _invalidate_cache_(self, scores_only: bool = True) -> None:
        '''Intended for internal use only so its
        name begins with an underline.

        Also, it changes `self` in-place.

        Currently, it is used only when the `scores` field
        are re-assigned.

        Args:
          scores_only:
            It True, it invalidates only cached entries related
            to scores. If False, the whole cache is invalidated.

        '''
        if scores_only is False:
            self.__dict__['_cache'] = dict()
        else:
            pattern = re.compile(r'score|arc_cdf|arc_post')
            to_remove = []

            for key in self.__dict__['_cache']:
                if pattern.search(key):
                    to_remove.append(key)

            if 'entering_arcs' in self.__dict__['_cache']:
                # We also need to remove "entering_arcs"
                # since it may be set in get_forward_scores()
                to_remove.append('entering_arcs')

            for key in to_remove:
                del self.__dict__['_cache'][key]

    def to_str(self, openfst: bool = False) -> str:
        extra_labels = []
        ragged_labels = []
        for name, value in sorted(self.named_tensor_attr(include_scores=False)):
            if isinstance(value, torch.Tensor) and value.dtype == torch.int32:
                extra_labels.append(value)
            elif isinstance(value, _k2.RaggedInt):
                ragged_labels.append(value)

        if self.arcs.num_axes() == 2:
            ans = 'k2.Fsa: ' + _k2.fsa_to_str(self.arcs, openfst=openfst,
                                              extra_labels=extra_labels,
                                              ragged_labels=ragged_labels)
        else:
            ans = 'k2.FsaVec: \n'
            for i in range(self.shape[0]):
                # get the i-th Fsa
                ragged_arc, start = self.arcs.index(0, i)
                end = start + ragged_arc.values().shape[0]
                ans += 'FsaVec[' + str(i) + ']: ' + _k2.fsa_to_str(
                    ragged_arc, openfst=openfst,
                    extra_labels=[x[start:end] for x in extra_labels],
                    ragged_labels=[_k2.ragged_int_arange(x, 0, start, end)
                                   for x in ragged_labels])
        ans += 'properties_str = ' + _k2.fsa_properties_as_str(
            self._properties) + '.'
        for name, value in self.named_tensor_attr(include_scores=False):
            sep = '\n'
            ans += f'{sep}{name}: {value}'
        for name, value in self.named_non_tensor_attr():
            if name == 'labels_sym':
                continue
            sep = '\n'
            ans += f'{sep}{name}: {value}'

        return ans

    def __str__(self) -> str:
        '''Return a string representation of this object

        For visualization and debug only.
        '''
        return self.to_str(openfst=False)

    def get_filler(self, attribute_name: str) -> Union[int, float]:
        '''Return the filler value associated with attribute names.

        This is 0 unless otherwise specified, but you can override this by
        for example, doing::

            fsa.foo_filler = -1

        which will mean the "filler" for attribute fsa.foo is -1; and this will
        get propagated when you do FSA operations, like any other non-tensor
        attribute.  The filler is the value that means "nothing is here" (like
        epsilon).

        Caution::
          you should use a value that is castable to float and back to integer
          without loss of precision, because currently the `default_value`
          parameter of `index_select` in ./ops.py is a float.
        '''

        ans = getattr(self, attribute_name + '_filler', 0)
        assert attribute_name != 'aux_labels' or ans == 0, \
                                 'you cannot set the filler for aux_labels'
        return ans

    def draw(self, filename: Optional[str],
             title: Optional[str] = None) -> 'Digraph':  # noqa
        '''
        Render FSA as an image via graphviz, and return the Digraph object;
        and optionally save to file `filename`.
        `filename` must have a suffix that graphviz understands, such as
        `pdf`, `svg` or `png`.

        Note:
          You need to install graphviz to use this function::

            pip install graphviz

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
                # fix_final_labels() will change 0's to -1's and vice versa to
                # ensure that constraints on where final-labels should appear,
                # are satisfied.
                _k2.fix_final_labels(self.arcs, None)
                self.__dict__['_properties'] = None
                # access self.properties which will do a validity check on the
                # modified FSA after getting the properties
                self.properties
                return

            self._tensor_attr[name] = value

            if name == 'scores':
                assert value.dtype == torch.float32
                # NOTE: we **reinterpret** the float patterns
                # to integer patterns here.
                self.arcs.values()[:, -1] = _k2.as_int(value.detach())
                self._invalidate_cache_()
        elif isinstance(value, _k2.RaggedInt):
            assert value.dim0() == self.arcs.values().shape[0], \
                    f'value.dim0(): {value.dim0()}, shape[0]: {self.arcs.values().shape[0]}'  # noqa
            self._tensor_attr[name] = value
        else:
            self._non_tensor_attr[name] = value

    @property
    def num_arcs(self) -> int:
        '''Return the number of arcs in this Fsa.
        '''
        return self.arcs.num_elements()

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
        # getters, or 'scores' which is special.
        assert name not in Fsa.__dict__ and name != 'scores'
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

    def _get_state_batches(self) -> _k2.RaggedInt:
        '''Get (and compute if necessary) cached property `state_batches`.

        For use by internal k2 code.  Used in many algorithms.
        '''
        name, cache = 'state_batches', self._cache
        if name not in cache:
            cache[name] = _k2.get_state_batches(self.arcs, transpose=True)
        return cache[name]

    def _get_dest_states(self) -> torch.Tensor:
        '''Get (and compute if necessary) cached property self.dest_states.

        For use by internal k2 code, relates to best-path.
        '''
        name, cache = 'dest_states', self._cache
        if name not in cache:
            cache[name] = _k2.get_dest_states(self.arcs, as_idx01=True)
        return cache[name]

    def _get_incoming_arcs(self) -> _k2.RaggedInt:
        '''Get (and compute if necessary) cached property self.incoming_arcs.

        For use by internal k2 code, relates to best-path
        '''
        name, cache = 'incoming_arcs', self._cache
        if name not in cache:
            cache[name] = _k2.get_incoming_arcs(self.arcs,
                                                self._get_dest_states())
        return cache[name]

    def _get_entering_arc_batches(self) -> _k2.RaggedInt:
        '''Get (and compute if necessary) cached property
        `self.entering_arc_batches`.

        For use by internal k2 code, used in many algorithms.
        '''
        name, cache = 'entering_arc_batches', self._cache
        if name not in cache:
            cache[name] = _k2.get_entering_arc_index_batches(
                self.arcs,
                incoming_arcs=self._get_incoming_arcs(),
                state_batches=self._get_state_batches())
        return cache[name]

    def _get_leaving_arc_batches(self) -> _k2.RaggedInt:
        '''Get (and compute if necessary) cached property
        `self.leaving_arc_batches`.

        For use by internal k2 code, used in many algorithms.
        '''
        name, cache = 'leaving_arc_batches', self._cache
        if name not in cache:
            cache[name] = _k2.get_leaving_arc_index_batches(
                self.arcs, self._get_state_batches())
        return cache[name]

    def _get_arc_cdf(self, use_double_scores: bool,
                     log_semiring: bool) -> torch.Tensor:
        name = 'arc_cdf_' + \
                ('double_' if use_double_scores else 'float_') + \
                ('log' if log_semiring else 'tropical')  # noqa
        cache = self._cache
        if name not in cache:
            arc_post = self._get_arc_post(use_double_scores, log_semiring)
            func = (_k2.get_arc_cdf_double
                    if use_double_scores else _k2.get_arc_cdf_float)
            arc_cdf = func(fsas=self.arcs, arc_post=arc_post)
            cache[name] = arc_cdf
        return cache[name]

    def _get_forward_scores(self, use_double_scores: bool,
                            log_semiring: bool) -> torch.Tensor:
        '''Get (and compute if necessary) cached property
        `self.forward_scores_xxx_yyy` (where xxx indicates float-type and
        yyy indicates semiring).

        For use by internal k2 code; returns the total score from start-state to
        each state.  Not differentiable; see :func:`get_forward_scores` which is
        the differentiable version.

        Args:
          use_double_scores:
            True to use `double precision` floating point.
            False to use `single precision`.
          log_semiring:
            True to use log semiring (log-sum), false to use tropical (i.e. max
            on scores).
        '''
        name = 'forward_scores_' + \
               ('double_' if use_double_scores else 'float_') + \
               ('log' if log_semiring else 'tropical')
        cache = self._cache
        if name not in cache:
            if use_double_scores:
                func = _k2.get_forward_scores_double
            else:
                func = _k2.get_forward_scores_float
            cache[name], entering_arcs = func(
                self.arcs,
                state_batches=self._get_state_batches(),
                entering_arc_batches=self._get_entering_arc_batches(),
                log_semiring=log_semiring)
            if not log_semiring:
                cache['entering_arcs'] = entering_arcs
        return cache[name]

    def get_forward_scores(self, use_double_scores: bool,
                           log_semiring: bool) -> torch.Tensor:
        '''Compute forward-scores, i.e. total weight (or best-path weight)
        from start state to each state.

        Supports autograd.

        Args:
          use_double_scores:
            if True, use double precision.
          log_semiring:
            if True, use log semiring, else tropical.
        Returns:
          A torch.Tensor with shape equal to (num_states,)
        '''
        # Caution: the reason we don't cache this is
        forward_scores = k2.autograd._GetForwardScoresFunction.apply(
            self, log_semiring, use_double_scores, self.scores)
        return forward_scores

    def _get_tot_scores(self, use_double_scores: bool,
                        log_semiring: bool) -> torch.Tensor:
        '''Compute total-scores (one per FSA) as the best-path score.

        This version is not differentiable; see also :func:`get_tot_scores`
        which is differentiable.

        Args:
          use_double_scores:
            If True, use `double precision` floating point; false;
            else single precision.
          log_semiring:
            True to use log semiring (log-sum), false to use tropical (i.e. max
            on scores).
        '''
        name = 'tot_scores_' + \
               ('double_' if use_double_scores else 'float_') + \
               ('log' if log_semiring else 'tropical')
        cache = self._cache
        if name not in cache:
            if use_double_scores is True:
                func = _k2.get_tot_scores_double
            else:
                func = _k2.get_tot_scores_float
            forward_scores = self._get_forward_scores(use_double_scores,
                                                      log_semiring)
            total_scores = func(self.arcs, forward_scores)
            cache[name] = total_scores
        return cache[name]

    def get_tot_scores(self, use_double_scores: bool,
                       log_semiring: bool) -> torch.Tensor:
        '''Compute total-scores (one per FSA) as the
        best-path score.

        This version is differentiable.

        Args:
          use_double_scores:
            True to use `double precision` floating point;
            False to use `single precision`.
          log_semiring:
            True to use log semiring (log-sum), false to use tropical (i.e. max
            on scores).
        '''
        tot_scores = k2.autograd._GetTotScoresFunction.apply(
            self, log_semiring, use_double_scores, self.scores)
        return tot_scores

    def _get_backward_scores(self, use_double_scores: bool,
                             log_semiring: bool) -> torch.Tensor:
        '''Compute backward-scores, i.e. total weight (or best-path weight)
        from each state to the final state.

        For internal k2 use. Not differentiable.

        See also :func:`get_backward_scores` which is differentiable.

        Args:
          use_double_scores:
            True to use `double precision` floating point.
            False to use `single precision`.
          log_semiring:
            True to use log semiring (log-sum), false to use tropical (i.e. max
            on scores).

        Returns:
          A torch.Tensor with shape equal to (num_states,)
        '''
        name = 'backward_scores_' + \
               ('double_' if use_double_scores else 'float_') + \
               ('log' if log_semiring else 'tropical')
        cache = self._cache
        if name not in cache:
            if use_double_scores:
                func = _k2.get_backward_scores_double
            else:
                func = _k2.get_backward_scores_float

            state_batches = self._get_state_batches()
            leaving_arc_batches = self._get_leaving_arc_batches()
            backward_scores = func(self.arcs,
                                   state_batches=state_batches,
                                   leaving_arc_batches=leaving_arc_batches,
                                   log_semiring=log_semiring)
            cache[name] = backward_scores
        return cache[name]

    def get_backward_scores(self, use_double_scores: bool,
                            log_semiring: bool) -> torch.Tensor:
        '''Compute backward-scores, i.e. total weight (or best-path weight)
        from each state to the final state.

        Supports autograd.

        Args:
          use_double_scores:
            if True, use double precision.
          log_semiring:
            if True, use log semiring, else tropical.

        Returns:
          A torch.Tensor with shape equal to (num_states,)
        '''
        backward_scores = k2.autograd._GetBackwardScoresFunction.apply(
            self, log_semiring, use_double_scores, self.scores)
        return backward_scores

    def _get_arc_post(self, use_double_scores: bool,
                      log_semiring: bool) -> torch.Tensor:
        '''Compute scores on arcs, representing log probabilities;
        with log_semiring=True you could call these log posteriors,
        but if log_semiring=False they can only be interpreted as the
        difference betwen the best-path score and the score of the
        best path that includes this arc.

        This version is not differentiable; see also :func:`get_arc_post`.

        Args:
          use_double_scores:
            if True, use double precision.
          log_semiring:
            if True, use log semiring, else tropical.
        Returns:
          A torch.Tensor with shape equal to (num_arcs,)
          and non-positive elements.
        '''
        name = 'arc_post_' + \
               ('double_' if use_double_scores else 'float_') + \
               ('log' if log_semiring else 'tropical')
        cache = self._cache
        if name not in cache:
            forward_scores = self._get_forward_scores(use_double_scores,
                                                      log_semiring)
            backward_scores = self._get_backward_scores(
                use_double_scores, log_semiring)
            func = (_k2.get_arc_post_double
                    if use_double_scores else _k2.get_arc_post_float)
            arc_post = func(fsas=self.arcs,
                            forward_scores=forward_scores,
                            backward_scores=backward_scores)
            cache[name] = arc_post
        return cache[name]

    def get_arc_post(self, use_double_scores: bool,
                     log_semiring: bool) -> torch.Tensor:
        '''Compute scores on arcs, representing log probabilities;
        with log_semiring=True you could call these log posteriors,
        but if log_semiring=False they can only be interpreted as the
        difference between the best-path score and the score of the
        best path that includes this arc.
        This version is differentiable; see also :func:`_get_arc_post`.

        Caution:
          Because of how the autograd mechanics works and the
          need to avoid circular references, this is not cached;
          it's best to store it if you'll need it multiple times.

        Args:
          use_double_scores:
            if True, use double precision.
          log_semiring:
            if True, use log semiring, else tropical.
        Returns:
          A torch.Tensor with shape equal to (num_arcs,)
          and non-positive elements.
        '''
        # We don't cache this!  User should store it if needed more than once,
        # to avoid duplicate code in backprop.  We may be able to partially fix
        # this at some point with a weak dictionary.
        forward_scores = self.get_forward_scores(use_double_scores,
                                                 log_semiring)
        backward_scores = self.get_backward_scores(use_double_scores,
                                                   log_semiring)

        # Below, the last 3 args are active w.r.t. autograd, the backward
        # function will return non-None derivatives for them.
        arc_post = k2.autograd._GetArcPostFunction.apply(
            self, log_semiring, use_double_scores, self.scores, forward_scores,
            backward_scores)
        return arc_post

    def _get_entering_arcs(self, use_double_scores: bool) -> torch.Tensor:
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
            self._get_forward_scores(use_double_scores, False)
        return cache[name]

    def requires_grad_(self, requires_grad: bool) -> 'Fsa':
        '''Change if autograd should record operations on this FSA:

        Sets the `scores`'s requires_grad attribute in-place.

        Returns this FSA.

        You can test whether this object has the requires_grad property
        true or false by accessing :py:attr:`requires_grad` (handled in
        :func:`__getattr__`).

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

    def rename_tensor_attribute_(self, src_name: str, dest_name: str) -> 'Fsa':
        '''Rename a tensor attribute (or, as a special case 'labels'),
        and also rename non-tensor attributes that are associated with it,
        i.e. that have it as a prefix.

        Args:
          src_name:
            The original name, exist as a tensor attribute, e.g. 'aux_labels',
            or, as a special case, equal 'labels'; special attributes 'labels'
            and 'scores' are allowed but won't be deleted.
          dest_name:
            The new name, that we are renaming it to. If it already existed as
            a tensor attribute, it will be rewritten; and any previously
            existing non-tensor attributes that have this as a prefix will be
            deleted.  As a special case, may equal 'labels'.
        Returns:
          Return `self`.

        Note::
          It is OK if src_name and/or dest_name equals 'labels' or 'scores',
          but these special attributes won't be deleted.
        '''
        assert src_name != dest_name
        assert src_name in self._tensor_attr or src_name == 'labels'
        try:
            value = getattr(self, src_name)
            if src_name == 'labels':
                value = value.clone()
            setattr(self, dest_name, value)
            if src_name != 'scores' and src_name != 'labels':
                del self._tensor_attr[src_name]
        except KeyError as e:
            raise ValueError(f'Name {src_name} does not exist as a tensor '
                             'attribute: exception was ' + str(e))

        src_name_len = len(src_name)
        dest_name_len = len(dest_name)
        to_move = []
        for name, value in list(self._non_tensor_attr.items()):
            if name[:src_name_len] == src_name:
                # remove src_name from prefix and replace with dest_name
                new_name = dest_name + name[src_name_len:]
                to_move.append((name, new_name, value))
            elif name[:dest_name_len] == dest_name:
                del self._non_tensor_attr[name]

        for name, new_name, value in to_move:
            self._non_tensor_attr[new_name] = value
            del self._non_tensor_attr[name]
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

        self.rename_tensor_attribute_('labels', '__temp')
        self.rename_tensor_attribute_('aux_labels', 'labels')
        self.rename_tensor_attribute_('__temp', 'aux_labels')
        return self

        # TODO(dan), maybe: instead of using the generic approach above, we
        # could use more specific code like the following (the old code), which
        # might be more efficient.  Or perhaps create a generic
        # swap_tensor_attribute_ function.
        #
        # aux_labels = self.aux_labels
        # self.aux_labels = self.labels.clone()
        # self.labels = aux_labels

        # labels_sym = getattr(self, 'labels_sym', None)
        # aux_labels_sym = getattr(self, 'aux_labels_sym', None)
        # if labels_sym is not None:
        #     del self.labels_sym
        # if aux_labels_sym is not None:
        #     del self.aux_labels_sym
        # if labels_sym is not None:
        #     self.aux_labels_sym = labels_sym
        # if aux_labels_sym is not None:
        #     self.labels_sym = aux_labels_sym
        # self.__dict__['_properties'] = None
        # # access self.properties which will do a validity check on the
        # # modified FSA after getting the properties
        # self.properties
        # return self

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
        '''Return the core part of the Fsa (the arcs) serialized to a Tensor
           of int32 type, with shape (num_arcs, 4); the floats are reinterpreted
           as int32 and will appear as garbage if printed.  This can be passed
           to the constructor, along with the aux_labels if present, to
           reconstruct this object.  A more convenient way to serialize a Tensor
           is to use :func:`as_dict` and :func:`from_dict`
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
            # Caution: We are not using `deepcopy` for `value`!
            setattr(ans, name, value)

        # Just copy elements of the _cache that we might already have..
        # These don't directly participate in autograd, and are not supposed to
        # be modified by the user, so this should be safe (i.e. it should
        # be safe to do this without clone(); these are mostly not tensors
        # anyway.
        for name, value in self._cache.items():
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
            if isinstance(value, torch.Tensor):
                setattr(ans, name, value.detach())
            else:
                assert isinstance(value, k2.RaggedInt)
                # For ragged tensors, they are copied over.
                # Caution: Deep copy is not used!
                setattr(ans, name, value)

        for name, value in self.named_non_tensor_attr():
            # Caution: We are not using `deepcopy` for `value`!
            setattr(ans, name, value)

        # Just copy elements of the _cache that we might already have..
        # These don't directly participate in autograd, and are not supposed to
        # be modified by the user, so this should be safe (i.e. it should
        # be safe to do this without clone(); these are mostly not tensors
        # anyway.
        for name, value in self._cache.items():
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
    def from_str(cls,
                 s: str,
                 acceptor: Optional[bool] = None,
                 num_aux_labels: Optional[int] = None,
                 aux_label_names: Optional[List[str]] = None,
                 ragged_label_names: List[str] = [],
                 openfst: bool = False) -> 'Fsa':
        '''Create an Fsa from a string in the k2 or OpenFst format.
        (See also :func:`from_openfst`).

        The given string `s` consists of lines with the following format::

          src_state dest_state label [aux_label1 aux_label2...] [score]

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

        Note:
          At most one of `acceptor`, `num_aux_labels`, and `aux_label_names`
          must be supplied; if none are supplied, acceptor format is assumed.

        Args:
          s:
            The input string. Refer to the above comment for its format.

          acceptor:
            Set to true to denote acceptor format which is num_aux_labels == 0,
            or false to denote transducer format (i.e. num_aux_labels == 1
            with name 'aux_labels').
          num_aux_labels:
            The number of auxiliary labels to expect on each line (in addition
            to the 'acceptor' label; is 1 for traditional transducers but can be
            any non-negative number.  The names of the aux_labels default to
            'aux_labels' then 'aux_labels2', 'aux_labels3' and so on.
          aux_label_names:
            If provided, the length of this list dictates the number of
            aux_labels and this list dictates their names.
          ragged_label_names:
            If provided, expect this number of ragged labels, in the order
            of this list.  It is advisable that this list be in
            alphabetical order, so that the format when we write back to
            a string will be unchanged.
          openfst:
            If true, will expect the OpenFST format (costs not scores, i.e.
            negated; final-probs rather than final-state specified).
        '''
        (num_aux_labels, aux_label_names) = \
                get_aux_label_info(acceptor, num_aux_labels, aux_label_names)
        num_ragged_labels = len(ragged_label_names)
        try:
            (arcs, aux_labels,
             ragged_labels) = _k2.fsa_from_str(s, num_aux_labels,
                                               num_ragged_labels,
                                               openfst=openfst)
            ans = Fsa(arcs)
            if aux_labels is not None:
                for i in range(aux_labels.shape[0]):
                    setattr(ans, aux_label_names[i], aux_labels[i, :])
            for name, value in zip(ragged_label_names, ragged_labels):
                setattr(ans, name, value)

            return ans
        except Exception:
            o = 'in the OpenFst format ' if openfst else ''
            raise ValueError(f'The following is not a valid Fsa {o}(with '
                             f'num_aux_labels={num_aux_labels}): {s}')

    @classmethod
    def from_openfst(cls,
                     s: str,
                     acceptor: Optional[bool] = None,
                     num_aux_labels: Optional[int] = None,
                     aux_label_names: Optional[List[str]] = None,
                     ragged_label_names: List[str] = []) -> 'Fsa':
        '''Create an Fsa from a string in OpenFST format (or a slightly more
        general format, if num_aux_labels > 1). See also :func:`from_str`.

        The given string `s` consists of lines with the following format::

           src_state dest_state label [aux_label1 aux_label2...] [cost]

       (the cost defaults to 0.0 if not present).

        The line for the final state consists of two fields::

           final_state [cost]

        Note:
          Fields are separated by space(s), tab(s) or both. The `cost`
          field is a float, while other fields are integers.

          There might be multiple final states. Also, OpenFst may omit the cost
          if it is 0.0.

        Caution:
          We use `cost` here to indicate that its value will be negated so that
          we can get `scores`. That is, `score = -1 * cost`.

        Note:
          At most one of `acceptor`, `num_aux_labels`, and `aux_label_names`
          must be supplied; if none are supplied, acceptor format is assumed.

        Args:
          s:
            The input string. Refer to the above comment for its format.
          acceptor:
            Set to true to denote acceptor format which is num_aux_labels == 0,
            or false to denote transducer format (i.e. num_aux_labels == 1
            with name 'aux_labels').
          num_aux_labels:
            The number of auxiliary labels to expect on each line (in addition
            to the 'acceptor' label; is 1 for traditional transducers but can be
            any non-negative number.
          aux_label_names:
            If provided, the length of this list dictates the number of
            aux_labels. By default the names are 'aux_labels', 'aux_labels2',
            'aux_labels3' and so on.
          ragged_label_names:
            If provided, expect this number of ragged labels, in the order
            of this list.  It is advisable that this list be in
            alphabetical order, so that the format when we write back to
            a string will be unchanged.
        '''
        return Fsa.from_str(s, acceptor, num_aux_labels,
                            aux_label_names, ragged_label_names,
                            openfst=True)

    @staticmethod
    def from_fsas(fsas: List['Fsa']) -> 'Fsa':
        '''Create an FsaVec from a list of FSAs.

        See also :func:`k2.create_fsa_vec`. This function is just
        a wrapper of that function.
        '''
        return k2.create_fsa_vec(fsas)

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
        ragged_scores = k2.ragged.normalize_scores(ragged_scores, use_log=True)

        # Note we use `to` here since `scores` and `self.scores` may not
        # be on the same device.
        self.scores = ragged_scores.values.to(self.scores.device)

    def convert_attr_to_ragged_(self, name: str,
                                remove_eps: bool = True) -> 'Fsa':
        '''Convert the attribute given by `name` from a 1-D torch.tensor
        to a k2.RaggedInt.

        Caution:
          This function ends with an underscore, meaning it changes the FSA
          **in-place**.

        Args:
          name:
            The attribute name. This attribute is expected to be a 1-D tensor
            with dtype torch.int32.
          remove_eps:
            True to remove 0s in the resulting ragged tensor.

        Returns:
          Return self.
        '''
        assert hasattr(self, name)
        value = getattr(self, name)
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 1
        assert value.dtype == torch.int32

        shape = k2.ragged.regular_ragged_shape(dim0=value.numel(),
                                               dim1=1).to(value.device)
        new_value = k2.RaggedInt(shape, value.contiguous())
        if remove_eps:
            new_value = k2.ragged.remove_values_eq(new_value, target=0)

        setattr(self, name, new_value)

        return self


def get_aux_label_info(acceptor: Optional[bool], num_aux_labels: Optional[int],
                       aux_label_names: Optional[List[str]]
                      ) -> Tuple[int, List[str]]:  # noqa
    '''Given either acceptor or num_aux_labels or aux_label_names, at most
    one of which should be supplied, returns a pair (num_aux_labels,
    aux_label_names) specifying the number of auxiliary labels and the
    names to use for them.

    Args:
      acceptor:
        None, or a bool; False means no aux_labels and True means aux_label
        with name 'aux_label'.
      num_aux_labels:
        None, or a number >= 0
      aux_label_names:
        None, or a list of names, such as ['aux_labels', 'aux_labels2']
    Returns:
       Returns a tuple (num_aux_labels, aux_label_names).
       out from the other.  If all inputs are None, we assume there
       are no aux-labels and return (0, []).
    '''
    if acceptor is not None:
        assert num_aux_labels is None and aux_label_names is None
        return (0, []) if acceptor else (1, ['aux_labels'])
    elif aux_label_names is not None:
        assert num_aux_labels is None
        return len(aux_label_names), aux_label_names
    elif num_aux_labels is not None:
        assert num_aux_labels >= 0
        aux_label_names = []
        if num_aux_labels != 0:
            aux_label_names.append('aux_labels')
        for i in range(1, num_aux_labels):
            aux_label_names.append(f'aux_labels{i+1}')
        return num_aux_labels, aux_label_names
    else:
        return (0, [])
