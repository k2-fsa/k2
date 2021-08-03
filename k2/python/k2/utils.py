# Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corporation (authors: Haowen Qiu)
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

from typing import Optional
from typing import Tuple
from typing import Union

import torch

from .fsa import Fsa
from .ops import index
from .ops import index_ragged
from .ops import index_select
from .symbol_table import SymbolTable
import k2
import k2.ragged
import _k2


def to_str(fsa: Fsa,
           openfst: bool = False) -> str:
    '''Convert an Fsa to a string.  This version prints out all integer
    labels and integer ragged labels on the same line as each arc, the
    same format accepted by Fsa.from_str().

    Note:
      The returned string can be used to construct an Fsa with Fsa.from_str(),
      but you would need to know the names of the auxiliary labels and ragged
      labels.

    Args:
      openfst:
        Optional. If true, we negate the scores during the conversion.

    Returns:
      A string representation of the Fsa.
    '''
    assert fsa.arcs.num_axes() == 2
    extra_labels = []
    ragged_labels = []
    for name, value in sorted(fsa.named_tensor_attr(include_scores=False)):
        if isinstance(value, torch.Tensor) and value.dtype == torch.int32:
            extra_labels.append(value)
        elif isinstance(value, _k2.RaggedInt):
            ragged_labels.append(value)

    return _k2.fsa_to_str(fsa.arcs, openfst=openfst,
                          extra_labels=extra_labels,
                          ragged_labels=ragged_labels)


def to_str_simple(fsa: Fsa,
                  openfst: bool = False) -> str:
    '''Convert an Fsa to a string.  This is less complete than Fsa.to_str(),
    fsa.__str__(), or to_str_full(), meaning it prints only fsa.aux_labels and
    no ragged labels, not printing any other attributes.  This is used in
    testing.

    Note:
      The returned string can be used to construct an Fsa.  See also to_str().

    Args:
      openfst:
        Optional. If true, we negate the scores during the conversion.

    Returns:
      A string representation of the Fsa.
    '''
    assert fsa.arcs.num_axes() == 2
    if hasattr(fsa, 'aux_labels') and isinstance(fsa.aux_labels, torch.Tensor):
        aux_labels = [fsa.aux_labels.to(torch.int32)]
    else:
        aux_labels = []
    return _k2.fsa_to_str(fsa.arcs, openfst, aux_labels, [])


def to_tensor(fsa: Fsa) -> torch.Tensor:
    '''Convert an Fsa to a Tensor.

    You can save the tensor to disk and read it later
    to construct an Fsa.

    Note:
      The returned Tensor contains only the transition rules, e.g.,
      arcs. You may want to save its aux_labels separately if any.

    Args:
      fsa:
        The input Fsa.
    Returns:
      A `torch.Tensor` of dtype `torch.int32`. It is a 2-D tensor
      if the input is a single FSA. It is a 1-D tensor if the input
      is a vector of FSAs.
    '''
    return _k2.fsa_to_tensor(fsa.arcs)


def to_dot(fsa: Fsa, title: Optional[str] = None) -> 'Digraph':  # noqa
    '''Visualize an Fsa via graphviz.

    Note:
      Graphviz is needed only when this function is called.

    Args:
      fsa:
        The input FSA to be visualized.

      title:
        Optional. The title of the resulting visualization.
    Returns:
      a Diagraph from grahpviz.
    '''

    try:
        import graphviz
    except Exception:
        print(
            'You cannot use `to_dot` unless the graphviz package is installed.'
        )
        raise

    assert len(fsa.shape) == 2, 'FsaVec is not supported'
    if hasattr(fsa, 'aux_labels'):
        aux_labels = fsa.aux_labels
        name = 'WFST'
    else:
        aux_labels = None
        name = 'WFSA'

    def convert_aux_label_to_symbol(
            aux_labels: Union[torch.Tensor, _k2.RaggedInt],
            arc_index: int,
            symbols: Optional[SymbolTable] = None) -> str:
        '''Convert aux_label(s) to symbol(s).

        Args:
          aux_labels:
            The aux_labels of an FSA.
          arc_index:
            The index of the arc.
          symbols:
            Symbol table of the FSA associated with the `aux_labels`.
        Returns:
          If `aux_labels` is a torch.Tensor, it returns a single string.
          If `aux_labels` is a ragged tensor, it returns a string with symbols
          separated by a space.
        '''
        if isinstance(aux_labels, torch.Tensor):
            ans = int(aux_labels[arc_index])
            if ans != -1 and symbols is not None:
                ans = symbols[ans]
            return f':{ans}'
        assert isinstance(aux_labels, _k2.RaggedInt)
        assert aux_labels.num_axes() == 2
        row_splits = aux_labels.row_splits(1).cpu()
        begin = row_splits[arc_index]
        end = row_splits[arc_index + 1]
        if end == begin:
            return ':<eps>'

        labels = aux_labels.values()[begin:end]
        ans = []
        for label in labels.tolist():
            if label == -1:
                ans.append('-1')
            elif symbols is not None:
                ans.append(symbols[label])
            else:
                ans.append(f'{label}')
        return f':{" ".join(ans)}'

    graph_attr = {
        'rankdir': 'LR',
        'size': '8.5,11',
        'center': '1',
        'orientation': 'Portrait',
        'ranksep': '0.4',
        'nodesep': '0.25',
    }
    if title is not None:
        graph_attr['label'] = title

    default_node_attr = {
        'shape': 'circle',
        'style': 'bold',
        'fontsize': '14',
    }

    final_state_attr = {
        'shape': 'doublecircle',
        'style': 'bold',
        'fontsize': '14',
    }

    final_state = -1
    dot = graphviz.Digraph(name=name, graph_attr=graph_attr)

    seen = set()
    i = -1
    for arc, weight in zip(fsa.arcs.values()[:, :-1], fsa.scores.tolist()):
        i += 1
        src_state, dst_state, label = arc.tolist()
        src_state = str(src_state)
        dst_state = str(dst_state)
        label = int(label)
        if label == -1:
            final_state = dst_state
        if src_state not in seen:
            dot.node(src_state, label=src_state, **default_node_attr)
            seen.add(src_state)

        if dst_state not in seen:
            if dst_state == final_state:
                dot.node(dst_state, label=dst_state, **final_state_attr)

            else:
                dot.node(dst_state, label=dst_state, **default_node_attr)
            seen.add(dst_state)
        if aux_labels is not None:
            if hasattr(fsa, 'aux_labels_sym'):
                aux_label = convert_aux_label_to_symbol(
                    aux_labels, i, fsa.aux_labels_sym)
            else:
                aux_label = convert_aux_label_to_symbol(aux_labels, i, None)
            aux_label = aux_label.replace('<eps>', 'ε')
        else:
            aux_label = ''

        if hasattr(fsa, 'labels_sym') and label != -1:
            label = fsa.labels_sym.get(label)
            if label == '<eps>':
                label = 'ε'

        weight = f'{weight:.2f}'.rstrip('0').rstrip('.')
        dot.edge(src_state, dst_state, label=f'{label}{aux_label}/{weight}')
    return dot


def create_fsa_vec(fsas):
    '''Create an FsaVec from a list of FSAs

    We use the following rules to set the attributes of the output FsaVec:

    - For tensor attributes, we assume that all input FSAs have the same
      attribute name and the values are concatenated.

    - For non-tensor attributes, if any two of the input FSAs have the same
      attribute name, then we assume that their attribute values are equal and
      the output FSA will inherit the attribute.

    Args:
      fsas:
        A list of `Fsa`. Each element must be a single FSA.

    Returns:
      An instance of :class:`Fsa` that represents a FsaVec.
    '''
    ragged_arc_list = list()
    for fsa in fsas:
        assert len(fsa.shape) == 2
        ragged_arc_list.append(fsa.arcs)

    ragged_arcs = _k2.create_fsa_vec(ragged_arc_list)
    fsa_vec = Fsa(ragged_arcs)

    tensor_attr_names = set(
        name for name, _ in fsa.named_tensor_attr() for fsa in fsas)
    for name in tensor_attr_names:
        values = []
        for fsa in fsas:
            values.append(getattr(fsa, name))
        if isinstance(values[0], torch.Tensor):
            value = torch.cat(values)
        else:
            assert isinstance(values[0], _k2.RaggedInt)
            value = k2.ragged.cat(values, axis=0)
        setattr(fsa_vec, name, value)

    non_tensor_attr_names = set(
        name for name, _ in fsa.named_non_tensor_attr() for fsa in fsas)

    for name in non_tensor_attr_names:
        if name == 'properties':
            continue

        for fsa in fsas:
            value = getattr(fsa, name, None)
            if value is not None:
                if hasattr(fsa_vec, name):
                    assert getattr(fsa_vec, name) == value
                else:
                    setattr(fsa_vec, name, value)
    return fsa_vec


def is_rand_equivalent(a: Fsa,
                       b: Fsa,
                       log_semiring: bool,
                       beam: float = float('inf'),
                       treat_epsilons_specially: bool = True,
                       delta: float = 1e-6,
                       npath: int = 100) -> bool:
    '''Check if the Fsa `a` appears to be equivalent to `b` by
    randomly checking some symbol sequences in them.

    Caution:
      It works only on CPU.

    Args:
      a:
        One of the input FSA. It can be either a single FSA or an FsaVec.
        Must be top-sorted and on CPU.
      b:
        The other input FSA. It must have the same NumAxes() as a.
        Must be top-sorted and on CPU.
      log_semiring:
        The semiring to be used for all weight measurements;
        if false then we use 'max' on alternative paths; if
        true we use 'log-add'.
      beam:
         beam > 0 that affects pruning; the algorithm will only check
         paths within `beam` of the total score of the lattice (for
         tropical semiring, it's max weight over all paths from start
         state to final state; for log semiring, it's log-sum probs over
         all paths) in `a` or `b`.
      treat_epsilons_specially:
         We'll do `intersection` between generated path and a or b when
         check equivalence. Generally, if it's true, we will treat
         epsilons as epsilon when doing intersection; Otherwise, epsilons
         will just be treated as any other symbol.
      delta:
         Tolerance for path weights to check the equivalence.
         If abs(weights_a, weights_b) <= delta, we say the two
         paths are equivalent.
      npath:
         The number of paths will be generated to check the
         equivalence of `a` and `b`
    Returns:
       True if the Fsa `a` appears to be equivalent to `b` by randomly
       generating `npath` paths from one of them and then checking if the symbol
       sequence exists in the other one and if the total weight for that symbol
       sequence is the same in both FSAs.
    '''
    return _k2.is_rand_equivalent(a.arcs, b.arcs, log_semiring, beam,
                                  treat_epsilons_specially, delta, npath)


def create_sparse(rows: torch.Tensor,
                  cols: torch.Tensor,
                  values: torch.Tensor,
                  size: Optional[Tuple[int, int]] = None,
                  min_col_index: Optional[int] = None):
    '''This is a utility function that creates a (torch) sparse matrix likely
    intended to represent posteriors.  The likely usage is something like
    (for example)::

        post = k2.create_sparse(fsa.seqframe, fsa.phones,
                                fsa.get_arc_post(True,True).exp(),
                                min_col_index=1)

    (assuming `seqframe` and `phones` were integer-valued attributes of `fsa`).

    Args:
      rows:
        Row indexes of the sparse matrix (a torch.Tensor), which must have
        values >= 0; likely `fsa.seqframe`.   Must have row_indexes.dim == 1.
        Will be converted to `dtype=torch.long`
      cols:
        Column indexes of the sparse matrix, with the same shape as `rows`.
        Will be converted to `dtype=torch.long`
      values:
        Values of the sparse matrix, likely of dtype float or double, with
        the same shape as `rows` and `cols`.
      size:
        Optional. If not None, it is assumed to be a tuple containing
        `(num_frames, highest_phone_plus_one)`
      min_col_index:
        If provided, before the sparse tensor is constructed we will filter out
        elements with `cols[i] < min_col_index`.  Will likely be 0 or 1, if
        set.  This is necessary if `col_indexes` may have values less than 0,
        or if you want to filter out 0 values (e.g. as representing blanks).

    Returns:
      Returns a torch.Tensor that is sparse with coo (coordinate) format,
      i.e. `layout=torch.sparse_coo` (which is actually the only sparse format
      that torch currently supports).
    '''
    assert rows.ndim == cols.ndim == 1
    assert rows.numel() == cols.numel() == values.numel()

    if min_col_index is not None:
        assert isinstance(min_col_index, int)
        kept_indexes = cols >= min_col_index
        rows = rows[kept_indexes]
        cols = cols[kept_indexes]
        values = values[kept_indexes]
    if size is not None:
        return torch.sparse_coo_tensor(torch.stack([rows, cols]),
                                       values,
                                       size=size,
                                       device=values.device,
                                       requires_grad=values.requires_grad)
    else:
        return torch.sparse_coo_tensor(torch.stack([rows, cols]),
                                       values,
                                       device=values.device,
                                       requires_grad=values.requires_grad)


def fsa_from_unary_function_tensor(src: Fsa, dest_arcs: _k2.RaggedArc,
                                   arc_map: torch.Tensor) -> Fsa:
    '''Create an Fsa object, including autograd logic and propagating
    properties from the source FSA.

    This is intended to be called from unary functions on FSAs where the arc_map
    is a Tensor of int32 (i.e. not ragged).

    Args:
      src:
        The source Fsa, i.e. the arg to the unary function.
      dest_arcs:
        The raw output of the unary function, as output by whatever C++
        algorithm we used.
      arc_map:
        A map from arcs in `dest_arcs` to the corresponding arc-index in `src`,
        or -1 if the arc had no source arc (e.g. added epsilon self-loops).
    Returns:
      Returns the resulting Fsa, with properties propagated appropriately, and
      autograd handled.
    '''
    dest = Fsa(dest_arcs)

    for name, value in src.named_tensor_attr(include_scores=False):
        if isinstance(value, torch.Tensor):
            filler = float(src.get_filler(name))
            setattr(dest, name, index_select(value, arc_map,
                                             default_value=filler))
        else:
            setattr(dest, name, index(value, arc_map))

    for name, value in src.named_non_tensor_attr():
        setattr(dest, name, value)

    k2.autograd_utils.phantom_index_select_scores(dest, src.scores, arc_map)
    return dest


def fsa_from_unary_function_ragged(src: Fsa, dest_arcs: _k2.RaggedArc,
                                   arc_map: _k2.RaggedInt,
                                   remove_filler: bool = True) -> Fsa:
    '''Create an Fsa object, including autograd logic and propagating
    properties from the source FSA.

    This is intended to be called from unary functions on FSAs where the arc_map
    is an instance of _k2.RaggedInt.

    Args:
      src:
        The source Fsa, i.e. the arg to the unary function.
      dest_arcs:
        The raw output of the unary function, as output by whatever C++
        algorithm we used.
      arc_map:
        A map from arcs in `dest_arcs` to the corresponding arc-index in `src`,
        or -1 if the arc had no source arc (e.g. :func:`remove_epsilon`).
      remove_filler:
        If true, for each attribute that is linear in `src` and ragged
        in the result, after turning it into a ragged tensor we will
        remove all items that are equal to the filler for that attribute
        (0 by default; see Fsa.get_filler()).  Attribute values on final-arcs
        that are equal to -1 will also be treated as fillers and removed,
        if remove_filler==True.
    Returns:
        Returns the resulting Fsa, with properties propagated appropriately, and
        autograd handled.
    '''
    dest = Fsa(dest_arcs)

    for name, value in src.named_tensor_attr(include_scores=False):
        if remove_filler and isinstance(value, torch.Tensor) and \
           value.dtype == torch.int32:
            filler = src.get_filler(name)
            # when removing fillers for `aux_labels`, we need to treat -1 as a
            # filler where it is on a final-arc (i.e. turn it into the actual
            # filler, so it will be later removed by remove_values_eq).  We
            # assume that `dest` has been checked for validity, so the presence
            # of -1 as the label precisely indicates final-arcs.
            if filler != -1:
                value = value.clone()
                if hasattr(torch, 'logical_and'):
                    # torch.logical_and requires torch>=1.5.0
                    value[torch.where(
                        torch.logical_and(src.labels == -1,
                                          value == -1))] = filler
                else:
                    value[torch.where((src.labels == -1) &
                                      (value == -1))] = filler
            new_value = index(value, arc_map)
            setattr(dest, name, k2.ragged.remove_values_eq(new_value, filler))
        else:
            new_value = index(value, arc_map)
            setattr(dest, name, new_value)

    for name, value in src.named_non_tensor_attr():
        setattr(dest, name, value)

    k2.autograd_utils.phantom_index_and_sum_scores(dest, src.scores, arc_map)

    return dest


def fsa_from_binary_function_tensor(
        a_fsa: Fsa,
        b_fsa: Fsa,
        dest_arcs: _k2.RaggedArc,
        a_arc_map: torch.Tensor,
        b_arc_map: torch.Tensor) -> Fsa:

    '''Create an Fsa object, including autograd logic and propagating
    properties from the source FSAs.

    This is intended to be called from binary functions on FSAs where the
    arc_map is a Tensor of int32 (i.e. not ragged).

    Caution: Only the attributes with dtype `torch.float32` will be merged,
             other kinds of attributes with the same name are discarded.

    Args:
      a_fsa:
        The source Fsa, i.e. the arg to the binary function.
      b_fsa:
        The other source Fsa.
      dest_arcs:
        The raw output of the binary function, as output by whatever C++
        algorithm we used.
      a_arc_map:
        A map from arcs in `dest_arcs` to the corresponding arc-index in `a_fsa`
        or -1 if the arc had no source arc (e.g. added epsilon self-loops).
      a_arc_map:
        A map from arcs in `dest_arcs` to the corresponding arc-index in `b_fsa`
        or -1 if the arc had no source arc (e.g. added epsilon self-loops).
    Returns:
      Returns the resulting Fsa, with properties propagated appropriately, and
      autograd handled.
    '''

    out_fsa = Fsa(dest_arcs)

    for name, a_value in a_fsa.named_tensor_attr():
        # we include 'scores' in the attributes; this enables the
        # autograd to work.
        filler = float(a_fsa.get_filler(name))
        if hasattr(b_fsa, name):
            # Both a_fsa and b_fsa have this attribute.
            # We only support attributes with dtype `torch.float32`.
            # Other kinds of attributes are discarded.
            if a_value.dtype != torch.float32:
                raise AttributeError("We don't support propagating two "
                                     "attributes with the same name that are "
                                     "not real-valued, in intersection: " +
                                     name)
            b_value = getattr(b_fsa, name)
            assert b_value.dtype == torch.float32
            # The following will actually overwrite `scores` with the same
            # value it had before; but this enables the autograd to work since
            # we do it using torch mechanisms.
            value = index_select(a_value, a_arc_map, default_value=filler) \
                    + index_select(b_value, b_arc_map, default_value=filler)
            setattr(out_fsa, name, value)
        else:
            # only a_fsa has this attribute, copy it via arc_map
            if isinstance(a_value, torch.Tensor):
                value = index_select(a_value, a_arc_map, default_value=filler)
            else:
                assert isinstance(a_value, _k2.RaggedInt)
                value = index_ragged(a_value, a_arc_map)
            setattr(out_fsa, name, value)

    for name, b_value in b_fsa.named_tensor_attr():
        if not hasattr(out_fsa, name):
            if isinstance(b_value, torch.Tensor):
                filler = float(b_fsa.get_filler(name))
                value = index_select(b_value, b_arc_map, default_value=filler)
            else:
                assert isinstance(b_value, _k2.RaggedInt)
                value = index_ragged(b_value, b_arc_map)
            setattr(out_fsa, name, value)

    for name, a_value in a_fsa.named_non_tensor_attr():
        setattr(out_fsa, name, a_value)

    for name, b_value in b_fsa.named_non_tensor_attr():
        if not hasattr(out_fsa, name):
            setattr(out_fsa, name, b_value)

    return out_fsa


def random_fsa(acyclic: bool = True,
               max_symbol: int = 50,
               min_num_arcs: int = 0,
               max_num_arcs: int = 1000) -> Fsa:
    '''Generate a random Fsa.

    Args:
      acyclic:
        If true, generated Fsa will be acyclic.
      max_symbol:
        Maximum symbol on arcs. Generated arc symbols will be in range
        [-1,max_symbol], note -1 is kFinalSymbol; must be at least 0;
       min_num_arcs:
         Minimum number of arcs; must be at least 0.
       max_num_arcs:
         Maximum number of arcs; must be >= min_num_arcs.
    '''
    random_arcs = _k2.random_fsa(acyclic, max_symbol, min_num_arcs,
                                 max_num_arcs)
    return Fsa(random_arcs)


def random_fsa_vec(min_num_fsas: int = 1,
                   max_num_fsas: int = 1000,
                   acyclic: bool = True,
                   max_symbol: int = 50,
                   min_num_arcs: int = 0,
                   max_num_arcs: int = 1000) -> Fsa:

    '''Generate a random FsaVec.

    Args:
      min_num_fsas:
        Minimum number of fsas we'll generated in the returned FsaVec;
        must be at least 1.
      max_num_fsas:
        Maximum number of fsas we'll generated in the returned FsaVec;
        must be >= min_num_fsas.
      acyclic:
        If true, generated Fsas will be acyclic.
      max_symbol:
        Maximum symbol on arcs. Generated arcs' symbols will be in range
        [-1,max_symbol], note -1 is kFinalSymbol; must be at least 0;
      min_num_arcs:
        Minimum number of arcs in each Fsa; must be at least 0.
      max_num_arcs:
        Maximum number of arcs in each Fsa; must be >= min_num_arcs.
    '''
    random_arcs = _k2.random_fsa_vec(min_num_fsas, max_num_fsas, acyclic,
                                     max_symbol, min_num_arcs, max_num_arcs)
    return Fsa(random_arcs)


def get_best_matching_stats(tokens: _k2.RaggedInt, scores: torch.Tensor,
                            counts: torch.Tensor, eos: int, min_token: int,
                            max_token: int, max_order: int
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa
    '''For "query" sentences, this function gets the mean and variance of
    scores from the best matching words-in-context in a set of provided "key"
    sentences. This matching process matches the word and the words preceding
    it, looking for the highest-order match it can find (it's intended for
    approximating the scores of models that see only left-context,
    like language models). The intended application is in estimating the scores
    of hypothesized transcripts, when we have actually computed the scores for
    only a subset of the hypotheses.

    CAUTION:
      This function only runs on CPU for now.

    Args:
      tokens:
        A ragged tensor of int32_t with 2 or 3 axes. If 2 axes, this represents
        a collection of key and query sequences. If 3 axes, this represents a
        set of such collections.

          2-axis example:
            [ [ the, cat, said, eos ], [ the, cat, fed, eos ] ]
          3-axis example:
            [ [ [ the, cat, said, eos ], [ the, cat, fed, eos ] ],
              [ [ hi, my, name, is, eos ], [ bye, my, name, is, eos ] ], ... ]

        where the words would actually be represented as integers,
        The eos symbol is required if this code is to work as intended
        (otherwise this code will not be able to recognize when we have reached
        the beginnings of sentences when comparing histories).
        bos symbols are allowed but not required.

      scores:
        A one dim torch.tensor with scores.size() == tokens.NumElements(),
        this is the item for which we are requesting best-matching values
        (as means and variances in case there are multiple best matches).
        In our anticipated use, these would represent scores of words in the
        sentences, but they could represent anything.
      counts:
        An one dim torch.tensor with counts.size() == tokens.NumElements(),
        containing 1 for words that are considered "keys" and 0 for
        words that are considered "queries".  Typically some entire
        sentences will be keys and others will be queries.
      eos:
        The value of the eos (end of sentence) symbol; internally, this
        is used as an extra padding value before the first sentence in each
        collection, so that it can act like a "bos" symbol.
      min_token:
        The lowest possible token value, including the bos
        symbol (e.g., might be -1).
      max_token:
        The maximum possible token value.  Be careful not to
        set this too large the implementation contains a part which
        takes time and space O(max_token - min_token).
      max_order:
        The maximum n-gram order to ever return in the
        `ngram_order` output; the output will be the minimum of max_order
        and the actual order matched; or max_order if we matched all the
        way to the beginning of both sentences. The main reason this is
        needed is that we need a finite number to return at the
        beginning of sentences.

    Returns:
      Returns a tuple of four torch.tensor (mean, var, counts_out, ngram_order)
        mean:
          For query positions, will contain the mean of the scores at the
          best matching key positions, or zero if that is undefined because
          there are no key positions at all.  For key positions,
          you can treat the output as being undefined (actually they
          are treated the same as queries, but won't match with only
          themselves because we don't match at singleton intervals).
        var:
          Like `mean`, but contains the (centered) variance
          of the best matching positions.
        counts_out:
          The number of key positions that contributed to the `mean`
          and `var` statistics.  This should only be zero if `counts`
          was all zero.
        ngram_order:
          The n-gram order corresponding to the best matching
          positions found at each query position, up to a maximum of
          `max_order`; will be `max_order` if we matched all
          the way to the beginning of a sentence.
    '''
    return _k2.get_best_matching_stats(tokens, scores, counts, eos,
                                       min_token, max_token, max_order)
