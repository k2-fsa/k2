# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#                      Xiaomi Corporation (authors: Haowen Qiu)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import Optional
from typing import Union

import torch

import _k2
from .fsa import Fsa
from .symbol_table import SymbolTable


def to_str(fsa: Fsa, openfst: bool = False) -> str:
    '''Convert an Fsa to a string.

    Note:
      The returned string can be used to construct an Fsa.

    Args:
      openfst:
        Optional. If true, we negate the scores during the conversion,

    Returns:
      A string representation of the Fsa.
    '''
    if hasattr(fsa, 'aux_labels'):
        aux_labels = fsa.aux_labels.to(torch.int32)
    else:
        aux_labels = None
    return _k2.fsa_to_str(fsa.arcs, openfst, aux_labels)


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
            if hasattr(fsa, 'aux_symbols'):
                aux_label = convert_aux_label_to_symbol(
                    aux_labels, i, fsa.aux_symbols)
            else:
                aux_label = convert_aux_label_to_symbol(aux_labels, i, None)
            aux_label = aux_label.replace('<eps>', 'ε')
        else:
            aux_label = ''

        if hasattr(fsa, 'symbols') and label != -1:
            label = fsa.symbols.get(label)
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
        value = torch.cat(values)
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
    return torch.sparse_coo_tensor(torch.stack([rows, cols]),
                                   values,
                                   device=values.device,
                                   requires_grad=values.requires_grad)
