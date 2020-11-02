# Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors

from typing import List
from typing import Optional

from .fsa import Fsa
from _k2 import _create_fsa_vec
from _k2 import _fsa_to_str
from _k2 import _fsa_to_tensor

import torch


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
    return _fsa_to_str(fsa.arcs, openfst, aux_labels)


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
      A ``torch.Tensor`` of dtype ``torch.int32``. It is a 2-D tensor
      if the input is a single FSA. It is a 1-D tensor if the input
      is a vector of FSAs.
    '''
    return _fsa_to_tensor(fsa.arcs)


def to_dot(fsa: Fsa, title: Optional[str] = None):
    '''Visualize an Fsa via graphviz.

    Note:
      The type hint for the return value is omitted.
      Graphviz is needed only when this function is called.

    Args:
      fsa:
        The input FSA to be visualized.

      title:
        Optional. The title of the resulting visualization.
    Returns:
      a Diagraph from grahpviz.
    '''
    from graphviz import Digraph
    assert len(fsa.shape) == 2, 'FsaVec is not supported'
    if hasattr(fsa, 'aux_labels'):
        aux_labels = fsa.aux_labels
        name = 'WFST'
    else:
        aux_labels = None
        name = 'WFSA'

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
    dot = Digraph(name=name, graph_attr=graph_attr)

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
            aux_label = int(aux_labels[i])
            if hasattr(fsa, 'aux_symbols') and aux_label != -1:
                aux_label = fsa.aux_symbols.get(aux_label)
                if aux_label == '<eps>':
                    aux_label = 'ε'
            aux_label = f':{aux_label}'
        else:
            aux_label = ''

        if hasattr(fsa, 'symbols') and label != -1:
            label = fsa.symbols.get(label)
            if label == '<eps>':
                label = 'ε'

        weight = f'{weight:.2f}'.rstrip('0').rstrip('.')
        dot.edge(src_state, dst_state, label=f'{label}{aux_label}/{weight}')
    return dot


def create_fsa_vec(fsas: List[Fsa]) -> Fsa:
    '''Create an FsaVec from a list of FSAs

    We use the following rules to set the attributes of the output FsaVec:

    - For tensor attributes, we assume that all input FSAs have the same attribute
    name and the values are concatenated.

    - For non-tensor attributes, if any two of the input FSAs have the same attribute
    name, then we assume that their attribute values are equal and the output FSA will
    inherit the attribute.

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

    ragged_arcs = _create_fsa_vec(ragged_arc_list)
    fsa_vec = Fsa.from_ragged_arc(ragged_arcs)

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
